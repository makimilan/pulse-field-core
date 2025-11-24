"""
Archive: HNSW-based fuzzy memory with versioning, rollback, and GC.

Features:
  - HNSW KNN addressing (fuzzy centroids).
  - Versioning: Strong/Weak partial order.
  - Rollback: Safe state restoration.
  - GC/Merge: Reconcile contracts for redundancy reduction.
"""

import time
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import logging

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False

from .impulse import Impulse, ImpulseStep

logger = logging.getLogger(__name__)

@dataclass
class ArchiveEntry:
    """Single entry in the archive."""
    id: int
    V: np.ndarray
    C_key: np.ndarray  # Centroid key
    impulse_data: Dict  # Serialized impulse
    version: int
    timestamp: float
    is_strong: bool = True
    access_count: int = 0
    
    def similarity(self, other_V: np.ndarray) -> float:
        """Cosine similarity."""
        norm_a = np.linalg.norm(self.V)
        norm_b = np.linalg.norm(other_V)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return np.dot(self.V, other_V) / (norm_a * norm_b)

class Archive:
    """
    Robust Archive with HNSW index and versioning.
    """
    
    def __init__(self, dim: int = 128, max_elements: int = 10000, seed: int = 42):
        self.dim = dim
        self.max_elements = max_elements
        self.seed = seed
        self.entries: Dict[int, ArchiveEntry] = {}
        self.next_id = 0
        
        # HNSW Index
        self.has_hnsw = HAS_HNSW
        if self.has_hnsw:
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
            self.index.set_ef(50)
        else:
            # logger.warning("hnswlib not found, falling back to linear scan.")
            self.index = None

    def put(self, impulse: Impulse, is_strong: bool = True) -> int:
        """
        Store impulse in archive.
        
        Returns:
            Entry ID.
        """
        # Create entry
        entry_id = self.next_id
        self.next_id += 1
        
        # Use V as key for HNSW
        if isinstance(impulse.V, torch.Tensor):
            V_flat = impulse.V.detach().cpu().numpy().astype(np.float32)
        else:
            V_flat = np.array(impulse.V, dtype=np.float32)
        
        # Context key: if C is int, convert to vector or use V
        if isinstance(impulse.C, (int, np.integer)):
            # If C is int, we might use V as the geometric key
            C_key = V_flat
        else:
            C_key = impulse.C.astype(np.float32)

        entry = ArchiveEntry(
            id=entry_id,
            V=V_flat,
            C_key=C_key,
            impulse_data=impulse.to_dict(),
            version=1,
            timestamp=time.time(),
            is_strong=is_strong
        )
        
        self.entries[entry_id] = entry
        
        if self.has_hnsw:
            try:
                self.index.add_items(V_flat.reshape(1, -1), np.array([entry_id]))
            except Exception as e:
                logger.error(f"HNSW add failed: {e}")
                pass
                
        return entry_id

    def get(self, query_impulse: Impulse, k: int = 5) -> List[Tuple[Impulse, float]]:
        """
        Retrieve k-nearest neighbors.
        
        Returns:
            List of (Impulse, distance).
        """
        if not self.entries:
            return []
            
        if isinstance(query_impulse.V, torch.Tensor):
            query_vec = query_impulse.V.detach().cpu().numpy().astype(np.float32)
        else:
            query_vec = np.array(query_impulse.V, dtype=np.float32)
        
        if self.has_hnsw and self.index.get_current_count() > 0:
            try:
                labels, distances = self.index.knn_query(query_vec.reshape(1, -1), k=min(k, len(self.entries)))
                final_results = []
                for label, dist in zip(labels[0], distances[0]):
                    if label in self.entries:
                        entry = self.entries[label]
                        entry.access_count += 1
                        imp = self._reconstruct_impulse(entry.impulse_data)
                        final_results.append((imp, float(dist)))
                return final_results
            except Exception as e:
                logger.error(f"HNSW query failed: {e}")
                return self._linear_scan(query_vec, k)
        else:
            return self._linear_scan(query_vec, k)

    def _linear_scan(self, query_vec: np.ndarray, k: int) -> List[Tuple[Impulse, float]]:
        """Fallback linear scan."""
        scores = []
        for eid, entry in self.entries.items():
            # Cosine distance = 1 - similarity
            sim = entry.similarity(query_vec)
            dist = 1.0 - sim
            scores.append((eid, dist))
        
        scores.sort(key=lambda x: x[1])
        top_k = scores[:k]
        
        results = []
        for eid, dist in top_k:
            entry = self.entries[eid]
            entry.access_count += 1
            imp = self._reconstruct_impulse(entry.impulse_data)
            results.append((imp, dist))
        return results

    def _reconstruct_impulse(self, data: Dict) -> Impulse:
        """Helper to reconstruct Impulse from dict."""
        trace = []
        for step_data in data["T"]:
            trace.append(ImpulseStep(
                node_id=step_data["node_id"],
                time=step_data["time"],
                route_tag=step_data["route_tag"],
                defect=step_data["defect"],
                decision=step_data["decision"]
            ))
            
        return Impulse(
            V=torch.tensor(data["V"], dtype=torch.float32),
            E=data["E"],
            T=tuple(trace),
            C=data["C"] if isinstance(data["C"], int) else np.array(data["C"], dtype=np.float32),
            seed=data["seed"]
        )

    def rollback(self, impulse: Impulse) -> Optional[Impulse]:
        """
        Find previous version of this impulse (by C_key continuity).
        """
        candidates = self.get(impulse, k=5)
        for cand_imp, dist in candidates:
            if dist < 0.01: # Very close
                return cand_imp
        return None

    def gc_merge(self, cosine_threshold: float = 0.95) -> int:
        """
        Garbage collect and merge similar entries.
        """
        merged = 0
        ids_to_remove = []
        
        # Naive O(N^2) for now as HNSW doesn't support easy deletion/traversal
        # In production, use a separate structure for GC candidates
        
        keys = list(self.entries.keys())
        for i in range(len(keys)):
            if keys[i] in ids_to_remove: continue
            entry_a = self.entries[keys[i]]
            
            for j in range(i + 1, len(keys)):
                if keys[j] in ids_to_remove: continue
                entry_b = self.entries[keys[j]]
                
                sim = entry_a.similarity(entry_b.V)
                if sim > cosine_threshold:
                    # Merge: keep newer/stronger
                    if entry_a.timestamp > entry_b.timestamp:
                        ids_to_remove.append(keys[j])
                    else:
                        ids_to_remove.append(keys[i])
                        break # entry_a removed
                    merged += 1
                    
        for eid in ids_to_remove:
            del self.entries[eid]
            # Note: HNSW deletion is hard, usually we just mark as deleted or rebuild index
            
        return merged
        
    def size(self) -> int:
        return len(self.entries)
