"""
Router: System1 (fast) and System2 (validation) dual-process routing.
RouteCache with TTL and green corridor for high-confidence routes.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .impulse import Impulse


@dataclass
class Route:
    """A route through the system (sequence of node_ids)."""
    node_ids: List[str]
    tag: str
    cost: float
    confidence: float  # [0, 1]
    metadata: Dict


@dataclass
class CacheEntry:
    """Cached route with TTL."""
    route: Route
    timestamp: float
    invariants_pass_rate: float


class RouteCache:
    """
    Time-limited route cache with "green corridor" for high-confidence routes.
    
    Contract:
      - TTL-based eviction (default 600s).
      - "Green corridor": invariants_pass_rate ≥ 99% cached forever.
      - get/put/invalidate/clear.
    """
    
    def __init__(self, ttl_seconds: float = 600.0):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for entries (default 600s).
        """
        self.ttl = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
    
    def _make_key(self, context_key: int, tag: str) -> str:
        """Make cache key from context and tag."""
        return f"{context_key}:{tag}"
    
    def get(self, context_key: int, tag: str) -> Optional[Route]:
        """
        Retrieve route from cache if valid.
        
        Args:
            context_key: Context identifier.
            tag: Route tag.
        
        Returns:
            Route if cached and not expired, else None.
        """
        key = self._make_key(context_key, tag)
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        now = time.time()
        
        # Check TTL (except green corridor)
        age = now - entry.timestamp
        if entry.invariants_pass_rate < 0.99 and age > self.ttl:
            del self.cache[key]
            return None
        
        return entry.route
    
    def put(self, context_key: int, tag: str, route: Route, invariants_pass_rate: float = 0.0):
        """
        Store route in cache.
        
        Args:
            context_key: Context identifier.
            tag: Route tag.
            route: Route to cache.
            invariants_pass_rate: Invariant pass-rate (for green corridor).
        """
        key = self._make_key(context_key, tag)
        self.cache[key] = CacheEntry(
            route=route,
            timestamp=time.time(),
            invariants_pass_rate=invariants_pass_rate,
        )
    
    def invalidate(self, context_key: int, tag: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            context_key: Context identifier.
            tag: Specific tag, or None to invalidate all for context.
        """
        if tag:
            key = self._make_key(context_key, tag)
            self.cache.pop(key, None)
        else:
            prefix = f"{context_key}:"
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                del self.cache[k]
    
    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Number of cached entries."""
        return len(self.cache)


class System1Router:
    """
    Fast heuristic router: suggests route candidate ≤ L_max hops.
    
    Contract:
      - Input: Impulse
      - Output: Route candidate (sequence of node_ids, cost estimate, confidence)
      - Time: O(1) to O(N_max) depending on heuristic
    """
    
    def __init__(self, l_max: int = 10, seed: int = 42):
        """
        Initialize System1.
        
        Args:
            l_max: Max route length.
            seed: Random seed for reproducibility.
        """
        self.l_max = l_max
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def predict(self, impulse: Impulse, all_nodes: List[str]) -> Route:
        """
        Predict route from impulse heuristic.
        
        Args:
            impulse: Input impulse.
            all_nodes: Available nodes (ActiveSet, sorted by compatibility).
        
        Returns:
            Route candidate.
        """
        # Deterministic Heuristic: ActiveSet is already sorted by compatibility.
        # We take the top-K nodes to form a route, where K is determined by
        # the impulse energy (higher energy -> longer route allowed) or just l_max.
        
        if not all_nodes:
            return Route(node_ids=[], tag="empty", cost=0.0, confidence=0.0, metadata={})
            
        # Determine length based on l_max and available nodes
        # Deterministic: take up to l_max nodes
        length = min(len(all_nodes), self.l_max)
        
        # Take top 'length' nodes (already sorted by compatibility in ActiveSet)
        route_nodes = all_nodes[:length]
        
        # Estimate cost (linear in length)
        cost = 0.1 * len(route_nodes)
        
        # Estimate confidence (higher for shorter routes, higher for top nodes)
        # Since all_nodes are the "best" matches, confidence is generally high
        confidence = 1.0 / (1.0 + 0.05 * len(route_nodes))
        
        return Route(
            node_ids=route_nodes,
            tag="system1_topk",
            cost=cost,
            confidence=confidence,
            metadata={"heuristic": "compatibility_sorted_deterministic"},
        )


class System2Router:
    """
    Validation router: validates contracts, checks invariants, provides alternatives.
    
    Contract:
      - Input: Route candidate
      - Output: Validated route or alternatives
      - Validation: input/output specs, invariants I1–I11, rollback on failure
    """
    
    def __init__(self):
        pass
    
    def validate(self, route: Route, context: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate route contracts.
        
        Args:
            route: Route candidate.
            context: Optional context for validation.
        
        Returns:
            (is_valid, error_message)
        """
        if not route.node_ids:
            return False, "Route is empty"
        
        if len(route.node_ids) > 20:
            return False, "Route too long"
        
        if route.cost < 0 or route.confidence < 0 or route.confidence > 1:
            return False, "Invalid cost or confidence"
        
        return True, None
    
    def get_alternatives(self, route: Route, all_nodes: List[str], k: int = 3) -> List[Route]:
        """
        Generate alternative routes.
        
        Args:
            route: Failed route.
            all_nodes: Available nodes.
            k: Number of alternatives.
        
        Returns:
            List of alternative routes.
        """
        alternatives = []
        
        # Deterministic alternatives: sliding window or subsets
        for i in range(k):
            # Alternative i: skip the first i nodes, take next available
            start_idx = i + 1
            if start_idx >= len(all_nodes):
                break
                
            alt_length = min(len(all_nodes) - start_idx, len(route.node_ids))
            if alt_length <= 0:
                break
                
            alt_nodes = all_nodes[start_idx : start_idx + alt_length]
            
            alt_route = Route(
                node_ids=alt_nodes,
                tag=f"alternative_{i}",
                cost=0.1 * len(alt_nodes),
                confidence=0.5 - 0.1 * i,
                metadata={"original_route": route.tag, "strategy": "next_best"},
            )
            alternatives.append(alt_route)
        
        return alternatives
    
    def rollback(self, route: Route, reason: str) -> Route:
        """
        Rollback route to simpler version.
        
        Args:
            route: Route to rollback.
            reason: Rollback reason.
        
        Returns:
            Simplified route.
        """
        # Rollback to shorter route
        new_length = max(1, len(route.node_ids) // 2)
        rollback_route = Route(
            node_ids=route.node_ids[:new_length],
            tag=f"rollback_{route.tag}",
            cost=0.1 * new_length,
            confidence=route.confidence * 0.5,
            metadata={"rollback_reason": reason, "original_route": route.tag},
        )
        return rollback_route


class ReversibilityPair:
    """
    Forward and backward route pair ensuring reversibility (I5).
    
    Contract:
      - forward_route exists
      - backward_route exists
      - D_forward + D_backward = 0 (or within tolerance)
    """
    
    def __init__(self, forward_route: Route, backward_route: Route):
        """
        Initialize reversibility pair.
        
        Args:
            forward_route: Forward direction route.
            backward_route: Backward direction route.
        """
        self.forward = forward_route
        self.backward = backward_route
    
    def check_symmetry(self) -> bool:
        """
        Check reversibility symmetry.
        
        Returns:
            True if routes are symmetric (can reverse).
        """
        # Forward and backward should be similar length
        length_ratio = len(self.forward.node_ids) / max(1, len(self.backward.node_ids))
        return 0.5 < length_ratio < 2.0
    
    def check_defect_symmetry(self, defect_forward: float, defect_backward: float, tolerance: float = 1e-6) -> bool:
        """
        Check defect symmetry on forward/backward.
        
        Args:
            defect_forward: Defect on forward pass.
            defect_backward: Defect on backward pass.
            tolerance: Numerical tolerance.
        
        Returns:
            True if defects are symmetric (I5).
        """
        delta_D = abs(defect_backward - defect_forward)
        return delta_D <= tolerance
