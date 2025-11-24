"""
Compatibility field and node selection for Pulse-Field.

Selects active nodes based on compatibility scoring and ActiveSet constraints.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Union
import numpy as np
import torch


@dataclass
class CompatibilityScore:
    """Score and metadata for node compatibility."""
    node_id: str
    score: float
    rank: int
    metadata: Dict


class CompatibilityField:
    """
    Computes compatible node set (ActiveSet) for a given Impulse.
    
    Contract:
      Input: V (vector), tau_compat (threshold), n_max (capacity)
      Output: ActiveSet of node_ids with compat ≥ tau_compat, |ActiveSet| ≤ n_max
      Invariant: I1 (selective activation)
    """
    
    def __init__(self, nodes: Dict[str, 'Node'], tau_compat: float = 0.7, n_max: int = 16):
        """
        Initialize compatibility field.
        
        Args:
            nodes: Dict[node_id -> Node] registry.
            tau_compat: Compatibility threshold.
            n_max: Max concurrent active nodes.
        """
        self.nodes = nodes
        self.tau_compat = tau_compat
        self.n_max = n_max
    
    def select_active_set(self, V: Union[np.ndarray, torch.Tensor], context_key: Optional[int] = None) -> List[str]:
        """
        Select active nodes based on compatibility.
        
        Args:
            V: Input vector.
            context_key: Optional context for caching.
        
        Returns:
            List of node_ids with compat ≥ tau_compat, sorted by score descending.
        
        Invariant:
          - I1: Only nodes with compat ≥ tau_compat are included.
          - Capacity: |ActiveSet| ≤ n_max.
        """
        scores = []
        for node_id in self.nodes.list_nodes():
            node = self.nodes.get(node_id)
            if node is None:
                continue
            try:
                compat = node.compat_fn(V)
                if compat >= self.tau_compat:
                    scores.append((node_id, compat))
            except Exception:
                # Skip nodes that fail compatibility check
                continue
        
        # Sort by compatibility score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Enforce capacity
        active_set = [node_id for node_id, _ in scores[:self.n_max]]
        
        return active_set
    
    def get_compatibility_scores(self, V: Union[np.ndarray, torch.Tensor]) -> List[CompatibilityScore]:
        """
        Get all compatibility scores (not just active set).
        
        Args:
            V: Input vector.
        
        Returns:
            List of CompatibilityScore for all nodes.
        """
        scores = []
        compat_list = []
        
        for node_id in self.nodes.list_nodes():
            node = self.nodes.get(node_id)
            if node is None:
                continue
            try:
                compat = node.compat_fn(V)
                compat_list.append((node_id, compat))
            except Exception:
                compat_list.append((node_id, 0.0))
        
        # Rank by score
        compat_list.sort(key=lambda x: x[1], reverse=True)
        for rank, (node_id, compat) in enumerate(compat_list):
            scores.append(CompatibilityScore(
                node_id=node_id,
                score=compat,
                rank=rank,
                metadata={"tau_compat": self.tau_compat, "above_threshold": compat >= self.tau_compat}
            ))
        
        return scores


class Node:
    """
    Base node in CGW graph with compatibility and local operator.
    
    Contract:
      - input_spec: Dict with 'shape', 'dtype', 'tags'
      - output_spec: Dict with 'shape', 'dtype', 'tags'
      - compat_fn: (V: np.ndarray) -> score ∈ [0, 1]
      - g: local operator (V) -> bounded_output
      - cost: energy cost to activate
    """
    
    def __init__(
        self,
        node_id: str,
        input_spec: Dict,
        output_spec: Dict,
        compat_fn: Callable[[Union[np.ndarray, torch.Tensor]], float],
        g: Callable[[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]],
        cost: float = 0.1,
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize node.
        
        Args:
            node_id: Unique identifier.
            input_spec: Input contract {shape, dtype, tags}.
            output_spec: Output contract {shape, dtype, tags}.
            compat_fn: Compatibility scorer (V) -> [0, 1].
            g: Local operator (V) -> output.
            cost: Energy cost to activate.
            metadata: Additional metadata.
        """
        self.node_id = node_id
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.compat_fn = compat_fn
        self.g = g
        self.cost = max(0.0, min(cost, 1.0))
        self.metadata = metadata or {}
    
    def validate_input(self, V: Union[np.ndarray, torch.Tensor]) -> bool:
        """Check if V matches input_spec."""
        if "shape" in self.input_spec:
            if tuple(V.shape) != tuple(self.input_spec["shape"]):
                return False
        # Relaxed dtype check for tensor/numpy interop
        return True
    
    def validate_output(self, output: Union[np.ndarray, torch.Tensor]) -> bool:
        """Check if output matches output_spec."""
        if "shape" in self.output_spec:
            if tuple(output.shape) != tuple(self.output_spec["shape"]):
                return False
        return True
    
    def activate(self, V: Union[np.ndarray, torch.Tensor], context: Optional[Dict] = None) -> Union[np.ndarray, torch.Tensor]:
        """
        Activate node; compute local operator.
        
        Args:
            V: Input vector.
            context: Optional context metadata.
        
        Returns:
            Output from local operator g(V).
        
        Raises:
            ValueError: If input does not match spec.
        """
        if not self.validate_input(V):
            raise ValueError(f"Input does not match spec for {self.node_id}")
        
        try:
            output = self.g(V)
            if not self.validate_output(output):
                raise ValueError(f"Output does not match spec for {self.node_id}")
            return output
        except Exception as e:
            raise RuntimeError(f"Node {self.node_id} activation failed: {e}")


class NodeRegistry:
    """Registry and factory for nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
    
    def register(self, node_id: str, node: Node):
        """Register node in registry."""
        self.nodes[node_id] = node
    
    def get(self, node_id: str) -> Optional[Node]:
        """Retrieve node by ID."""
        return self.nodes.get(node_id)
    
    def list_nodes(self) -> List[str]:
        """List all registered node IDs."""
        return list(self.nodes.keys())
    
    def get_by_tag(self, tag: str) -> List[Node]:
        """Get all nodes with given semantic tag."""
        result = []
        for node in self.nodes.values():
            if tag in node.metadata.get("tags", []):
                result.append(node)
        return result
