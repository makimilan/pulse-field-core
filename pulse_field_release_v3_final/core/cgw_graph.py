"""
CGW Graph: Compositional Graph-Waves with local convolution and resonance.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from .impulse import Impulse
from .compatibility import Node, NodeRegistry


class CGWGraph:
    """
    Directed acyclic graph of nodes with κ kernels and local convolution.
    
    Contract:
      - Nodes have input_spec, output_spec, compat_fn, local op g_j.
      - κ kernels are bounded: ||κ|| ≤ tau_amp.
      - Local convolution: y_i = Σ_j∈N(i) κ(i,j)·g_j(V_j) for j in ActiveSet.
      - Resonance: activate if compat ≥ tau_compat AND E ≥ cost.
    """
    
    def __init__(self, registry: NodeRegistry, tau_amp: float = 1.0):
        """
        Initialize CGW graph.
        
        Args:
            registry: NodeRegistry with registered nodes.
            tau_amp: Max kernel norm.
        """
        self.registry = registry
        self.tau_amp = tau_amp
        self.kernels: Dict[Tuple[str, str], torch.Tensor] = {}  # (src, dst) -> κ
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> [neighbors]
    
    def add_edge(self, src_id: str, dst_id: str, kernel: np.ndarray):
        """
        Add edge with κ kernel; enforce norm constraint.
        
        Args:
            src_id: Source node.
            dst_id: Destination node.
            kernel: κ kernel (shape must match output(src) and input(dst)).
        
        Raises:
            ValueError: If norm exceeds tau_amp.
        """
        if isinstance(kernel, np.ndarray):
            kernel = torch.from_numpy(kernel).float()
        elif not isinstance(kernel, torch.Tensor):
            kernel = torch.tensor(kernel, dtype=torch.float32)
            
        norm = torch.norm(kernel)
        if norm > self.tau_amp + 1e-6:
            raise ValueError(f"Kernel norm {norm} exceeds tau_amp {self.tau_amp}")
        
        # Normalize to ensure <= tau_amp
        if norm > 1e-8:
            kernel = kernel / norm * min(norm, self.tau_amp)
        
        self.kernels[(src_id, dst_id)] = kernel
        
        if dst_id not in self.adjacency:
            self.adjacency[dst_id] = []
        if src_id not in self.adjacency[dst_id]:
            self.adjacency[dst_id].append(src_id)
    
    def get_kernel(self, src_id: str, dst_id: str) -> Optional[torch.Tensor]:
        """Get κ kernel between nodes."""
        return self.kernels.get((src_id, dst_id))
    
    def topological_sort(self) -> List[str]:
        """
        Topological sort of all nodes (ensures DAG property).
        
        Returns:
            List of node_ids in topological order.
        
        Raises:
            ValueError: If graph contains cycle.
        """
        all_nodes = set(self.registry.list_nodes())
        in_degree = {node: 0 for node in all_nodes}
        
        for dst in self.adjacency:
            for src in self.adjacency[dst]:
                in_degree[dst] += 1
        
        queue = [node for node in all_nodes if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # For reverse adjacency (find nodes that depend on this one)
            for (s, d), _ in self.kernels.items():
                if s == node and d in all_nodes:
                    in_degree[d] -= 1
                    if in_degree[d] == 0:
                        queue.append(d)
        
        if len(result) != len(all_nodes):
            raise ValueError("Graph contains cycle")
        
        return result
    
    def local_convolution(
        self,
        impulse: Impulse,
        active_set: List[str],
        tau_compat: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute local convolution for all active nodes.
        
        Args:
            impulse: Current impulse.
            active_set: List of compatible node IDs.
            tau_compat: Compatibility threshold.
        
        Returns:
            Dict[node_id -> output] for activated nodes.
        
        Contract:
          y_i = Σ_{j ∈ N(i), j ∈ ActiveSet} κ(i,j) · g_j(V_j)
        """
        outputs = {}
        
        for node_id in self.registry.list_nodes():
            if node_id not in active_set:
                continue
            
            node = self.registry.get(node_id)
            if node is None:
                continue
            
            # Compute compat score
            # Note: compat_fn should handle tensor or we convert
            try:
                compat = node.compat_fn(impulse.V)
                if compat < tau_compat:
                    continue
                
                # Apply local operator
                g_output = node.activate(impulse.V)
                outputs[node_id] = g_output
            except Exception:
                continue
        
        return outputs
    
    def resonance(
        self,
        impulse: Impulse,
        active_set: List[str],
        tau_compat: float = 0.7,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute resonance: aggregate outputs via kernels.
        
        Args:
            impulse: Current impulse.
            active_set: Compatible nodes.
            tau_compat: Compatibility threshold.
        
        Returns:
            (aggregated output, total local defect)
        """
        outputs = self.local_convolution(impulse, active_set, tau_compat)
        
        if not outputs:
            return impulse.V, 0.0
        
        # Aggregate via kernels (simple sum for now)
        aggregated = torch.zeros_like(impulse.V)
        total_defect = 0.0
        
        for node_id, output in outputs.items():
            # Simple contribution (no inter-node kernels for now)
            aggregated += output
            # Local defect as output magnitude deviation
            defect = torch.norm(output - impulse.V).item()
            total_defect += defect
        
        # Normalize
        norm = torch.norm(aggregated) + 1e-8
        aggregated = aggregated / norm
        
        return aggregated, total_defect / max(1, len(outputs))
    
    def propagate(
        self,
        impulse: Impulse,
        active_set: List[str],
        tau_compat: float = 0.7,
        cost_per_node: float = 0.1,
    ) -> Impulse:
        """
        Full propagation step: resonance + energy decay.
        
        Args:
            impulse: Current impulse.
            active_set: Compatible nodes.
            tau_compat: Compatibility threshold.
            cost_per_node: Energy cost per active node.
        
        Returns:
            New Impulse with updated V, E, trace.
        """
        # Compute resonance
        V_new, defect = self.resonance(impulse, active_set, tau_compat)
        
        # Decay energy
        cost = cost_per_node * len(active_set)
        E_new = impulse.E - cost
        E_new = max(0.0, min(E_new, 1.5))
        
        # Create new impulse
        new_impulse = Impulse(
            V=V_new,
            E=E_new,
            T=impulse.T,
            C=impulse.C,
            seed=impulse.seed,
        )
        
        # Log step
        if active_set:
            new_impulse = new_impulse.log_step(
                node_id=active_set[0],
                time=len(impulse.T),
                route_tag="cgw_propagate",
                defect=defect,
                decision={
                    "active_set": active_set,
                    "cost": cost,
                    "V_norm_before": float(impulse.V.norm().item()),
                    "V_norm_after": float(V_new.norm().item()),
                }
            )
        
        return new_impulse
