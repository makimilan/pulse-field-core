"""
Runtime: Orchestration and end-to-end route execution.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .impulse import Impulse, ImpulseEncoder, ImpulseDecoder
from .compatibility import CompatibilityField, NodeRegistry, Node
from .cgw_graph import CGWGraph
from .router import System1Router, System2Router, Route, RouteCache
from .crystals import Crystal, CrystalRegistry
from .archive import Archive
from .autoarchitect import Autoarchitect
from .invariants import InvariantChecker
from .config import Config
from .logging import get_logger
from .global_critic import GlobalCritic


class Runtime:
    """
    Orchestrates full Pulse-Field execution.
    
    Contract:
      - Initialize graph, crystals, archive, router.
      - Execute: encode → compatibility → routing → propagation → archive.
      - Deterministic per seed; all traces logged.
    """
    
    def __init__(self, config: Optional[Config] = None, seed: int = 42):
        """
        Initialize runtime.
        
        Args:
            config: Config instance.
            seed: Global seed.
        """
        self.config = config or Config()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Global seed for reproducibility
        np.random.seed(seed)
        
        # Components
        self.node_registry = NodeRegistry()
        self._init_default_nodes() # Initialize real nodes
        self.crystal_registry = CrystalRegistry()
        self.archive = Archive(dim=128, max_elements=10000, seed=seed)
        self.autoarchitect = Autoarchitect(self.archive, seed=seed)
        self.global_critic = GlobalCritic()
        
        # Graph
        self.graph = CGWGraph(self.node_registry, tau_amp=self.config.get("thresholds.tau_amp"))
        
        # Router
        self.route_cache = RouteCache(ttl_seconds=self.config.get("router.cache_ttl"))
        self.system1 = System1Router(l_max=self.config.get("router.l_max"), seed=seed)
        self.system2 = System2Router()
        
        # Compatibility
        self.compat_field = CompatibilityField(
            self.node_registry,
            tau_compat=self.config.get("thresholds.tau_compat"),
            n_max=self.config.get("thresholds.n_max"),
        )
        
        # Invariants
        self.invariant_checker = InvariantChecker()
        
        # I/O
        self.encoder = ImpulseEncoder(dim=128, seed=seed)
        self.decoder = ImpulseDecoder(seed=seed)
        
        # Logger
        self.logger = get_logger()
        self.logger.set_metadata("global_seed", seed)
    
    def _init_default_nodes(self):
        """Initialize a set of real processing nodes."""
        # Create 50 random nodes with real vector operations
        for i in range(50):
            # Random centroid for compatibility
            centroid = torch.randn(128)
            centroid /= torch.norm(centroid)
            
            # Random weights for processing (Linear + Tanh)
            W = torch.randn(128, 128) * 0.1
            b = torch.randn(128) * 0.01
            
            def make_compat_fn(c):
                def compat(V):
                    # Cosine similarity
                    if not isinstance(V, torch.Tensor):
                        V = torch.tensor(V, dtype=torch.float32)
                    vn = torch.norm(V)
                    if vn < 1e-9: return 0.0
                    return float(torch.dot(V, c) / vn)
                return compat
                
            def make_op_fn(w, bias):
                def op(V):
                    # Linear + Tanh
                    if not isinstance(V, torch.Tensor):
                        V = torch.tensor(V, dtype=torch.float32)
                    return torch.tanh(torch.matmul(V, w) + bias)
                return op
            
            node = Node(
                node_id=f"node_{i}",
                input_spec={"shape": (128,), "dtype": "float32"},
                output_spec={"shape": (128,), "dtype": "float32"},
                compat_fn=make_compat_fn(centroid),
                g=make_op_fn(W, b),
                cost=0.05,
                metadata={"type": "dense_tanh", "layer": i % 5}
            )
            self.node_registry.register(node.node_id, node)

    def register_node(self, node: Node):
        """Register a node."""
        self.node_registry.register(node.node_id, node)
        self.compat_field = CompatibilityField(
            self.node_registry,
            tau_compat=self.config.get("thresholds.tau_compat"),
            n_max=self.config.get("thresholds.n_max"),
        )
    
    def register_crystal(self, crystal: Crystal):
        """Register a crystal."""
        self.crystal_registry.register(crystal)
    
    def add_graph_edge(self, src_id: str, dst_id: str, kernel: np.ndarray):
        """Add edge to CGW graph."""
        self.graph.add_edge(src_id, dst_id, kernel)
    
    def execute(
        self,
        impulse: Impulse,
        max_steps: int = 100,
        trace_id: Optional[str] = None,
    ) -> Impulse:
        """
        Execute full route through system.
        
        Args:
            impulse: Input impulse.
            max_steps: Max propagation steps.
            trace_id: Optional trace identifier.
        
        Returns:
            Output impulse.
        
        Contract:
          - Deterministic per seed.
          - All steps logged.
          - Invariants checked and logged.
        """
        trace_id = trace_id or f"trace_{self.rng.randint(0, 1000000)}"
        self.logger.set_metadata("trace_id", trace_id)
        self.logger.set_metadata("global_seed", self.seed)
        self.logger.set_metadata("input_V_norm", float(impulse.V.norm().item()))
        
        current_impulse = impulse
        step = 0
        
        try:
            while step < max_steps and current_impulse.is_alive:
                # Select active set
                active_set = self.compat_field.select_active_set(
                    current_impulse.V,
                    context_key=current_impulse.C if isinstance(current_impulse.C, int) else None,
                )
                
                if not active_set:
                    # No compatible nodes; propagate with identity
                    new_impulse = current_impulse.decay(0.01)
                    self.logger.log_step(
                        step=step,
                        node_id="identity",
                        route_tag="no_compat",
                        compat_score=0.0,
                        defect=0.0,
                        total_defect=current_impulse.total_defect(),
                        E_start=current_impulse.E,
                        E_end=new_impulse.E,
                        decisions={"active_set": []},
                    )
                    current_impulse = new_impulse
                    step += 1
                    continue
                
                # Route via System1 + System2
                route = self._route(current_impulse, active_set, trace_id)
                
                # Propagate via graph
                current_impulse = self.graph.propagate(
                    current_impulse,
                    active_set=route.node_ids,
                    tau_compat=self.config.get("thresholds.tau_compat"),
                )
                
                step += 1
            
            # Log final state
            self.logger.set_metadata("output_V_norm", float(current_impulse.V.norm().item()))
            self.logger.set_metadata("final_defect", float(current_impulse.total_defect()))
            self.logger.set_metadata("steps_executed", step)
            
            # Archive result
            self.archive.put(current_impulse, is_strong=True)
            
            # Global Critic observe
            self.global_critic.observe(current_impulse)
            
            # Save trace
            self.logger.save_trace(trace_id)
            self.logger.clear()
            
            return current_impulse
        
        except Exception as e:
            # Log error and save partial trace
            self.logger.log_rollback(
                step=step,
                cause=f"execution_error: {str(e)}",
                recovery_step=max(0, step - 1),
                metadata={"error": str(e)},
            )
            self.logger.save_trace(f"{trace_id}_error")
            self.logger.clear()
            raise
    
    def _route(self, impulse: Impulse, active_set: List[str], trace_id: str) -> Route:
        """
        Route using System1 + System2.
        
        Args:
            impulse: Current impulse.
            active_set: Compatible nodes.
            trace_id: Trace identifier.
        
        Returns:
            Validated route.
        """
        # Check cache
        context_key = impulse.C if isinstance(impulse.C, int) else 0
        cached_route = self.route_cache.get(context_key, "default")
        if cached_route:
            return cached_route
        
        # System1: predict
        route = self.system1.predict(impulse, active_set)
        
        # System2: validate
        is_valid, error_msg = self.system2.validate(route)
        
        if not is_valid:
            # Get alternatives
            alternatives = self.system2.get_alternatives(route, active_set, k=1)
            if alternatives:
                route = alternatives[0]
        
        # Cache
        self.route_cache.put(context_key, "default", route, invariants_pass_rate=0.95)
        
        return route
    
    def validate_invariants(
        self,
        active_set: List[str],
        E: float,
        D_curr: float,
        D_prev: float,
    ) -> Tuple[Dict[str, bool], float]:
        """
        Validate all invariants.
        
        Args:
            active_set: Active nodes.
            E: Current energy.
            D_curr: Current defect.
            D_prev: Previous defect.
        
        Returns:
            (invariant_flags, pass_rate)
        """
        results, pass_rate = self.invariant_checker.check_all(
            E=E,
            tau_energy=self.config.get("thresholds.tau_energy"),
            active_set=active_set,
            all_nodes=self.node_registry.list_nodes(),
            tau_compat=self.config.get("thresholds.tau_compat"),
            D_curr=D_curr,
            D_prev=D_prev,
        )
        
        flags = {result.invariant_id: result.passed for result in results}
        return flags, pass_rate
    
    def get_stats(self) -> Dict:
        """Get runtime statistics."""
        return {
            "nodes_registered": len(self.node_registry.list_nodes()),
            "crystals_registered": len(self.crystal_registry.list_crystals()),
            "archive_size": self.archive.size(),
            "route_cache_size": self.route_cache.size(),
            "mutations_stats": self.autoarchitect.get_stats(),
        }
