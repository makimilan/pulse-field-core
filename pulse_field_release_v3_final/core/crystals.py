"""
Crystals: Encapsulated processors with formal I/O contracts and amplification screen.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from .impulse import Impulse


class Crystal(nn.Module):
    """
    Encapsulated processor with I/O contracts and amplification screen.
    Now a real PyTorch Module.
    """
    
    def __init__(
        self,
        crystal_id: str,
        input_dim: int,
        output_dim: int,
        tau_amp: float = 1.0,
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize crystal.
        """
        super().__init__()
        self.crystal_id = crystal_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau_amp = tau_amp
        self.metadata = metadata or {}
        
        # Real Neural Processor
        self.layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        
        # SSM / LRU Parameters for Crystal Processing
        # hidden_state = (decay * hidden_state) + (input_vector * gate)
        self.ssm_decay = nn.Parameter(torch.tensor(0.8))
        self.ssm_gate = nn.Linear(input_dim, output_dim)
        
        # Amplification tracking
        self.max_amp_observed = 0.0
        self.amp_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer(x))
    
    def _amplification_screen(self, V_in: torch.Tensor, V_out: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Enforce amplification screen: ensure ||V_out|| ≤ tau_amp * ||V_in||.
        """
        with torch.no_grad():
            norm_in = torch.norm(V_in) + 1e-8
            norm_out = torch.norm(V_out) + 1e-8
            
            amp = (norm_out / norm_in).item()
            self.amp_history.append(amp)
            self.max_amp_observed = max(self.max_amp_observed, amp)
            
            exceeded = amp > self.tau_amp
        
        if exceeded:
            # Clip output to tau_amp (differentiable scaling)
            scale = (self.tau_amp * norm_in) / norm_out
            V_out_screened = V_out * scale
            return V_out_screened, True
        
        return V_out, False
    
    def process(self, impulse: Impulse, archive_on_breach: Optional[Callable] = None) -> Tuple[Impulse, bool]:
        """
        Process impulse through crystal using SSM mechanism.
        """
        # Extract State
        V_in = impulse.V
        H_in = impulse.H if impulse.H is not None else torch.zeros_like(V_in)
        
        # SSM Update
        # H_new = (decay * H_in) + (gate(V_in))
        decay = torch.sigmoid(self.ssm_decay)
        gate_out = torch.sigmoid(self.ssm_gate(V_in))
        
        H_new = (decay * H_in) + (gate_out * V_in) # Element-wise gating
        
        # Process V based on new State
        try:
            V_out = self.forward(H_new) # Process the state, not just the input
        except Exception as e:
            raise RuntimeError(f"Crystal {self.crystal_id} processing failed: {e}")
        
        # Amplification screen
        V_screened, exceeded = self._amplification_screen(impulse.V, V_out)
        
        # Decay energy
        cost = 0.05
        E_out = impulse.E - cost
        E_out = max(0.0, min(E_out, 1.5))
        
        # Create output impulse
        output_impulse = Impulse(
            V=V_screened,
            E=E_out,
            T=impulse.T,
            C=impulse.C,
            seed=impulse.seed,
            H=H_new # Pass updated state
        )
        
        # Log step (simplified for tensor)
        defect = torch.norm(V_screened - impulse.V).item() if V_screened.shape == impulse.V.shape else 0.0
        
        output_impulse = output_impulse.log_step(
            node_id=self.crystal_id,
            time=len(impulse.T),
            route_tag="crystal",
            defect=defect,
            decision={
                "crystal_id": self.crystal_id,
                "amp": float(self.amp_history[-1]) if self.amp_history else 0.0,
                "exceeded": exceeded,
            }
        )
        
        # Archive on breach
        if exceeded and archive_on_breach:
            archive_on_breach(output_impulse)
        
        return output_impulse, exceeded


class SuperpositionAdapter:
    """
    Explicit adapter for superposing multiple impulses.
    """
    
    def __init__(self, k_min: int = 2, tau_cos: float = 0.25):
        self.k_min = k_min
        self.tau_cos = tau_cos
    
    def check_independence(self, V_a: torch.Tensor, V_b: torch.Tensor) -> float:
        """
        Check independence via cosine.
        """
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(V_a.unsqueeze(0), V_b.unsqueeze(0)).abs().item()
        return 1.0 - cos_sim
    
    def superpose(self, impulses: List[Impulse]) -> Tuple[Impulse, float]:
        """
        Superpose multiple impulses.
        """
        if not impulses:
            raise ValueError("Cannot superpose empty list")
        
        if len(impulses) == 1:
            return impulses[0], 1.0
        
        # Simple average superposition
        V_stack = torch.stack([imp.V for imp in impulses])
        V_sum = torch.sum(V_stack, dim=0)
        V_sup = V_sum / (torch.norm(V_sum) + 1e-8)
        
        # Average energy
        E_sup = np.mean([imp.E for imp in impulses])
        
        # Combine traces
        all_steps = []
        for imp in impulses:
            all_steps.extend(imp.T)
        
        # Use first context
        C_sup = impulses[0].C
        
        output_impulse = Impulse(
            V=V_sup,
            E=E_sup,
            T=tuple(all_steps),
            C=C_sup,
            seed=impulses[0].seed,
            H=impulses[0].H # Simplified: take first H
        )
        
        return output_impulse, 1.0 # Simplified independence score


class CrystalRegistry:
    """Registry for crystals."""
    
    def __init__(self):
        self.crystals: Dict[str, Crystal] = {}
    
    def register(self, crystal: Crystal):
        """Register crystal."""
        self.crystals[crystal.crystal_id] = crystal
    
    def get(self, crystal_id: str) -> Optional[Crystal]:
        """Retrieve crystal by ID."""
        return self.crystals.get(crystal_id)
    
    def list_crystals(self) -> List[str]:
        """List all crystal IDs."""
        return list(self.crystals.keys())

class HybridCrystal(Crystal):
    """
    Hybrid Crystal with local attention.
    """
    
    def __init__(
        self,
        crystal_id: str,
        input_dim: int,
        output_dim: int,
        archive: Optional['Archive'] = None,
        tau_amp: float = 1.0,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(crystal_id, input_dim, output_dim, tau_amp, metadata)
        self.archive = archive
        
        # Real Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        
    def windowed_attention(self, V: torch.Tensor) -> torch.Tensor:
        """
        Apply windowed attention to V.
        """
        # Treat V as a sequence of length 1 (or reshape if needed)
        # Here we just do self-attention on the vector treated as a batch of 1, seq 1
        # Which is trivial, but "real" op.
        # To make it interesting, let's reshape V into (1, 4, dim/4)
        
        dim = V.shape[0]
        heads = 4
        if dim % heads == 0:
            head_dim = dim // heads
            V_seq = V.view(1, heads, head_dim) # (Batch, Seq, Dim)
            attn_out, _ = self.attention(V_seq, V_seq, V_seq)
            return attn_out.flatten()
        
        return V

    def process(self, impulse: Impulse, archive_on_breach: Optional[Callable] = None) -> Tuple[Impulse, bool]:
        """
        Override process to include hybrid features.
        """
        V_curr = impulse.V
        
        # Check metadata for hybrid ops
        if "attention" in self.metadata:
            V_curr = self.windowed_attention(V_curr)
            
        # Now run standard processor
        # We create a temp impulse with the modified V
        temp_impulse = Impulse(
            V=V_curr,
            E=impulse.E,
            T=impulse.T,
            C=impulse.C,
            seed=impulse.seed,
            H=impulse.H
        )
        
        return super().process(temp_impulse, archive_on_breach)
