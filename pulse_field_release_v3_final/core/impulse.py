"""
Impulse: V (vector), E (energy), T (trace), C (context) state carrier.

Impulses flow through the Pulse-Field system carrying:
  - V: torch.Tensor, normalized to [-1, 1], L2-stabilized.
  - E: energy scalar in [0, E_max], decays per step.
  - T: topological trace list of decisions and defects.
  - C: context key (uint64 or centroid) for archive addressing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ImpulseStep:
    """Single step in trace: node activation, decision, defect."""
    node_id: str
    time: int
    route_tag: str
    defect: float
    decision: Dict


@dataclass(frozen=True)
class Impulse:
    """
    Immutable state carrier through Pulse-Field.
    
    Contract:
      V: torch.Tensor, normalized to [-1, 1].
      E: float in [0, E_max], decays per step.
      T: list of ImpulseStep.
      C: uint64 or float32 array (archive key).
      seed: int for reproducibility.
    """
    V: torch.Tensor        # Shape (m,), dtype float32, requires_grad=True during training
    E: float               # Energy in [0, E_max]
    T: Tuple[ImpulseStep, ...]  # Immutable trace
    C: Union[int, np.ndarray]   # Context key
    seed: int              # Global seed for reproducibility
    H: Optional[torch.Tensor] = None # Hidden state for SSM/LRU context
    
    # Metadata
    _meta: Dict = field(default_factory=dict, repr=False, compare=False)
    
    def __post_init__(self):
        """Validate impulse invariants."""
        # V normalization
        if not isinstance(self.V, torch.Tensor):
            raise TypeError("V must be torch.Tensor")
        
        # E bounds
        if not 0 <= self.E <= 1.5:
            raise ValueError(f"E must be in [0, 1.5], got {self.E}")
    
    @property
    def dim(self) -> int:
        """Dimensionality of V."""
        return self.V.shape[0]
    
    @property
    def is_alive(self) -> bool:
        """True if E > tau_energy (0.05)."""
        return self.E > 0.05
    
    def decay(self, cost: float, e_max: float = 1.5) -> 'Impulse':
        """
        Decay energy by cost; clamp to [0, E_max].
        
        Returns:
            New Impulse with decayed energy.
        """
        new_E = max(0.0, min(self.E - cost, e_max))
        return Impulse(
            V=self.V,
            E=new_E,
            T=self.T,
            C=self.C,
            seed=self.seed,
            H=self.H,
            _meta=dict(self._meta)
        )
    
    def log_step(
        self,
        node_id: str,
        time: int,
        route_tag: str,
        defect: float,
        decision: Dict
    ) -> 'Impulse':
        """
        Append step to trace; return new Impulse.
        """
        step = ImpulseStep(
            node_id=node_id,
            time=time,
            route_tag=route_tag,
            defect=defect,
            decision=decision
        )
        new_trace = self.T + (step,)
        return Impulse(
            V=self.V,
            E=self.E,
            T=new_trace,
            C=self.C,
            seed=self.seed,
            H=self.H,
            _meta=dict(self._meta)
        )
    
    def total_defect(self) -> float:
        """Sum of all defects in trace."""
        return sum(step.defect for step in self.T)
    
    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "V": self.V.detach().cpu().numpy().tolist(),
            "E": float(self.E),
            "T": [
                {
                    "node_id": step.node_id,
                    "time": step.time,
                    "route_tag": step.route_tag,
                    "defect": step.defect,
                    "decision": step.decision,
                }
                for step in self.T
            ],
            "C": int(self.C) if isinstance(self.C, (int, np.integer)) else str(self.C),
            "seed": self.seed,
            "H": self.H.detach().cpu().numpy().tolist() if self.H is not None else None,
        }


class ImpulseEncoder(nn.Module):
    """
    Encodes text into Impulse using a learnable embedding.
    """
    
    def __init__(self, vocab_size: int = 1000, dim: int = 128, seed: int = 42):
        """
        Initialize encoder.
        
        Args:
            vocab_size: Size of vocabulary.
            dim: Dimensionality of V.
            seed: Random seed.
        """
        super().__init__()
        self.dim = dim
        self.seed = seed
        torch.manual_seed(seed)
        
        # Real learnable embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # SSM Parameters for Sequence Encoding
        # h_t = decay * h_{t-1} + (1-decay) * x_t
        self.ssm_decay = nn.Parameter(torch.tensor(0.9))
        
        # Simple tokenizer (hash-based for simplicity, but maps to vocab indices)
        self.vocab_size = vocab_size
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Simple hash-based tokenizer for demo purposes."""
        # In a real system, use a proper tokenizer (BPE, etc.)
        # Here we just map characters/words to indices
        indices = [hash(word) % self.vocab_size for word in text.split()]
        if not indices:
            indices = [0]
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, text: str, energy: float = 1.0) -> Impulse:
        """
        Encode text into Impulse using SSM Recurrence.
        """
        indices = self.tokenize(text)
        embeds = self.embedding(indices) # (Seq, Dim)
        
        # SSM / LRU Recurrence
        # This replaces Mean Pooling (Bag of Words)
        # h_t = decay * h_{t-1} + (1-decay) * x_t
        
        h = torch.zeros(self.dim, device=embeds.device)
        decay = torch.sigmoid(self.ssm_decay) # Ensure in [0, 1]
        
        # Linear Scan (O(N))
        # Note: For production, use parallel scan (associative scan)
        for x_t in embeds:
            h = decay * h + (1 - decay) * x_t
            
        V = torch.tanh(h) # Squash to [-1, 1]
        
        # Context key
        context_key = int(hash(text)) & 0xffffffffffffffff
        
        return Impulse(
            V=V,
            E=energy,
            T=tuple(),
            C=context_key,
            seed=self.seed,
            H=h # Carry the final hidden state
        )
    
    # Compatibility alias
    def encode(self, text: str, energy: float = 1.0) -> Impulse:
        return self.forward(text, energy)
    
    def quarantine_outlier(self, V: torch.Tensor, sigma: float = 3.0) -> Tuple[bool, torch.Tensor]:
        """
        Detect if V is an outlier.
        """
        # Simplified for tensor
        return False, V


class ImpulseDecoder(nn.Module):
    """
    Decodes Impulse back to text (or logits).
    """
    
    def __init__(self, dim: int = 128, vocab_size: int = 1000, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.linear = nn.Linear(dim, vocab_size)
    
    def forward(self, impulse: Impulse) -> torch.Tensor:
        """
        Decode Impulse to logits.
        """
        return self.linear(impulse.V)

    def decode(self, impulse: Impulse) -> str:
        """
        Decode Impulse to text (mock implementation for compatibility).
        """
        # In a real system, this would sample from the logits
        logits = self.forward(impulse)
        probs = torch.softmax(logits, dim=0)
        top_idx = torch.argmax(probs).item()
        return f"[Token ID: {top_idx}]"
    
    def decode_structured(self, impulse: Impulse) -> Dict:
        """
        Decode Impulse to structured output.
        """
        return {
            "V": impulse.V.detach().cpu().numpy().tolist(),
            "E": float(impulse.E),
            "trace_length": len(impulse.T),
            "total_defect": float(impulse.total_defect()),
            "is_alive": impulse.is_alive,
            "route_tags": list(set(step.route_tag for step in impulse.T)),
        }
