"""
I/O Encoder: Text/number/DSL → Impulse with deterministic preprocessing.
"""

import numpy as np
import torch
from core.impulse import Impulse, ImpulseEncoder as BaseEncoder


class TextEncoder(BaseEncoder):
    """Enhanced text encoder."""
    
    def encode_text(self, text: str, energy: float = 1.0) -> Impulse:
        """Encode text to impulse."""
        return self.encode(text, energy)
    
    def encode_number(self, value: float, energy: float = 1.0) -> Impulse:
        """Encode number to impulse."""
        # Normalize number to text representation
        text = f"num:{value:.6f}"
        return self.encode(text, energy)
    
    def encode_dsl(self, dsl_expr: str, energy: float = 1.0) -> Impulse:
        """Encode DSL expression to impulse."""
        text = f"dsl:{dsl_expr}"
        return self.encode(text, energy)


class NumberEncoder:
    """Dedicated number encoder."""
    
    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def encode(self, value: float, energy: float = 1.0) -> Impulse:
        """
        Encode number to impulse.
        
        Args:
            value: Numeric value.
            energy: Initial energy.
        
        Returns:
            Impulse with V representing the number.
        """
        # Hash value to seed
        value_seed = int(hash(value)) & 0x7fffffff
        rng = np.random.RandomState(self.seed ^ value_seed)
        
        # Initialize V
        V = rng.randn(self.dim).astype(np.float32) * (1 + abs(value) / 10.0)
        
        # Percentile clipping
        p1, p99 = np.percentile(V, [1, 99])
        V = np.clip(V, p1, p99)
        
        # Normalize
        V_min, V_max = V.min(), V.max()
        if V_max - V_min > 1e-6:
            V = 2.0 * (V - V_min) / (V_max - V_min) - 1.0
        
        # L2 normalization
        norm = np.linalg.norm(V)
        if norm > 1e-6:
            V = V / norm
        
        # Clamp energy
        energy = max(0.0, min(energy, 1.5))
        
        # Context as uint64
        context_key = int(hash(value)) & 0xffffffffffffffff
        
        return Impulse(
            V=torch.from_numpy(V.astype(np.float32)),
            E=energy,
            T=tuple(),
            C=context_key,
            seed=self.seed ^ value_seed
        )


class DSLEncoder:
    """DSL expression encoder."""
    
    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def encode(self, expr: str, energy: float = 1.0) -> Impulse:
        """
        Encode DSL expression to impulse.
        
        Args:
            expr: DSL expression string.
            energy: Initial energy.
        
        Returns:
            Impulse with V representing expression.
        """
        # Parse DSL (simple tokenization)
        tokens = expr.split()
        
        # Initialize V from tokens
        V = np.zeros(self.dim, dtype=np.float32)
        for i, token in enumerate(tokens):
            token_hash = hash(token) & 0xffffffff
            idx = token_hash % self.dim
            V[idx] += (i + 1) / len(tokens)
        
        # Percentile clipping
        if np.any(V != 0):
            p1, p99 = np.percentile(V[V != 0], [1, 99])
            V = np.clip(V, p1, p99)
        
        # Normalize
        norm = np.linalg.norm(V) + 1e-8
        V = V / norm
        
        # Clamp energy
        energy = max(0.0, min(energy, 1.5))
        
        # Context as uint64
        context_key = int(hash(expr)) & 0xffffffffffffffff
        
        return Impulse(
            V=torch.from_numpy(V.astype(np.float32)),
            E=energy,
            T=tuple(),
            C=context_key,
            seed=self.seed ^ (hash(expr) & 0x7fffffff)
        )


# Re-export from impulse
from core.impulse import ImpulseDecoder

__all__ = [
    'TextEncoder',
    'NumberEncoder',
    'DSLEncoder',
    'ImpulseDecoder',
]
