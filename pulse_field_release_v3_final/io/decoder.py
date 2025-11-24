"""
I/O Decoder: Impulse → text/structured outputs.
"""

from core.impulse import ImpulseDecoder as BaseDecoder
from core.impulse import Impulse
from typing import Dict, Any
import numpy as np
import torch


class TextDecoder(BaseDecoder):
    """Decode impulse to text."""
    
    def decode_text(self, impulse: Impulse) -> str:
        """Decode to natural text."""
        return self.decode(impulse)
    
    def decode_verbose(self, impulse: Impulse) -> str:
        """Verbose text output with all details."""
        parts = []
        parts.append("=== Impulse Output ===")
        parts.append(f"Energy: {impulse.E:.4f}")
        parts.append(f"Total Defect: {impulse.total_defect():.6f}")
        parts.append(f"Vector Norm: {impulse.V.norm().item():.6f}")
        parts.append(f"Trace Length: {len(impulse.T)}")
        
        if impulse.T:
            parts.append("\nTrace:")
            for step in impulse.T[-3:]:  # Last 3 steps
                parts.append(f"  - {step.node_id} @ {step.time}: δ={step.defect:.4f}")
        
        return "\n".join(parts)


class StructuredDecoder:
    """Decode impulse to structured data."""
    
    def decode_dict(self, impulse: Impulse) -> Dict[str, Any]:
        """Decode to dictionary."""
        return impulse.to_dict()
    
    def decode_metrics(self, impulse: Impulse) -> Dict[str, Any]:
        """Decode to metrics."""
        return {
            "energy_final": float(impulse.E),
            "total_defect": float(impulse.total_defect()),
            "vector_norm": float(impulse.V.norm().item()),
            "is_alive": impulse.is_alive,
            "trace_length": len(impulse.T),
            "avg_defect_per_step": float(impulse.total_defect() / max(1, len(impulse.T))),
        }
    
    def decode_summary(self, impulse: Impulse) -> str:
        """Decode to summary text."""
        summary = {
            "status": "alive" if impulse.is_alive else "exhausted",
            "energy": f"{impulse.E:.3f}",
            "defect": f"{impulse.total_defect():.6f}",
            "steps": len(impulse.T),
        }
        
        items = [f"{k}: {v}" for k, v in summary.items()]
        return " | ".join(items)


# Re-exports
__all__ = [
    'TextDecoder',
    'StructuredDecoder',
]
