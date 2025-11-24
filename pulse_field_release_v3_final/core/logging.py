"""
Structured JSON logging for Pulse-Field execution traces.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class StructuredLogger:
    """
    Structured JSON logging with per-step traces and metadata.
    
    Logs:
      - Seeds (global, step).
      - Decisions (node_id, compat_score, route_tag).
      - Defects (δ_i, D_sum).
      - Energies (E_start, E_decay, E_end).
      - Rollbacks (cause, recovery).
      - Invariant flags (I1–I11 status).
    """
    
    def __init__(self, log_dir: str = "reports/routes_traces", enabled: bool = True):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for trace files.
            enabled: Enable logging.
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.current_trace: List[Dict] = []
        self.metadata: Dict[str, Any] = {}
        
        if enabled and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def log_step(
        self,
        step: int,
        node_id: str,
        route_tag: str,
        compat_score: float,
        defect: float,
        total_defect: float,
        E_start: float,
        E_end: float,
        decisions: Dict,
        invariant_flags: Optional[Dict[str, bool]] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Log single execution step.
        
        Args:
            step: Step number.
            node_id: Activated node.
            route_tag: Route identifier.
            compat_score: Compatibility score.
            defect: Node defect δ_i.
            total_defect: Cumulative defect D.
            E_start: Energy at start.
            E_end: Energy at end.
            decisions: Decision metadata.
            invariant_flags: Dict[invariant_id -> passed].
            metadata: Additional metadata.
        """
        if not self.enabled:
            return
        
        step_entry = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": node_id,
            "route_tag": route_tag,
            "compatibility": {
                "score": float(compat_score),
            },
            "defect": {
                "node_defect": float(defect),
                "total_defect": float(total_defect),
            },
            "energy": {
                "E_start": float(E_start),
                "E_end": float(E_end),
                "E_decay": float(E_start - E_end),
            },
            "decisions": decisions,
            "invariants": invariant_flags or {},
            "metadata": metadata or {},
        }
        
        self.current_trace.append(step_entry)
    
    def log_rollback(
        self,
        step: int,
        cause: str,
        recovery_step: int,
        metadata: Optional[Dict] = None,
    ):
        """
        Log rollback event.
        
        Args:
            step: Step number when rollback occurred.
            cause: Reason for rollback.
            recovery_step: Step to rollback to.
            metadata: Additional context.
        """
        if not self.enabled:
            return
        
        rollback_entry = {
            "type": "rollback",
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "cause": cause,
            "recovery_step": recovery_step,
            "metadata": metadata or {},
        }
        
        self.current_trace.append(rollback_entry)
    
    def set_metadata(self, key: str, value: Any):
        """Set global trace metadata."""
        if not self.enabled:
            return
        self.metadata[key] = value
    
    def save_trace(self, trace_id: str):
        """
        Save current trace to JSON file.
        
        Args:
            trace_id: Identifier for trace file.
        """
        if not self.enabled:
            return
        
        trace_data = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": self.metadata,
            "steps": self.current_trace,
        }
        
        filename = f"{self.log_dir}/{trace_id}.json"
        with open(filename, 'w') as f:
            json.dump(trace_data, f, indent=2)
    
    def clear(self):
        """Clear current trace."""
        self.current_trace = []
        self.metadata = {}
    
    def get_current_trace(self) -> List[Dict]:
        """Get current trace."""
        return self.current_trace.copy()


# Global logger instance
_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get or create global logger."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger


def configure_logger(log_dir: str = "reports/routes_traces", enabled: bool = True):
    """Configure global logger."""
    global _logger
    _logger = StructuredLogger(log_dir=log_dir, enabled=enabled)
