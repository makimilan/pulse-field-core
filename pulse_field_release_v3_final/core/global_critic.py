"""
Global Critic: Rare alignment signal and energy governance.

Features:
  - Rare global signal (every N tasks).
  - Alignment checks: misaligned local success vs global failure.
  - Energy penalties for cancerous clusters.
"""

from typing import List, Dict, Set
import numpy as np
from .impulse import Impulse

class GlobalCritic:
    """
    Global alignment critic.
    
    Contract:
      - Evaluate alignment every N tasks.
      - Penalize nodes involved in misaligned outcomes.
      - Enforce system-wide integrity.
    """
    
    def __init__(self, alignment_interval: int = 10, penalty_factor: float = 0.5):
        """
        Initialize critic.
        
        Args:
            alignment_interval: Check alignment every N tasks.
            penalty_factor: Energy penalty multiplier (0.0-1.0).
        """
        self.alignment_interval = alignment_interval
        self.penalty_factor = penalty_factor
        self.task_counter = 0
        self.history: List[Impulse] = []
        self.penalized_nodes: Set[str] = set()
    
    def observe(self, impulse: Impulse):
        """
        Observe completed task impulse.
        """
        self.history.append(impulse)
        self.task_counter += 1
        
        if self.task_counter >= self.alignment_interval:
            self._evaluate_alignment()
            self.task_counter = 0
            self.history.clear()
            
    def _evaluate_alignment(self):
        """
        Evaluate alignment of recent history.
        """
        # Simple heuristic: if average defect is high but some nodes claim low local defect,
        # they might be "lying" or misaligned.
        
        # Calculate global average defect
        global_defects = [imp.total_defect() for imp in self.history]
        avg_global_defect = np.mean(global_defects) if global_defects else 0.0
        
        # Identify misaligned nodes
        # Misalignment: Node reports low defect (in trace) but global outcome is bad (high defect)
        
        misaligned_candidates = {}
        
        for imp, global_d in zip(self.history, global_defects):
            if global_d > 0.5: # "Bad" outcome
                for step in imp.T:
                    # If node reported very low defect locally
                    if step.defect < 0.01:
                        misaligned_candidates[step.node_id] = misaligned_candidates.get(step.node_id, 0) + 1
                        
        # Apply penalties to frequent offenders
        threshold = len(self.history) * 0.3 # 30% of bad tasks involved
        for node_id, count in misaligned_candidates.items():
            if count > threshold:
                self.penalize(node_id)
                
    def penalize(self, node_id: str):
        """
        Apply penalty to node.
        """
        self.penalized_nodes.add(node_id)
        # In a real system, this would reduce the node's energy budget or compatibility
        
    def is_penalized(self, node_id: str) -> bool:
        """Check if node is penalized."""
        return node_id in self.penalized_nodes
        
    def get_energy_modifier(self, node_id: str) -> float:
        """Get energy cost modifier for node."""
        if node_id in self.penalized_nodes:
            return 1.0 + self.penalty_factor
        return 1.0

