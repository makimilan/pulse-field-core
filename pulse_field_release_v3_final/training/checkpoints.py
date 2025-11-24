import os
import json
import time
from typing import Dict, Any

class CheckpointManager:
    """
    Manage archive snapshots and training state.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save(self, step: int, state: Dict[str, Any], metrics: Dict[str, float]):
        """Save checkpoint."""
        filename = os.path.join(self.checkpoint_dir, f"ckpt_{step}.json")
        data = {
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics,
            # In real scenario, we'd save the archive state here
            # For now, we save metadata
            "archive_size": state.get("archive_size", 0)
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
            
    def load(self, step: int) -> Dict[str, Any]:
        """Load checkpoint."""
        filename = os.path.join(self.checkpoint_dir, f"ckpt_{step}.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        return {}
        
    def list_checkpoints(self):
        """List all checkpoints."""
        files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("ckpt_")]
        return sorted(files)
