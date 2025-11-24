"""
Learning: Distillation and RL Routing.

Features:
  - DistillationTeacher: Mock teacher for route distillation.
  - RLRoutingEnv: Gym-like environment for routing optimization.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from .impulse import Impulse, ImpulseEncoder

class DistillationTeacher:
    """
    Teacher model for route distillation.
    Provides target outputs or preferred routes.
    """
    
    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        
    def get_target(self, text_input: str) -> torch.Tensor:
        """
        Get target output vector for input.
        """
        # For now, we still use a random target, but return a Tensor.
        # In a real system, this would query a larger model (e.g. GPT-4).
        h = hash(text_input)
        rng = np.random.RandomState(h & 0xffffffff)
        target = rng.randn(self.dim).astype(np.float32)
        target = torch.tensor(target)
        target = target / torch.norm(target)
        return target
        
    def get_preferred_route(self, text_input: str) -> List[str]:
        """
        Get teacher's preferred route (oracle).
        """
        # Mock: return a fixed route based on input length
        if len(text_input) < 10:
            return ["node_A", "node_B"]
        else:
            return ["node_A", "node_C", "node_D"]

class RLRoutingEnv:
    """
    RL Environment for routing optimization.
    State: Impulse V (Tensor)
    Action: Next node selection
    Reward: -Energy + Success
    """
    
    def __init__(self, nodes: List[str], dim: int = 128):
        self.nodes = nodes
        self.dim = dim
        self.current_impulse: Optional[Impulse] = None
        self.steps = 0
        self.max_steps = 10
        self.encoder = ImpulseEncoder(dim=dim)
        
    def reset(self, text_input: str = "start") -> torch.Tensor:
        """Reset env with new task."""
        self.current_impulse = self.encoder.encode(text_input)
        self.steps = 0
        return self.current_impulse.V
        
    def step(self, action_node_idx: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take step (route to node).
        """
        if self.current_impulse is None:
            raise RuntimeError("Call reset first")
            
        node_id = self.nodes[action_node_idx]
        
        # Simulate processing (energy decay)
        cost = 0.1
        self.current_impulse = self.current_impulse.decay(cost)
        
        # In a real system, we'd call the actual node (Crystal)
        # Here we just simulate state change with noise for the RL env
        noise = torch.randn(self.dim) * 0.01
        new_V = self.current_impulse.V + noise
        new_V = new_V / torch.norm(new_V)
        
        # Update impulse V
        self.current_impulse = Impulse(
            V=new_V,
            E=self.current_impulse.E,
            T=self.current_impulse.T, 
            C=self.current_impulse.C,
            seed=self.current_impulse.seed
        )
        
        self.steps += 1
        done = not self.current_impulse.is_alive or self.steps >= self.max_steps
        
        # Reward: -Energy cost + Bonus if "solved" (mock)
        reward = -cost
        if done and self.current_impulse.E > 0:
            reward += 1.0 # Survival bonus
            
        return self.current_impulse.V, reward, done, {"node": node_id}

