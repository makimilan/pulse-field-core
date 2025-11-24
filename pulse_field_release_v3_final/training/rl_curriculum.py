import random
from typing import Dict, List

class RLCurriculum:
    """
    Reinforcement Learning Curriculum for task selection.
    """
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.scores = {t: 0.5 for t in tasks} # Initial confidence
        self.epsilon = 0.2 # Exploration rate
        
    def select_task(self) -> str:
        """Select next task based on learning progress."""
        if random.random() < self.epsilon:
            return random.choice(self.tasks)
            
        # Select task with lowest score (hardest) to focus on
        # Or highest score to reinforce? 
        # Usually we want to focus on what we are bad at, but not too bad.
        # Let's pick the one with lowest score to improve it.
        return min(self.scores, key=self.scores.get)
        
    def update(self, task: str, success: bool, defect: float):
        """Update task score."""
        # Simple moving average
        alpha = 0.1
        reward = 1.0 if success else 0.0
        # Penalize high defect
        reward -= defect
        
        current = self.scores[task]
        self.scores[task] = (1 - alpha) * current + alpha * reward
