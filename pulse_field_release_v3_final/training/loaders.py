import random
from typing import List, Dict

class DataLoader:
    """
    Mock data loader for synthetic tasks.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.tasks = [
            "qa", "summarization", "reasoning", "code"
        ]
        
    def get_batch(self, batch_size: int = 32) -> List[Dict]:
        """Generate a batch of synthetic tasks."""
        batch = []
        for _ in range(batch_size):
            task_type = random.choice(self.tasks)
            item = self._generate_item(task_type)
            batch.append(item)
        return batch
        
    def _generate_item(self, task_type: str) -> Dict:
        if task_type == "qa":
            return {
                "type": "qa",
                "input": "Context: The cat sat on the mat. Question: Where is the cat?",
                "target": "on the mat"
            }
        elif task_type == "summarization":
            return {
                "type": "summarization",
                "input": "Long text about a topic... " * 10,
                "target": "Summary of topic."
            }
        elif task_type == "reasoning":
            a, b = random.randint(1, 10), random.randint(1, 10)
            return {
                "type": "reasoning",
                "input": f"Solve: {a} + {b} =",
                "target": str(a + b)
            }
        elif task_type == "code":
            return {
                "type": "code",
                "input": "def add(a, b): return a + b",
                "target": "pass"
            }
        return {}

class Tokenizer:
    """Simple whitespace tokenizer."""
    def encode(self, text: str) -> List[int]:
        return [hash(w) % 10000 for w in text.split()]
        
    def decode(self, tokens: List[int]) -> str:
        return " ".join([str(t) for t in tokens])
