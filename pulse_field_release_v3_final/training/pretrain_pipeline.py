import time
import numpy as np
from typing import Dict, List
from pulse_field.core import Runtime, Config
from pulse_field.training.loaders import DataLoader
from pulse_field.training.metrics import Metrics
from pulse_field.training.checkpoints import CheckpointManager
from pulse_field.training.rl_curriculum import RLCurriculum
from pulse_field.training.evaluators import QAEvaluator, SummarizationEvaluator, ReasoningEvaluator, CodeEvaluator

class PretrainPipeline:
    """
    Full pretraining pipeline with RL curriculum and multi-task evaluation.
    """
    def __init__(self):
        self.config = Config()
        self.runtime = Runtime(self.config)
        self.loader = DataLoader({})
        self.checkpoints = CheckpointManager()
        self.metrics = Metrics()
        
        self.tasks = ["qa", "summarization", "reasoning", "code"]
        self.curriculum = RLCurriculum(self.tasks)
        
        self.evaluators = {
            "qa": QAEvaluator(),
            "summarization": SummarizationEvaluator(),
            "reasoning": ReasoningEvaluator(),
            "code": CodeEvaluator()
        }
        
        self.history = {
            "step": [],
            "defect": [],
            "accuracy": [],
            "rouge": [],
            "bleu": [],
            "ppl": []
        }
        self.step = 0
        
    def train(self, total_steps: int = 1000):
        print(f"Starting Pretraining for {total_steps} steps...")
        
        for i in range(total_steps):
            # 1. Select task via RL
            task_type = self.curriculum.select_task()
            
            # 2. Get data
            # We need to force the loader to give us this task type
            # For now, we'll just filter a batch
            batch = self.loader.get_batch(10)
            item = next((x for x in batch if x["type"] == task_type), batch[0])
            
            # 3. Forward pass
            start_time = time.time()
            impulse = self.runtime.encoder.encode(item["input"])
            output = self.runtime.execute(impulse, max_steps=10)
            
            # 4. Calculate Metrics (Real)
            # Use the actual defect from the output impulse
            current_defect = output.total_defect()
            
            # Decode output to text for evaluation
            decoded_text = self.runtime.decoder.decode(output)
            target_text = item["target"]
            
            # Calculate real accuracy/metrics based on task
            if task_type == "qa":
                accuracy = self.evaluators["qa"].evaluate(decoded_text, target_text)
            elif task_type == "summarization":
                metrics = self.evaluators["summarization"].evaluate(decoded_text, target_text)
                accuracy = metrics.get("rouge_l", 0.0) # Use ROUGE as proxy for accuracy
            elif task_type == "reasoning":
                accuracy = self.evaluators["reasoning"].evaluate(decoded_text, target_text)
            elif task_type == "code":
                accuracy = self.evaluators["code"].evaluate(decoded_text, target_text)
            else:
                accuracy = 0.0
            
            # Ensure accuracy is a float
            if isinstance(accuracy, dict):
                accuracy = float(accuracy.get("accuracy", 0.0))
            elif not isinstance(accuracy, (int, float)):
                accuracy = 0.0

            # 5. Update Curriculum
            success = accuracy > 0.8
            self.curriculum.update(task_type, success, current_defect)
            
            # 6. Log
            self.history["step"].append(self.step)
            self.history["defect"].append(current_defect)
            self.history["accuracy"].append(accuracy)
            self.history["ppl"].append(Metrics.perplexity(current_defect))
            
            if task_type == "summarization":
                # Real metrics if available
                self.history["rouge"].append(metrics.get("rouge_l", 0.0) if 'metrics' in locals() else 0.0)
                self.history["bleu"].append(metrics.get("bleu", 0.0) if 'metrics' in locals() else 0.0)
            else:
                self.history["rouge"].append(0.0)
                self.history["bleu"].append(0.0)
                
            if i % 100 == 0:
                print(f"Step {self.step}: Task={task_type}, Defect={current_defect:.4f}, Acc={accuracy:.4f}, PPL={self.history['ppl'][-1]:.2f}")
                
            if i % 500 == 0:
                self.checkpoints.save(
                    self.step,
                    {"archive_size": self.runtime.archive.size()},
                    {"defect": current_defect, "accuracy": accuracy}
                )
        
        # Save history to JSON for reporting
        import json
        import os
        os.makedirs("pulse_field/reports", exist_ok=True)
        with open("pulse_field/reports/pretrain_history.json", "w") as f:
            json.dump(self.history, f)
            
        return self.history

if __name__ == "__main__":
    pipeline = PretrainPipeline()
    pipeline.train(100)
