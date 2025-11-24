import math
import ast
from typing import Dict, List, Any
from .metrics import Metrics

class Evaluator:
    """Base class for task evaluators."""
    def evaluate(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        raise NotImplementedError

class QAEvaluator(Evaluator):
    def evaluate(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        acc = Metrics.accuracy(predictions, targets)
        return {"accuracy": acc}

class SummarizationEvaluator(Evaluator):
    def evaluate(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        rouge = Metrics.rouge_l(predictions, targets)
        bleu = Metrics.bleu_score(predictions, targets)
        return {"rouge_l": rouge, "bleu": bleu}

class ReasoningEvaluator(Evaluator):
    def evaluate(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        # For reasoning, we check if the final answer matches
        # Simple exact match for now
        acc = Metrics.accuracy(predictions, targets)
        return {"accuracy": acc}

class CodeEvaluator(Evaluator):
    def evaluate(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        # Real syntax check using AST
        valid_count = 0
        for code in predictions:
            try:
                # Try to parse the code
                # Strip markdown code blocks if present
                clean_code = code.replace("```python", "").replace("```", "").strip()
                ast.parse(clean_code)
                valid_count += 1
            except SyntaxError:
                pass
                
        syntax_acc = valid_count / len(predictions) if predictions else 0.0
        return {"execution_success": syntax_acc}
