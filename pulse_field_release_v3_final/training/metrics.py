import numpy as np
import math
from typing import List, Dict

class Metrics:
    """
    Quality metrics for evaluation.
    """
    
    @staticmethod
    def accuracy(predictions: List[str], targets: List[str]) -> float:
        """Exact match accuracy."""
        correct = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
        return correct / len(targets) if targets else 0.0
        
    @staticmethod
    def rouge_l(predictions: List[str], targets: List[str]) -> float:
        """Simplified ROUGE-L (Longest Common Subsequence)."""
        scores = []
        for p, t in zip(predictions, targets):
            p_tokens = p.split()
            t_tokens = t.split()
            if not p_tokens or not t_tokens:
                scores.append(0.0)
                continue
                
            # LCS dynamic programming
            m, n = len(p_tokens), len(t_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if p_tokens[i-1] == t_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            lcs = dp[m][n]
            prec = lcs / m
            rec = lcs / n
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            scores.append(f1)
            
        return np.mean(scores) if scores else 0.0

    @staticmethod
    def bleu_score(predictions: List[str], targets: List[str]) -> float:
        """Simplified BLEU-1 score."""
        scores = []
        for p, t in zip(predictions, targets):
            p_tokens = p.split()
            t_tokens = t.split()
            if not p_tokens:
                scores.append(0.0)
                continue
            
            # Count matches
            matches = sum(1 for token in p_tokens if token in t_tokens)
            precision = matches / len(p_tokens)
            
            # Brevity penalty
            bp = 1.0
            if len(p_tokens) < len(t_tokens):
                bp = math.exp(1 - len(t_tokens) / len(p_tokens))
                
            scores.append(bp * precision)
            
        return np.mean(scores) if scores else 0.0
        
    @staticmethod
    def defect_rate(defects: List[float]) -> float:
        """Mean defect."""
        return np.mean(defects) if defects else 0.0

    @staticmethod
    def perplexity(defect: float) -> float:
        """
        Calculate Perplexity from Defect (Energy Loss).
        Assumption: Defect is proportional to Cross-Entropy Loss.
        PPL = exp(Defect * ScaleFactor)
        ScaleFactor is calibrated to align 0.05 defect with ~1.5 PPL.
        """
        # Calibration: exp(0.05 * X) = 1.5 => 0.05 * X = ln(1.5) => X = ln(1.5)/0.05 ~= 8.1
        scale_factor = 8.1
        return math.exp(defect * scale_factor)
