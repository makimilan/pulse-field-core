"""
Autoarchitect: Mutation engine with safe evolution and adversarial immunity.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from .impulse import Impulse
from .crystals import Crystal
from .archive import Archive


class Mutation:
    """Single mutation proposal."""
    
    def __init__(
        self,
        mutation_type: str,
        target_id: str,
        parameters: Dict,
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize mutation.
        
        Args:
            mutation_type: Type (kappa_change, route_rearrange, merge, split, threshold_tune).
            target_id: Target node/crystal ID.
            parameters: Mutation parameters.
            metadata: Additional metadata.
        """
        self.mutation_type = mutation_type
        self.target_id = target_id
        self.parameters = parameters
        self.metadata = metadata or {}


class Autoarchitect:
    """
    Adaptive architecture engine with mutations, shadow A/B, and immunity.
    
    Contract:
      - Mutations: κ change, route rearrangement, merge/split, threshold tuning.
      - Shadow A/B: simulate mutation; commit iff ΔD<0 AND pass_rate≥95%.
      - Immunity: red-team adversarial inputs; infection quarantine.
      - Invariant: I11 (safe evolution).
    """
    
    def __init__(self, archive: Archive, seed: int = 42):
        """
        Initialize autoarchitect.
        
        Args:
            archive: Archive instance for versioning.
            seed: Random seed.
        """
        self.archive = archive
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Mutation history
        self.mutations_proposed = 0
        self.mutations_committed = 0
        self.mutations_failed = []
    
    def propose_mutation(self, mutation_type: str, target_id: str, **kwargs) -> Mutation:
        """
        Propose a mutation.
        
        Args:
            mutation_type: Type of mutation.
            target_id: Target to mutate.
            **kwargs: Mutation-specific parameters.
        
        Returns:
            Mutation proposal.
        """
        self.mutations_proposed += 1
        
        return Mutation(
            mutation_type=mutation_type,
            target_id=target_id,
            parameters=kwargs,
        )
    
    def shadow_ab(
        self,
        mutation: Mutation,
        representative_tasks: List[Tuple[Impulse, float]],
        validator: Callable[[List[float]], bool],
    ) -> Tuple[bool, float, Dict]:
        """
        Validate mutation via shadow A/B testing.
        
        Args:
            mutation: Mutation to test.
            representative_tasks: List of (impulse, expected_defect) pairs.
            validator: Function to validate results (ΔD<0, pass_rate≥95%).
        
        Returns:
            (success, delta_D, details)
        
        Invariant:
          I11: Commit only if ΔD<0 AND invariants_pass_rate≥95%.
        """
        results_before = []
        results_after = []
        
        for impulse, expected_defect in representative_tasks:
            # Run task without mutation (before)
            defect_before = impulse.total_defect()
            results_before.append(defect_before)
            
            # Run task with mutation (after)
            # For now, simulate with noise
            defect_after = defect_before * (1 + self.rng.randn() * 0.1)
            results_after.append(defect_after)
        
        # Compute delta_D
        delta_D = np.mean(results_after) - np.mean(results_before)
        
        # Check validator
        success = validator(results_after) and delta_D < 0
        
        return success, delta_D, {
            "defect_before": float(np.mean(results_before)),
            "defect_after": float(np.mean(results_after)),
            "delta_D": float(delta_D),
            "sample_size": len(representative_tasks),
        }
    
    def commit_mutation(
        self,
        mutation: Mutation,
        delta_D: float,
        invariants_pass_rate: float,
        threshold: float = 0.95,
    ) -> bool:
        """
        Commit mutation if safe (I11).
        
        Args:
            mutation: Mutation to commit.
            delta_D: Change in defect.
            invariants_pass_rate: Pass-rate of invariants I1–I11.
            threshold: Pass-rate threshold (default 95%).
        
        Returns:
            True if committed.
        
        Invariant:
          I11: Commit iff ΔD<0 AND pass_rate≥95%.
        """
        defect_improves = delta_D < 0
        passes_threshold = invariants_pass_rate >= threshold - 1e-6
        
        success = defect_improves and passes_threshold
        
        if success:
            self.mutations_committed += 1
        else:
            self.mutations_failed.append({
                "mutation": mutation,
                "delta_D": delta_D,
                "pass_rate": invariants_pass_rate,
                "reason": "defect_no_improve" if not defect_improves else "pass_rate_threshold",
            })
        
        return success
    
    def red_team(
        self,
        n_samples: int = 10,
        intensity: float = 1.0,
    ) -> List[Impulse]:
        """
        Generate adversarial impulses for robustness testing.
        
        Args:
            n_samples: Number of adversarial samples.
            intensity: Adversarial intensity [0, 1].
        
        Returns:
            List of adversarial Impulses.
        
        Contract:
          - Adversarial impulses designed to break safety invariants.
          - Infection marked; quarantine and retrain.
        """
        adversarial_samples = []
        
        for i in range(n_samples):
            # Generate pathological impulse
            dim = 128
            V_adv = torch.randn(dim)
            
            # Add intensity
            V_adv *= (1 + intensity * 10.0)
            
            # Clip to increase non-uniformity
            V_adv = torch.clamp(V_adv, -1, 1)
            
            # Normalize
            norm = torch.norm(V_adv) + 1e-8
            V_adv = V_adv / norm
            
            impulse = Impulse(
                V=V_adv,
                E=0.01,  # Low energy
                T=tuple(),
                C=int(hash(f"adversarial_{i}")) & 0xffffffffffffffff,
                seed=self.seed ^ i,
                _meta={"infection": True, "adversarial_intensity": intensity},
            )
            adversarial_samples.append(impulse)
        
        return adversarial_samples
    
    def quarantine(self, impulse: Impulse) -> bool:
        """
        Quarantine suspicious impulse.
        
        Args:
            impulse: Impulse to quarantine.
        
        Returns:
            True if quarantined.
        
        Contract:
          - Infected impulses prevented from affecting main system.
          - Logged for analysis.
        """
        if impulse._meta.get("infection", False):
            # Archive with quarantine flag
            # Create a copy with quarantine metadata
            quarantine_impulse = Impulse(
                V=impulse.V,
                E=impulse.E,
                T=impulse.T,
                C=impulse.C,
                seed=impulse.seed,
                _meta={"quarantine_reason": "infection", **impulse._meta}
            )
            self.archive.put(quarantine_impulse, is_strong=False)
            return True
        
        return False
    
    def retrain_on_clean(self, clean_impulses: List[Impulse]):
        """
        Retrain system on clean, non-adversarial impulses.
        
        Args:
            clean_impulses: List of verified clean impulses.
        """
        for impulse in clean_impulses:
            # Create copy with retrain metadata
            retrain_impulse = Impulse(
                V=impulse.V,
                E=impulse.E,
                T=impulse.T,
                C=impulse.C,
                seed=impulse.seed,
                _meta={"retrain": True, **impulse._meta}
            )
            self.archive.put(retrain_impulse, is_strong=True)
    
    def get_stats(self) -> Dict:
        """Get mutation statistics."""
        return {
            "mutations_proposed": self.mutations_proposed,
            "mutations_committed": self.mutations_committed,
            "mutations_failed": len(self.mutations_failed),
            "commit_rate": (
                self.mutations_committed / self.mutations_proposed
                if self.mutations_proposed > 0 else 0.0
            ),
        }
