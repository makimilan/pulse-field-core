
import pytest
import torch
import torch.nn as nn
from core.impulse import ImpulseEncoder, Impulse
from core.crystals import Crystal

class TestAbsoluteIntelligence:
    
    def test_sequence_understanding(self):
        """
        THE ABSOLUTE TEST:
        Verify that the architecture can distinguish between:
        A: "The dog bit the man"
        B: "The man bit the dog"
        
        In a Bag-of-Words model, these are identical.
        In Pulse-Field v4 (SSM/LRU), they must be different.
        """
        print("\n--- Running ABSOLUTE Intelligence Test ---")
        
        dim = 128
        encoder = ImpulseEncoder(vocab_size=1000, dim=dim, seed=42)
        
        # Ensure we have different words mapped to different indices
        # "The", "dog", "bit", "the", "man"
        # "The", "man", "bit", "the", "dog"
        
        text_a = "The dog bit the man"
        text_b = "The man bit the dog"
        
        # Encode
        impulse_a = encoder(text_a)
        impulse_b = encoder(text_b)
        
        # Check vectors
        V_a = impulse_a.V
        V_b = impulse_b.V
        
        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(V_a.unsqueeze(0), V_b.unsqueeze(0)).item()
        
        print(f"Vector A (First 5): {V_a[:5].detach().numpy()}")
        print(f"Vector B (First 5): {V_b[:5].detach().numpy()}")
        print(f"Cosine Similarity: {cos_sim:.4f}")
        
        # Assertion
        # If they are identical, cos_sim will be 1.0 (or very close due to float error)
        # We want them to be distinct.
        assert cos_sim < 0.99, f"Failed: Vectors are too similar ({cos_sim}). Sequence order ignored!"
        
        print("[PASS] The model successfully distinguishes sequence order.")
        
    def test_crystal_ssm_state(self):
        """
        Verify that Crystals update the hidden state H.
        """
        dim = 32
        crystal = Crystal(crystal_id="test_ssm", input_dim=dim, output_dim=dim)
        
        # Create impulse with initial state
        V = torch.randn(dim)
        H = torch.zeros(dim)
        impulse = Impulse(V=V, E=1.0, T=(), C=0, seed=42, H=H)
        
        # Process
        out_impulse, _ = crystal.process(impulse)
        
        # Check if H changed
        H_new = out_impulse.H
        assert H_new is not None
        assert not torch.allclose(H, H_new), "Hidden state H did not change after Crystal processing!"
        
        print("\n[PASS] Crystal SSM state update works.")

if __name__ == "__main__":
    t = TestAbsoluteIntelligence()
    t.test_sequence_understanding()
    t.test_crystal_ssm_state()
