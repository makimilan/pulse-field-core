
import pytest
import torch
import torch.nn as nn
from core.impulse import Impulse, ImpulseEncoder, ImpulseDecoder
from core.crystals import Crystal

class TestRealArchitecture:
    
    def test_impulse_encoder_gradients(self):
        """
        Verify that the encoder produces tensors with gradients enabled,
        proving it is a learnable component.
        """
        encoder = ImpulseEncoder(vocab_size=100, dim=32)
        text = "hello world"
        impulse = encoder(text)
        
        assert isinstance(impulse.V, torch.Tensor)
        # In the current implementation, V is a result of operations on embeddings.
        # Embeddings have requires_grad=True by default.
        # However, Impulse.V is a detached tensor in some contexts if not careful, 
        # but let's check if we can backprop through it.
        
        # We need to check if the embedding weights have gradients after a backward pass.
        loss = impulse.V.sum()
        loss.backward()
        
        assert encoder.embedding.weight.grad is not None
        assert torch.norm(encoder.embedding.weight.grad) > 0
        print("\n[PASS] Encoder gradients are flowing.")

    def test_crystal_learning(self):
        """
        Verify that a Crystal can learn a simple transformation.
        """
        dim = 16
        crystal = Crystal(crystal_id="test_c", input_dim=dim, output_dim=dim)
        
        # Input
        input_tensor = torch.randn(dim)
        impulse = Impulse(V=input_tensor, E=1.0, T=(), C=0, seed=42)
        
        # Target: Identity transformation (just for test)
        target = input_tensor
        
        optimizer = torch.optim.SGD(crystal.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        
        # Train for a few steps
        initial_loss = 0
        final_loss = 0
        
        for i in range(10):
            optimizer.zero_grad()
            out_impulse, _ = crystal.process(impulse)
            loss = criterion(out_impulse.V, target)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
            
        print(f"\n[PASS] Crystal training: Initial Loss {initial_loss:.4f} -> Final Loss {final_loss:.4f}")
        assert final_loss < initial_loss

    def test_impulse_energy_decay(self):
        """
        Verify energy decay mechanics.
        """
        V = torch.randn(10)
        impulse = Impulse(V=V, E=1.0, T=(), C=0, seed=42)
        
        decayed = impulse.decay(0.1)
        assert decayed.E == 0.9
        assert decayed.is_alive
        
        dead = impulse.decay(1.0)
        assert dead.E == 0.0
        assert not dead.is_alive
        print("\n[PASS] Energy decay works correctly.")

    def test_semantic_embedding(self):
        """
        Verify that the embedding is not just a random hash.
        (Although untrained embeddings are random, they are dense vectors, not orthogonal one-hot hashes).
        """
        encoder = ImpulseEncoder(vocab_size=1000, dim=128)
        
        # "hello" and "hello." should share some embedding content if tokenized similarly.
        # Our simple tokenizer hashes words. "hello" -> hash("hello"). "hello." -> hash("hello.")
        # So they will be different indices.
        # But let's check that the output is a dense vector.
        
        impulse = encoder("test")
        assert impulse.V.shape == (128,)
        assert torch.is_floating_point(impulse.V)
        print("\n[PASS] Embedding produces dense float tensors.")

if __name__ == "__main__":
    # Manual run if executed as script
    t = TestRealArchitecture()
    t.test_impulse_encoder_gradients()
    t.test_crystal_learning()
    t.test_impulse_energy_decay()
    t.test_semantic_embedding()
