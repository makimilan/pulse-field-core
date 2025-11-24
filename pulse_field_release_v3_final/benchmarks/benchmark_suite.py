
import torch
import torch.nn as nn
import time
import numpy as np
import tracemalloc
import psutil
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.impulse import ImpulseEncoder, ImpulseDecoder, Impulse
from core.crystals import Crystal
from experiments.baseline_transformer import SimpleTransformer

class PulseFieldWrapper(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.encoder = ImpulseEncoder(vocab_size=vocab_size, dim=dim)
        self.crystal = Crystal(crystal_id="bench", input_dim=dim, output_dim=dim)
        self.decoder = ImpulseDecoder(dim=dim, vocab_size=vocab_size)
        
    def forward(self, input_ids):
        # input_ids: (Batch, Seq_Len)
        # Pulse-Field processes the whole sequence into one Impulse (SSM style)
        
        embeds = self.encoder.embedding(input_ids) # (B, S, D)
        
        # SSM / LRU Recurrence (v4.0 Logic)
        # h_t = decay * h_{t-1} + (1-decay) * x_t
        
        B, S, D = embeds.shape
        h = torch.zeros(B, D, device=embeds.device)
        decay = torch.sigmoid(self.encoder.ssm_decay)
        
        # Linear Scan (O(S))
        # We iterate over the sequence length dimension
        for t in range(S):
            x_t = embeds[:, t, :]
            h = decay * h + (1 - decay) * x_t
            
        V = torch.tanh(h) # Squash to [-1, 1]
        
        # Crystal Process (Linear Layer + SSM State Update)
        # Note: In this simplified wrapper, we just pass V. 
        # The Crystal in v4 expects an Impulse with H, but for raw throughput benchmarking
        # of the core math, passing V to the layer is the main cost.
        # However, to be precise, let's simulate the Crystal's internal SSM too.
        
        # Crystal SSM Logic (Simplified for benchmark)
        # H_new = (decay * H_in) + (gate(V_in) * V_in)
        # Here H_in is the output of the encoder (h)
        
        crystal_decay = torch.sigmoid(self.crystal.ssm_decay)
        crystal_gate = torch.sigmoid(self.crystal.ssm_gate(V))
        
        H_crystal = (crystal_decay * h) + (crystal_gate * V)
        
        # Then the actual layer
        V_processed = self.crystal.forward(H_crystal)
        
        # Decoder (Linear Layer)
        # We create a dummy Impulse just to satisfy the interface if needed, 
        # but here we just call the linear layer directly for speed.
        logits = self.decoder.linear(V_processed)
        
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_suite():
    print("--- Starting Comprehensive Benchmark Suite ---")
    
    # Config
    vocab_size = 1000
    dim = 128
    batch_size = 1
    context_lengths = [128, 512, 1024, 2048, 4096]
    
    # Models
    pf_model = PulseFieldWrapper(vocab_size=vocab_size, dim=dim)
    tf_model = SimpleTransformer(vocab_size=vocab_size, d_model=dim, n_head=4, n_layer=2, max_len=5000)
    
    print(f"Pulse-Field Params: {count_parameters(pf_model)}")
    print(f"Transformer Params: {count_parameters(tf_model)}")
    
    results = {
        "context": [],
        "pf_latency": [],
        "tf_latency": [],
        "pf_ram": [],
        "tf_ram": [],
        "pf_flops": [],
        "tf_flops": []
    }
    
    process = psutil.Process(os.getpid())
    
    for seq_len in context_lengths:
        print(f"\nTesting Context Length: {seq_len}")
        results["context"].append(seq_len)
        
        input_data = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # --- Pulse-Field Benchmark ---
        tracemalloc.start()
        start_time = time.time()
        # Run 10 times for better timing
        with torch.no_grad():
            for _ in range(10):
                _ = pf_model(input_data)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        pf_latency = ((end_time - start_time) * 1000) / 10
        pf_ram = peak / (1024 * 1024) # MB
        
        # Estimate FLOPS for PF
        # Encoder: B*S*D (additions for mean)
        # Crystal: B*D*D (matmul)
        # Decoder: B*D*V (matmul)
        pf_flops = (batch_size * seq_len * dim) + (batch_size * dim * dim) + (batch_size * dim * vocab_size)
        
        results["pf_latency"].append(pf_latency)
        results["pf_ram"].append(pf_ram)
        results["pf_flops"].append(pf_flops)
        
        print(f"Pulse-Field: {pf_latency:.2f}ms | {pf_ram:.2f}MB | ~{pf_flops/1e6:.2f} MFLOPs")
        
        # --- Transformer Benchmark ---
        tracemalloc.start()
        start_time = time.time()
        # Run 10 times
        with torch.no_grad():
            for _ in range(10):
                _ = tf_model(input_data)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        tf_latency = ((end_time - start_time) * 1000) / 10
        tf_ram = peak / (1024 * 1024) # MB
        
        # Estimate FLOPS for TF (Approx)
        # 2 * N_layer * (4 * B * S * D^2 + 2 * B * S^2 * D)
        # This is a rough approximation for Attention + FFN
        tf_flops = 2 * 2 * (4 * batch_size * seq_len * dim**2 + 2 * batch_size * seq_len**2 * dim)
        
        results["tf_latency"].append(tf_latency)
        results["tf_ram"].append(tf_ram)
        results["tf_flops"].append(tf_flops)
        
        print(f"Transformer: {tf_latency:.2f}ms | {tf_ram:.2f}MB | ~{tf_flops/1e6:.2f} MFLOPs")
        if pf_latency > 0:
            print(f"Speedup: {tf_latency/pf_latency:.1f}x")
        else:
            print(f"Speedup: Infinite (PF < 0.1ms)")

    # --- PPL Test (Synthetic) ---
    print("\n--- Running PPL Comparison (Synthetic Task) ---")
    # Task: Predict next token in a repeating sequence "0 1 2 3 0 1 2 3..."
    # This favors models that can attend to local context.
    
    seq_len = 32
    data_size = 20 # Reduced for speed
    data = []
    for _ in range(data_size):
        start = np.random.randint(0, 4)
        seq = [(start + i) % 4 for i in range(seq_len)]
        data.append(seq)
    data = torch.tensor(data, dtype=torch.long) # (20, 32)
    
    # Train Pulse-Field
    print("Training Pulse-Field...")
    optimizer = torch.optim.Adam(pf_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    pf_model.train()
    
    for epoch in range(5): # Reduced epochs
        total_loss = 0
        for i in range(data_size):
            # Input: 0..N-1, Target: N
            # PF is not autoregressive by default, so we train it to predict the LAST token given the prefix
            # This is a fair "Next Token Prediction" test for a specific context window
            
            # Let's just train on predicting the last token from the first N-1
            inputs = data[i, :-1].unsqueeze(0)
            target = data[i, -1].unsqueeze(0)
            
            optimizer.zero_grad()
            logits = pf_model(inputs) # (1, Vocab)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"Pulse-Field Final Loss: {total_loss/data_size:.4f}")
    pf_ppl = np.exp(total_loss/data_size)
    
    # Train Transformer
    print("Training Transformer...")
    optimizer = torch.optim.Adam(tf_model.parameters(), lr=0.01)
    tf_model.train()
    
    for epoch in range(5): # Reduced epochs
        total_loss = 0
        for i in range(data_size):
            inputs = data[i, :-1].unsqueeze(0)
            target = data[i, -1].unsqueeze(0)
            
            optimizer.zero_grad()
            logits = tf_model(inputs) # (1, S, V)
            # We only care about the last token prediction for fair comparison
            last_token_logits = logits[:, -1, :]
            
            loss = criterion(last_token_logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"Transformer Final Loss: {total_loss/data_size:.4f}")
    tf_ppl = np.exp(total_loss/data_size)
    
    print("\n--- Final Results ---")
    print(f"Pulse-Field PPL: {pf_ppl:.2f}")
    print(f"Transformer PPL: {tf_ppl:.2f}")
    
    if pf_ppl < tf_ppl:
        print("Winner: Pulse-Field (Surprisingly!)")
    else:
        print("Winner: Transformer (Expected for sequential tasks)")

if __name__ == "__main__":
    benchmark_suite()
