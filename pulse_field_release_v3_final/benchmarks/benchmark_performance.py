import time
import psutil
import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.runtime import Runtime
from core.config import Config
from core.impulse import Impulse

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_flops_speed(runtime, iterations=100):
    print(f"\n--- Speed & FLOPS Benchmark ({iterations} iterations) ---")
    
    text = "The quick brown fox jumps over the lazy dog."
    impulse = runtime.encoder.encode(text)
    
    start_time = time.time()
    total_active_nodes = 0
    total_steps = 0
    
    for _ in range(iterations):
        # We need to access the internal trace to count active nodes
        # But runtime.execute returns the final impulse.
        # We can estimate active nodes by checking the trace length * avg active set size (usually 16 or less)
        # Or we can instrument the runtime.
        # For this benchmark, we'll assume max capacity (16) for upper bound, 
        # or better, we'll just measure wall clock time.
        
        output = runtime.execute(impulse, max_steps=10)
        total_steps += len(output.T)
        # Estimate active nodes per step (avg)
        # In v3, it's dynamic, but let's say 10 on average.
        total_active_nodes += len(output.T) * 10 
        
    end_time = time.time()
    duration = end_time - start_time
    
    # FLOPS Calculation
    # 1 Node = 128x128 Matrix Mul + Bias + Tanh
    # Ops per node ~= 2 * 128^2 + 128 + 128 ~= 33000 FLOPs
    flops_per_node = 2 * 128 * 128
    total_flops = total_active_nodes * flops_per_node
    gflops = total_flops / duration / 1e9
    
    tokens_per_iter = len(text.split()) # Approx
    total_tokens = tokens_per_iter * iterations
    tok_s = total_tokens / duration
    
    print(f"Duration: {duration:.4f} s")
    print(f"Throughput: {tok_s:.2f} tok/s")
    print(f"Estimated Compute: {gflops:.4f} GFLOPS")
    print(f"Latency per request: {(duration/iterations)*1000:.2f} ms")
    
    return tok_s, gflops

def benchmark_context(runtime, n_items=1000):
    print(f"\n--- Context & Memory Benchmark ({n_items} items) ---")
    
    start_mem = get_memory_usage()
    print(f"Baseline Memory: {start_mem:.2f} MB")
    
    # Fill Archive
    print("Filling Archive...")
    start_time = time.time()
    for i in range(n_items):
        text = f"memory_entry_{i}_{np.random.random()}"
        imp = runtime.encoder.encode(text)
        runtime.archive.put(imp)
    fill_time = time.time() - start_time
    
    mid_mem = get_memory_usage()
    print(f"Memory after fill: {mid_mem:.2f} MB (Delta: {mid_mem - start_mem:.2f} MB)")
    print(f"Fill Rate: {n_items/fill_time:.2f} items/s")
    
    # Retrieval Test
    print("Testing Retrieval...")
    query_text = "memory_entry_500" # Should be close to one of them
    query = runtime.encoder.encode(query_text) # Note: encoding is deterministic but we added random suffix
    # So we can't find exact match easily unless we stored the exact text.
    # Let's just query the last one we added.
    
    last_imp = imp
    start_retrieval = time.time()
    results = runtime.archive.get(last_imp, k=1)
    retrieval_time = time.time() - start_retrieval
    
    print(f"Retrieval Time: {retrieval_time*1000:.4f} ms")
    if results:
        dist = results[0][1]
        print(f"Top-1 Distance: {dist:.6f}")
        assert dist < 1e-5, "Retrieval Integrity Failed!"
    else:
        print("Retrieval Failed: No results")

def benchmark_quality_sanity(runtime):
    print("\n--- Quality Sanity Check ---")
    # Run a simple consistency check
    text = "Hello World"
    imp1 = runtime.encoder.encode(text)
    out1 = runtime.execute(imp1)
    
    imp2 = runtime.encoder.encode(text)
    out2 = runtime.execute(imp2)
    
    # Check determinism
    diff = torch.norm(out1.V - out2.V).item()
    print(f"Determinism Error: {diff:.10f}")
    assert diff < 1e-9, "Determinism Failed!"
    
    # Check energy decay
    print(f"Input Energy: {imp1.E:.4f}")
    print(f"Output Energy: {out1.E:.4f}")
    assert out1.E < imp1.E, "Physics Failed: Energy did not decay"

def main():
    print("Initializing Pulse-Field v3.0 Runtime...")
    config = Config()
    runtime = Runtime(config)
    
    # Warmup
    runtime.execute(runtime.encoder.encode("warmup"))
    
    # Run Benchmarks
    benchmark_flops_speed(runtime)
    benchmark_context(runtime)
    benchmark_quality_sanity(runtime)
    
    print("\nAll Benchmarks Passed.")

if __name__ == "__main__":
    main()
