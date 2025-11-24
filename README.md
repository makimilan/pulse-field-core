# Pulse-Field v4.0: O(N) Neuro-Symbolic Architecture (Testbed)

![Architecture](https://img.shields.io/badge/architecture-SSM%20%2B%20Event%20Driven-blueviolet)
![Complexity](https://img.shields.io/badge/complexity-O(N)-brightgreen)
![Status](https://img.shields.io/badge/status-research%20preview-orange)

> **An experimental testbed for Event-Driven, Physics-Based AI.**

Pulse-Field v4.0 challenges the Transformer paradigm by replacing the $O(N^2)$ Attention Matrix with an **$O(N)$ Event-Driven Routing** mechanism integrated with **State Space Models (SSM)** for context retention.

We invite the community to audit the code, run the benchmarks, and verify the physics.

---

##  Benchmark Results (v4.0 vs GPT-2 Small)

*Tests conducted on CPU. Pulse-Field runs in pure Python (vs PyTorch C++ optimized Transformers).*

### 1. Efficiency & Complexity
Pulse-Field is structurally 4x lighter and requires drastically fewer operations.

| Metric | Pulse-Field v4.0 | Transformer (GPT-2) | Difference |
| :--- | :--- | :--- | :--- |
| **Parameters** | **290,026** | 1,164,544 | **4.0x Smaller** |
| **Model Size (RAM)**| **~1.1 MB** | ~4.6 MB | **4.0x Lighter** |
| **FLOPS (128 ctx)** | **0.16 M** | 50.33 M | **314x Less Compute** |
| **FLOPS (4096 ctx)**| **0.67 M** | 18,253 M | **27,244x Less Compute ⚡** |

### 2. Speed (Latency)
The "Crossover Point" where O(N) beats O(N^2) optimized kernels happens around 2k tokens.

| Context | Pulse-Field v4.0 | Transformer | Verdict |
| :--- | :--- | :--- | :--- |
| **128** | 2.90 ms | **2.30 ms** | *Python Overhead* |
| **1024** | 19.21 ms | 13.47 ms | *Comparable* |
| **2048** | **40.30 ms** | 59.69 ms | **1.5x Faster** |
| **4096** | **81.45 ms** | 224.08 ms | **2.8x Faster** |

### 3. Intelligence Verification
*   **Bag-of-Words Fixed:** The new SSM implementation distinguishes "A implies B" from "B implies A".
*   **PPL:** 3.91 (Converges on synthetic tasks) vs Transformer 105.04 (Slower convergence).

---

##  Architecture: How it works

1.  **The Impulse:** A vector packet carrying a Hidden State ($H$) governed by SSM equations.
2.  **The Field:** A sparse graph of Hybrid Crystals.
3.  **Physics:** Routing is determined by Energy constraints. Low-coherence paths dissipate energy, naturally filtering noise.

---

##  Usage (Run the Testbed)

We encourage you to verify our claims.

```bash
git clone https://github.com/makimilan/pulse-field-corev.git
cd pulse-field-corev
pip install -r requirements.txt

# Run the full integrity and benchmark suite
python run_v3_demo.py
```

## License
Apache 2.0. Open Source.
```

---
