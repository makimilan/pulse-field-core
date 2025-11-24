# Pulse-Field v3.0 Project Structure

## Root Directory
- `README.md`: Main documentation and usage guide.
- `RELEASE_NOTES.md`: Version history and key features.
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT License.

## Core Architecture (`core/`)
The heart of the system, containing the real neural components.
- `impulse.py`: `Impulse` class (Tensor carrier) and `ImpulseEncoder`/`ImpulseDecoder`.
- `crystals.py`: `Crystal` class (`nn.Module` with `nn.Linear` layers).
- `runtime.py`: `Runtime` engine orchestrating execution.
- `cgw_graph.py`: Compositional Graph-Waves logic.
- `archive.py`: HNSW-based memory system.
- `autoarchitect.py`: Evolutionary mutation engine.
- `compatibility.py`: Node selection logic.
- `router.py`: System1/System2 routing.
- `global_critic.py`: Alignment verification.
- `config.py`: Configuration management.

## Input/Output (`io/`)
- `encoder.py`: Specialized encoders (Text, Number, DSL).
- `decoder.py`: Specialized decoders.

## Serving (`serving/`)
- `api.py`: FastAPI server for production deployment.

## Benchmarks (`benchmarks/`)
- `benchmark_suite.py`: Comprehensive comparison vs Transformer.
- `benchmark_performance.py`: Speed and memory tests.

## Experiments (`experiments/`)
- `baseline_transformer.py`: PyTorch Transformer baseline for comparison.

## Training (`training/`)
- `distillation_pipeline.py`: Pipeline for training via distillation.
- `checkpoints.py`, `loaders.py`, `metrics.py`: Training utilities.

## Tests (`tests/`)
- `test_real_architecture.py`: Unit tests verifying the real architecture.

## Reports (`reports/`)
- `FINAL_METRICS.md`: Summary of performance metrics.
- `routes_traces/`: Directory for execution logs.
