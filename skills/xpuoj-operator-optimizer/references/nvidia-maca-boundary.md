# Historical NVIDIA Proxy Versus MACA/C500

New NVIDIA runs are disabled. Read this reference only to interpret existing historical proxy evidence; use `c500-native-optimization.md` for new candidates.

## Evidence Classes

| Class | Valid conclusions | Invalid conclusions |
| --- | --- | --- |
| CPU/reference | Mathematical semantics, edge cases, numerical comparison | GPU performance |
| NVIDIA proxy | Algorithm viability, memory safety, relative screening on that exact NVIDIA environment | MACA compilation, C500 latency, OJ score/rank |
| C500 local | Target hardware profiling in the recorded local environment | Exact OJ result if evaluator differs |
| XPU-OJ/C500 | Official correctness, score and leaderboard evidence for that submission | Generalization beyond disclosed/hidden evaluator |

## What Usually Transfers

- Algebraic transformations that preserve semantics.
- Removal of redundant work or global memory traffic.
- Better data reuse and fewer kernel launches, subject to target confirmation.
- Correct handling of shapes, strides, layouts, quantization and edge cases.
- Test harnesses, reference implementations and experiment automation.

## What Often Does Not Transfer

- Warp-size assumptions, occupancy heuristics and block shapes.
- Register pressure, shared-memory/LDS capacity and bank behavior.
- Tensor-core/intrinsic availability and instruction throughput.
- Cache hierarchy, asynchronous copy, compiler scheduling and launch overhead.
- CUDA-specific extensions accepted by nvcc but not MXMACA, or vice versa.

Keep target-specific code behind explicit backends. Do not hide an NVIDIA-only fast path under a generic name and submit it unreviewed.

## Proxy Benchmark Rules

- Pin GPU model, driver, CUDA, compiler, clocks/power state when observable, and all package versions.
- Use the same inputs, warmup, repetitions and statistic for baseline and candidate.
- Synchronize only where required by the benchmark methodology; do not add synchronization inside timed `run_kernel` unless the OJ contract requires it.
- Store raw per-run values, not only an average.
- Report variance and failures/OOM.
- Prefix conclusions with `NVIDIA proxy:`.

## Target Validation Plan

Use NVIDIA to narrow candidates, then use XPU-OJ sparingly for target feedback. Once C500 access is available, rerun correctness, compile, profiler and benchmark gates there before claiming reproducibility.
