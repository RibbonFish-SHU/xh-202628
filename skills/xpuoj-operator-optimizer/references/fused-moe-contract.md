# Fused MoE Live Contract Snapshot

Status: authenticated XPU-OJ page verified read-only on 2026-08-18 (Asia/Shanghai). Recheck immediately before implementation and every submission because the live page remains authoritative.

## Identity And Environment

- URL: `https://xpuoj.com/contest/12/problem/1`
- Visible title: `1. Agent 推理算子库优化 - Fused MoE i8 tn`
- Available implementations: CUDA Maca, Triton, TileLang.
- Displayed limits: 10000 ms and 4096 MiB.
- Evaluator image: `maca torch 2.8.0+metax 3.7.1.5`.

## CUDA Maca Signature

```cpp
extern "C" void run_kernel(
    const int8_t* a,
    const int8_t* b_col_major,
    const float* scale_a,
    const float* scale_b,
    const float* moe_weights,
    const int32_t* token_ids,
    const int32_t* expert_ids,
    int64_t topk,
    __nv_bfloat16* out
);
```

Parameter order is fixed. The function writes `out` in place; all other inputs are read-only.

## Semantics

For routed row `r` and output column `n`:

```text
expert(r) = expert_ids[r // 128]
out[r,n] = sum_k(a[r,k] * b_col_major[expert(r),n,k])
           * scale_a[r] * scale_b[expert(r),n] * moe_weights[r]
```

- `a`: int8 `[EM, K]`, row-major and already expanded by routed row.
- `b_col_major`: int8 `[num_experts, N, K]` with logical layout `[expert,n,k]`.
- `scale_a`: float32 `[EM]`, already expanded by routed row.
- `scale_b`: float32 `[num_experts, N]`.
- `moe_weights`: float32 `[EM]`.
- `token_ids`: int32 `[EM]`; provenance only. Do not gather `a`/`scale_a` again.
- `expert_ids`: int32 `[EM/128]`; one expert for each 128 routed rows.
- `topk`: integer, always 8.
- `out`: bfloat16 `[EM,N]`.
- `num_experts=256`, `EM=num_tokens*topk`, and `EM % 128 == 0`.

The input generator has already applied `raw_a[token_ids[r] // topk]` and the corresponding `scale_a` gather. Sparse routing does not guarantee every expert is selected.

## Evaluated Shapes

| Case | Workload | num_tokens | EM | Tiles | N | K | Expert distribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | decode gate-up | 512 | 4096 | 32 | 4096 | 7168 | uniform |
| 2 | prefill gate-up | 4096 | 32768 | 256 | 4096 | 7168 | skewed |
| 3 | decode down | 512 | 4096 | 32 | 7168 | 2048 | uniform |
| 4 | prefill down | 4096 | 32768 | 256 | 7168 | 2048 | skewed |

The live page states that int8 inputs are in `[-20,20]`. Accumulate the exact integer reduction in int32 where practical before applying scales and the MoE weight.

## Correctness Gate

The live page defines:

```python
close = torch.isclose(out_target.float(), out_ref.float(), rtol=2e-2, atol=5e-3)
matched_ratio = close.float().mean()
passed = matched_ratio >= 0.99
```

Do not substitute tolerance values from older local documents. Cover both gate-up/down, decode/prefill, unselected experts, tile boundaries, and large-`K` accumulation in local tests.

## Recheck Before Submission

Capture the selected language tab's exact interface, allowed libraries/intrinsics, score and stability rules, quota/cooldown, and any changed limits. Save a timestamped, privacy-reviewed evidence artifact without account data or credentials.
