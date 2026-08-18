# Fused MoE i8 tn Contract

Captured from the authenticated XPU-OJ problem page at
`https://xpuoj.com/contest/12/problem/1` on `2026-08-18T22:44:53+08:00`.
The page is authoritative and must be checked again before every submission.

## Selected Interface

- Language: CUDA Maca.
- Target: C500.
- Limits displayed by the problem: 10000 ms and 4096 MiB.
- Evaluator image: `maca torch 2.8.0+metax 3.7.1.5`.
- The page displayed zero accepted runs and zero submissions for the current account.
- No quota or cooldown was visible on the problem page; this remains a submission preflight gate.

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

The argument order is fixed. Only `out` may be written.

## Semantics And Shapes

```text
expert(r) = expert_ids[r / 128]
out[r,n] = int32_sum_k(a[r,k] * b_col_major[expert(r),n,k])
           * scale_a[r] * scale_b[expert(r),n] * moe_weights[r]
```

- `a`: contiguous INT8 `[EM,K]`, already expanded by routed row.
- `b_col_major`: contiguous INT8 `[256,N,K]`, logical layout `[expert,n,k]`.
- `scale_a`: FP32 `[EM]`, already expanded by routed row.
- `scale_b`: FP32 `[256,N]`.
- `moe_weights`: FP32 `[EM]`.
- `token_ids`: INT32 `[EM]`, provenance only; do not gather `a` again.
- `expert_ids`: INT32 `[EM/128]`.
- `topk`: always 8.
- `out`: BF16 `[EM,N]`.

| Case | EM | N | K | Distribution |
| --- | ---: | ---: | ---: | --- |
| decode gate-up | 4096 | 4096 | 7168 | uniform |
| prefill gate-up | 32768 | 4096 | 7168 | skewed |
| decode down | 4096 | 7168 | 2048 | uniform |
| prefill down | 32768 | 7168 | 2048 | skewed |

Input INT8 values are in `[-20,20]`; INT32 accumulation is exact for the
disclosed reductions. Sparse routing does not guarantee that every expert is
selected.

## Correctness Gate

```python
close = torch.isclose(out_target.float(), out_ref.float(), rtol=2e-2, atol=5e-3)
matched_ratio = close.float().mean()
passed = matched_ratio >= 0.99
```

The older local tutorial contains stale summaries (`torch.allclose` and a
pre-gather `token_ids` FAQ). The live page explicitly says `a` and `scale_a`
are already gathered and defines the matched-ratio gate above.

## Raw-Pointer Dimension Constraint

The CUDA ABI contains no shape arguments. The current official CUDA Maca
starter recognizes the four disclosed shapes through device allocation sizes
and then falls back to generated input values. This project keeps the
allocation-size mechanism, which is required by the ABI, but deliberately
removes the input-value fallback so the implementation does not identify a
generated sample. Failure to expose exact allocation ranges remains a target
correctness risk to verify through XPU-OJ.
