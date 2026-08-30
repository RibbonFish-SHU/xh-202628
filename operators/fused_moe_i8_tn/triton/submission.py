import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_kernel(
    a_ptr, b_ptr, out_ptr,
    scale_a_ptr, scale_b_ptr, moe_weights_ptr,
    token_ids_ptr, expert_ids_ptr,
    n_dim, k_dim, em, num_pid_m,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_om, stride_on,
    stride_sbe, stride_sbn,
    PRE_EXPANDED: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n_dim, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_mask = offs_m < em

    # one expert per 128-row tile (BLOCK_M == 128 == expert tile size)
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # a-row index: direct (pre-expanded) or gather via token_ids // top_k
    if PRE_EXPANDED:
        a_row = offs_m.to(tl.int64)
    else:
        tid = tl.load(token_ids_ptr + offs_m, mask=token_mask, other=0)
        a_row = (tid // top_k).to(tl.int64)

    # a_ptrs: a[a_row, k] -> [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    # b_ptrs: b[expert, n, k] read as [BLOCK_K, BLOCK_N] for tl.dot(a, b)
    b_ptrs = (b_ptr
              + expert * stride_be
              + offs_n[None, :] * stride_bn
              + offs_k[:, None] * stride_bk)

    # int8 x int8 -> int32 native IMMA, int32 accumulator (exact for these ranges)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_idx in range(0, tl.cdiv(k_dim, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0, eviction_policy="evict_last")
        b = tl.load(b_ptrs, eviction_policy="evict_first")
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # dequant: int32 -> fp32 once, then per-row a_scale * moe_w, per-col b_scale.
    acc = acc.to(tl.float32)
    a_scale = tl.load(scale_a_ptr + a_row, mask=token_mask, other=0.0)
    b_scale = tl.load(scale_b_ptr + expert * stride_sbe + offs_n * stride_sbn)
    moe_w = tl.load(moe_weights_ptr + offs_m, mask=token_mask, other=0.0)

    acc = acc * a_scale[:, None] * b_scale[None, :] * moe_w[:, None]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=token_mask[:, None])


def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights,
               token_ids, expert_ids, topk, out):
    em = moe_weights.shape[0]
    num_experts, n_dim, k_dim = b_col_major.shape

    # detect a convention: pre-expanded [EM,K] (OJ spec) vs un-gathered [EM//topk,K]
    if a.shape[0] == em:
        pre_expanded = True
    elif topk != 0 and a.shape[0] == em // topk:
        pre_expanded = False
    else:
        pre_expanded = True

    num_pid_m = em // 128
    num_pid_n = triton.cdiv(n_dim, 128)
    grid = (num_pid_m * num_pid_n,)
    _fused_moe_kernel[grid](
        a, b_col_major, out,
        scale_a, scale_b, moe_weights,
        token_ids, expert_ids,
        n_dim, k_dim, em, num_pid_m,
        a.stride(0), a.stride(1),
        b_col_major.stride(0), b_col_major.stride(1), b_col_major.stride(2),
        out.stride(0), out.stride(1),
        scale_b.stride(0), scale_b.stride(1),
        PRE_EXPANDED=pre_expanded,
        top_k=topk,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        GROUP_M=1,
        num_warps=8, num_stages=2,
    )
    return out
