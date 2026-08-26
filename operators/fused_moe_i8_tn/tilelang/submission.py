import tilelang
import tilelang.language as T
from tilelang import jit

K_TILE_M = 128

_kernel_cache = {}


@jit
def fused_moe_i8_tn_kernel(EM, N, K, E, block_N=128, block_K=128, num_stages=1, threads=256):
    @T.prim_func
    def kernel(
        A: T.Tensor((EM, K), "int8"),
        B: T.Tensor((E, N, K), "int8"),
        ScaleA: T.Tensor((EM,), "float32"),
        Sb: T.Tensor((E, N), "float32"),
        MoeW: T.Tensor((EM,), "float32"),
        Eid: T.Tensor((EM // K_TILE_M,), "int32"),
        Out: T.Tensor((EM, N), "bfloat16"),
    ):
        block_M = K_TILE_M
        num_tiles = EM // block_M

        with T.Kernel(num_tiles, N // block_N, threads=threads) as (bt, bn):
            A_shared = T.alloc_shared((block_M, block_K), "int8")
            B_shared = T.alloc_shared((block_N, block_K), "int8")
            C_local = T.alloc_fragment((block_M, block_N), "int32")

            e = Eid[bt]
            row0 = bt * block_M
            col0 = bn * block_N

            T.clear(C_local)
            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[row0, k * block_K], A_shared)
                T.copy(B[e, col0, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                Out[row0 + i, col0 + j] = T.Cast(
                    "bfloat16",
                    T.Cast("float32", C_local[i, j])
                    * ScaleA[row0 + i]
                    * MoeW[row0 + i]
                    * Sb[e, col0 + j],
                )

    return kernel


@jit
def fused_moe_i8_tn_prefill_sorted_kernel(
    EM, N, K, E, block_N=128, block_K=128, num_stages=1, threads=256
):
    @T.prim_func
    def kernel(
        A: T.Tensor((EM, K), "int8"),
        B: T.Tensor((E, N, K), "int8"),
        ScaleA: T.Tensor((EM,), "float32"),
        Sb: T.Tensor((E, N), "float32"),
        MoeW: T.Tensor((EM,), "float32"),
        Eid: T.Tensor((EM // K_TILE_M,), "int32"),
        Out: T.Tensor((EM, N), "bfloat16"),
    ):
        block_M = K_TILE_M
        num_tiles = EM // block_M
        RankMap = T.alloc_global((num_tiles,), "int32")

        with T.Kernel(1, 1, threads=threads) as (_, __):
            rank = T.alloc_var("int32")
            for logical in T.Parallel(num_tiles):
                expert = Eid[logical]
                rank[0] = 0
                for candidate in T.serial(num_tiles):
                    candidate_expert = Eid[candidate]
                    if candidate_expert < expert:
                        rank[0] = rank[0] + 1
                    else:
                        if candidate_expert == expert:
                            if candidate < logical:
                                rank[0] = rank[0] + 1
                RankMap[rank[0]] = logical

        with T.Kernel(8, N // block_N, num_tiles // 8, threads=threads) as (bx, bn, bz):
            physical = bx + 8 * bz
            bt = RankMap[physical]
            A_shared = T.alloc_shared((block_M, block_K), "int8")
            B_shared = T.alloc_shared((block_N, block_K), "int8")
            C_local = T.alloc_fragment((block_M, block_N), "int32")

            e = Eid[bt]
            row0 = bt * block_M
            col0 = bn * block_N

            T.clear(C_local)
            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[row0, k * block_K], A_shared)
                T.copy(B[e, col0, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                Out[row0 + i, col0 + j] = T.Cast(
                    "bfloat16",
                    T.Cast("float32", C_local[i, j])
                    * ScaleA[row0 + i]
                    * MoeW[row0 + i]
                    * Sb[e, col0 + j],
                )

    return kernel


def _cached_kernel(EM, N, K, E):
    key = (EM, N, K, E)
    kernel = _kernel_cache.get(key)
    if kernel is None:
        builder = fused_moe_i8_tn_prefill_sorted_kernel if EM == 32768 else fused_moe_i8_tn_kernel
        kernel = builder(EM=EM, N=N, K=K, E=E)
        _kernel_cache[key] = kernel
    return kernel


def run_kernel(a, b_col_major, scale_a, scale_b, moe_weights, token_ids, expert_ids, topk, out):
    EM = out.shape[0]
    E, N, K = b_col_major.shape

    kernel = _cached_kernel(int(EM), int(N), int(K), int(E))
    kernel(a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out)
    return out
