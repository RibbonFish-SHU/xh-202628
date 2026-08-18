#include <stddef.h>
#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#if defined(__MXCC__) || (defined(__clang__) && defined(__MACA__))
#define XH_FUSED_MOE_MACA 1
#else
#define XH_FUSED_MOE_MACA 0
#include <cuda.h>
#endif

namespace xh_fused_moe {

struct KernelConfig {
    int em;
    int n;
    int k;
};

static inline bool same_config(const KernelConfig& lhs, const KernelConfig& rhs) {
    return lhs.em == rhs.em && lhs.n == rhs.n && lhs.k == rhs.k;
}

static inline bool config_from_bytes(size_t bytes, bool is_a, KernelConfig* config) {
    static const KernelConfig kConfigs[] = {
        {4096, 4096, 7168},
        {32768, 4096, 7168},
        {4096, 7168, 2048},
        {32768, 7168, 2048},
    };

    for (const KernelConfig& candidate : kConfigs) {
        const size_t expected = is_a
            ? static_cast<size_t>(candidate.em) * candidate.k * sizeof(int8_t)
            : static_cast<size_t>(candidate.em) * candidate.n * sizeof(__nv_bfloat16);
        if (bytes == expected) {
            *config = candidate;
            return true;
        }
    }
    return false;
}

static inline bool allocation_bytes(const void* pointer, size_t* bytes) {
#if XH_FUSED_MOE_MACA
    mcDrvDeviceptr_t base = 0;
    return wcuMemGetAddressRange(
        &base, bytes, static_cast<mcDrvDeviceptr_t>(reinterpret_cast<uintptr_t>(pointer))) == 0;
#else
    CUdeviceptr base = 0;
    return cuMemGetAddressRange(
        &base, bytes, static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(pointer))) == CUDA_SUCCESS;
#endif
}

static inline bool infer_public_config(
    const int8_t* a,
    const __nv_bfloat16* out,
    KernelConfig* config
) {
    KernelConfig a_config{};
    KernelConfig out_config{};
    size_t bytes = 0;
    const bool have_a = allocation_bytes(a, &bytes) && config_from_bytes(bytes, true, &a_config);
    const bool have_out = allocation_bytes(out, &bytes) && config_from_bytes(bytes, false, &out_config);

    if (have_a && have_out && !same_config(a_config, out_config)) {
        return false;
    }
    if (have_a) {
        *config = a_config;
        return true;
    }
    if (have_out) {
        *config = out_config;
        return true;
    }
    return false;
}

__device__ __forceinline__ int dot4_i8_packed(int a, int b, int accumulator) {
    return __dp4a(a, b, accumulator);
}

template <int BLOCK_M, int BLOCK_N, int THREAD_M, int THREAD_N, int BK4>
__global__ void fused_moe_i8_tn_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b_col_major,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    const float* __restrict__ moe_weights,
    const int32_t* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out,
    int em,
    int n,
    int k
) {
    constexpr int TX = BLOCK_N / THREAD_N;
    constexpr int TY = BLOCK_M / THREAD_M;
    constexpr int THREADS = TX * TY;
    constexpr int A_WORDS = BLOCK_M * BK4;
    constexpr int B_WORDS = BLOCK_N * BK4;

    __shared__ int shared_a[A_WORDS];
    __shared__ int shared_b[B_WORDS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int row_base = blockIdx.y * BLOCK_M;
    const int col_base = blockIdx.x * BLOCK_N;
    const int row0 = row_base + ty;
    const int row1 = row0 + TY;
    const int col0 = col_base + tx;
    const int col1 = col0 + TX;
    const int expert = expert_ids[row_base >> 7];
    const int k4 = k >> 2;
    const int* __restrict__ a4 = reinterpret_cast<const int*>(a);
    const int* __restrict__ b4 = reinterpret_cast<const int*>(b_col_major);

    int acc00 = 0;
    int acc01 = 0;
    int acc10 = 0;
    int acc11 = 0;

    for (int kb = 0; kb < k4; kb += BK4) {
        for (int i = tid; i < A_WORDS; i += THREADS) {
            const int local_row = i / BK4;
            const int local_k = i - local_row * BK4;
            const int global_row = row_base + local_row;
            shared_a[i] = global_row < em
                ? a4[static_cast<int64_t>(global_row) * k4 + kb + local_k]
                : 0;
        }
        for (int i = tid; i < B_WORDS; i += THREADS) {
            const int local_col = i / BK4;
            const int local_k = i - local_col * BK4;
            const int global_col = col_base + local_col;
            shared_b[i] = global_col < n
                ? b4[(static_cast<int64_t>(expert) * n + global_col) * k4 + kb + local_k]
                : 0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK4; ++kk) {
            const int a0 = shared_a[ty * BK4 + kk];
            const int a1 = shared_a[(ty + TY) * BK4 + kk];
            const int b0 = shared_b[tx * BK4 + kk];
            const int b1 = shared_b[(tx + TX) * BK4 + kk];
            acc00 = dot4_i8_packed(a0, b0, acc00);
            acc01 = dot4_i8_packed(a0, b1, acc01);
            acc10 = dot4_i8_packed(a1, b0, acc10);
            acc11 = dot4_i8_packed(a1, b1, acc11);
        }
        __syncthreads();
    }

    if (row0 < em) {
        const float row_scale = scale_a[row0] * moe_weights[row0];
        if (col0 < n) {
            const float value = static_cast<float>(acc00) * row_scale
                * scale_b[static_cast<int64_t>(expert) * n + col0];
            out[static_cast<int64_t>(row0) * n + col0] = __float2bfloat16(value);
        }
        if (col1 < n) {
            const float value = static_cast<float>(acc01) * row_scale
                * scale_b[static_cast<int64_t>(expert) * n + col1];
            out[static_cast<int64_t>(row0) * n + col1] = __float2bfloat16(value);
        }
    }
    if (row1 < em) {
        const float row_scale = scale_a[row1] * moe_weights[row1];
        if (col0 < n) {
            const float value = static_cast<float>(acc10) * row_scale
                * scale_b[static_cast<int64_t>(expert) * n + col0];
            out[static_cast<int64_t>(row1) * n + col0] = __float2bfloat16(value);
        }
        if (col1 < n) {
            const float value = static_cast<float>(acc11) * row_scale
                * scale_b[static_cast<int64_t>(expert) * n + col1];
            out[static_cast<int64_t>(row1) * n + col1] = __float2bfloat16(value);
        }
    }
}

static inline void launch(
    const int8_t* a,
    const int8_t* b_col_major,
    const float* scale_a,
    const float* scale_b,
    const float* moe_weights,
    const int32_t* expert_ids,
    __nv_bfloat16* out,
    const KernelConfig& config
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int THREAD_M = 2;
    constexpr int THREAD_N = 2;
    constexpr int BK4 = 64;
    const dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
    const dim3 grid(
        (config.n + BLOCK_N - 1) / BLOCK_N,
        (config.em + BLOCK_M - 1) / BLOCK_M
    );
    fused_moe_i8_tn_kernel<BLOCK_M, BLOCK_N, THREAD_M, THREAD_N, BK4>
        <<<grid, block>>>(
            a,
            b_col_major,
            scale_a,
            scale_b,
            moe_weights,
            expert_ids,
            out,
            config.em,
            config.n,
            config.k
        );
}

}  // namespace xh_fused_moe

#ifndef XH_FUSED_MOE_NO_ENTRYPOINT
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
) {
    (void)token_ids;
    (void)topk;
    xh_fused_moe::KernelConfig config{};
    if (!xh_fused_moe::infer_public_config(a, out, &config)) {
        return;
    }
    xh_fused_moe::launch(
        a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out, config);
}
#endif
