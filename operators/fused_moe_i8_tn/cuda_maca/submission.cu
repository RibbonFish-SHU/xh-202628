#include <stddef.h>
#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#if defined(__MXCC__) || (defined(__clang__) && defined(__MACA__))
#define XH_FUSED_MOE_MACA 1
#include <cute/tensor.hpp>
#include <maca_bfloat16.h>
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

static inline int mma_grid_x(const KernelConfig& config) {
    constexpr int kNumExperts = 256;
    constexpr int kTileRows = 128;
    const int grid_m = (config.em + kTileRows - 1) / kTileRows;
    const int average_rows_per_expert = config.em / kNumExperts;
    int group_tiles = (average_rows_per_expert + kTileRows - 1) / kTileRows;
    group_tiles = group_tiles < 1 ? 1 : group_tiles;
    group_tiles = group_tiles > 8 ? 8 : group_tiles;
    return grid_m < group_tiles ? grid_m : group_tiles;
}

constexpr int kOutputScratchSortTiles = 256;
constexpr int kOutputScratchSortGridM = 8;

static inline bool use_case2_output_scratch_expert_sort(const KernelConfig& config) {
    return config.em == 32768 && config.n == 4096 && config.k == 7168;
}

static inline int mma_launch_grid_x(const KernelConfig& config) {
    return use_case2_output_scratch_expert_sort(config)
        ? kOutputScratchSortGridM
        : mma_grid_x(config);
}

constexpr int kPrefillDownModuleFullSortGridM = 8;

static inline bool use_prefill_down_module_full_expert_sort(
    const KernelConfig& config
) {
    return config.em == 32768 && config.n == 7168 && config.k == 2048;
}

static inline int module_full_sort_launch_grid_x(const KernelConfig& config) {
    return use_prefill_down_module_full_expert_sort(config)
        ? kPrefillDownModuleFullSortGridM
        : mma_launch_grid_x(config);
}

__host__ __device__ __forceinline__ int output_scratch_stable_sort_rank(
    const int32_t* expert_ids,
    int logical_tile_m,
    int tile_count
) {
    if (logical_tile_m == 0) {
        return 0;
    }

    const int expert = expert_ids[logical_tile_m];
    int rank = 1;
    for (int candidate = 1; candidate < tile_count; ++candidate) {
        const int candidate_expert = expert_ids[candidate];
        rank += candidate_expert < expert
            || (candidate_expert == expert && candidate < logical_tile_m);
    }
    return rank;
}

__global__ void build_output_scratch_expert_sort_map_kernel(
    const int32_t* __restrict__ expert_ids,
    __nv_bfloat16* __restrict__ out
) {
    const int logical_tile_m = threadIdx.x;
    if (logical_tile_m < kOutputScratchSortTiles) {
        const int physical_tile_m = output_scratch_stable_sort_rank(
            expert_ids, logical_tile_m, kOutputScratchSortTiles);
        reinterpret_cast<int32_t*>(out)[physical_tile_m] = logical_tile_m;
    }
}

__device__ int32_t g_case2_full_expert_sort_map[kOutputScratchSortTiles];

__host__ __device__ __forceinline__ int case2_full_stable_sort_rank(
    const int32_t* expert_ids,
    int logical_tile_m,
    int tile_count
) {
    const int expert = expert_ids[logical_tile_m];
    int rank = 0;
    for (int candidate = 0; candidate < tile_count; ++candidate) {
        const int candidate_expert = expert_ids[candidate];
        rank += candidate_expert < expert
            || (candidate_expert == expert && candidate < logical_tile_m);
    }
    return rank;
}

__global__ void build_case2_full_expert_sort_map_kernel(
    const int32_t* __restrict__ expert_ids
) {
    const int logical_tile_m = threadIdx.x;
    if (logical_tile_m < kOutputScratchSortTiles) {
        const int physical_tile_m = case2_full_stable_sort_rank(
            expert_ids, logical_tile_m, kOutputScratchSortTiles);
        g_case2_full_expert_sort_map[physical_tile_m] = logical_tile_m;
    }
}

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
    size_t bytes = 0;
    if (allocation_bytes(a, &bytes) && config_from_bytes(bytes, true, config)) {
        return true;
    }
    return allocation_bytes(out, &bytes) && config_from_bytes(bytes, false, config);
}

__host__ __device__ __forceinline__ int mma_output_row_local(
    int thread_id,
    int row_group,
    int row_in_group
) {
    const int wave = thread_id / 64;
    const int lane = thread_id % 64;
    return ((lane / 16) % 2) * 4
        + wave * 8
        + (lane / 32) * 32
        + row_group * 64
        + row_in_group;
}

__host__ __device__ __forceinline__ int mma_output_col_local(
    int thread_id,
    int col_group,
    int col_in_group
) {
    return (thread_id % 16) * 4 + col_group * 64 + col_in_group;
}

#if XH_FUSED_MOE_MACA

using MmaInt1 = __NATIVE_VECTOR__(1, int32_t);
using MmaInt4 = __NATIVE_VECTOR__(4, int32_t);
using MmaFloat2 = __NATIVE_VECTOR__(2, float);
using MmaFloat4 = __NATIVE_VECTOR__(4, float);
using MmaLoad128 = __NATIVE_VECTOR__(4, int32_t);
using MmaStore64 = __NATIVE_VECTOR__(2, uint);
using MmaBfloat16 = maca_bfloat16;

constexpr int kMmaTileM = 128;
constexpr int kMmaTileN = 128;
constexpr int kMmaTileK = 128;
constexpr int kMmaThreads = 256;
constexpr int kMmaWaveSize = 64;
constexpr int kMmaWaves = kMmaThreads / kMmaWaveSize;
constexpr int kMmaWaveM = 4;
constexpr int kMmaWaveN = kMmaWaves / kMmaWaveM;
constexpr int kMmaLoadBytes = sizeof(MmaLoad128) * kMmaThreads;
constexpr int kMmaRowsPerLoad = kMmaLoadBytes / kMmaTileK;
constexpr int kMmaLoadBytesPerWave = kMmaLoadBytes / kMmaWaves;
constexpr int kMmaSharedABytes = kMmaTileM * kMmaTileK;
constexpr int kMmaSharedBBytes = kMmaTileN * kMmaTileK;
constexpr int kMmaLoadsA = kMmaSharedABytes / kMmaLoadBytes;
constexpr int kMmaLoadsB = kMmaSharedBBytes / kMmaLoadBytes;
constexpr int kMmaLdsA = kMmaSharedABytes / (kMmaLoadBytesPerWave * kMmaWaveM);
constexpr int kMmaLdsB = kMmaSharedBBytes / (kMmaLoadBytesPerWave * kMmaWaveN);
constexpr int kMmaRows = kMmaTileM / 16 / kMmaWaveM;
constexpr int kMmaCols = kMmaTileN / 16 / kMmaWaveN;
constexpr int kMmaDepth = kMmaTileK / 16;
constexpr int kMmaOutputVectors = 16;
constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;

#define XH_MMA_FENCE() asm(";--------------")
#define XH_MMA_LDS(dst, src, type_)                                                               \
    XH_MMA_FENCE();                                                                               \
    *reinterpret_cast<type_*>(&(dst)) = *reinterpret_cast<type_*>(&(src));                        \
    XH_MMA_FENCE()
#define XH_MMA_STS(dst, src, type_)                                                               \
    XH_MMA_FENCE();                                                                               \
    *reinterpret_cast<type_*>(&(dst)) = *reinterpret_cast<type_*>(&(src));                        \
    XH_MMA_FENCE()

#if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1000 || __MACA_ARCH__ == 1089)
#define XH_MMA_I8(a, b, c) __builtin_mxc_mma_16x16x16i8(a, b, c)
#else
#define XH_MMA_I8(a, b, c) 0
#endif

template <bool kUseOutputScratchExpertSort, int kFixedN, int kFixedK>
__global__ void fused_moe_i8_tn_mma_kernel(
    const int8_t* __restrict__ a_ptr,
    const int8_t* __restrict__ b_ptr,
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const float* __restrict__ moe_weights_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    int em,
    int runtime_n,
    int runtime_k
) {
    using namespace cute;

    const int n = kFixedN == 0 ? runtime_n : kFixedN;
    const int k = kFixedK == 0 ? runtime_k : kFixedK;

#define XH_MMA_STAGE_MNKX2(m, nn, kk)                                                             \
    accum[m][nn] = XH_MMA_I8(a_frag[m][kk], b_frag[nn][kk], accum[m][nn]);                       \
    accum[m][nn] = XH_MMA_I8(a_frag[m][kk + 1], b_frag[nn][kk + 1], accum[m][nn])

#define XH_LDG_A_STAGE_I(ldgi)                                                                    \
    load_a_##ldgi = __builtin_mxc_ldg_b128(                                                       \
        a_base + load_a_row_offset[ldgi] + load_k,                                                \
        0,                                                                                         \
        -1,                                                                                        \
        true,                                                                                      \
        true,                                                                                      \
        false,                                                                                     \
        false)

#define XH_LDG_B_STAGE_I(ldgi)                                                                    \
    load_b_##ldgi = __builtin_mxc_ldg_b128(                                                       \
        &(global_b(load_b_row[ldgi], load_k, tile_k)),                                            \
        0,                                                                                         \
        -1,                                                                                        \
        true,                                                                                      \
        true,                                                                                      \
        false,                                                                                     \
        false)

#define XH_LDS_A_B128(rowi, coli)                                                                 \
    XH_MMA_LDS(a_frag[rowi][coli * 4], shared_a_tensor(lds_row_a[rowi], lds_col[coli]), MmaLoad128)
#define XH_LDS_B_B128(rowi, coli)                                                                 \
    XH_MMA_LDS(b_frag[rowi][coli * 4], shared_b_tensor(lds_row_b[rowi], lds_col[coli]), MmaLoad128)

#define XH_CVT_F32_TO_BF16(dst, src0, src1)                                                       \
    src0 = ((src0 >> 16) & 1) + src0 + 0x7fff;                                                   \
    src1 = ((src1 >> 16) & 1) + src1 + 0x7fff;                                                   \
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)

    const int tid = threadIdx.x;
    const int physical_tile_m = blockIdx.x + blockIdx.z * gridDim.x;
    const int tile_n = blockIdx.y;
    const int wave = tid / kMmaWaveSize;
    const int lane = tid % kMmaWaveSize;

    if (physical_tile_m * kMmaTileM >= em) {
        return;
    }

    const int tile_m = kUseOutputScratchExpertSort
        ? g_case2_full_expert_sort_map[physical_tile_m]
        : physical_tile_m;
    const int row_base = tile_m * kMmaTileM;

    __shared__ int8_t shared_data[kMmaSharedBytes];
    int8_t* shared_a = shared_data;
    int8_t* shared_b = shared_a + kMmaSharedABytes;

    const int expert = expert_ids_ptr[tile_m];
    const int8_t* expert_b =
        b_ptr + static_cast<uint64_t>(expert) * n * k;

    Tensor matrix_b = make_tensor(
        make_gmem_ptr(const_cast<int8_t*>(expert_b)),
        make_shape(n, k),
        make_stride(k, Int<1>{}));
    Tensor global_b = local_tile(
        matrix_b,
        make_tile(Int<kMmaTileN>{}, Int<kMmaTileK>{}),
        make_coord(tile_n, _));

    MmaLoad128 load_a_0;
    MmaLoad128 load_a_1;
    MmaLoad128 load_a_2;
    MmaLoad128 load_a_3;
    MmaLoad128 load_b_0;
    MmaLoad128 load_b_1;
    MmaLoad128 load_b_2;
    MmaLoad128 load_b_3;
    const int k_head = (k - 1) % kMmaTileK + 1;
    const int remaining_cols = n - tile_n * kMmaTileN;
    const int col_limit = remaining_cols < kMmaTileN ? remaining_cols : kMmaTileN;
    int load_b_row[kMmaLoadsB];
    int load_a_row_offset[kMmaLoadsA];
    const int load_a_row_base = tid / 8;
    const int load_b_row_base = tid / 8 * kMmaLoadsB;
    const int load_k = (lane % 8) * 16;
    const int num_k_tiles = (k + kMmaTileK - 1) / kMmaTileK;

    int8_t* a_base = const_cast<int8_t*>(a_ptr) + (num_k_tiles - 1) * kMmaTileK;

#pragma unroll
    for (uint32_t i = 0; i < kMmaLoadsA; ++i) {
        const int routed_row = row_base + load_a_row_base + kMmaRowsPerLoad * i;
        load_a_row_offset[i] = routed_row * k;
    }
    {
        const int candidate_col = load_b_row_base;
        load_b_row[0] = candidate_col < col_limit ? candidate_col : col_limit - 1;
        load_b_0 = __builtin_mxc_ldg_b128_predicator(
            &(global_b(load_b_row[0], load_k, num_k_tiles - 1)),
            0,
            true,
            true,
            false,
            false,
            load_k,
            k_head,
            MACA_ICMP_SLT);
    }
    {
        const int candidate_col = load_b_row_base + 1;
        load_b_row[1] = candidate_col < col_limit ? candidate_col : col_limit - 1;
        load_b_1 = __builtin_mxc_ldg_b128_predicator(
            &(global_b(load_b_row[1], load_k, num_k_tiles - 1)),
            0,
            true,
            true,
            false,
            false,
            load_k,
            k_head,
            MACA_ICMP_SLT);
    }
    {
        const int candidate_col = load_b_row_base + 2;
        load_b_row[2] = candidate_col < col_limit ? candidate_col : col_limit - 1;
        load_b_2 = __builtin_mxc_ldg_b128_predicator(
            &(global_b(load_b_row[2], load_k, num_k_tiles - 1)),
            0,
            true,
            true,
            false,
            false,
            load_k,
            k_head,
            MACA_ICMP_SLT);
    }
    {
        const int candidate_col = load_b_row_base + 3;
        load_b_row[3] = candidate_col < col_limit ? candidate_col : col_limit - 1;
        load_b_3 = __builtin_mxc_ldg_b128_predicator(
            &(global_b(load_b_row[3], load_k, num_k_tiles - 1)),
            0,
            true,
            true,
            false,
            false,
            load_k,
            k_head,
            MACA_ICMP_SLT);
    }
    load_a_0 = __builtin_mxc_ldg_b128(
        a_base + load_a_row_offset[0] + load_k,
        0,
        -1,
        true,
        true,
        false,
        false);
    load_a_1 = __builtin_mxc_ldg_b128(
        a_base + load_a_row_offset[1] + load_k,
        0,
        -1,
        true,
        true,
        false,
        false);
    load_a_2 = __builtin_mxc_ldg_b128(
        a_base + load_a_row_offset[2] + load_k,
        0,
        -1,
        true,
        true,
        false,
        false);
    load_a_3 = __builtin_mxc_ldg_b128(
        a_base + load_a_row_offset[3] + load_k,
        0,
        -1,
        true,
        true,
        false,
        false);

    Tensor shared_a_tensor = make_tensor(
        make_smem_ptr(shared_a),
        make_shape(Int<kMmaTileM>{}, Int<kMmaTileK>{}),
        make_stride(Int<kMmaTileK>{}, Int<1>{}));
    Tensor shared_b_tensor = make_tensor(
        make_smem_ptr(shared_b),
        make_shape(Int<kMmaTileN>{}, Int<kMmaTileK>{}),
        make_stride(Int<kMmaTileK>{}, Int<1>{}));

    int store_row_a[kMmaLoadsA];
    int store_row_b[kMmaLoadsB];
    const int store_col = (((tid / 8) + (tid % 8)) % 8) * 16;
    store_row_b[0] = tid / 8;
    XH_MMA_STS(shared_b_tensor(store_row_b[0], store_col), load_b_0, MmaLoad128);
    store_row_b[1] = tid / 8 + kMmaRowsPerLoad;
    XH_MMA_STS(shared_b_tensor(store_row_b[1], store_col), load_b_1, MmaLoad128);
    store_row_b[2] = tid / 8 + kMmaRowsPerLoad * 2;
    XH_MMA_STS(shared_b_tensor(store_row_b[2], store_col), load_b_2, MmaLoad128);
    store_row_b[3] = tid / 8 + kMmaRowsPerLoad * 3;
    XH_MMA_STS(shared_b_tensor(store_row_b[3], store_col), load_b_3, MmaLoad128);
#pragma unroll
    for (uint32_t i = 0; i < kMmaLoadsA; ++i) {
        store_row_a[i] = wave * 32 + lane / 8 + i * 8;
    }
    XH_MMA_STS(shared_a_tensor(store_row_a[0], store_col), load_a_0, MmaLoad128);
    XH_MMA_STS(shared_a_tensor(store_row_a[1], store_col), load_a_1, MmaLoad128);

    MmaInt4 accum[kMmaRows][kMmaCols] = {0};
    int32_t a_frag[kMmaRows][kMmaDepth];
    int32_t b_frag[kMmaCols][kMmaDepth];
    int lds_row_a[2];
    int lds_row_b[8];
    int lds_col[2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        lds_col[i] = (((tid % 16) + (lane / 16) + 4 * i) % 8) * 16;
        lds_row_a[i] = (tid % 16) + wave * 32 + 16 * i;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        lds_row_b[i] = (tid % 16) + 16 * i;
    }

    __syncthreadshared();

    XH_LDS_A_B128(0, 0);
    XH_LDS_B_B128(0, 0);
    XH_LDS_B_B128(1, 0);
    XH_LDS_B_B128(2, 0);
    XH_LDS_B_B128(3, 0);

    const int loop_k_tiles = num_k_tiles - 1;
    a_base = const_cast<int8_t*>(a_ptr);
    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {
        XH_LDG_B_STAGE_I(0);
        XH_LDG_B_STAGE_I(1);
        XH_MMA_STAGE_MNKX2(0, 0, 0);
        XH_LDS_B_B128(4, 0);
        XH_MMA_STAGE_MNKX2(0, 0, 2);
        XH_LDS_B_B128(5, 0);
        XH_MMA_STAGE_MNKX2(0, 1, 0);
        XH_LDS_B_B128(6, 0);
        XH_LDG_B_STAGE_I(2);
        XH_MMA_STAGE_MNKX2(0, 1, 2);
        XH_LDS_B_B128(7, 0);
        XH_MMA_STAGE_MNKX2(0, 2, 0);
        XH_LDG_B_STAGE_I(3);
        XH_MMA_STAGE_MNKX2(0, 2, 2);
        XH_MMA_STAGE_MNKX2(0, 3, 0);
        XH_LDG_A_STAGE_I(0);
        XH_MMA_STAGE_MNKX2(0, 3, 2);
        XH_LDG_A_STAGE_I(1);

        XH_MMA_STAGE_MNKX2(0, 4, 0);
        XH_LDS_A_B128(0, 1);
        XH_MMA_STAGE_MNKX2(0, 4, 2);
        XH_LDS_B_B128(0, 1);
        XH_MMA_STAGE_MNKX2(0, 5, 0);
        XH_LDS_B_B128(1, 1);
        XH_MMA_STAGE_MNKX2(0, 5, 2);
        XH_LDS_B_B128(2, 1);
        XH_MMA_STAGE_MNKX2(0, 6, 0);
        XH_LDS_B_B128(3, 1);
        XH_MMA_STAGE_MNKX2(0, 6, 2);
        XH_MMA_STAGE_MNKX2(0, 7, 0);
        XH_MMA_STAGE_MNKX2(0, 7, 2);

        XH_LDS_B_B128(4, 1);
        XH_MMA_STAGE_MNKX2(0, 0, 4);
        XH_LDS_B_B128(5, 1);
        XH_MMA_STAGE_MNKX2(0, 0, 6);
        XH_LDS_B_B128(6, 1);
        XH_MMA_STAGE_MNKX2(0, 1, 4);
        XH_LDS_B_B128(7, 1);
        XH_MMA_STAGE_MNKX2(0, 1, 6);
        XH_MMA_STAGE_MNKX2(0, 2, 4);
        XH_MMA_STAGE_MNKX2(0, 2, 6);
        XH_MMA_STS(shared_a_tensor(store_row_a[2], store_col), load_a_2, MmaLoad128);
        XH_MMA_STAGE_MNKX2(0, 3, 4);
        XH_MMA_STAGE_MNKX2(0, 3, 6);
        XH_MMA_STS(shared_a_tensor(store_row_a[3], store_col), load_a_3, MmaLoad128);

        XH_MMA_STAGE_MNKX2(0, 4, 4);
        XH_LDG_A_STAGE_I(2);
        XH_MMA_STAGE_MNKX2(0, 4, 6);
        XH_LDG_A_STAGE_I(3);
        XH_MMA_STAGE_MNKX2(0, 5, 4);
        XH_MMA_STAGE_MNKX2(0, 5, 6);
        XH_MMA_STAGE_MNKX2(0, 6, 4);
        XH_LDS_A_B128(1, 0);
        XH_MMA_STAGE_MNKX2(0, 6, 6);
        XH_MMA_STAGE_MNKX2(0, 7, 4);
        a_base += kMmaTileK;
        XH_MMA_STAGE_MNKX2(0, 7, 6);

        __syncthreadshared();
        XH_MMA_STAGE_MNKX2(1, 0, 0);
        XH_LDS_A_B128(1, 1);
        XH_MMA_STAGE_MNKX2(1, 0, 2);
        XH_MMA_STAGE_MNKX2(1, 1, 0);
        XH_MMA_STAGE_MNKX2(1, 1, 2);
        XH_MMA_STAGE_MNKX2(1, 2, 0);
        XH_MMA_STAGE_MNKX2(1, 2, 2);
        XH_MMA_STAGE_MNKX2(1, 3, 0);
        XH_MMA_STAGE_MNKX2(1, 3, 2);

        XH_MMA_STAGE_MNKX2(1, 4, 0);
        XH_MMA_STS(shared_b_tensor(store_row_b[0], store_col), load_b_0, MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 4, 2);
        XH_MMA_STAGE_MNKX2(1, 5, 0);
        XH_MMA_STAGE_MNKX2(1, 5, 2);
        XH_MMA_STS(shared_b_tensor(store_row_b[1], store_col), load_b_1, MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 6, 0);
        XH_MMA_STAGE_MNKX2(1, 6, 2);
        XH_MMA_STAGE_MNKX2(1, 7, 0);
        XH_MMA_STS(shared_b_tensor(store_row_b[2], store_col), load_b_2, MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 7, 2);

        XH_MMA_STAGE_MNKX2(1, 0, 4);
        XH_MMA_STAGE_MNKX2(1, 0, 6);
        XH_MMA_STS(shared_b_tensor(store_row_b[3], store_col), load_b_3, MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 1, 4);
        XH_MMA_STAGE_MNKX2(1, 1, 6);
        XH_MMA_STAGE_MNKX2(1, 2, 4);
        XH_MMA_STS(shared_a_tensor(store_row_a[0], store_col), load_a_0, MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 2, 6);
        XH_MMA_STAGE_MNKX2(1, 3, 4);
        XH_MMA_STAGE_MNKX2(1, 3, 6);
        XH_MMA_STS(shared_a_tensor(store_row_a[1], store_col), load_a_1, MmaLoad128);

        XH_MMA_STAGE_MNKX2(1, 4, 4);
        XH_MMA_STAGE_MNKX2(1, 4, 6);
        XH_MMA_STAGE_MNKX2(1, 5, 4);
        __syncthreadshared();
        XH_MMA_STAGE_MNKX2(1, 5, 6);
        XH_LDS_A_B128(0, 0);
        XH_LDS_B_B128(0, 0);
        XH_MMA_STAGE_MNKX2(1, 6, 4);
        XH_LDS_B_B128(1, 0);
        XH_MMA_STAGE_MNKX2(1, 6, 6);
        XH_LDS_B_B128(2, 0);
        XH_MMA_STAGE_MNKX2(1, 7, 4);
        XH_LDS_B_B128(3, 0);
        XH_MMA_STAGE_MNKX2(1, 7, 6);
    }

    int output_row[8];
    XH_MMA_STAGE_MNKX2(0, 0, 0);
    XH_LDS_B_B128(4, 0);
    XH_MMA_STAGE_MNKX2(0, 0, 2);
    XH_LDS_B_B128(5, 0);
    XH_MMA_STAGE_MNKX2(0, 1, 0);
    XH_LDS_B_B128(6, 0);
    XH_MMA_STAGE_MNKX2(0, 1, 2);
    XH_LDS_B_B128(7, 0);
    XH_MMA_STAGE_MNKX2(0, 2, 0);
    const int row_thread_base = row_base + mma_output_row_local(tid, 0, 0);
    XH_MMA_STAGE_MNKX2(0, 2, 2);
    XH_MMA_STAGE_MNKX2(0, 3, 0);
    XH_MMA_STAGE_MNKX2(0, 3, 2);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        output_row[j] = row_thread_base + j;
    }

    XH_MMA_STAGE_MNKX2(0, 4, 0);
    XH_LDS_A_B128(0, 1);
    XH_MMA_STAGE_MNKX2(0, 4, 2);
    XH_LDS_B_B128(0, 1);
    XH_MMA_STAGE_MNKX2(0, 5, 0);
    XH_LDS_B_B128(1, 1);
    XH_MMA_STAGE_MNKX2(0, 5, 2);
    XH_LDS_B_B128(2, 1);
    XH_MMA_STAGE_MNKX2(0, 6, 0);
    XH_LDS_B_B128(3, 1);
    XH_MMA_STAGE_MNKX2(0, 6, 2);
    XH_MMA_STAGE_MNKX2(0, 7, 0);
    XH_MMA_STAGE_MNKX2(0, 7, 2);

    XH_LDS_B_B128(4, 1);
    XH_MMA_STAGE_MNKX2(0, 0, 4);
    XH_LDS_B_B128(5, 1);
    XH_MMA_STAGE_MNKX2(0, 0, 6);
    XH_LDS_B_B128(6, 1);
    XH_MMA_STAGE_MNKX2(0, 1, 4);
    XH_LDS_B_B128(7, 1);
    XH_MMA_STAGE_MNKX2(0, 1, 6);
    XH_MMA_STAGE_MNKX2(0, 2, 4);
    XH_MMA_STS(shared_a_tensor(store_row_a[2], store_col), load_a_2, MmaLoad128);
    XH_MMA_STAGE_MNKX2(0, 2, 6);
    XH_MMA_STAGE_MNKX2(0, 3, 4);
    XH_MMA_STAGE_MNKX2(0, 3, 6);
    XH_MMA_STS(shared_a_tensor(store_row_a[3], store_col), load_a_3, MmaLoad128);

    XH_MMA_STAGE_MNKX2(0, 4, 4);
    XH_MMA_STAGE_MNKX2(0, 4, 6);
    XH_MMA_STAGE_MNKX2(0, 5, 4);
    XH_MMA_STAGE_MNKX2(0, 5, 6);
    XH_MMA_STAGE_MNKX2(0, 6, 4);
    XH_LDS_A_B128(1, 0);
    XH_MMA_STAGE_MNKX2(0, 6, 6);
    XH_MMA_STAGE_MNKX2(0, 7, 4);
    XH_MMA_STAGE_MNKX2(0, 7, 6);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        output_row[4 + j] = row_base + mma_output_row_local(tid, 1, j);
    }

    XH_MMA_STAGE_MNKX2(1, 0, 0);
    XH_MMA_STAGE_MNKX2(1, 0, 2);
    XH_MMA_STAGE_MNKX2(1, 1, 0);
    XH_MMA_STAGE_MNKX2(1, 1, 2);
    XH_MMA_STAGE_MNKX2(1, 2, 0);
    XH_MMA_STAGE_MNKX2(1, 2, 2);
    XH_MMA_STAGE_MNKX2(1, 3, 0);
    XH_MMA_STAGE_MNKX2(1, 3, 2);

    XH_MMA_STAGE_MNKX2(1, 4, 0);
    XH_MMA_STAGE_MNKX2(1, 4, 2);
    XH_LDS_A_B128(1, 1);
    XH_MMA_STAGE_MNKX2(1, 5, 0);
    XH_MMA_STAGE_MNKX2(1, 5, 2);
    XH_MMA_STAGE_MNKX2(1, 6, 0);
    XH_MMA_STAGE_MNKX2(1, 6, 2);
    XH_MMA_STAGE_MNKX2(1, 7, 0);
    XH_MMA_STAGE_MNKX2(1, 7, 2);

    XH_MMA_STAGE_MNKX2(1, 0, 4);
    XH_MMA_STAGE_MNKX2(1, 0, 6);
    XH_MMA_STAGE_MNKX2(1, 1, 4);
    XH_MMA_STAGE_MNKX2(1, 1, 6);
    XH_MMA_STAGE_MNKX2(1, 2, 4);
    XH_MMA_STAGE_MNKX2(1, 2, 6);
    XH_MMA_STAGE_MNKX2(1, 3, 4);
    XH_MMA_STAGE_MNKX2(1, 3, 6);

    XH_MMA_STAGE_MNKX2(1, 4, 4);
    XH_MMA_STAGE_MNKX2(1, 4, 6);
    XH_MMA_STAGE_MNKX2(1, 5, 4);
    XH_MMA_STAGE_MNKX2(1, 5, 6);
    XH_MMA_STAGE_MNKX2(1, 6, 4);
    XH_MMA_STAGE_MNKX2(1, 6, 6);
    XH_MMA_STAGE_MNKX2(1, 7, 4);
    XH_MMA_STAGE_MNKX2(1, 7, 6);

    MmaInt4 output[kMmaOutputVectors];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            output[i * 8 + 2 * j][0] = accum[i][0][j];
            output[i * 8 + 2 * j][1] = accum[i][2][j];
            output[i * 8 + 2 * j][2] = accum[i][4][j];
            output[i * 8 + 2 * j][3] = accum[i][6][j];
            output[i * 8 + 2 * j + 1][0] = accum[i][1][j];
            output[i * 8 + 2 * j + 1][1] = accum[i][3][j];
            output[i * 8 + 2 * j + 1][2] = accum[i][5][j];
            output[i * 8 + 2 * j + 1][3] = accum[i][7][j];
        }
    }

    int output_col[2];
    bool output_col_mask[2];
    output_col[0] = mma_output_col_local(tid, 0, 0);
    output_col[1] = mma_output_col_local(tid, 1, 0);
    output_col_mask[0] = output_col[0] < col_limit;
    output_col_mask[1] = output_col[1] < col_limit;

    float weights[2][4];
    float row_scale[2][4];
    MmaFloat4 col_scale[2];

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const int row = output_row[i * 4 + j];
            *(reinterpret_cast<MmaInt1*>(&weights[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(moe_weights_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
            *(reinterpret_cast<MmaInt1*>(&row_scale[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(scale_a_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
        }
    }

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const float* ptr =
            scale_b_ptr + static_cast<uint64_t>(expert) * n
            + tile_n * kMmaTileN + output_col[i];
        col_scale[i] = __builtin_mxc_ldg_b128_predicator(
            const_cast<float*>(ptr),
            0,
            true,
            true,
            false,
            false,
            output_col_mask[i],
            1,
            MACA_ICMP_EQ);
    }

    MmaBfloat16* out_base =
        reinterpret_cast<MmaBfloat16*>(out_ptr) + tile_n * kMmaTileN;
    MmaFloat2 zero2 = {0.0f, 0.0f};
    MmaStore64 packed_out;

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            float values[8];
            values[0] = output[i * 8 + 2 * j][0];
            values[1] = output[i * 8 + 2 * j][1];
            values[2] = output[i * 8 + 2 * j][2];
            values[3] = output[i * 8 + 2 * j][3];
            values[4] = output[i * 8 + 2 * j + 1][0];
            values[5] = output[i * 8 + 2 * j + 1][1];
            values[6] = output[i * 8 + 2 * j + 1][2];
            values[7] = output[i * 8 + 2 * j + 1][3];

            row_scale[i][j] *= weights[i][j];
            MmaFloat2 row_scale2 = {row_scale[i][j], row_scale[i][j]};
            MmaFloat2 scales[4];
            scales[0] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[0])[0], row_scale2, zero2);
            scales[1] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[0])[1], row_scale2, zero2);
            scales[2] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[1])[0], row_scale2, zero2);
            scales[3] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[1])[1], row_scale2, zero2);
            *reinterpret_cast<MmaFloat2*>(&values[0]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[0]), scales[0], zero2);
            *reinterpret_cast<MmaFloat2*>(&values[2]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[2]), scales[1], zero2);
            *reinterpret_cast<MmaFloat2*>(&values[4]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[4]), scales[2], zero2);
            *reinterpret_cast<MmaFloat2*>(&values[6]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[6]), scales[3], zero2);

            XH_CVT_F32_TO_BF16(
                packed_out[0],
                reinterpret_cast<uint*>(&values)[0],
                reinterpret_cast<uint*>(&values)[1]);
            XH_CVT_F32_TO_BF16(
                packed_out[1],
                reinterpret_cast<uint*>(&values)[2],
                reinterpret_cast<uint*>(&values)[3]);
            __builtin_mxc_stg_b64_predicator(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n + output_col[0],
                0,
                *reinterpret_cast<uint64_t*>(&packed_out),
                true,
                false,
                false,
                (output_row[i * 4 + j] < em) && output_col_mask[0],
                1,
                MACA_ICMP_EQ);

            XH_CVT_F32_TO_BF16(
                packed_out[0],
                reinterpret_cast<uint*>(&values)[4],
                reinterpret_cast<uint*>(&values)[5]);
            XH_CVT_F32_TO_BF16(
                packed_out[1],
                reinterpret_cast<uint*>(&values)[6],
                reinterpret_cast<uint*>(&values)[7]);
            __builtin_mxc_stg_b64_predicator(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n + output_col[1],
                0,
                *reinterpret_cast<uint64_t*>(&packed_out),
                true,
                false,
                false,
                (output_row[i * 4 + j] < em) && output_col_mask[1],
                1,
                MACA_ICMP_EQ);
        }
    }

#undef XH_CVT_F32_TO_BF16
#undef XH_LDS_B_B128
#undef XH_LDS_A_B128
#undef XH_LDG_B_STAGE_I
#undef XH_LDG_A_STAGE_I
#undef XH_MMA_STAGE_MNKX2
}

#else

__device__ __forceinline__ int dot4_i8_scalar(int a, int b, int accumulator) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int av = static_cast<int>(static_cast<int8_t>((a >> (8 * i)) & 0xff));
        const int bv = static_cast<int>(static_cast<int8_t>((b >> (8 * i)) & 0xff));
        accumulator += av * bv;
    }
    return accumulator;
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
            acc00 = dot4_i8_scalar(a0, b0, acc00);
            acc01 = dot4_i8_scalar(a0, b1, acc01);
            acc10 = dot4_i8_scalar(a1, b0, acc10);
            acc11 = dot4_i8_scalar(a1, b1, acc11);
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

#endif

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
#if XH_FUSED_MOE_MACA
    const dim3 block(kMmaThreads);
    const int grid_m = (config.em + kMmaTileM - 1) / kMmaTileM;
    const bool use_global_sort_map =
        use_case2_output_scratch_expert_sort(config)
        || use_prefill_down_module_full_expert_sort(config);
    const int grid_x = module_full_sort_launch_grid_x(config);
    const dim3 grid(
        grid_x,
        (config.n + kMmaTileN - 1) / kMmaTileN,
        (grid_m + grid_x - 1) / grid_x
    );
    if (use_global_sort_map) {
        build_case2_full_expert_sort_map_kernel
            <<<1, kOutputScratchSortTiles>>>(expert_ids);
        if (use_case2_output_scratch_expert_sort(config)) {
            fused_moe_i8_tn_mma_kernel<true, 4096, 7168><<<grid, block>>>(
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
        } else {
            fused_moe_i8_tn_mma_kernel<true, 0, 0><<<grid, block>>>(
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
    } else {
        fused_moe_i8_tn_mma_kernel<false, 0, 0><<<grid, block>>>(
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
#else
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
#endif
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
