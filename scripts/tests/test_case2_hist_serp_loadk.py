#!/usr/bin/env python3
"""No-device proof for the exp-228 exact-case2 composition."""

from __future__ import annotations

import hashlib
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260829-228.sh"
BASELINE_SHA256 = "083eb1262dbe220aea0c2b324a00f2cf9b14720dc2b8c1701b261f794eaf8cf1"

CASE1 = (4096, 4096, 7168)
CASE2 = (32768, 4096, 7168)
CASE3 = (4096, 7168, 2048)
CASE4 = (32768, 7168, 2048)
TILE_M = 128
TILE_N = 128
TILE_K = 128
CLUSTER_WIDTH = 8

CONSTANTS = """constexpr int kFullSortPayloadTileBits = 8;
constexpr uint32_t kFullSortPayloadTileMask =
    (1U << kFullSortPayloadTileBits) - 1U;
"""

HELPERS = """__host__ __device__ __forceinline__ int32_t pack_full_sort_payload(
    int expert,
    int logical_tile_m
) {
    return static_cast<int32_t>(
        (static_cast<uint32_t>(expert) << kFullSortPayloadTileBits)
        | static_cast<uint32_t>(logical_tile_m));
}

__host__ __device__ __forceinline__ int unpack_full_sort_tile(int32_t payload) {
    return static_cast<int>(
        static_cast<uint32_t>(payload) & kFullSortPayloadTileMask);
}

__host__ __device__ __forceinline__ int unpack_full_sort_expert(int32_t payload) {
    return static_cast<int>(
        static_cast<uint32_t>(payload) >> kFullSortPayloadTileBits);
}

"""

HISTOGRAM_BUILDER = """    const int logical_tile_m = threadIdx.x;
    __shared__ int32_t expert_offsets[kOutputScratchSortTiles];

    expert_offsets[logical_tile_m] = 0;
    __syncthreads();

    const int expert = expert_ids[logical_tile_m];
    atomicAdd(&expert_offsets[expert], 1);
    __syncthreads();

    if (logical_tile_m == 0) {
        int32_t next_offset = 0;
        for (int current_expert = 0;
             current_expert < kOutputScratchSortTiles;
             ++current_expert) {
            const int32_t expert_count = expert_offsets[current_expert];
            expert_offsets[current_expert] = next_offset;
            next_offset += expert_count;
        }
    }
    __syncthreads();

    const int physical_tile_m = atomicAdd(&expert_offsets[expert], 1);
    g_case2_full_expert_sort_map[physical_tile_m] =
        pack_full_sort_payload(expert, logical_tile_m);"""

BASELINE_BUILDER = """    const int logical_tile_m = threadIdx.x;
    if (logical_tile_m < kOutputScratchSortTiles) {
        const int physical_tile_m = case2_full_stable_sort_rank(
            expert_ids, logical_tile_m, kOutputScratchSortTiles);
        g_case2_full_expert_sort_map[physical_tile_m] = logical_tile_m;
    }"""

CANDIDATE_TEMPLATE = """template <
    bool kUseOutputScratchExpertSort,
    int kFixedEm,
    int kFixedN,
    int kFixedK,
    bool kUseClusterNSerpentine = false>"""
BASELINE_TEMPLATE = (
    "template <bool kUseOutputScratchExpertSort, int kFixedEm, "
    "int kFixedN, int kFixedK>"
)
CANDIDATE_TILE_N = """    const int tile_n = kUseClusterNSerpentine && (blockIdx.z & 1)
        ? gridDim.y - 1 - blockIdx.y
        : blockIdx.y;"""
BASELINE_TILE_N = "    const int tile_n = blockIdx.y;"
CANDIDATE_CASE2_LAUNCH = """fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168, true>
                <<<grid, block>>>"""
BASELINE_CASE2_LAUNCH = (
    "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168><<<grid, block>>>"
)
PACKED_TILE = """    const int32_t sort_payload = kUseOutputScratchExpertSort
        ? g_case2_full_expert_sort_map[physical_tile_m]
        : 0;
    const int tile_m = kUseOutputScratchExpertSort
        ? unpack_full_sort_tile(sort_payload)
        : physical_tile_m;"""
BASELINE_TILE = """    const int tile_m = kUseOutputScratchExpertSort
        ? g_case2_full_expert_sort_map[physical_tile_m]
        : physical_tile_m;"""
PACKED_EXPERT = """    const int expert = kUseOutputScratchExpertSort
        ? unpack_full_sort_expert(sort_payload)
        : expert_ids_ptr[tile_m];"""
BASELINE_EXPERT = "    const int expert = expert_ids_ptr[tile_m];"

CANDIDATE_A_MACRO = """        a_base + load_a_row_offset[ldgi]                                                          \\
            + (kUseCase2FixedNkU32BLocalOffsets ? 0 : load_k),                                   \\
"""
BASELINE_A_MACRO = """        a_base + load_a_row_offset[ldgi] + load_k,                                                \\
"""
CANDIDATE_A_OFFSET = """        load_a_row_offset[i] = routed_row * k
            + (kUseCase2FixedNkU32BLocalOffsets ? load_k : 0);"""
BASELINE_A_OFFSET = "        load_a_row_offset[i] = routed_row * k;"

FORMAL_GUARD = """    if (physical_tile_m * kMmaTileM >= em) {
        return;
    }"""
FORMAL_METADATA = """            *(reinterpret_cast<MmaInt1*>(&weights[i]) + j) =
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
                    MACA_ICMP_SLT);"""
FORMAL_COL_SCALE = """        col_scale[i] = __builtin_mxc_ldg_b128_predicator(
            const_cast<float*>(ptr),
            0,
            true,
            true,
            false,
            false,
            output_col_mask[i],
            1,
            MACA_ICMP_EQ);"""


def replace_once(source: str, candidate: str, baseline: str) -> str:
    count = source.count(candidate)
    if count != 1:
        raise AssertionError(f"candidate fragment count is {count}: {candidate!r}")
    return source.replace(candidate, baseline, 1)


def reverse_to_baseline(source: str) -> str:
    replacements = [
        (CONSTANTS, ""),
        (HELPERS, ""),
        (HISTOGRAM_BUILDER, BASELINE_BUILDER),
        (CANDIDATE_TEMPLATE, BASELINE_TEMPLATE),
        (CANDIDATE_TILE_N, BASELINE_TILE_N),
        (CANDIDATE_CASE2_LAUNCH, BASELINE_CASE2_LAUNCH),
        (PACKED_TILE, BASELINE_TILE),
        (PACKED_EXPERT, BASELINE_EXPERT),
        (CANDIDATE_A_MACRO, BASELINE_A_MACRO),
        (CANDIDATE_A_OFFSET, BASELINE_A_OFFSET),
    ]
    for candidate, baseline in replacements:
        source = replace_once(source, candidate, baseline)
    for index in range(4):
        source = replace_once(
            source,
            f"""        a_base + load_a_row_offset[{index}]
            + (kUseCase2FixedNkU32BLocalOffsets ? 0 : load_k),""",
            f"        a_base + load_a_row_offset[{index}] + load_k,",
        )
    return source


def histogram_map(experts: list[int], order: list[int]) -> list[tuple[int, int]]:
    counts = [0] * 256
    for expert in experts:
        counts[expert] += 1
    next_rank = [0] * 256
    for expert in range(1, 256):
        next_rank[expert] = next_rank[expert - 1] + counts[expert - 1]
    result: list[tuple[int, int] | None] = [None] * len(experts)
    for tile in order:
        expert = experts[tile]
        rank = next_rank[expert]
        next_rank[expert] += 1
        result[rank] = (expert, tile)
    if any(item is None for item in result):
        raise AssertionError("incomplete histogram map")
    return [item for item in result if item is not None]


def template_for(shape: tuple[int, int, int]) -> tuple[bool, int, int, int, bool]:
    if shape == CASE2:
        return (True, 0, 4096, 7168, True)
    if shape == CASE4:
        return (True, 32768, 7168, 2048, False)
    return (False, 0, 0, 0, False)


class Case2HistogramSerpentineLoadKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")
        cls.kernel = cls.source.split("fused_moe_i8_tn_mma_kernel(", 1)[1].split(
            "\n}\n\n#else", 1
        )[0]

    def test_restricted_inverse_recovers_formal_best(self) -> None:
        baseline = reverse_to_baseline(self.source)
        self.assertEqual(hashlib.sha256(baseline.encode()).hexdigest(), BASELINE_SHA256)

    def test_histogram_packed_map_is_grouped_bijection_for_atomic_orders(self) -> None:
        rng = random.Random(228)
        patterns = [
            [0] * 256,
            list(range(256)),
            [0 if tile < 224 else 255 for tile in range(256)],
        ]
        patterns.extend(
            [[rng.randrange(256) for _ in range(256)] for _ in range(128)]
        )
        for experts in patterns:
            order = list(range(256))
            rng.shuffle(order)
            payloads = histogram_map(experts, order)
            self.assertEqual(sorted(tile for _, tile in payloads), list(range(256)))
            self.assertEqual([expert for expert, _ in payloads], sorted(experts))
            self.assertTrue(all(expert == experts[tile] for expert, tile in payloads))

    def test_case2_serpentine_is_per_cluster_n_bijection(self) -> None:
        n_tiles = CASE2[1] // TILE_N
        for cluster in range((CASE2[0] // TILE_M) // CLUSTER_WIDTH):
            sequence = [
                n_tiles - 1 - tile if cluster & 1 else tile
                for tile in range(n_tiles)
            ]
            self.assertEqual(sorted(sequence), list(range(n_tiles)))
            if cluster:
                previous_last = n_tiles - 1 if (cluster - 1) % 2 == 0 else 0
                self.assertEqual(previous_last, sequence[0])

    def test_all_case2_a_load_addresses_are_identical(self) -> None:
        em, _n, k = CASE2
        checked = 0
        for row_base in range(0, em, TILE_M):
            for tid in range(256):
                load_k = (tid % 64 % 8) * 16
                for load_index in range(4):
                    row = row_base + tid // 8 + 32 * load_index
                    folded_offset = row * k + load_k
                    for k_tile in range(k // TILE_K):
                        baseline = row * k + k_tile * TILE_K + load_k
                        candidate = folded_offset + k_tile * TILE_K
                        self.assertEqual(candidate, baseline)
                        self.assertEqual(candidate % 16, 0)
                        self.assertLessEqual(candidate + 15, em * k - 1)
                        checked += 1
        self.assertEqual(checked, 256 * 256 * 4 * 56)

    def test_only_exact_case2_enables_serpentine(self) -> None:
        expected = {
            CASE1: (False, 0, 0, 0, False),
            CASE2: (True, 0, 4096, 7168, True),
            CASE3: (False, 0, 0, 0, False),
            CASE4: (True, 32768, 7168, 2048, False),
        }
        self.assertEqual({shape: template_for(shape) for shape in expected}, expected)
        self.assertEqual(self.source.count(CANDIDATE_CASE2_LAUNCH), 1)
        self.assertIn(
            "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>",
            self.source,
        )
        self.assertIn("fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>", self.source)

    def test_formal_predicated_paths_are_exact(self) -> None:
        self.assertEqual(self.kernel.count(FORMAL_GUARD), 1)
        self.assertEqual(self.kernel.count(FORMAL_METADATA), 1)
        self.assertEqual(self.kernel.count(FORMAL_COL_SCALE), 1)
        self.assertNotIn("if constexpr (!kUseCase2FixedNkU32BLocalOffsets)", self.kernel)
        self.assertNotIn("__builtin_mxc_ldg_b32(", self.kernel)
        self.assertNotIn(
            "kUseCase2FixedNkU32BLocalOffsets\n                    ? true",
            self.kernel,
        )
        for column in range(2):
            predicate = (
                "                (output_row[i * 4 + j] < em) && output_col_mask["
                + str(column)
                + "],"
            )
            self.assertEqual(self.kernel.count(predicate), 1)

    def test_schedule_abi_and_remote_job_are_protected(self) -> None:
        self.assertEqual(self.kernel.count("__syncthreadshared();"), 3)
        self.assertEqual(self.kernel.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(self.kernel.count("XH_LDG_A_STAGE_I("), 5)
        self.assertEqual(self.kernel.count("XH_LDG_B_STAGE_I("), 5)
        self.assertEqual(self.source.count('extern "C" void run_kernel('), 1)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaMemcpy", self.source)
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
