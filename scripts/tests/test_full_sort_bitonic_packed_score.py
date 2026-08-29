#!/usr/bin/env python3
"""No-device proof for the exp-214 bitonic packed-map builder."""

from __future__ import annotations

import hashlib
import itertools
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260829-221.sh"
TILE_COUNT = 256
BASELINE_SHA256 = "083eb1262dbe220aea0c2b324a00f2cf9b14720dc2b8c1701b261f794eaf8cf1"


def pack(expert: int, tile: int) -> int:
    return (expert << 8) | tile


def unpack(payload: int) -> tuple[int, int]:
    return payload >> 8, payload & 0xFF


def bitonic_payloads(experts: list[int] | tuple[int, ...]) -> list[int]:
    values = [pack(expert, tile) for tile, expert in enumerate(experts)]
    sort_size = 2
    while sort_size <= len(values):
        stride = sort_size >> 1
        while stride:
            for tile in range(len(values)):
                partner = tile ^ stride
                if partner <= tile:
                    continue
                ascending = (tile & sort_size) == 0
                swap = values[tile] > values[partner] if ascending else values[tile] < values[partner]
                if swap:
                    values[tile], values[partner] = values[partner], values[tile]
            stride >>= 1
        sort_size <<= 1
    return values


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

BITONIC_BUILDER = """    const int logical_tile_m = threadIdx.x;
    __shared__ int32_t sort_payloads[kOutputScratchSortTiles];

    const int expert = expert_ids[logical_tile_m];
    sort_payloads[logical_tile_m] =
        pack_full_sort_payload(expert, logical_tile_m);
    __syncthreads();

#pragma unroll
    for (int sort_size = 2;
         sort_size <= kOutputScratchSortTiles;
         sort_size <<= 1) {
#pragma unroll
        for (int stride = sort_size >> 1; stride > 0; stride >>= 1) {
            const int partner = logical_tile_m ^ stride;
            if (partner > logical_tile_m) {
                const int32_t left = sort_payloads[logical_tile_m];
                const int32_t right = sort_payloads[partner];
                const bool ascending = (logical_tile_m & sort_size) == 0;
                const bool swap = ascending ? left > right : left < right;
                if (swap) {
                    sort_payloads[logical_tile_m] = right;
                    sort_payloads[partner] = left;
                }
            }
            __syncthreads();
        }
    }

    g_case2_full_expert_sort_map[logical_tile_m] =
        sort_payloads[logical_tile_m];"""

BASELINE_BUILDER = """    const int logical_tile_m = threadIdx.x;
    if (logical_tile_m < kOutputScratchSortTiles) {
        const int physical_tile_m = case2_full_stable_sort_rank(
            expert_ids, logical_tile_m, kOutputScratchSortTiles);
        g_case2_full_expert_sort_map[physical_tile_m] = logical_tile_m;
    }"""

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


def reverse_to_baseline(source: str) -> str:
    replacements = [
        (CONSTANTS, ""),
        (HELPERS, ""),
        (BITONIC_BUILDER, BASELINE_BUILDER),
        (PACKED_TILE, BASELINE_TILE),
        (PACKED_EXPERT, BASELINE_EXPERT),
    ]
    for candidate, baseline in replacements:
        if source.count(candidate) != 1:
            raise AssertionError(f"candidate fragment count is {source.count(candidate)}")
        source = source.replace(candidate, baseline)
    return source


class FullSortBitonicPackedScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")

    def verify_distribution(self, experts: list[int] | tuple[int, ...]) -> None:
        payloads = bitonic_payloads(experts)
        expected = sorted(pack(expert, tile) for tile, expert in enumerate(experts))
        self.assertEqual(payloads, expected)
        self.assertEqual(
            [unpack(payload) for payload in payloads],
            sorted((expert, tile) for tile, expert in enumerate(experts)),
        )

    def test_network_has_exact_256_item_stable_coverage(self) -> None:
        distributions = [
            [7] * TILE_COUNT,
            list(range(TILE_COUNT)),
            [(tile * 73) % 256 for tile in range(TILE_COUNT)],
            [(tile * tile + 3 * tile) % 16 for tile in range(TILE_COUNT)],
        ]
        for seed in range(64):
            generator = random.Random(seed)
            distributions.append([generator.randrange(256) for _ in range(TILE_COUNT)])
        for experts in distributions:
            self.verify_distribution(experts)

    def test_short_exhaustive_prefixes_preserve_tie_order(self) -> None:
        patterns = 0
        for width in range(1, 9):
            for prefix in itertools.product(range(3), repeat=width):
                experts = list(prefix) + [255] * (TILE_COUNT - width)
                self.verify_distribution(experts)
                patterns += 1
        self.assertEqual(patterns, 9840)

    def test_pack_unpack_exhausts_both_fields(self) -> None:
        for expert in range(256):
            for tile in range(256):
                self.assertEqual(unpack(pack(expert, tile)), (expert, tile))

    def test_restricted_inverse_recovers_formal_best(self) -> None:
        baseline = reverse_to_baseline(self.source)
        self.assertEqual(hashlib.sha256(baseline.encode()).hexdigest(), BASELINE_SHA256)

    def test_static_network_and_protected_paths(self) -> None:
        builder = self.source.split(
            "__global__ void build_case2_full_expert_sort_map_kernel", 1
        )[1].split("static inline bool same_config", 1)[0]
        self.assertIn("__shared__ int32_t sort_payloads[kOutputScratchSortTiles]", builder)
        self.assertIn("const int partner = logical_tile_m ^ stride", builder)
        self.assertNotIn("atomic", builder)
        self.assertEqual(self.source.count("pack_full_sort_payload(expert, logical_tile_m)"), 1)
        self.assertEqual(self.source.count("unpack_full_sort_tile(sort_payload)"), 1)
        self.assertEqual(self.source.count("unpack_full_sort_expert(sort_payload)"), 1)
        self.assertIn("sync_m4::launch_decode(", self.source)
        self.assertIn("fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168>", self.source)
        self.assertIn("fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>", self.source)
        self.assertEqual(self.source.count('extern "C" void run_kernel('), 1)
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
