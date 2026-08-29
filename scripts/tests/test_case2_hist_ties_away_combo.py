#!/usr/bin/env python3
"""No-device identity and mapping proof for exp-238."""

from __future__ import annotations

import hashlib
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260830-238.sh"
FORMAL_SHA256 = "083eb1262dbe220aea0c2b324a00f2cf9b14720dc2b8c1701b261f794eaf8cf1"

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

FORMAL_BUILDER = """    const int logical_tile_m = threadIdx.x;
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

FORMAL_TILE = """    const int tile_m = kUseOutputScratchExpertSort
        ? g_case2_full_expert_sort_map[physical_tile_m]
        : physical_tile_m;"""

PACKED_EXPERT = """    const int expert = kUseOutputScratchExpertSort
        ? unpack_full_sort_expert(sort_payload)
        : expert_ids_ptr[tile_m];"""

FORMAL_EXPERT = "    const int expert = expert_ids_ptr[tile_m];"

TIES_AWAY = """#define XH_CVT_F32_TO_BF16(dst, src0, src1)                                                       \\
    src0 += 0x8000;                                                                               \\
    src1 += 0x8000;                                                                               \\
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)"""

FORMAL_RNE = """#define XH_CVT_F32_TO_BF16(dst, src0, src1)                                                       \\
    src0 = ((src0 >> 16) & 1) + src0 + 0x7fff;                                                   \\
    src1 = ((src1 >> 16) & 1) + src1 + 0x7fff;                                                   \\
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)"""


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"fragment count is {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def reverse_to_formal(source: str) -> str:
    replacements = (
        (CONSTANTS, ""),
        (HELPERS, ""),
        (HISTOGRAM_BUILDER, FORMAL_BUILDER),
        (PACKED_TILE, FORMAL_TILE),
        (PACKED_EXPERT, FORMAL_EXPERT),
        (TIES_AWAY, FORMAL_RNE),
    )
    for candidate, formal in replacements:
        source = replace_once(source, candidate, formal)
    return source


def histogram_map(experts: list[int], order: list[int]) -> list[tuple[int, int]]:
    counts = [0] * 256
    for expert in experts:
        counts[expert] += 1
    offsets = [0] * 256
    for expert in range(1, 256):
        offsets[expert] = offsets[expert - 1] + counts[expert - 1]
    result: list[tuple[int, int] | None] = [None] * 256
    for tile in order:
        expert = experts[tile]
        rank = offsets[expert]
        offsets[expert] += 1
        result[rank] = (expert, tile)
    return [item for item in result if item is not None]


class Case2HistogramTiesAwayComboTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")

    def test_restricted_inverse_recovers_formal_source(self) -> None:
        formal = reverse_to_formal(self.source)
        self.assertEqual(hashlib.sha256(formal.encode()).hexdigest(), FORMAL_SHA256)

    def test_histogram_payload_is_grouped_bijection(self) -> None:
        rng = random.Random(238)
        patterns = [
            [0] * 256,
            list(range(256)),
            [0 if tile < 224 else 255 for tile in range(256)],
        ]
        patterns.extend([[rng.randrange(256) for _ in range(256)] for _ in range(128)])
        for experts in patterns:
            order = list(range(256))
            rng.shuffle(order)
            payloads = histogram_map(experts, order)
            self.assertEqual(len(payloads), 256)
            self.assertEqual(sorted(tile for _, tile in payloads), list(range(256)))
            self.assertEqual([expert for expert, _ in payloads], sorted(experts))
            self.assertTrue(all(expert == experts[tile] for expert, tile in payloads))

    def test_two_components_and_decode_isolation(self) -> None:
        self.assertEqual(self.source.count("pack_full_sort_payload(expert, logical_tile_m)"), 1)
        self.assertEqual(self.source.count("atomicAdd(&expert_offsets[expert], 1)"), 2)
        self.assertEqual(self.source.count(TIES_AWAY), 1)
        self.assertIn("src0 = ((src0 >> 16) & 1) + src0 + 0x7fff", self.source)
        self.assertIn("direct_moe_kernel_m4<<<grid, block>>>(args);", self.source)

    def test_abi_and_remote_job_are_protected(self) -> None:
        self.assertEqual(self.source.count('extern "C" void run_kernel('), 1)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaMemcpy", self.source)
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
