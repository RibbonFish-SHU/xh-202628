#!/usr/bin/env python3
"""Source-lock proof for the exp-239 leaderboard-first composition."""

from __future__ import annotations

import hashlib
import random
import unittest
from pathlib import Path

from scripts.tests.test_case2_hist_serp_loadk import (
    BASELINE_SHA256,
    CASE1,
    CASE2,
    CASE3,
    CASE4,
    CLUSTER_WIDTH,
    FORMAL_COL_SCALE,
    FORMAL_GUARD,
    FORMAL_METADATA,
    TILE_K,
    TILE_M,
    TILE_N,
    histogram_map,
    reverse_to_baseline,
    template_for,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260830-239.sh"

TIES_AWAY = """#define XH_CVT_F32_TO_BF16(dst, src0, src1)                                                       \\
    src0 += 0x8000;                                                                               \\
    src1 += 0x8000;                                                                               \\
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)"""
FORMAL_RNE = """#define XH_CVT_F32_TO_BF16(dst, src0, src1)                                                       \\
    src0 = ((src0 >> 16) & 1) + src0 + 0x7fff;                                                   \\
    src1 = ((src1 >> 16) & 1) + src1 + 0x7fff;                                                   \\
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)"""


class Case2HistogramSerpentineLoadKTiesAwayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")
        cls.kernel = cls.source.split("fused_moe_i8_tn_mma_kernel(", 1)[1].split(
            "\n}\n\n#else", 1
        )[0]

    def test_restricted_inverse_recovers_formal_best(self) -> None:
        self.assertEqual(self.source.count(TIES_AWAY), 1)
        without_rounding = self.source.replace(TIES_AWAY, FORMAL_RNE, 1)
        baseline = reverse_to_baseline(without_rounding)
        self.assertEqual(hashlib.sha256(baseline.encode()).hexdigest(), BASELINE_SHA256)

    def test_exact_case2_contains_all_four_components(self) -> None:
        self.assertIn("pack_full_sort_payload(expert, logical_tile_m)", self.source)
        self.assertIn("bool kUseClusterNSerpentine = false", self.source)
        self.assertIn("fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168, true>", self.source)
        self.assertIn("kUseCase2FixedNkU32BLocalOffsets ? load_k : 0", self.source)
        self.assertEqual(self.source.count(TIES_AWAY), 1)

    def test_histogram_payload_is_grouped_bijection_for_atomic_orders(self) -> None:
        rng = random.Random(239)
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

    def test_serpentine_preserves_each_case2_cluster_n_bijection(self) -> None:
        n_tiles = CASE2[1] // TILE_N
        cluster_count = (CASE2[0] // TILE_M) // CLUSTER_WIDTH
        for cluster in range(cluster_count):
            sequence = [
                n_tiles - 1 - tile if cluster & 1 else tile
                for tile in range(n_tiles)
            ]
            self.assertEqual(sorted(sequence), list(range(n_tiles)))
            if cluster:
                previous_last = n_tiles - 1 if (cluster - 1) % 2 == 0 else 0
                self.assertEqual(previous_last, sequence[0])

    def test_all_case2_folded_a_load_addresses_match_formal(self) -> None:
        em, _n, k = CASE2
        checked = 0
        for row_base in range(0, em, TILE_M):
            for tid in range(256):
                load_k = (tid % 64 % 8) * 16
                for load_index in range(4):
                    row = row_base + tid // 8 + 32 * load_index
                    folded_offset = row * k + load_k
                    for k_tile in range(k // TILE_K):
                        formal = row * k + k_tile * TILE_K + load_k
                        candidate = folded_offset + k_tile * TILE_K
                        self.assertEqual(candidate, formal)
                        self.assertEqual(candidate % 16, 0)
                        self.assertLessEqual(candidate + 15, em * k - 1)
                        checked += 1
        self.assertEqual(checked, 256 * 256 * 4 * 56)

    def test_serpentine_and_loadk_folding_are_exact_case2_only(self) -> None:
        expected = {
            CASE1: (False, 0, 0, 0, False),
            CASE2: (True, 0, 4096, 7168, True),
            CASE3: (False, 0, 0, 0, False),
            CASE4: (True, 32768, 7168, 2048, False),
        }
        self.assertEqual({shape: template_for(shape) for shape in expected}, expected)
        self.assertEqual(
            self.source.count(
                "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168, true>"
            ),
            1,
        )

    def test_formal_predicates_schedule_and_abi_are_unchanged(self) -> None:
        self.assertEqual(self.kernel.count(FORMAL_GUARD), 1)
        self.assertEqual(self.kernel.count(FORMAL_METADATA), 1)
        self.assertEqual(self.kernel.count(FORMAL_COL_SCALE), 1)
        self.assertEqual(self.kernel.count("__syncthreadshared();"), 3)
        self.assertEqual(self.kernel.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(self.kernel.count("XH_LDG_A_STAGE_I("), 5)
        self.assertEqual(self.kernel.count("XH_LDG_B_STAGE_I("), 5)
        self.assertEqual(self.source.count('extern "C" void run_kernel('), 1)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaMemcpy", self.source)

    def test_decode_rounding_and_formal_predicates_remain_protected(self) -> None:
        self.assertIn("#define CVT_F32_TO_BF16(dst, src0, src1)", self.source)
        self.assertIn("src0 = ((src0 >> 16) & 1) + src0 + 0x7fff", self.source)
        self.assertNotIn("if constexpr (!kUseCase2FixedNkU32BLocalOffsets)", self.source)
        self.assertNotIn("__builtin_mxc_ldg_b32(", self.kernel)

    def test_remote_job_matches_trusted_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
