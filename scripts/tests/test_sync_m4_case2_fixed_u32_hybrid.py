#!/usr/bin/env python3
"""No-device proof for the score-targeted exp-202 composition."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from scripts.tests.test_case2_fixed_nk_u32_brow import (
    CANDIDATE_CASE2_LAUNCH,
    LOADS_B,
    POLICY_BLOCK,
    function_body,
    reverse_to_formal_best,
    row_and_k_offset,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260828-202.sh"
BASELINE_SHA256 = "eb94ca383647ce166fca1e568ad75b4c6c143b816b01f65f2d346d3c7a1effb4"


class SyncM4Case2FixedU32HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")

    def test_restricted_inverse_recovers_current_formal_best(self) -> None:
        baseline = reverse_to_formal_best(self.source)
        self.assertEqual(hashlib.sha256(baseline.encode()).hexdigest(), BASELINE_SHA256)

    def test_dispatches_keep_decode_m4_and_only_specialize_case2(self) -> None:
        self.assertEqual(self.source.count(POLICY_BLOCK), 1)
        self.assertEqual(self.source.count(CANDIDATE_CASE2_LAUNCH), 1)
        self.assertIn(
            "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>", self.source
        )
        self.assertIn("namespace sync_m4", self.source)
        self.assertEqual(self.source.count("sync_m4::launch_decode("), 1)
        self.assertIn("constexpr int kTileK = 256;", self.source)

    def test_all_case2_vector_starts_match_original_addresses(self) -> None:
        n, k = 4096, 7168
        starts = 0
        maximum = 0
        for tile_n in range(n // 128):
            for tid in range(256):
                for load_index in range(LOADS_B):
                    raw_row, load_k = row_and_k_offset(tid, load_index)
                    row = min(raw_row, 127)
                    carrier = (tile_n * 128 + row) * k + load_k
                    self.assertLess(carrier, 1 << 31)
                    for tile_k in range(k // 128):
                        candidate = (carrier & 0xFFFFFFFF) + tile_k * 128
                        original = (tile_n * 128 + row) * k + tile_k * 128 + load_k
                        self.assertEqual(candidate, original)
                        self.assertEqual(candidate % 16, 0)
                        self.assertLess(candidate + 15, n * k)
                        maximum = max(maximum, candidate)
                        starts += 1
        self.assertEqual(starts, 1_835_008)
        self.assertEqual(maximum, n * k - 16)

    def test_schedule_abi_and_job_are_protected(self) -> None:
        formal_kernel = function_body(self.source, "fused_moe_i8_tn_mma_kernel(")
        self.assertEqual(formal_kernel.count("__syncthreadshared();"), 3)
        self.assertEqual(formal_kernel.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(formal_kernel.count("XH_LDG_B_STAGE_I("), 5)
        self.assertEqual(self.source.count('extern "C" void run_kernel('), 1)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaMemcpy", self.source)
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
