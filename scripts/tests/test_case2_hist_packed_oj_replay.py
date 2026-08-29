#!/usr/bin/env python3
"""Identity proof for exp-220's exact exp-204 score-policy replay."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260829-220.sh"
TESTED_SOURCE_SHA256 = "2405dfc415ddf1b60c69cb158e1c72eb768111718fb4daa00b03c75919e26dab"


class Case2HistogramPackedOjReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")

    def test_source_is_exact_exp204_tested_source(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.source.encode()).hexdigest(), TESTED_SOURCE_SHA256
        )

    def test_histogram_packed_map_contract(self) -> None:
        self.assertIn("__shared__ int32_t expert_offsets[kOutputScratchSortTiles];", self.source)
        self.assertEqual(self.source.count("atomicAdd(&expert_offsets[expert], 1)"), 2)
        self.assertEqual(self.source.count("pack_full_sort_payload(expert, logical_tile_m)"), 1)
        self.assertEqual(self.source.count("unpack_full_sort_tile(sort_payload)"), 1)
        self.assertEqual(self.source.count("unpack_full_sort_expert(sort_payload)"), 1)
        self.assertIn("config.em == 32768 && config.n == 4096 && config.k == 7168", self.source)

    def test_abi_and_trusted_job_are_protected(self) -> None:
        self.assertEqual(self.source.count('extern "C" void run_kernel('), 1)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaMemcpy", self.source)
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
