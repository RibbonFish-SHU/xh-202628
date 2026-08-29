#!/usr/bin/env python3
"""Identity proof for exp-221's exact exp-214 score-policy replay."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260829-221.sh"
TESTED_SOURCE_SHA256 = "a3b58083a6a2d50b38dc500882f6060423812f7966c6bda53799130d7c5ca031"


class Case2BitonicPackedOjReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")

    def test_source_is_exact_exp214_tested_source(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.source.encode()).hexdigest(), TESTED_SOURCE_SHA256
        )

    def test_bitonic_packed_map_contract(self) -> None:
        self.assertIn("__shared__ int32_t sort_payloads[kOutputScratchSortTiles];", self.source)
        self.assertIn("const int partner = logical_tile_m ^ stride;", self.source)
        self.assertIn("const bool ascending = (logical_tile_m & sort_size) == 0;", self.source)
        self.assertEqual(self.source.count("pack_full_sort_payload(expert, logical_tile_m)"), 1)
        self.assertEqual(self.source.count("unpack_full_sort_tile(sort_payload)"), 1)
        self.assertEqual(self.source.count("unpack_full_sort_expert(sort_payload)"), 1)

    def test_abi_and_trusted_job_are_protected(self) -> None:
        self.assertEqual(self.source.count('extern "C" void run_kernel('), 1)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaMemcpy", self.source)
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
