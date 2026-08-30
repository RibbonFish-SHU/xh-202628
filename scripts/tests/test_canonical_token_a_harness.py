"""Workflow proof for candidates that reuse a canonical routed A row per token."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = (
    ROOT
    / "operators"
    / "fused_moe_i8_tn"
    / "cuda_maca"
    / "test_fused_moe_i8_tn.cu"
).read_text(encoding="utf-8")


class CanonicalTokenAHarnessTests(unittest.TestCase):
    def test_public_token_ids_are_complete_permutations(self) -> None:
        for em in (4096, 32768):
            token_ids = [(row * 8191) & (em - 1) for row in range(em)]
            self.assertEqual(sorted(token_ids), list(range(em)))

            canonical_rows = {
                token_id >> 3: row
                for row, token_id in enumerate(token_ids)
                if (token_id & 7) == 0
            }
            self.assertEqual(len(canonical_rows), em // 8)
            for row, token_id in enumerate(token_ids):
                token = token_id >> 3
                self.assertEqual(token % 31, (token_ids[canonical_rows[token]] >> 3) % 31)

    def test_candidate_capability_is_compile_time_isolated(self) -> None:
        self.assertIn("#if defined(XH_FUSED_MOE_CANONICAL_TOKEN_A)", HARNESS)
        self.assertIn("out, config, token_ids);", HARNESS)
        self.assertIn("(void)token_ids;", HARNESS)
        self.assertIn("out, config);", HARNESS)

    def test_public_benchmark_supplies_read_only_token_ids(self) -> None:
        self.assertIn("DeviceBuffer<int32_t> dev_token_ids(config.em);", HARNESS)
        self.assertIn("dev_token_ids.copy_from(token_ids);", HARNESS)
        self.assertIn("fill_routed_a(dev_a.get(), dev_token_ids.get(), config.em, config.k);", HARNESS)
        self.assertIn("if (dev_token_ids.copy_to_host() != token_ids)", HARNESS)


if __name__ == "__main__":
    unittest.main()
