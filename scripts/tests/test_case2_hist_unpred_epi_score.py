"""Focused proof for histogram-packed exact-case2 unpredicated epilogue."""

from __future__ import annotations

import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "operators/fused_moe_i8_tn/cuda_maca/submission.cu").read_text(
    encoding="utf-8"
)


def histogram_map(experts: list[int], order: list[int]) -> list[tuple[int, int]]:
    counts = [0] * 256
    for expert in experts:
        counts[expert] += 1
    offsets = [0] * 256
    for expert in range(1, 256):
        offsets[expert] = offsets[expert - 1] + counts[expert - 1]
    result: list[tuple[int, int] | None] = [None] * len(experts)
    next_rank = offsets[:]
    for tile in order:
        expert = experts[tile]
        rank = next_rank[expert]
        next_rank[expert] += 1
        result[rank] = (expert, tile)
    if any(item is None for item in result):
        raise AssertionError("histogram rank map is incomplete")
    return [item for item in result if item is not None]


class Case2HistogramUnpredicatedEpilogueTests(unittest.TestCase):
    def test_histogram_packed_map_is_bijective_and_expert_sorted(self) -> None:
        rng = random.Random(208)
        patterns = [
            [0] * 256,
            list(range(256)),
            [0 if i < 224 else 255 for i in range(256)],
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
            self.assertTrue(all(0 <= expert <= 255 for expert, _ in payloads))

    def test_exact_case2_rows_columns_and_ctas_are_in_bounds(self) -> None:
        em = 32768
        n = 4096
        seen_rows: set[int] = set()
        for tile_m in range(256):
            row_base = tile_m * 128
            for thread_id in range(256):
                for group in range(2):
                    for row_in_group in range(4):
                        wave = thread_id // 64
                        lane = thread_id % 64
                        local = (
                            ((lane // 16) % 2) * 4
                            + wave * 8
                            + (lane // 32) * 32
                            + group * 64
                            + row_in_group
                        )
                        row = row_base + local
                        self.assertLess(row, em)
                        seen_rows.add(row)
                for col_group in range(2):
                    col = (thread_id % 16) * 4 + col_group * 64
                    self.assertLess(col, 128)
        self.assertEqual(seen_rows, set(range(em)))
        self.assertEqual(8 * 32 * 1, 256)
        self.assertEqual(n % 128, 0)

    def test_source_limits_unpredicated_path_to_exact_case2(self) -> None:
        self.assertIn("if constexpr (!kUseCase2FixedNkU32BLocalOffsets)", SOURCE)
        self.assertEqual(SOURCE.count("__builtin_mxc_ldg_b32("), 3)
        self.assertIn(
            "if constexpr (kUseCase2FixedNkU32BLocalOffsets) {\n"
            "            col_scale[i] = __builtin_mxc_ldg_b128(",
            SOURCE,
        )
        self.assertEqual(
            SOURCE.count("kUseCase2FixedNkU32BLocalOffsets\n                    ? true"),
            2,
        )
        self.assertIn("pack_full_sort_payload(expert, logical_tile_m)", SOURCE)
        self.assertIn("unpack_full_sort_expert(sort_payload)", SOURCE)


if __name__ == "__main__":
    unittest.main()
