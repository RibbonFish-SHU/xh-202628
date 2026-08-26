#!/usr/bin/env python3
"""Focused no-device regression for the case-4 module full-sort dispatch."""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TILE_COUNT = 256
GRID_X = 8


def stable_full_map(experts: list[int]) -> list[int]:
    if len(experts) != TILE_COUNT:
        raise ValueError("module full-sort model requires 256 tiles")
    if any(expert < 0 or expert >= TILE_COUNT for expert in experts):
        raise ValueError("expert id is outside [0,255]")
    return sorted(range(TILE_COUNT), key=lambda tile: (experts[tile], tile))


def verify_cover(experts: list[int], n_tiles: int) -> None:
    rank_map = stable_full_map(experts)
    if sorted(rank_map) != list(range(TILE_COUNT)):
        raise AssertionError("full-sort rank map is not a bijection")
    mapped_experts = [experts[logical_tile] for logical_tile in rank_map]
    if mapped_experts != sorted(mapped_experts):
        raise AssertionError("expert rank intervals are not contiguous")

    visits = [0] * (TILE_COUNT * n_tiles)
    for z in range(TILE_COUNT // GRID_X):
        for x in range(GRID_X):
            physical_rank = x + GRID_X * z
            logical_tile = rank_map[physical_rank]
            for tile_n in range(n_tiles):
                visits[logical_tile * n_tiles + tile_n] += 1
    if any(count != 1 for count in visits):
        raise AssertionError("module full-sort grid does not cover M/N exactly once")


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


class Case4ModuleFullSortTests(unittest.TestCase):
    def test_exact_case2_and_case4_cover(self) -> None:
        distributions = [
            [7] * TILE_COUNT,
            [(tile * 73) % TILE_COUNT for tile in range(TILE_COUNT)],
            [(tile * tile + 3 * tile) % 16 for tile in range(TILE_COUNT)],
        ]
        for experts in distributions:
            verify_cover(experts, 32)
            verify_cover(experts, 56)

        skewed = distributions[2]
        rank_map = stable_full_map(skewed)
        pure_clusters = sum(
            len({skewed[tile] for tile in rank_map[start : start + GRID_X]}) == 1
            for start in range(0, TILE_COUNT, GRID_X)
        )
        self.assertEqual(pure_clusters, 32)
        self.assertEqual(TILE_COUNT * 32, 8192)
        self.assertEqual(TILE_COUNT * 56, 14336)

    def test_short_distributions_preserve_full_stable_bijection(self) -> None:
        pattern_count = 0
        for width in range(1, 9):
            for prefix in itertools.product(range(3), repeat=width):
                experts = list(prefix) + [255] * (TILE_COUNT - width)
                verify_cover(experts, 32)
                verify_cover(experts, 56)
                pattern_count += 1
        self.assertEqual(pattern_count, 9840)

    def test_dispatch_uses_the_existing_module_map_without_cleanup(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        case2_selector = function_body(
            source, "use_case2_output_scratch_expert_sort("
        )
        case4_selector = function_body(
            source, "use_prefill_down_module_full_expert_sort("
        )
        grid_selector = function_body(source, "module_full_sort_launch_grid_x(")
        launch = function_body(source, "static inline void launch(")
        builder = function_body(source, "build_case2_full_expert_sort_map_kernel(")

        self.assertIn("config.n == 4096 && config.k == 7168", case2_selector)
        self.assertNotIn("7168 && config.k == 2048", case2_selector)
        self.assertIn("config.n == 7168 && config.k == 2048", case4_selector)
        self.assertIn("kPrefillDownModuleFullSortGridM", grid_selector)
        self.assertIn("mma_launch_grid_x(config)", grid_selector)
        self.assertIn(
            "use_case2_output_scratch_expert_sort(config)\n"
            "        || use_prefill_down_module_full_expert_sort(config)",
            launch,
        )
        self.assertIn("module_full_sort_launch_grid_x(config)", launch)
        self.assertIn("build_case2_full_expert_sort_map_kernel", launch)
        self.assertEqual(
            launch.count(
                "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>"
            ),
            1,
        )
        self.assertEqual(
            launch.count(
                "fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>"
            ),
            1,
        )
        self.assertEqual(
            launch.count("fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"),
            1,
        )
        self.assertNotIn("<true, 32768, 4096, 7168>", launch)
        self.assertNotIn("<true, 0, 4096, 7168>", launch)
        self.assertNotIn("kFixedEm", launch)
        self.assertNotIn("tile_zero", launch)
        self.assertIn("case2_full_stable_sort_rank(", builder)
        self.assertIn(
            "g_case2_full_expert_sort_map[physical_tile_m] = logical_tile_m;",
            builder,
        )

    def test_module_map_lifecycle_and_working_sets(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "__device__ int32_t "
            "g_case2_full_expert_sort_map[kOutputScratchSortTiles];",
            source,
        )
        self.assertIn("? g_case2_full_expert_sort_map[physical_tile_m]", source)
        self.assertNotIn("cudaDeviceSynchronize", source)
        self.assertNotIn("cudaMemcpy", source)
        self.assertEqual(GRID_X * 128 * 7168 + 128 * 7168, 8257536)
        self.assertEqual(GRID_X * 128 * 2048 + 128 * 2048, 2359296)


if __name__ == "__main__":
    unittest.main(verbosity=2)
