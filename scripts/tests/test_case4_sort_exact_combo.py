#!/usr/bin/env python3
"""Focused no-device regression for the case-4 sort/exact combination."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
PROTECTED_BODY_SHA256 = "04df1339e7aa9a5b52bb1124eaa2bae700a412cf7e5fa4ebb98770e31d98c6ac"


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


def resolve_dimensions(
    fixed: tuple[int, int, int], runtime: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        runtime_value if fixed_value == 0 else fixed_value
        for fixed_value, runtime_value in zip(fixed, runtime)
    )


class Case4SortExactComboTests(unittest.TestCase):
    def test_only_case4_selects_fixed_dimensions(self) -> None:
        public_shapes = [
            (4096, 4096, 7168),
            (32768, 4096, 7168),
            (4096, 7168, 2048),
            (32768, 7168, 2048),
        ]
        fixed_case4 = (32768, 7168, 2048)
        selections = [shape == fixed_case4 for shape in public_shapes]
        self.assertEqual(selections, [False, False, False, True])
        for shape, selected in zip(public_shapes, selections):
            fixed = fixed_case4 if selected else (0, 0, 0)
            self.assertEqual(resolve_dimensions(fixed, shape), shape)

        fallback_shapes = [
            (128, 128, 128),
            (8192, 4096, 7168),
            (32768, 7168, 4096),
            (32768, 4096, 2048),
        ]
        for shape in fallback_shapes:
            self.assertEqual(resolve_dimensions((0, 0, 0), shape), shape)

    def test_case4_keeps_sorted_grid_and_exact_ownership(self) -> None:
        case2_grid = (8, 32, 32)
        case4_grid = (8, 56, 32)
        self.assertEqual(case2_grid[0] * case2_grid[1] * case2_grid[2], 8192)
        self.assertEqual(case4_grid[0] * case4_grid[1] * case4_grid[2], 14336)

    def test_dispatch_combines_sort_with_only_the_case4_fixed_instance(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        selector = function_body(source, "use_case4_exact_shape_specialization(")
        launch = function_body(source, "static inline void launch(")
        self.assertIn("config.em == 32768", selector)
        self.assertIn("config.n == 7168", selector)
        self.assertIn("config.k == 2048", selector)
        self.assertIn(
            "use_case2_output_scratch_expert_sort(config)\n"
            "        || use_prefill_down_module_full_expert_sort(config)",
            launch,
        )
        self.assertIn("module_full_sort_launch_grid_x(config)", launch)
        self.assertEqual(launch.count("build_case2_full_expert_sort_map_kernel"), 1)
        self.assertIn("use_case4_exact_shape_specialization(config)", launch)
        self.assertEqual(
            launch.count(
                "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>"
            ),
            1,
        )
        self.assertEqual(
            launch.count("fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>"), 1
        )
        self.assertEqual(
            launch.count("fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"), 1
        )
        self.assertNotIn("tile_zero", launch)

    def test_dimension_bindings_preserve_the_complete_mma_body(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        kernel = function_body(source, "__global__ void fused_moe_i8_tn_mma_kernel(")
        self.assertIn(
            "const int em = kFixedEm == 0 ? runtime_em : kFixedEm;", kernel
        )
        self.assertIn(
            "const int n = kFixedN == 0 ? runtime_n : kFixedN;", kernel
        )
        self.assertIn(
            "const int k = kFixedK == 0 ? runtime_k : kFixedK;", kernel
        )
        protected = kernel[kernel.index("#define XH_MMA_STAGE_MNKX2") :]
        self.assertEqual(
            hashlib.sha256(protected.encode("utf-8")).hexdigest(),
            PROTECTED_BODY_SHA256,
        )
        self.assertEqual(protected.count("__syncthreadshared();"), 3)

    def test_public_inputs_and_fallback_remain_read_only(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        start = source.index("__global__ void fused_moe_i8_tn_mma_kernel(")
        kernel_header = source[start : source.index(") {", start)]
        for name in (
            "a_ptr",
            "b_ptr",
            "scale_a_ptr",
            "scale_b_ptr",
            "moe_weights_ptr",
            "expert_ids_ptr",
        ):
            self.assertRegex(kernel_header, rf"const [^,\n]+\b{name}\b")
        self.assertNotIn("cudaDeviceSynchronize", source)
        self.assertNotIn("cudaMemcpy", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
