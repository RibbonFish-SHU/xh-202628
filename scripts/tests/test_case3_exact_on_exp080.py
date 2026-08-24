#!/usr/bin/env python3
"""No-device proof for the case-3-only exact dimensions on exp-080."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
CASE1 = (4096, 4096, 7168)
CASE2 = (32768, 4096, 7168)
CASE3 = (4096, 7168, 2048)
CASE4 = (32768, 7168, 2048)
FALLBACK = (384, 2048, 1024)
PROTECTED_BODY_SHA256 = (
    "80bc6355f0a2fd09ad37f721ae6ef493d1948976b9dfe341c329dddefea59e6c"
)


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


def dispatch(shape: tuple[int, int, int]) -> tuple[bool, tuple[int, int, int]]:
    use_global_sort_map = shape in (CASE2, CASE4)
    if use_global_sort_map:
        return True, (0, 0, 0)
    return False, CASE3 if shape == CASE3 else (0, 0, 0)


class Case3ExactOnExp080Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")
        cls.launch = function_body(cls.source, "static inline void launch(")

    def test_only_public_case3_uses_fixed_dimensions(self) -> None:
        expected = {
            CASE1: (False, (0, 0, 0)),
            CASE2: (True, (0, 0, 0)),
            CASE3: (False, CASE3),
            CASE4: (True, (0, 0, 0)),
            FALLBACK: (False, (0, 0, 0)),
        }
        self.assertEqual({shape: dispatch(shape) for shape in expected}, expected)

        selector = function_body(self.source, "use_decode_down_exact_dimensions(")
        self.assertEqual(
            " ".join(selector.split()),
            "return config.em == 4096 && config.n == 7168 && config.k == 2048;",
        )
        fixed = "fused_moe_i8_tn_mma_kernel<false, 4096, 7168, 2048>"
        self.assertEqual(self.source.count(fixed), 1)
        self.assertNotIn("<true, 4096", self.source)
        self.assertNotIn("<false, 32768", self.source)

    def test_runtime_dimension_carrier_and_dispatch_are_exact(self) -> None:
        template = (
            "template <bool kUseOutputScratchExpertSort, int kFixedEm, "
            "int kFixedN, int kFixedK>"
        )
        self.assertIn(template, self.source)
        self.assertIn(
            "const int em = kFixedEm == 0 ? runtime_em : kFixedEm;",
            self.kernel,
        )
        self.assertIn(
            "const int n = kFixedN == 0 ? runtime_n : kFixedN;", self.kernel
        )
        self.assertIn(
            "const int k = kFixedK == 0 ? runtime_k : kFixedK;", self.kernel
        )
        self.assertIn("if (use_global_sort_map)", self.launch)
        self.assertIn(
            "else if (use_decode_down_exact_dimensions(config))", self.launch
        )
        self.assertEqual(
            self.launch.count("fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>"), 1
        )
        self.assertEqual(
            self.launch.count("fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"), 1
        )
        self.assertEqual(
            self.launch.count(
                "fused_moe_i8_tn_mma_kernel<false, 4096, 7168, 2048>"
            ),
            1,
        )

    def test_exp080_body_payload_and_geometry_are_unchanged(self) -> None:
        protected = self.kernel[self.kernel.index("#define XH_MMA_STAGE_MNKX2") :]
        digest = hashlib.sha256(protected.encode("utf-8")).hexdigest()
        self.assertEqual(digest, PROTECTED_BODY_SHA256)
        self.assertEqual(protected.count("__syncthreadshared()"), 3)
        self.assertEqual(protected.count("XH_MMA_I8("), 2)
        self.assertEqual(protected.count("XH_MMA_STAGE_MNKX2("), 129)
        for prefix in ("load_a_", "load_b_"):
            for index in range(4):
                self.assertEqual(protected.count(f"{prefix}{index}"), 4)
        self.assertIsNone(re.search(r"load_[ab]\s*\[", protected))
        self.assertIn("constexpr int kMmaThreads = 256;", self.source)
        self.assertIn("constexpr int kMmaSharedBytes =", self.source)

    def test_full_sort_paths_remain_runtime_dimensioned(self) -> None:
        case2_selector = function_body(
            self.source, "use_case2_output_scratch_expert_sort("
        )
        case4_selector = function_body(
            self.source, "use_prefill_down_module_full_expert_sort("
        )
        self.assertEqual(
            " ".join(case2_selector.split()),
            "return config.em == 32768 && config.n == 4096 && config.k == 7168;",
        )
        self.assertEqual(
            " ".join(case4_selector.split()),
            "return config.em == 32768 && config.n == 7168 && config.k == 2048;",
        )
        self.assertIn("build_case2_full_expert_sort_map_kernel", self.launch)
        self.assertIn("module_full_sort_launch_grid_x(config)", self.launch)
        self.assertNotIn("<true, 32768", self.launch)

    def test_read_only_abi_and_no_host_synchronization(self) -> None:
        start = self.source.index("fused_moe_i8_tn_mma_kernel(")
        opening = self.source.index("{", start)
        signature = self.source[start:opening]
        for name in (
            "a_ptr",
            "b_ptr",
            "scale_a_ptr",
            "scale_b_ptr",
            "moe_weights_ptr",
            "expert_ids_ptr",
        ):
            self.assertRegex(signature, rf"const [^,]+\b{name}\b")
        self.assertIn("__nv_bfloat16* __restrict__ out_ptr", signature)
        self.assertNotIn("cudaMemcpy", self.source)
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaStreamSynchronize", self.source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
