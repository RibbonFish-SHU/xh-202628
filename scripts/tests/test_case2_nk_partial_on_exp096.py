#!/usr/bin/env python3
"""No-device proof for case-2 partial N/K constants on exp-096."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260825-097.sh"

CASE1 = (4096, 4096, 7168)
CASE2 = (32768, 4096, 7168)
CASE3 = (4096, 7168, 2048)
CASE4 = (32768, 7168, 2048)
FALLBACK = (384, 2048, 1024)
BASE_SOURCE_SHA256 = (
    "3d80de91345e19098158cbeb2972381c07e445b698c9c8c584051fca5b2ed7b1"
)
CANDIDATE_SOURCE_SHA256 = (
    "34b6b34f12aa603b6502d8f981254620fe92cc732eb7d7d80a6e33d9c664bf96"
)
PROTECTED_BODY_SHA256 = (
    "80bc6355f0a2fd09ad37f721ae6ef493d1948976b9dfe341c329dddefea59e6c"
)
MAP_GRID_EPILOGUE_SHA256 = (
    "000b2911b9f41f5f1067b6d1079caa25ef01afd869477fddd382f2b239759e66"
)

BASE_CASE2_INSTANCE = "fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>"
CANDIDATE_CASE2_INSTANCE = (
    "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168>"
)
CASE4_INSTANCE = (
    "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>"
)
RUNTIME_INSTANCE = "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def replace_exactly_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise AssertionError(f"expected exactly one replacement for {old!r}")
    return source.replace(old, new, 1)


def reverse_to_exp096(source: str) -> str:
    return replace_exactly_once(
        source, CANDIDATE_CASE2_INSTANCE, BASE_CASE2_INSTANCE
    )


def dispatch(shape: tuple[int, int, int]) -> tuple[bool, tuple[int, int, int]]:
    if shape == CASE2:
        return True, (0, 4096, 7168)
    if shape == CASE4:
        return True, CASE4
    return False, (0, 0, 0)


def resolve_dimensions(
    fixed: tuple[int, int, int], runtime: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        runtime_value if fixed_value == 0 else fixed_value
        for fixed_value, runtime_value in zip(fixed, runtime)
    )


class Case2NkPartialOnExp096Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.baseline = reverse_to_exp096(cls.source)
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")
        cls.launch = function_body(cls.source, "static inline void launch(")

    def test_five_way_dispatch_is_exact(self) -> None:
        expected = {
            CASE1: (False, (0, 0, 0)),
            CASE2: (True, (0, 4096, 7168)),
            CASE3: (False, (0, 0, 0)),
            CASE4: (True, CASE4),
            FALLBACK: (False, (0, 0, 0)),
        }
        self.assertEqual({shape: dispatch(shape) for shape in expected}, expected)
        for shape, (_, fixed) in expected.items():
            self.assertEqual(resolve_dimensions(fixed, shape), shape)

        self.assertEqual(self.launch.count(CANDIDATE_CASE2_INSTANCE), 1)
        self.assertEqual(self.launch.count(CASE4_INSTANCE), 1)
        self.assertEqual(self.launch.count(RUNTIME_INSTANCE), 1)
        self.assertNotIn(BASE_CASE2_INSTANCE, self.launch)
        self.assertNotIn("<true, 32768, 4096, 7168>", self.launch)

    def test_case2_keeps_runtime_em_and_has_56_full_k_tiles(self) -> None:
        self.assertIn(
            "template <bool kUseOutputScratchExpertSort, int kFixedEm, "
            "int kFixedN, int kFixedK>",
            self.source,
        )
        self.assertIn(
            "const int em = kFixedEm == 0 ? runtime_em : kFixedEm;",
            self.kernel,
        )
        self.assertEqual(CASE2[2] // 128, 56)
        self.assertEqual((CASE2[2] - 1) % 128 + 1, 128)
        self.assertIn(
            "const int num_k_tiles = (k + kMmaTileK - 1) / kMmaTileK;",
            self.kernel,
        )
        self.assertIn("constexpr int kMmaTileK = 128;", self.source)

    def test_kernel_body_and_all_barriers_are_exp096_identical(self) -> None:
        protected = self.kernel[self.kernel.index("#define XH_MMA_STAGE_MNKX2") :]
        self.assertEqual(digest(protected), PROTECTED_BODY_SHA256)
        self.assertEqual(protected.count("__syncthreadshared()"), 3)
        self.assertEqual(protected.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(protected.count("XH_MMA_I8("), 2)
        for prefix in ("load_a_", "load_b_"):
            for index in range(4):
                self.assertEqual(protected.count(f"{prefix}{index}"), 4)
        self.assertIsNone(re.search(r"load_[ab]\s*\[", protected))

    def test_map_grid_geometry_and_epilogue_are_unchanged(self) -> None:
        protected = (
            function_body(self.source, "build_case2_full_expert_sort_map_kernel(")
            + function_body(self.source, "module_full_sort_launch_grid_x(")
            + self.kernel[self.kernel.index("    int output_row[8];") :]
        )
        self.assertEqual(digest(protected), MAP_GRID_EPILOGUE_SHA256)
        self.assertEqual(8 * 32 * 32, 8192)
        self.assertEqual(8 * 56 * 32, 14336)
        self.assertEqual(
            self.launch.count("build_case2_full_expert_sort_map_kernel"), 1
        )
        self.assertIn("module_full_sort_launch_grid_x(config)", self.launch)
        self.assertIn("constexpr int kMmaThreads = 256;", self.source)
        self.assertIn("constexpr int kMmaSharedBytes =", self.source)

    def test_reverse_transform_restores_exact_exp096_source(self) -> None:
        self.assertEqual(digest(self.source), CANDIDATE_SOURCE_SHA256)
        self.assertEqual(digest(self.baseline), BASE_SOURCE_SHA256)
        self.assertEqual(
            self.source.count(CANDIDATE_CASE2_INSTANCE), 1
        )
        self.assertEqual(self.baseline.count(BASE_CASE2_INSTANCE), 1)
        self.assertNotIn(CANDIDATE_CASE2_INSTANCE, self.baseline)

    def test_abi_read_only_inputs_and_fallback_are_unchanged(self) -> None:
        signature_start = self.source.index("fused_moe_i8_tn_mma_kernel(")
        signature_end = self.source.index("{", signature_start)
        signature = self.source[signature_start:signature_end]
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
        self.assertIn(RUNTIME_INSTANCE, self.launch)
        for forbidden in (
            "cudaMemcpy",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "ldg_b128_bsm",
            "#pragma unroll 2",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_remote_job_is_byte_identical_to_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
