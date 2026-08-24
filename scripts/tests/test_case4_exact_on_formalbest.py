#!/usr/bin/env python3
"""No-device proof for exact case-4 dimensions on the formal-best source."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260825-096.sh"

CASE1 = (4096, 4096, 7168)
CASE2 = (32768, 4096, 7168)
CASE3 = (4096, 7168, 2048)
CASE4 = (32768, 7168, 2048)
FALLBACK = (384, 2048, 1024)
BASELINE_SOURCE_SHA256 = (
    "762b7581919a9cb75cde0e4732c8ac580ad236f86620f2d3f34c565d7c5a3204"
)
CANDIDATE_SOURCE_SHA256 = (
    "3d80de91345e19098158cbeb2972381c07e445b698c9c8c584051fca5b2ed7b1"
)
PROTECTED_BODY_SHA256 = (
    "80bc6355f0a2fd09ad37f721ae6ef493d1948976b9dfe341c329dddefea59e6c"
)
MAP_GRID_EPILOGUE_SHA256 = (
    "000b2911b9f41f5f1067b6d1079caa25ef01afd869477fddd382f2b239759e66"
)

BASE_TEMPLATE = "template <bool kUseOutputScratchExpertSort>"
CANDIDATE_TEMPLATE = (
    "template <bool kUseOutputScratchExpertSort, int kFixedEm, "
    "int kFixedN, int kFixedK>"
)
BASE_PARAMETERS = """    int em,
    int n,
    int k
"""
CANDIDATE_PARAMETERS = """    int runtime_em,
    int runtime_n,
    int runtime_k
"""
DIMENSION_BINDINGS = """    const int em = kFixedEm == 0 ? runtime_em : kFixedEm;
    const int n = kFixedN == 0 ? runtime_n : kFixedN;
    const int k = kFixedK == 0 ? runtime_k : kFixedK;

"""

BASE_SORTED_LAUNCH = """        fused_moe_i8_tn_mma_kernel<true><<<grid, block>>>(
            a,
            b_col_major,
            scale_a,
            scale_b,
            moe_weights,
            expert_ids,
            out,
            config.em,
            config.n,
            config.k
        );
"""
CANDIDATE_SORTED_LAUNCH = """        if (use_prefill_down_module_full_expert_sort(config)) {
            fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048><<<grid, block>>>(
                a,
                b_col_major,
                scale_a,
                scale_b,
                moe_weights,
                expert_ids,
                out,
                config.em,
                config.n,
                config.k
            );
        } else {
            fused_moe_i8_tn_mma_kernel<true, 0, 0, 0><<<grid, block>>>(
                a,
                b_col_major,
                scale_a,
                scale_b,
                moe_weights,
                expert_ids,
                out,
                config.em,
                config.n,
                config.k
            );
        }
"""


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


def reverse_to_formal_best(source: str) -> str:
    reverted = replace_exactly_once(source, CANDIDATE_TEMPLATE, BASE_TEMPLATE)
    reverted = replace_exactly_once(
        reverted, CANDIDATE_PARAMETERS, BASE_PARAMETERS
    )
    reverted = replace_exactly_once(reverted, DIMENSION_BINDINGS, "")
    reverted = replace_exactly_once(
        reverted, CANDIDATE_SORTED_LAUNCH, BASE_SORTED_LAUNCH
    )
    return replace_exactly_once(
        reverted,
        "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>",
        "fused_moe_i8_tn_mma_kernel<false>",
    )


def dispatch(shape: tuple[int, int, int]) -> tuple[bool, tuple[int, int, int]]:
    use_sort = shape in (CASE2, CASE4)
    fixed = CASE4 if shape == CASE4 else (0, 0, 0)
    return use_sort, fixed


def resolve_dimensions(
    fixed: tuple[int, int, int], runtime: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        runtime_value if fixed_value == 0 else fixed_value
        for fixed_value, runtime_value in zip(fixed, runtime)
    )


class Case4ExactOnFormalBestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.baseline = reverse_to_formal_best(cls.source)
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")
        cls.launch = function_body(cls.source, "static inline void launch(")

    def test_only_exact_sorted_case4_selects_fixed_dimensions(self) -> None:
        expected = {
            CASE1: (False, (0, 0, 0)),
            CASE2: (True, (0, 0, 0)),
            CASE3: (False, (0, 0, 0)),
            CASE4: (True, CASE4),
            FALLBACK: (False, (0, 0, 0)),
        }
        self.assertEqual({shape: dispatch(shape) for shape in expected}, expected)
        for shape, (_, fixed) in expected.items():
            self.assertEqual(resolve_dimensions(fixed, shape), shape)

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
        self.assertEqual(
            self.launch.count(
                "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>"
            ),
            1,
        )
        self.assertEqual(
            self.launch.count("fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>"),
            1,
        )
        self.assertEqual(
            self.launch.count("fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"),
            1,
        )
        self.assertNotIn("<true, 32768, 4096, 7168>", self.launch)
        self.assertNotIn("<true, 4096, 7168>", self.launch)

    def test_reverse_transform_matches_exact_formal_best_source(self) -> None:
        self.assertEqual(digest(self.source), CANDIDATE_SOURCE_SHA256)
        self.assertEqual(digest(self.baseline), BASELINE_SOURCE_SHA256)
        self.assertNotIn("kFixedEm", self.baseline)
        self.assertNotIn("kFixedN", self.baseline)
        self.assertNotIn("kFixedK", self.baseline)

    def test_dimension_bindings_protect_the_complete_kernel_body(self) -> None:
        self.assertEqual(self.source.count(CANDIDATE_TEMPLATE), 1)
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
        protected = self.kernel[self.kernel.index("#define XH_MMA_STAGE_MNKX2") :]
        self.assertEqual(digest(protected), PROTECTED_BODY_SHA256)
        self.assertEqual(protected.count("__syncthreadshared()"), 3)
        self.assertEqual(protected.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(protected.count("XH_MMA_I8("), 2)
        self.assertNotIn("kSteadyKUnrollFactor", self.source)
        self.assertNotIn("#pragma unroll 2", self.source)
        for prefix in ("load_a_", "load_b_"):
            for index in range(4):
                self.assertEqual(protected.count(f"{prefix}{index}"), 4)
        self.assertIsNone(re.search(r"load_[ab]\s*\[", protected))

    def test_map_grid_stable_rank_geometry_and_epilogue_are_unchanged(self) -> None:
        protected = (
            function_body(self.source, "build_case2_full_expert_sort_map_kernel(")
            + function_body(self.source, "module_full_sort_launch_grid_x(")
            + self.kernel[self.kernel.index("    int output_row[8];") :]
        )
        self.assertEqual(digest(protected), MAP_GRID_EPILOGUE_SHA256)
        builder = function_body(
            self.source, "build_case2_full_expert_sort_map_kernel("
        )
        self.assertIn("case2_full_stable_sort_rank(", builder)
        self.assertIn(
            "g_case2_full_expert_sort_map[physical_tile_m] = logical_tile_m;",
            builder,
        )
        self.assertEqual(8 * 32 * 32, 8192)
        self.assertEqual(8 * 56 * 32, 14336)
        self.assertEqual(
            self.launch.count("build_case2_full_expert_sort_map_kernel"), 1
        )
        self.assertIn("module_full_sort_launch_grid_x(config)", self.launch)
        self.assertIn("constexpr int kMmaThreads = 256;", self.source)
        self.assertIn("constexpr int kMmaSharedBytes =", self.source)
        self.assertNotIn("tile_zero", self.launch)

    def test_abi_inputs_and_runtime_fallback_are_unchanged(self) -> None:
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
        self.assertIn("int runtime_em", signature)
        self.assertIn("int runtime_n", signature)
        self.assertIn("int runtime_k", signature)
        self.assertIn(
            "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>", self.launch
        )
        for forbidden in (
            "cudaMemcpy",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "ldg_b128_bsm",
            "kMmaTileK = 64",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_remote_job_is_byte_identical_to_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
