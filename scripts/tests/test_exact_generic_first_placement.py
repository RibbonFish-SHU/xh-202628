#!/usr/bin/env python3
"""No-device proof for generic-first native placement of exact down kernels."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260826-107.sh"

CASE1 = (4096, 4096, 7168)
CASE2 = (32768, 4096, 7168)
CASE3 = (4096, 7168, 2048)
CASE4 = (32768, 7168, 2048)
FALLBACK = (384, 2048, 1024)

FORMAL_BEST_SHA256 = (
    "762b7581919a9cb75cde0e4732c8ac580ad236f86620f2d3f34c565d7c5a3204"
)
EXP105_SHA256 = (
    "0f922248b25d563d38ec8e5392f1be5804b832870b8ca90ba79294a2cafebd68"
)
CANDIDATE_SHA256 = (
    "d0de8b33b509e3ca76092a6d326e9fd7a29228f6a54b0c7671f288e4025cb554"
)
PROTECTED_BODY_SHA256 = (
    "80bc6355f0a2fd09ad37f721ae6ef493d1948976b9dfe341c329dddefea59e6c"
)
MAP_GRID_EPILOGUE_SHA256 = (
    "000b2911b9f41f5f1067b6d1079caa25ef01afd869477fddd382f2b239759e66"
)

CASE3_SELECTOR = """static inline bool use_decode_down_exact_dimensions(const KernelConfig& config) {
    return config.em == 4096 && config.n == 7168 && config.k == 2048;
}

"""
BASE_TEMPLATE = "template <bool kUseOutputScratchExpertSort>"
FULL_TEMPLATE = (
    "template <bool kUseOutputScratchExpertSort, int kFixedEm, "
    "int kFixedN, int kFixedK>"
)
BASE_PARAMETERS = """    int em,
    int n,
    int k
"""
FULL_PARAMETERS = """    int runtime_em,
    int runtime_n,
    int runtime_k
"""
FULL_BINDINGS = """    const int em = kFixedEm == 0 ? runtime_em : kFixedEm;
    const int n = kFixedN == 0 ? runtime_n : kFixedN;
    const int k = kFixedK == 0 ? runtime_k : kFixedK;

"""
BASE_DISPATCH = """    if (use_global_sort_map) {
        build_case2_full_expert_sort_map_kernel
            <<<1, kOutputScratchSortTiles>>>(expert_ids);
        fused_moe_i8_tn_mma_kernel<true><<<grid, block>>>(
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
        fused_moe_i8_tn_mma_kernel<false><<<grid, block>>>(
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
EXP105_DISPATCH = """    if (use_global_sort_map) {
        build_case2_full_expert_sort_map_kernel
            <<<1, kOutputScratchSortTiles>>>(expert_ids);
        if (use_case2_output_scratch_expert_sort(config)) {
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
        } else {
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
        }
    } else if (use_decode_down_exact_dimensions(config)) {
        fused_moe_i8_tn_mma_kernel<false, 4096, 7168, 2048><<<grid, block>>>(
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
        fused_moe_i8_tn_mma_kernel<false, 0, 0, 0><<<grid, block>>>(
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
CANDIDATE_DISPATCH = """    if (use_global_sort_map) {
        build_case2_full_expert_sort_map_kernel
            <<<1, kOutputScratchSortTiles>>>(expert_ids);
        if (use_case2_output_scratch_expert_sort(config)) {
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
        } else {
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
        }
    } else if (!use_decode_down_exact_dimensions(config)) {
        fused_moe_i8_tn_mma_kernel<false, 0, 0, 0><<<grid, block>>>(
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
        fused_moe_i8_tn_mma_kernel<false, 4096, 7168, 2048><<<grid, block>>>(
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


def reconstruct_exp105(source: str) -> str:
    return replace_exactly_once(source, CANDIDATE_DISPATCH, EXP105_DISPATCH)


def reverse_to_formal_best(source: str) -> str:
    reverted = reconstruct_exp105(source)
    reverted = replace_exactly_once(reverted, CASE3_SELECTOR, "")
    reverted = replace_exactly_once(reverted, FULL_TEMPLATE, BASE_TEMPLATE)
    reverted = replace_exactly_once(reverted, FULL_PARAMETERS, BASE_PARAMETERS)
    reverted = replace_exactly_once(reverted, FULL_BINDINGS, "")
    return replace_exactly_once(reverted, EXP105_DISPATCH, BASE_DISPATCH)


def dispatch_actions(shape: tuple[int, int, int]) -> tuple[str, ...]:
    use_case2 = shape == CASE2
    use_case3 = shape == CASE3
    use_case4 = shape == CASE4
    use_global_sort_map = use_case2 or use_case4
    if use_case2:
        return "map", "runtime-sorted"
    if not use_global_sort_map and not use_case3:
        return ("runtime-unsorted",)
    if use_case4:
        return "map", "fixed-sorted"
    return ("fixed-unsorted",)


class ExactGenericFirstPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.exp105 = reconstruct_exp105(cls.source)
        cls.baseline = reverse_to_formal_best(cls.source)
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")
        cls.launch = function_body(cls.source, "static inline void launch(")

    def test_inverse_transform_recovers_exp105_and_formal_best(self) -> None:
        self.assertEqual(digest(self.source), CANDIDATE_SHA256)
        self.assertEqual(digest(self.exp105), EXP105_SHA256)
        self.assertEqual(digest(self.baseline), FORMAL_BEST_SHA256)
        self.assertEqual(
            replace_exactly_once(
                self.exp105, EXP105_DISPATCH, CANDIDATE_DISPATCH
            ),
            self.source,
        )

    def test_dispatch_is_exclusive_and_preserves_map_prerequisites(self) -> None:
        expected = {
            CASE1: ("runtime-unsorted",),
            CASE2: ("map", "runtime-sorted"),
            CASE3: ("fixed-unsorted",),
            CASE4: ("map", "fixed-sorted"),
            FALLBACK: ("runtime-unsorted",),
        }
        self.assertEqual(
            {shape: dispatch_actions(shape) for shape in expected}, expected
        )
        self.assertEqual(self.launch.count("build_case2_full_expert_sort_map_kernel"), 1)
        self.assertEqual(self.source.count(CANDIDATE_DISPATCH), 1)

        selectors = {
            "use_case2_output_scratch_expert_sort(": CASE2,
            "use_decode_down_exact_dimensions(": CASE3,
            "use_prefill_down_module_full_expert_sort(": CASE4,
        }
        for signature, shape in selectors.items():
            body = " ".join(function_body(self.source, signature).split())
            self.assertEqual(
                body,
                f"return config.em == {shape[0]} && config.n == {shape[1]} "
                f"&& config.k == {shape[2]};",
            )

    def test_native_references_keep_sorted_pair_then_runtime_unsorted(self) -> None:
        references = (
            "fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>",
            "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>",
            "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>",
            "fused_moe_i8_tn_mma_kernel<false, 4096, 7168, 2048>",
        )
        positions = []
        for reference in references:
            self.assertEqual(self.launch.count(reference), 1)
            positions.append(self.launch.index(reference))
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("<true, 0, 4096, 7168>", self.launch)
        self.assertNotIn("<true, 32768, 4096, 7168>", self.launch)
        self.assertNotIn("<false, 4096, 4096, 7168>", self.launch)

    def test_kernel_body_address_mma_lds_sts_and_barriers_are_unchanged(self) -> None:
        protected = self.kernel[self.kernel.index("#define XH_MMA_STAGE_MNKX2") :]
        self.assertEqual(digest(protected), PROTECTED_BODY_SHA256)
        self.assertEqual(protected.count("__syncthreadshared()"), 3)
        self.assertEqual(protected.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(protected.count("XH_MMA_I8("), 2)
        self.assertGreater(protected.count("XH_MMA_LDS("), 0)
        self.assertGreater(protected.count("XH_MMA_STS("), 0)
        for prefix in ("load_a_", "load_b_"):
            for index in range(4):
                self.assertEqual(protected.count(f"{prefix}{index}"), 4)
        self.assertIsNone(re.search(r"load_[ab]\s*\[", protected))

    def test_map_grid_epilogue_geometry_abi_and_closed_families_are_unchanged(self) -> None:
        protected = (
            function_body(self.source, "build_case2_full_expert_sort_map_kernel(")
            + function_body(self.source, "module_full_sort_launch_grid_x(")
            + self.kernel[self.kernel.index("    int output_row[8];") :]
        )
        self.assertEqual(digest(protected), MAP_GRID_EPILOGUE_SHA256)
        self.assertIn("module_full_sort_launch_grid_x(config)", self.launch)
        self.assertIn("constexpr int kMmaThreads = 256;", self.source)
        self.assertIn("constexpr int kMmaSharedBytes =", self.source)
        self.assertEqual(8 * 32 * 32, 8192)
        self.assertEqual(8 * 56 * 32, 14336)

        signature_start = self.source.index('extern "C" void run_kernel(')
        signature_end = self.source.index("{", signature_start)
        baseline_start = self.baseline.index('extern "C" void run_kernel(')
        baseline_end = self.baseline.index("{", baseline_start)
        self.assertEqual(
            self.source[signature_start:signature_end],
            self.baseline[baseline_start:baseline_end],
        )
        for forbidden in (
            "ldg_b128_bsm",
            "kMmaTileK = 64",
            "kSteadyKUnrollFactor",
            "u32_offset_product",
            "kPrefillDownCluster",
            "#pragma unroll 2",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_remote_job_is_byte_identical_to_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
