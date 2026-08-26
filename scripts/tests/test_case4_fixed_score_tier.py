#!/usr/bin/env python3
"""No-device proof for the score-tier case-4 fixed clone replay."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260827-166.sh"

CASE1 = (4096, 4096, 7168)
CASE2 = (32768, 4096, 7168)
CASE3 = (4096, 7168, 2048)
CASE4 = (32768, 7168, 2048)
FALLBACK = (384, 2048, 1024)

FORMAL_SOURCE_SHA256 = (
    "4fe1962ccbade0e598c39c7ac0369cb85b62ed74fc97967a7e018c198acd58c7"
)
CANDIDATE_SOURCE_SHA256 = (
    "0edbf3ba0efea172fa1958e25c6904ec4314ff86b9249deaaac56a98092341b3"
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
            fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>
                <<<grid, block>>>(
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


def function_signature(source: str, signature: str) -> str:
    start = source.index(signature)
    return source[start : source.index("{", start)]


def reverse_to_formal(source: str) -> str:
    reverted = source.replace(CANDIDATE_TEMPLATE, BASE_TEMPLATE, 1)
    reverted = reverted.replace(CANDIDATE_PARAMETERS, BASE_PARAMETERS, 1)
    reverted = reverted.replace(DIMENSION_BINDINGS, "", 1)
    reverted = reverted.replace(CANDIDATE_SORTED_LAUNCH, BASE_SORTED_LAUNCH, 1)
    return reverted.replace(
        "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>",
        "fused_moe_i8_tn_mma_kernel<false>",
        1,
    )


def dispatch(shape: tuple[int, int, int]) -> tuple[bool, tuple[int, int, int]]:
    use_case4 = shape == CASE4
    use_sort = shape == CASE2 or use_case4
    if use_case4:
        return True, CASE4
    return use_sort, (0, 0, 0)


def invocation_lines(source: str, name: str) -> tuple[str, ...]:
    pattern = re.compile(rf"^[ \t]*({name}\([^;]+\);)$", re.MULTILINE)
    return tuple(pattern.findall(source))


class Case4FixedScoreTierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.formal = reverse_to_formal(cls.source)
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")
        cls.formal_kernel = function_body(
            cls.formal, "fused_moe_i8_tn_mma_kernel("
        )
        cls.launch = function_body(cls.source, "static inline void launch(")

    def test_exact_replay_reverse_projection_and_job_identity(self) -> None:
        self.assertEqual(digest(self.source), CANDIDATE_SOURCE_SHA256)
        self.assertEqual(digest(self.formal), FORMAL_SOURCE_SHA256)
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())

    def test_only_exact_case4_uses_full_fixed_dimensions(self) -> None:
        expected = {
            CASE1: (False, (0, 0, 0)),
            CASE2: (True, (0, 0, 0)),
            CASE3: (False, (0, 0, 0)),
            CASE4: (True, CASE4),
            FALLBACK: (False, (0, 0, 0)),
        }
        self.assertEqual({shape: dispatch(shape) for shape in expected}, expected)

        selector = function_body(
            self.source, "use_prefill_down_module_full_expert_sort("
        )
        self.assertEqual(
            " ".join(selector.split()),
            "return config.em == 32768 && config.n == 7168 && config.k == 2048;",
        )
        calls = (
            "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>",
            "fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>",
            "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>",
        )
        for call in calls:
            self.assertEqual(self.launch.count(call), 1)
        self.assertLess(self.launch.index(calls[0]), self.launch.index(calls[1]))
        self.assertLess(self.launch.index(calls[1]), self.launch.index(calls[2]))
        self.assertNotIn("<true, 32768, 4096, 7168>", self.launch)
        self.assertNotIn("<false, 32768", self.launch)

    def test_runtime_clones_and_exp150_kernel_suffix_are_exact(self) -> None:
        self.assertIn(CANDIDATE_TEMPLATE, self.source)
        for binding in (
            "const int em = kFixedEm == 0 ? runtime_em : kFixedEm;",
            "const int n = kFixedN == 0 ? runtime_n : kFixedN;",
            "const int k = kFixedK == 0 ? runtime_k : kFixedK;",
        ):
            self.assertIn(binding, self.kernel)

        protected = self.kernel[self.kernel.index("#define XH_MMA_STAGE_MNKX2") :]
        formal_protected = self.formal_kernel[
            self.formal_kernel.index("#define XH_MMA_STAGE_MNKX2") :
        ]
        self.assertEqual(protected, formal_protected)
        for name in (
            "XH_LDG_A_STAGE_I",
            "XH_LDG_B_STAGE_I",
            "XH_MMA_STS",
            "XH_LDS_A_B128",
            "XH_LDS_B_B128",
            "XH_MMA_STAGE_MNKX2",
        ):
            self.assertEqual(
                invocation_lines(protected, name),
                invocation_lines(formal_protected, name),
            )
        self.assertEqual(protected.count("__syncthreadshared();"), 3)
        self.assertEqual(len(invocation_lines(protected, "XH_MMA_STAGE_MNKX2")), 128)

    def test_late_a00_a01_pair_anchors_remain_exact(self) -> None:
        steady_start = self.kernel.index(
            "for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k)"
        )
        tail_start = self.kernel.index("    int output_row[8];", steady_start)
        steady = self.kernel[steady_start:tail_start]
        tail = self.kernel[tail_start:]
        self.assertIn(
            "XH_LDG_B_STAGE_I(1);\n        XH_LDS_A_B128(0, 0);\n"
            "        XH_MMA_STAGE_MNKX2(0, 0, 0);",
            steady,
        )
        self.assertIn(
            "XH_LDS_B_B128(4, 1);\n        XH_LDS_A_B128(0, 1);\n"
            "        XH_MMA_STAGE_MNKX2(0, 0, 4);",
            steady,
        )
        self.assertTrue(
            tail.startswith(
                "    int output_row[8];\n    XH_LDS_A_B128(0, 0);\n"
                "    XH_MMA_STAGE_MNKX2(0, 0, 0);"
            )
        )
        self.assertIn(
            "XH_LDS_B_B128(4, 1);\n    XH_LDS_A_B128(0, 1);\n"
            "    XH_MMA_STAGE_MNKX2(0, 0, 4);",
            tail,
        )

    def test_map_geometry_public_abi_and_read_only_inputs_are_preserved(self) -> None:
        self.assertIn("module_full_sort_launch_grid_x(config)", self.launch)
        self.assertEqual(
            self.launch.count("build_case2_full_expert_sort_map_kernel"), 1
        )
        self.assertEqual(8 * 32 * 32, 8192)
        self.assertEqual(8 * 56 * 32, 14336)

        public = function_signature(self.source, 'extern "C" void run_kernel(')
        formal_public = function_signature(
            self.formal, 'extern "C" void run_kernel('
        )
        self.assertEqual(public, formal_public)
        kernel_signature = function_signature(
            self.source, "fused_moe_i8_tn_mma_kernel("
        )
        for name in (
            "a_ptr",
            "b_ptr",
            "scale_a_ptr",
            "scale_b_ptr",
            "moe_weights_ptr",
            "expert_ids_ptr",
        ):
            self.assertRegex(kernel_signature, rf"const [^,]+\b{name}\b")
        self.assertIn("__nv_bfloat16* __restrict__ out_ptr", kernel_signature)
        for forbidden in (
            "cudaMemcpy",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
        ):
            self.assertNotIn(forbidden, self.source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
