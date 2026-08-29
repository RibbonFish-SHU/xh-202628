import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators/fused_moe_i8_tn/cuda_maca/submission.cu"
TEMPLATE_JOB = ROOT / "templates/remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs/exp-20260830-240.sh"

FORMAL_LF_SHA256 = (
    "083eb1262dbe220aea0c2b324a00f2cf9b14720dc2b8c1701b261f794eaf8cf1"
)

FORMAL_DISPATCH = """        if (use_prefill_down_module_full_expert_sort(config)) {
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
            fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168><<<grid, block>>>(
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

CANDIDATE_DISPATCH = """        if (!use_prefill_down_module_full_expert_sort(config)) {
            fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168><<<grid, block>>>(
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
        }
"""


def function_body(source: str, signature: str) -> str:
    start = source.rindex(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


class Case2K128FirstLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.launch = function_body(cls.source, "static inline void launch(")

    def test_only_dispatch_source_order_changes(self) -> None:
        self.assertEqual(self.source.count(CANDIDATE_DISPATCH), 1)
        restored = self.source.replace(CANDIDATE_DISPATCH, FORMAL_DISPATCH)
        self.assertEqual(
            hashlib.sha256(restored.encode("utf-8")).hexdigest(),
            FORMAL_LF_SHA256,
        )

    def test_case2_instance_precedes_case4_and_generic(self) -> None:
        case2 = self.launch.index(
            "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168>"
        )
        case4 = self.launch.index(
            "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>"
        )
        generic = self.launch.index(
            "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"
        )
        self.assertLess(case2, case4)
        self.assertLess(case4, generic)
        self.assertEqual(
            self.source.count("fused_moe_i8_tn_mma_kernel<"),
            3,
        )

    def test_inverted_polarity_preserves_sorted_dispatch(self) -> None:
        case2 = lambda em, n, k: em == 32768 and n == 4096 and k == 7168
        case4 = lambda em, n, k: em == 32768 and n == 7168 and k == 2048
        shapes = [
            (4096, 4096, 7168),
            (32768, 4096, 7168),
            (4096, 7168, 2048),
            (32768, 7168, 2048),
            (384, 2048, 1024),
        ]
        for shape in shapes:
            use_sorted = case2(*shape) or case4(*shape)
            if use_sorted:
                self.assertEqual(not case4(*shape), case2(*shape))
        self.assertIn(
            "if (!use_prefill_down_module_full_expert_sort(config))",
            self.launch,
        )

    def test_remote_job_matches_trusted_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main()
