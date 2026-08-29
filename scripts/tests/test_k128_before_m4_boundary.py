import hashlib
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators/fused_moe_i8_tn/cuda_maca/submission.cu"
TEMPLATE_JOB = ROOT / "templates/remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs/exp-20260830-243.sh"
FORMAL_COMMIT = "8341aa38e55659285673111df90e786c1fba08df"
FORMAL_LF_SHA256 = (
    "083eb1262dbe220aea0c2b324a00f2cf9b14720dc2b8c1701b261f794eaf8cf1"
)
SOURCE_RELATIVE = "operators/fused_moe_i8_tn/cuda_maca/submission.cu"


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


def committed_source(commit: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{SOURCE_RELATIVE}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def dedent_four(text: str) -> str:
    return "\n".join(line[4:] if line.startswith("    ") else line for line in text.split("\n"))


class K128BeforeM4BoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.formal = committed_source(FORMAL_COMMIT)
        cls.launch = function_body(cls.source, "static inline void launch(")
        cls.formal_launch = function_body(cls.formal, "static inline void launch(")

    def test_only_launch_changed_and_inverse_restores_formal(self) -> None:
        self.assertEqual(self.source.count(self.launch), 1)
        restored = self.source.replace(self.launch, self.formal_launch)
        self.assertEqual(
            hashlib.sha256(restored.encode("utf-8")).hexdigest(),
            FORMAL_LF_SHA256,
        )

    def test_formal_k128_block_is_only_wrapped_and_moved(self) -> None:
        formal_start = self.formal_launch.index("    const dim3 block(kMmaThreads);")
        formal_end = self.formal_launch.index("#else")
        candidate_start = self.launch.index("        const dim3 block(kMmaThreads);")
        candidate_end = self.launch.index("        return;", candidate_start)
        candidate_core = self.launch[candidate_start:candidate_end]
        self.assertEqual(dedent_four(candidate_core), self.formal_launch[formal_start:formal_end])
        self.assertEqual(self.launch.count("const bool use_decode_m4"), 1)
        self.assertEqual(self.launch.count("if (!use_decode_m4)"), 1)

    def test_k128_references_precede_decode_m4(self) -> None:
        case4 = self.launch.index(
            "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>"
        )
        case2 = self.launch.index(
            "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168>"
        )
        generic = self.launch.index(
            "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"
        )
        decode = self.launch.index("sync_m4::launch_decode(")
        self.assertLess(case4, case2)
        self.assertLess(case2, generic)
        self.assertLess(generic, decode)
        self.assertEqual(self.source.count("fused_moe_i8_tn_mma_kernel<"), 3)
        builder = self.launch.index("build_case2_full_expert_sort_map_kernel")
        self.assertLess(builder, case4)
        self.assertLess(builder, case2)

    def test_public_and_fallback_dispatch_actions_are_equal(self) -> None:
        def action(em: int, n: int, k: int) -> str:
            if em == 4096 and (n, k) in ((4096, 7168), (7168, 2048)):
                return "decode-m4"
            if (em, n, k) == (32768, 7168, 2048):
                return "case4-k128-sorted"
            if (em, n, k) == (32768, 4096, 7168):
                return "case2-k128-sorted"
            return "generic-k128"

        shapes = [
            (4096, 4096, 7168),
            (32768, 4096, 7168),
            (4096, 7168, 2048),
            (32768, 7168, 2048),
            (384, 2048, 1024),
        ]
        expected = [
            "decode-m4",
            "case2-k128-sorted",
            "decode-m4",
            "case4-k128-sorted",
            "generic-k128",
        ]
        self.assertEqual([action(*shape) for shape in shapes], expected)

    def test_remote_job_matches_trusted_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main()
