from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "summarize-c500-abba.py"


class C500AbbaSummaryTests(unittest.TestCase):
    CASES = ("decode-gate-up", "prefill-gate-up", "decode-down", "prefill-down")

    def write_log(
        self,
        root: Path,
        name: str,
        values: dict[str, float],
        *,
        correctness: str = "PASS",
        sample_count: int = 5,
    ) -> Path:
        path = root / name
        lines = [
            "BENCHMARK device=c500-local "
            f"case={case} samples_ms="
            + ",".join(f"{value:.3f}" for _ in range(sample_count))
            + f" median_ms={value:.3f} effective_TOPS=1.000 sampled_correctness={correctness}"
            for case, value in values.items()
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "C500_CANDIDATE_COMMIT": "1" * 40,
            "C500_BASELINE_COMMIT": "2" * 40,
            "C500_SUBMISSION_SOURCE": "operators/example/submission.cu",
            "C500_CANDIDATE_SOURCE_SHA256": "3" * 64,
            "C500_BASELINE_SOURCE_SHA256": "4" * 64,
        }

    def test_summarizes_matching_abba_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_a = self.write_log(root, "baseline-a.log", dict.fromkeys(self.CASES, 10.0))
            candidate_a = self.write_log(root, "candidate-a.log", dict.fromkeys(self.CASES, 8.0))
            candidate_b = self.write_log(root, "candidate-b.log", dict.fromkeys(self.CASES, 8.2))
            baseline_b = self.write_log(root, "baseline-b.log", dict.fromkeys(self.CASES, 10.2))
            output = root / "summary.json"

            subprocess.run(
                [
                    "python", str(SCRIPT),
                    "--baseline-a", str(baseline_a),
                    "--candidate-a", str(candidate_a),
                    "--candidate-b", str(candidate_b),
                    "--baseline-b", str(baseline_b),
                    "--output", str(output),
                ],
                env=self.environment(),
                check=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["device_class"], "c500-local")
            self.assertEqual(payload["benchmark_design"], "ABBA")
            case = payload["cases"]["decode-gate-up"]
            self.assertAlmostEqual(case["baseline_median_ms"], 10.1)
            self.assertAlmostEqual(case["candidate_median_ms"], 8.1)
            self.assertAlmostEqual(case["speedup"], 10.1 / 8.1)
            self.assertEqual(len(case["candidate_raw_samples_ms_ab"][0]), 5)

    def test_rejects_mismatched_case_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = [
                self.write_log(root, "baseline-a.log", dict.fromkeys(self.CASES, 10.0)),
                self.write_log(root, "candidate-a.log", dict.fromkeys(self.CASES, 9.0)),
                self.write_log(root, "candidate-b.log", {"decode-gate-up": 9.0}),
                self.write_log(root, "baseline-b.log", dict.fromkeys(self.CASES, 10.0)),
            ]
            output = root / "summary.json"
            result = subprocess.run(
                [
                    "python", str(SCRIPT),
                    "--baseline-a", str(logs[0]),
                    "--candidate-a", str(logs[1]),
                    "--candidate-b", str(logs[2]),
                    "--baseline-b", str(logs[3]),
                    "--output", str(output),
                ],
                env=self.environment(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must contain exactly these cases", result.stderr)

    def test_rejects_incomplete_samples_or_missing_correctness(self) -> None:
        for kwargs in ({"sample_count": 4}, {"correctness": "FAIL"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = [
                    self.write_log(root, f"run-{index}.log", dict.fromkeys(self.CASES, 10.0), **kwargs)
                    for index in range(4)
                ]
                result = subprocess.run(
                    [
                        "python", str(SCRIPT),
                        "--baseline-a", str(paths[0]),
                        "--candidate-a", str(paths[1]),
                        "--candidate-b", str(paths[2]),
                        "--baseline-b", str(paths[3]),
                        "--output", str(root / "summary.json"),
                    ],
                    env=self.environment(),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("malformed C500 benchmark record", result.stderr)


if __name__ == "__main__":
    unittest.main()
