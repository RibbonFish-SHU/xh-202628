#!/usr/bin/env python3
"""Summarize an ABBA pair of C500 benchmark logs without hiding raw samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


EXPECTED_CASES = (
    "decode-gate-up",
    "prefill-gate-up",
    "decode-down",
    "prefill-down",
)
FLOAT_PATTERN = r"[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?"
BENCHMARK_RE = re.compile(
    rf"^BENCHMARK device=c500-local case=(?P<case>[^ ]+) "
    rf"samples_ms=(?P<samples>{FLOAT_PATTERN}(?:,{FLOAT_PATTERN}){{4}}) "
    rf"median_ms=(?P<median>{FLOAT_PATTERN}) "
    rf"effective_TOPS=(?P<tops>{FLOAT_PATTERN}) sampled_correctness=PASS$"
)


@dataclass(frozen=True)
class BenchmarkRecord:
    samples_ms: tuple[float, ...]
    median_ms: float


def parse_log(path: Path) -> dict[str, BenchmarkRecord]:
    values: dict[str, BenchmarkRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = BENCHMARK_RE.search(line)
        if line.startswith("BENCHMARK device=c500-local") and match is None:
            raise SystemExit(f"malformed C500 benchmark record in {path}: {line}")
        if match:
            case = match.group("case")
            if case in values:
                raise SystemExit(f"duplicate benchmark case {case!r} in {path}")
            samples = tuple(float(value) for value in match.group("samples").split(","))
            median = float(match.group("median"))
            tops = float(match.group("tops"))
            if not all(math.isfinite(value) and value > 0.0 for value in (*samples, median, tops)):
                raise SystemExit(f"non-positive or non-finite benchmark value for {case!r} in {path}")
            calculated_median = statistics.median(samples)
            if not math.isclose(calculated_median, median, rel_tol=0.0, abs_tol=0.0005):
                raise SystemExit(
                    f"reported median does not match five raw samples for {case!r} in {path}"
                )
            values[case] = BenchmarkRecord(samples, median)
    if set(values) != set(EXPECTED_CASES):
        raise SystemExit(
            f"C500 benchmark log {path} must contain exactly these cases: "
            + ", ".join(EXPECTED_CASES)
        )
    return values


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"required environment variable is missing: {name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-a", type=Path, required=True)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-b", type=Path, required=True)
    parser.add_argument("--baseline-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ordered = {
        "baseline_a": parse_log(args.baseline_a),
        "candidate_a": parse_log(args.candidate_a),
        "candidate_b": parse_log(args.candidate_b),
        "baseline_b": parse_log(args.baseline_b),
    }
    cases: dict[str, dict[str, object]] = {}
    for case in EXPECTED_CASES:
        baseline_records = [ordered["baseline_a"][case], ordered["baseline_b"][case]]
        candidate_records = [ordered["candidate_a"][case], ordered["candidate_b"][case]]
        baseline_medians = [record.median_ms for record in baseline_records]
        candidate_medians = [record.median_ms for record in candidate_records]
        baseline_median = statistics.median(baseline_medians)
        candidate_median = statistics.median(candidate_medians)
        speedup = baseline_median / candidate_median
        cases[case] = {
            "baseline_ms_ab": baseline_medians,
            "candidate_ms_ab": candidate_medians,
            "baseline_raw_samples_ms_ab": [list(record.samples_ms) for record in baseline_records],
            "candidate_raw_samples_ms_ab": [list(record.samples_ms) for record in candidate_records],
            "baseline_median_ms": baseline_median,
            "candidate_median_ms": candidate_median,
            "speedup": speedup,
            "candidate_delta_percent": (candidate_median / baseline_median - 1.0) * 100.0,
            "baseline_drift_percent": (baseline_medians[1] / baseline_medians[0] - 1.0) * 100.0,
            "candidate_drift_percent": (candidate_medians[1] / candidate_medians[0] - 1.0) * 100.0,
        }

    payload = {
        "schema_version": 1,
        "device_class": "c500-local",
        "benchmark_design": "ABBA",
        "interpretation": "paired-relative-only",
        "candidate_commit": required_environment("C500_CANDIDATE_COMMIT"),
        "baseline_commit": required_environment("C500_BASELINE_COMMIT"),
        "submission_source": required_environment("C500_SUBMISSION_SOURCE"),
        "candidate_source_sha256": required_environment("C500_CANDIDATE_SOURCE_SHA256"),
        "baseline_source_sha256": required_environment("C500_BASELINE_SOURCE_SHA256"),
        "logs": {name: str(path) for name, path in {
            "baseline_a": args.baseline_a,
            "candidate_a": args.candidate_a,
            "candidate_b": args.candidate_b,
            "baseline_b": args.baseline_b,
        }.items()},
        "cases": cases,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
