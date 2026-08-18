#!/usr/bin/env python3
"""Append one reproducible operator experiment to a JSONL ledger."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[3] / "state" / "experiments.jsonl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a measured XH-202628 operator experiment to JSONL."
    )
    parser.add_argument("--id", required=True, help="Unique experiment ID")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--device",
        required=True,
        choices=("cpu", "nvidia", "c500-local", "xpuoj-c500"),
    )
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--correctness", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--decision", required=True, choices=("keep", "revert", "investigate")
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument(
        "--artifact", action="append", default=[], help="Relative artifact path"
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--timestamp", default=None, help="ISO-8601; defaults to UTC now")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return parser


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict) and isinstance(value.get("id"), str):
            ids.add(value["id"])
    return ids


def main() -> int:
    args = build_parser().parse_args()
    ledger = args.ledger.resolve()
    if args.id in existing_ids(ledger):
        raise SystemExit(f"Experiment ID already exists: {args.id}")

    record = {
        "schema_version": 1,
        "id": args.id,
        "timestamp": args.timestamp or utc_now(),
        "operator": args.operator,
        "language": args.language,
        "device": args.device,
        "environment": args.environment,
        "baseline_commit": args.baseline_commit,
        "candidate_commit": args.candidate_commit,
        "hypothesis": args.hypothesis,
        "change": args.change,
        "correctness": args.correctness,
        "benchmark": args.benchmark,
        "result": args.result,
        "decision": args.decision,
        "artifacts": args.artifact,
        "notes": args.notes,
    }

    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"Recorded experiment {args.id} in {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
