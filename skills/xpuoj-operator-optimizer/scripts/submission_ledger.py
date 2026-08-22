#!/usr/bin/env python3
"""Enforce the report-before-next-XPU-OJ-submission gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE = (
    Path(__file__).resolve().parents[3] / "state" / "submission-state.json"
)
CONTROLLER = Path(__file__).resolve().with_name("submission_controller.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_state() -> dict[str, Any]:
    return {"version": 1, "pending_report": None, "submissions": []}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid submission state {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("submissions"), list):
        raise SystemExit(f"Invalid submission state schema: {path}")
    data.setdefault("version", 1)
    data.setdefault("pending_report", None)
    return data


def write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def add_state_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)


def centralized_runtime() -> tuple[Path, Path, Path] | None:
    """Find the current/script repository's initialized centralized controller."""
    checked: set[Path] = set()
    for start in (Path.cwd(), DEFAULT_STATE.parent):
        process = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.returncode != 0:
            continue
        worktree = Path(process.stdout.strip()).resolve()
        if worktree in checked:
            continue
        checked.add(worktree)
        common = subprocess.run(
            [
                "git", "-C", str(worktree), "rev-parse", "--path-format=absolute",
                "--git-common-dir",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if common.returncode != 0:
            continue
        common_dir = Path(common.stdout.strip()).resolve()
        database = common_dir / "xh-202628" / "submission-control.sqlite3"
        if database.exists():
            primary = common_dir.parent.resolve()
            mirror = primary / "state" / "submission-state.json"
            return primary, database, mirror
    return None


def centralized_check(repo: Path, database: Path, state_path: Path) -> int:
    process = subprocess.run(
        [
            sys.executable, str(CONTROLLER), "--repo", str(repo), "--db", str(database),
            "--mirror", str(state_path), "check",
        ],
        check=False,
    )
    return process.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track XPU-OJ submissions and block unreported follow-ups."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Check whether another submit is allowed")
    add_state_argument(check)

    show = commands.add_parser("show", help="Print the complete state")
    add_state_argument(show)

    record = commands.add_parser(
        "record", help="Record one terminal OJ result and require a user report"
    )
    add_state_argument(record)
    record.add_argument("--id", required=True, help="XPU-OJ submission ID")
    record.add_argument("--operator", required=True)
    record.add_argument("--language", required=True)
    record.add_argument("--commit", required=True)
    record.add_argument("--status", required=True, help="Terminal OJ status")
    record.add_argument("--score", default="n/a")
    record.add_argument("--rank", default="n/a")
    record.add_argument("--url", default="")
    record.add_argument("--evidence", required=True)
    record.add_argument("--experiment", required=True)
    record.add_argument("--previous-score", default="n/a")
    record.add_argument("--notes", default="")
    record.add_argument("--timestamp", default=None)

    report = commands.add_parser(
        "report", help="Mark a submission reported after the message was sent"
    )
    add_state_argument(report)
    report.add_argument("--id", required=True)
    report.add_argument("--summary", required=True)
    report.add_argument("--timestamp", default=None)

    return parser


def find_submission(data: dict[str, Any], submission_id: str) -> dict[str, Any] | None:
    for item in data["submissions"]:
        if isinstance(item, dict) and item.get("id") == submission_id:
            return item
    return None


def command_check(data: dict[str, Any]) -> int:
    pending = data.get("pending_report")
    if pending:
        print(
            f"BLOCKED: submission {pending} has not been reported to the user.",
            file=os.sys.stderr,
        )
        return 2
    print("PASS: no unreported XPU-OJ submission; preflight may continue.")
    return 0


def command_record(args: argparse.Namespace, path: Path, data: dict[str, Any]) -> int:
    pending = data.get("pending_report")
    if pending:
        raise SystemExit(
            f"Cannot record another submission while {pending} awaits a user report."
        )
    if find_submission(data, args.id):
        raise SystemExit(f"Submission ID already exists: {args.id}")

    item = {
        "id": args.id,
        "submitted_at": args.timestamp or utc_now(),
        "operator": args.operator,
        "language": args.language,
        "commit": args.commit,
        "experiment": args.experiment,
        "status": args.status,
        "score": args.score,
        "previous_score": args.previous_score,
        "rank": args.rank,
        "url": args.url,
        "evidence": args.evidence,
        "notes": args.notes,
        "reported_to_user": False,
        "reported_at": None,
        "report_summary": None,
    }
    data["submissions"].append(item)
    data["pending_report"] = args.id
    write_state(path, data)
    print(f"Recorded submission {args.id}; user report is now mandatory.")
    return 0


def command_report(args: argparse.Namespace, path: Path, data: dict[str, Any]) -> int:
    item = find_submission(data, args.id)
    if item is None:
        raise SystemExit(f"Unknown submission ID: {args.id}")
    if item.get("reported_to_user"):
        print(f"Submission {args.id} was already marked reported.")
        return 0
    if data.get("pending_report") != args.id:
        raise SystemExit(
            f"Submission {args.id} is not the active report gate: "
            f"{data.get('pending_report')!r}"
        )

    item["reported_to_user"] = True
    item["reported_at"] = args.timestamp or utc_now()
    item["report_summary"] = args.summary
    data["pending_report"] = None
    write_state(path, data)
    print(f"Marked submission {args.id} reported; gate cleared.")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    path = args.state.resolve()
    centralized = centralized_runtime()
    if centralized:
        repo, database, mirror = centralized
        if args.command == "check":
            return centralized_check(repo, database, mirror)
        if args.command in ("record", "report"):
            raise SystemExit(
                "The centralized submission controller is active. Legacy record/report "
                "is disabled; use submission_controller.py from the primary main worktree."
            )
    data = load_state(path)

    if args.command == "check":
        return command_check(data)
    if args.command == "show":
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "record":
        return command_record(args, path, data)
    if args.command == "report":
        return command_report(args, path, data)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
