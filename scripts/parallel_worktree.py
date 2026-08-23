#!/usr/bin/env python3
"""Create and inspect isolated candidate worktrees for parallel optimization lanes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Sequence


SCRIPT_REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_RE = re.compile(r"^exp-[0-9]{8}-[0-9]{3}$")
BATCH_RE = re.compile(r"^batch-[0-9]{8}-[0-9]{2}$")
LANE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_WORKFLOW_FILES = (
    "AGENTS.md",
    "runbooks/c500-execution.md",
    "runbooks/parallel-orchestration.md",
    "scripts/invoke-c500-run.ps1",
    "scripts/c500-runner.sh",
    "scripts/c500-stage.sh",
    "scripts/run-c500-fused-moe-paired.sh",
    "scripts/summarize-c500-abba.py",
    "scripts/parallel_worktree.py",
    "skills/xpuoj-operator-optimizer/SKILL.md",
    "skills/xpuoj-operator-optimizer/scripts/submission_controller.py",
    "state/c500-execution.json",
    "templates/remote-job.sh",
    "templates/subagent-handoff.md",
    "templates/subagent-task.md",
)


def git(
    repo: Path,
    args: Sequence[str],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(repo), *args]
    if input_text is None:
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    else:
        binary_process = subprocess.run(
            command,
            input=input_text.encode("ascii"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        process = subprocess.CompletedProcess(
            binary_process.args,
            binary_process.returncode,
            binary_process.stdout.decode("utf-8", errors="replace"),
            binary_process.stderr.decode("utf-8", errors="replace"),
        )
    if check and process.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process


def repo_root(value: Path) -> Path:
    root = value.resolve()
    top = Path(git(root, ["rev-parse", "--show-toplevel"]).stdout.strip()).resolve()
    if top != root:
        raise SystemExit(f"--repo must be a Git worktree root: {root}")
    return root


def require_primary_main(repo: Path) -> None:
    branch = git(repo, ["branch", "--show-current"]).stdout.strip()
    if branch != "main":
        raise SystemExit(f"Worktree creation requires the primary main branch, found {branch!r}.")
    git_dir = Path(
        git(repo, ["rev-parse", "--path-format=absolute", "--git-dir"]).stdout.strip()
    ).resolve()
    common_dir = Path(
        git(repo, ["rev-parse", "--path-format=absolute", "--git-common-dir"]).stdout.strip()
    ).resolve()
    if git_dir != common_dir:
        raise SystemExit("Worktree creation must run from the primary worktree, not a linked worktree.")
    status = git(repo, ["status", "--porcelain=v1", "--untracked-files=all"]).stdout.strip()
    if status:
        raise SystemExit(f"Primary main worktree must be clean:\n{status}")


def safe_worktree_root(repo: Path) -> Path:
    expected_parent = repo.parent.resolve()
    root = (expected_parent / f"{repo.name}-worktrees").resolve()
    return root


def safe_source_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise SystemExit(f"Unsafe repository-relative source path: {value}")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", normalized):
        raise SystemExit(f"Unsupported characters in source path: {value}")
    return normalized


def require_workflow_files(repo: Path, commit: str) -> None:
    missing = [
        path
        for path in REQUIRED_WORKFLOW_FILES
        if git(repo, ["cat-file", "-e", f"{commit}:{path}"], check=False).returncode
        != 0
    ]
    if missing:
        raise SystemExit(
            "Worktree base is missing required parallel-workflow files: "
            + ", ".join(missing)
        )


def require_current_workflow_base(repo: Path, commit: str) -> None:
    head = git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    if commit != head:
        raise SystemExit(
            "--worktree-base-commit must equal the current clean primary main HEAD so "
            "every lane contains the latest orchestration tools."
        )
    require_workflow_files(repo, commit)


def require_matching_performance_source(
    repo: Path, worktree_base: str, performance_baseline: str, source: str
) -> str:
    git(repo, ["cat-file", "-e", f"{performance_baseline}^{{commit}}"])
    worktree_blob = git(
        repo, ["rev-parse", f"{worktree_base}:{source}"]
    ).stdout.strip()
    performance_blob = git(
        repo, ["rev-parse", f"{performance_baseline}:{source}"]
    ).stdout.strip()
    if worktree_blob != performance_blob:
        raise SystemExit(
            "Current main does not contain the exact performance-baseline source. "
            "Restore and commit that source before allocating the batch."
        )
    return performance_blob


def command_create(args: argparse.Namespace) -> int:
    repo = repo_root(args.repo)
    require_primary_main(repo)
    if not BATCH_RE.fullmatch(args.batch):
        raise SystemExit(f"Invalid batch ID: {args.batch}")
    if not LANE_RE.fullmatch(args.lane):
        raise SystemExit(f"Invalid lane name: {args.lane}")
    if not EXPERIMENT_RE.fullmatch(args.experiment):
        raise SystemExit(f"Invalid experiment ID: {args.experiment}")
    if not COMMIT_RE.fullmatch(args.worktree_base_commit):
        raise SystemExit(
            "--worktree-base-commit is required as a full 40-character commit."
        )
    if not COMMIT_RE.fullmatch(args.performance_baseline_commit):
        raise SystemExit(
            "--performance-baseline-commit is required as a full 40-character commit."
        )
    source = safe_source_path(args.source)
    git(repo, ["cat-file", "-e", f"{args.worktree_base_commit}^{{commit}}"])
    require_current_workflow_base(repo, args.worktree_base_commit)
    performance_blob = require_matching_performance_source(
        repo,
        args.worktree_base_commit,
        args.performance_baseline_commit,
        source,
    )
    existing_job = git(
        repo,
        ["cat-file", "-e", f"HEAD:remote-jobs/{args.experiment}.sh"],
        check=False,
    )
    if existing_job.returncode == 0:
        raise SystemExit(f"Experiment ID already exists on main: {args.experiment}")
    reservation = f"refs/xh-202628/experiments/{args.experiment}"
    baseline_reservation = f"refs/xh-202628/baselines/{args.experiment}"
    transaction = (
        "start\n"
        f"create {reservation} {args.worktree_base_commit}\n"
        f"create {baseline_reservation} {args.performance_baseline_commit}\n"
        "prepare\n"
        "commit\n"
    )
    reserved = git(
        repo,
        ["update-ref", "--stdin", "-m", f"reserve {args.batch}/{args.lane}"],
        check=False,
        input_text=transaction,
    )
    if reserved.returncode != 0:
        existing = git(repo, ["rev-parse", "--verify", reservation], check=False)
        existing_baseline = git(
            repo, ["rev-parse", "--verify", baseline_reservation], check=False
        )
        owner = existing.stdout.strip() if existing.returncode == 0 else "unreserved"
        baseline_owner = (
            existing_baseline.stdout.strip()
            if existing_baseline.returncode == 0
            else "unreserved"
        )
        raise SystemExit(
            f"Experiment ID is already reserved: {args.experiment} "
            f"(workflow={owner}, baseline={baseline_owner}); "
            f"transaction error: {reserved.stderr.strip()}"
        )

    root = safe_worktree_root(repo)
    path = (root / args.batch / f"{args.lane}-{args.experiment}").resolve()
    if path.parent.parent != root or path.exists():
        raise SystemExit(f"Candidate worktree path already exists or escaped its root: {path}")
    branch = f"candidate/{args.batch}/{args.lane}-{args.experiment}"
    if git(repo, ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False).returncode == 0:
        raise SystemExit(f"Candidate branch already exists: {branch}")

    root.mkdir(parents=True, exist_ok=True)
    try:
        git(
            repo,
            [
                "worktree", "add", "-b", branch, str(path),
                args.worktree_base_commit,
            ],
        )
    except SystemExit as exc:
        raise SystemExit(
            f"{exc}\nExperiment {args.experiment} remains reserved; allocate a new ID."
        ) from exc
    result = {
        "batch": args.batch,
        "lane": args.lane,
        "experiment": args.experiment,
        "worktree_base_commit": args.worktree_base_commit,
        "performance_baseline_commit": args.performance_baseline_commit,
        "performance_baseline_blob": performance_blob,
        "source": source,
        "branch": branch,
        "reservation_ref": reservation,
        "baseline_reservation_ref": baseline_reservation,
        "worktree": str(path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_audit_create(args: argparse.Namespace) -> int:
    repo = repo_root(args.repo)
    require_primary_main(repo)
    if not BATCH_RE.fullmatch(args.batch):
        raise SystemExit(f"Invalid batch ID: {args.batch}")
    if not LANE_RE.fullmatch(args.lane):
        raise SystemExit(f"Invalid lane name: {args.lane}")
    if not EXPERIMENT_RE.fullmatch(args.experiment):
        raise SystemExit(f"Invalid experiment ID: {args.experiment}")
    if not COMMIT_RE.fullmatch(args.candidate_commit):
        raise SystemExit("--candidate-commit must be a full 40-character commit.")
    if not COMMIT_RE.fullmatch(args.performance_baseline_commit):
        raise SystemExit(
            "--performance-baseline-commit must be a full 40-character commit."
        )
    source = safe_source_path(args.source)
    git(repo, ["cat-file", "-e", f"{args.candidate_commit}^{{commit}}"])
    reservation = git(
        repo,
        ["rev-parse", "--verify", f"refs/xh-202628/experiments/{args.experiment}"],
    ).stdout.strip()
    baseline_reservation = git(
        repo,
        ["rev-parse", "--verify", f"refs/xh-202628/baselines/{args.experiment}"],
    ).stdout.strip()
    if baseline_reservation != args.performance_baseline_commit:
        raise SystemExit(
            "Audit performance baseline does not match the reserved baseline commit."
        )
    require_workflow_files(repo, reservation)
    performance_blob = require_matching_performance_source(
        repo, reservation, args.performance_baseline_commit, source
    )
    expected_branch = f"candidate/{args.batch}/{args.lane}-{args.experiment}"
    branch_tip = git(
        repo, ["rev-parse", "--verify", f"refs/heads/{expected_branch}"]
    ).stdout.strip()
    if branch_tip != args.candidate_commit:
        raise SystemExit(
            "Audit commit must be the exact tip of the allocated candidate branch."
        )
    if git(
        repo,
        ["merge-base", "--is-ancestor", reservation, args.candidate_commit],
        check=False,
    ).returncode != 0:
        raise SystemExit("Audit commit must descend from the reserved experiment base.")

    root = safe_worktree_root(repo)
    path = (root / args.batch / f"audit-{args.lane}-{args.experiment}").resolve()
    if path.parent.parent != root or path.exists():
        raise SystemExit(f"Audit worktree path already exists or escaped its root: {path}")
    root.mkdir(parents=True, exist_ok=True)
    git(repo, ["worktree", "add", "--detach", str(path), args.candidate_commit])
    print(json.dumps({
        "batch": args.batch,
        "lane": args.lane,
        "experiment": args.experiment,
        "candidate_commit": args.candidate_commit,
        "candidate_branch": expected_branch,
        "worktree_base_commit": reservation,
        "performance_baseline_commit": args.performance_baseline_commit,
        "baseline_reservation_ref": f"refs/xh-202628/baselines/{args.experiment}",
        "performance_baseline_blob": performance_blob,
        "source": source,
        "detached": True,
        "worktree": str(path),
    }, indent=2, sort_keys=True))
    return 0


def command_list(args: argparse.Namespace) -> int:
    repo = repo_root(args.repo)
    output = git(repo, ["worktree", "list", "--porcelain"]).stdout
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in output.splitlines() + [""]:
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=SCRIPT_REPO)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--batch", required=True)
    create.add_argument("--lane", required=True)
    create.add_argument("--experiment", required=True)
    create.add_argument("--worktree-base-commit", required=True)
    create.add_argument("--performance-baseline-commit", required=True)
    create.add_argument("--source", required=True)
    audit = commands.add_parser("audit-create")
    audit.add_argument("--batch", required=True)
    audit.add_argument("--lane", required=True)
    audit.add_argument("--experiment", required=True)
    audit.add_argument("--candidate-commit", required=True)
    audit.add_argument("--performance-baseline-commit", required=True)
    audit.add_argument("--source", required=True)
    commands.add_parser("list")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "create":
        return command_create(args)
    if args.command == "audit-create":
        return command_audit_create(args)
    if args.command == "list":
        return command_list(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
