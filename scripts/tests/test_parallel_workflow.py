#!/usr/bin/env python3
"""Black-box tests for parallel worktrees and centralized submission control."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = (
    ROOT
    / "skills"
    / "xpuoj-operator-optimizer"
    / "scripts"
    / "submission_controller.py"
)
LEGACY_LEDGER = CONTROLLER.with_name("submission_ledger.py")
WORKTREE = ROOT / "scripts" / "parallel_worktree.py"


def run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if check and process.returncode != 0:
        raise AssertionError(
            f"Command failed ({process.returncode}): {' '.join(command)}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    return process


class RepositoryFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="xh-parallel-test-")
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        run(["git", "init", "-b", "main"], cwd=self.repo)
        run(["git", "config", "user.name", "Workflow Test"], cwd=self.repo)
        run(["git", "config", "user.email", "workflow@example.invalid"], cwd=self.repo)
        (self.repo / "operators").mkdir()
        (self.repo / "remote-jobs").mkdir()
        (self.repo / "handoffs").mkdir()
        (self.repo / "state").mkdir()
        (self.repo / "operators" / "submission.cu").write_text(
            "extern \"C\" void run_kernel() {}\n", encoding="utf-8", newline="\n"
        )
        run(["git", "add", "operators/submission.cu"], cwd=self.repo)
        run(["git", "commit", "-m", "performance baseline"], cwd=self.repo)
        self.performance_commit = run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo
        ).stdout.strip()
        (self.repo / "operators" / "submission.cu").write_text(
            "extern \"C\" void run_kernel() { /* slower baseline */ }\n",
            encoding="utf-8",
            newline="\n",
        )
        run(["git", "add", "operators/submission.cu"], cwd=self.repo)
        run(["git", "commit", "-m", "different performance source"], cwd=self.repo)
        self.mismatched_performance_commit = run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo
        ).stdout.strip()
        run(
            [
                "git", "restore", f"--source={self.performance_commit}", "--",
                "operators/submission.cu",
            ],
            cwd=self.repo,
        )
        required_workflow_files = (
            "AGENTS.md",
            "runbooks/parallel-orchestration.md",
            "scripts/invoke-remote-gpu-run.ps1",
            "scripts/parallel_worktree.py",
            "skills/xpuoj-operator-optimizer/SKILL.md",
            "skills/xpuoj-operator-optimizer/scripts/submission_controller.py",
            "templates/subagent-handoff.md",
            "templates/subagent-task.md",
        )
        for relative in required_workflow_files:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"fixture workflow file: {relative}\n", encoding="utf-8", newline="\n"
            )
        for number in range(100, 116):
            experiment = f"exp-20260822-{number:03d}"
            (self.repo / "remote-jobs" / f"{experiment}.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8", newline="\n"
            )
            (self.repo / "handoffs" / f"{experiment}.md").write_text(
                f"# {experiment}\n\nVerified candidate handoff.\n",
                encoding="utf-8",
                newline="\n",
            )
        historical = {
            "version": 1,
            "pending_report": None,
            "submissions": [
                {
                    "id": "100001",
                    "submitted_at": "2026-08-20T00:00:00+00:00",
                    "operator": "Fused MoE i8 tn",
                    "language": "CUDA Maca",
                    "commit": "0" * 40,
                    "experiment": "exp-20260820-001",
                    "status": "Accepted",
                    "score": "82.25",
                    "previous_score": "n/a",
                    "rank": "20",
                    "url": "",
                    "evidence": "artifacts/raw/xpuoj/100001",
                    "notes": "historical fixture",
                    "reported_to_user": True,
                    "reported_at": "2026-08-20T00:01:00+00:00",
                    "report_summary": "reported",
                }
            ],
        }
        (self.repo / "state" / "submission-state.json").write_text(
            json.dumps(historical, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        run(["git", "add", "."], cwd=self.repo)
        run(["git", "commit", "-m", "fixture"], cwd=self.repo)
        self.commit = run(["git", "rev-parse", "HEAD"], cwd=self.repo).stdout.strip()
        self.db = self.repo / ".git" / "xh-202628" / "submission-control.sqlite3"
        self.mirror = self.repo / "state" / "submission-state.json"

    def close(self) -> None:
        self.temporary.cleanup()

    def controller(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return run(
            [
                sys.executable,
                str(CONTROLLER),
                "--repo",
                str(self.repo),
                "--db",
                str(self.db),
                "--mirror",
                str(self.mirror),
                *arguments,
            ],
            cwd=self.repo,
            check=check,
        )

    def acquire(self, owner: str = "main-test") -> str:
        value = json.loads(
            self.controller("controller-acquire", "--owner", owner).stdout
        )
        return value["controller_token"]

    def recovery_token(self) -> str:
        return self.db.with_name("controller-recovery.key").read_text(
            encoding="utf-8"
        ).strip()

    def enqueue_arguments(
        self,
        candidate: str,
        experiment: str,
        producer: str = "subagent-a",
    ) -> list[str]:
        batch = "batch-20260822-01"
        lane = f"lane-{experiment[-3:]}"
        reservation = f"refs/xh-202628/experiments/{experiment}"
        branch = f"candidate/{batch}/{lane}-{experiment}"
        run(["git", "update-ref", reservation, self.commit, "0" * 40], cwd=self.repo)
        run(["git", "branch", branch, self.commit], cwd=self.repo)
        return [
            "candidate-enqueue",
            "--candidate",
            candidate,
            "--producer",
            producer,
            "--experiment",
            experiment,
            "--batch",
            batch,
            "--lane",
            lane,
            "--worktree-base-commit",
            self.commit,
            "--performance-baseline-commit",
            self.performance_commit,
            "--commit",
            self.commit,
            "--source",
            "operators/submission.cu",
            "--language",
            "CUDA Maca",
            "--hypothesis",
            "Independent memory-pipeline candidate",
            "--evidence",
            f"handoffs/{experiment}.md",
        ]

    def enqueue_and_promote(self, token: str, candidate: str = "cand-exp-100-a") -> None:
        self.controller(
            *self.enqueue_arguments(candidate, "exp-20260822-100"),
        )
        self.controller(
            "candidate-promote",
            "--candidate",
            candidate,
            "--commit",
            self.commit,
            "--controller-token",
            token,
        )


class SubmissionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RepositoryFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_init_imports_legacy_pending_report_as_deferred(self) -> None:
        mirror = json.loads(self.fixture.mirror.read_text(encoding="utf-8"))
        mirror["pending_report"] = "100001"
        mirror["submissions"][0]["reported_to_user"] = False
        self.fixture.mirror.write_text(
            json.dumps(mirror, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        initialized = json.loads(self.fixture.controller("init").stdout)
        self.assertEqual(initialized["imported_submissions"], 1)
        deferred = json.loads(self.fixture.controller("unreported-list").stdout)
        self.assertEqual([item["id"] for item in deferred], ["100001"])
        self.assertEqual(self.fixture.controller("check").returncode, 0)

    def test_parallel_claim_and_full_submission_lifecycle(self) -> None:
        initialized = json.loads(self.fixture.controller("init").stdout)
        self.assertEqual(initialized["imported_submissions"], 1)
        token = self.fixture.acquire()
        self.fixture.enqueue_and_promote(token)
        self.fixture.controller(
            *self.fixture.enqueue_arguments(
                "cand-exp-101-b", "exp-20260822-101", "subagent-b"
            )
        )
        self.fixture.controller(
            "candidate-promote",
            "--candidate",
            "cand-exp-101-b",
            "--commit",
            self.fixture.commit,
            "--controller-token",
            token,
        )
        self.assertEqual(
            run(["git", "status", "--porcelain"], cwd=self.fixture.repo).stdout,
            "",
        )

        base = [
            sys.executable,
            str(CONTROLLER),
            "--repo",
            str(self.fixture.repo),
            "--db",
            str(self.fixture.db),
            "--mirror",
            str(self.fixture.mirror),
            "claim",
            "--candidate",
            "cand-exp-100-a",
            "--controller-token",
            token,
        ]
        first = subprocess.Popen(
            [*base, "--request-id", "req-concurrent-a"],
            cwd=self.fixture.repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        second = subprocess.Popen(
            [*base, "--request-id", "req-concurrent-b"],
            cwd=self.fixture.repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        first_out, first_err = first.communicate(timeout=20)
        second_out, second_err = second.communicate(timeout=20)
        outcomes = [(first.returncode, first_out, first_err), (second.returncode, second_out, second_err)]
        successes = [item for item in outcomes if item[0] == 0]
        failures = [item for item in outcomes if item[0] != 0]
        self.assertEqual(len(successes), 1, outcomes)
        self.assertEqual(len(failures), 1, outcomes)
        claim_result = json.loads(successes[0][1])
        claim = claim_result["claim_id"]
        retry = json.loads(
            self.fixture.controller(
                "claim",
                "--candidate",
                "cand-exp-100-a",
                "--request-id",
                claim_result["request_id"],
                "--controller-token",
                token,
            ).stdout
        )
        self.assertEqual(retry["claim_id"], claim)

        self.fixture.controller(
            "arm",
            "--claim",
            claim,
            "--problem-url",
            "https://xpuoj.com/contest/12/problem/1",
            "--language",
            "CUDA Maca",
            "--controller-token",
            token,
        )
        blocked_abandon = self.fixture.controller(
            "abandon",
            "--claim",
            claim,
            "--reason",
            "test",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(blocked_abandon.returncode, 0)

        self.fixture.controller(
            "bind",
            "--claim",
            claim,
            "--submission-id",
            "200001",
            "--controller-token",
            token,
        )
        blocked = self.fixture.controller("check", check=False)
        self.assertEqual(blocked.returncode, 2)
        impossible_abandon = self.fixture.controller(
            "abandon",
            "--claim",
            claim,
            "--reason",
            "cannot discard submitted work",
            "--confirmed-no-submit",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(impossible_abandon.returncode, 0)

        self.fixture.controller(
            "finalize",
            "--claim",
            claim,
            "--operator",
            "Fused MoE i8 tn",
            "--status",
            "Accepted",
            "--score",
            "83.00",
            "--previous-score",
            "82.25",
            "--rank",
            "19",
            "--url",
            "https://xpuoj.com/record/200001",
            "--evidence",
            "artifacts/raw/xpuoj/200001",
            "--controller-token",
            token,
        )
        self.fixture.controller(
            "finalize",
            "--claim",
            claim,
            "--operator",
            "Fused MoE i8 tn",
            "--status",
            "Accepted",
            "--score",
            "83.00",
            "--previous-score",
            "82.25",
            "--rank",
            "19",
            "--url",
            "https://xpuoj.com/record/200001",
            "--evidence",
            "artifacts/raw/xpuoj/200001",
            "--controller-token",
            token,
        )
        conflicting_final = self.fixture.controller(
            "finalize",
            "--claim",
            claim,
            "--operator",
            "Fused MoE i8 tn",
            "--status",
            "Accepted",
            "--score",
            "84.00",
            "--previous-score",
            "82.25",
            "--rank",
            "19",
            "--url",
            "https://xpuoj.com/record/200001",
            "--evidence",
            "artifacts/raw/xpuoj/200001",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(conflicting_final.returncode, 0)
        self.assertEqual(self.fixture.controller("check").returncode, 0)
        deferred = json.loads(self.fixture.controller("unreported-list").stdout)
        self.assertEqual([item["id"] for item in deferred], ["200001"])
        run(
            ["git", "add", "state/submission-state.json"], cwd=self.fixture.repo
        )
        run(
            ["git", "commit", "-m", "persist finalized submission"],
            cwd=self.fixture.repo,
        )
        next_claim = json.loads(
            self.fixture.controller(
                "claim",
                "--candidate",
                "cand-exp-101-b",
                "--request-id",
                "req-continuous-b",
                "--controller-token",
                token,
            ).stdout
        )["claim_id"]
        self.fixture.controller(
            "report",
            "--submission-id",
            "200001",
            "--summary",
            "Accepted at 83.00",
            "--message-ref",
            "user-update-20260822T120000Z",
            "--controller-token",
            token,
        )
        active_during_deferred_report = json.loads(
            self.fixture.controller("controller-status").stdout
        )["active_claim"]
        self.assertEqual(active_during_deferred_report["claim_id"], next_claim)
        self.assertEqual(active_during_deferred_report["phase"], "claimed")
        self.fixture.controller(
            "abandon",
            "--claim",
            next_claim,
            "--reason",
            "continuous-mode claim proof complete",
            "--controller-token",
            token,
        )
        self.fixture.controller(
            "controller-release", "--controller-token", token
        )
        token = self.fixture.acquire("main-after-interrupt")
        self.fixture.controller(
            "report",
            "--submission-id",
            "200001",
            "--summary",
            "Accepted at 83.00",
            "--message-ref",
            "user-update-20260822T120000Z",
            "--controller-token",
            token,
        )
        conflicting_report = self.fixture.controller(
            "report",
            "--submission-id",
            "200001",
            "--summary",
            "different summary",
            "--message-ref",
            "user-update-20260822T120000Z",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(conflicting_report.returncode, 0)
        self.assertEqual(self.fixture.controller("check").returncode, 0)
        self.assertEqual(
            json.loads(self.fixture.controller("unreported-list").stdout), []
        )
        mirror = json.loads(self.fixture.mirror.read_text(encoding="utf-8"))
        self.assertEqual(mirror["version"], 2)
        self.assertIsNone(mirror["pending_report"])
        self.assertEqual([item["id"] for item in mirror["submissions"]], ["100001", "200001"])
        self.assertTrue(mirror["submissions"][-1]["reported_to_user"])

    def test_takeover_fences_old_controller_and_preserves_armed_claim(self) -> None:
        token = self.fixture.acquire("main-old")
        self.fixture.enqueue_and_promote(token)
        claim = json.loads(
            self.fixture.controller(
                "claim",
                "--candidate",
                "cand-exp-100-a",
                "--request-id",
                "req-takeover-a",
                "--controller-token",
                token,
            ).stdout
        )["claim_id"]
        self.fixture.controller(
            "arm",
            "--claim",
            claim,
            "--problem-url",
            "https://xpuoj.com/contest/12/problem/1",
            "--language",
            "CUDA Maca",
            "--controller-token",
            token,
        )
        missing_recovery = self.fixture.controller(
            "controller-takeover",
            "--owner",
            "main-new",
            "--reason",
            "resume after coordinator crash",
            "--expected-owner",
            "main-old",
            "--expected-epoch",
            "1",
            check=False,
        )
        self.assertNotEqual(missing_recovery.returncode, 0)
        changed_precondition = self.fixture.controller(
            "controller-takeover",
            "--owner",
            "main-new",
            "--reason",
            "resume after coordinator crash",
            "--expected-owner",
            "some-other-owner",
            "--expected-epoch",
            "1",
            "--recovery-token",
            self.fixture.recovery_token(),
            check=False,
        )
        self.assertNotEqual(changed_precondition.returncode, 0)
        takeover = json.loads(
            self.fixture.controller(
                "controller-takeover",
                "--owner",
                "main-new",
                "--reason",
                "resume after coordinator crash",
                "--expected-owner",
                "main-old",
                "--expected-epoch",
                "1",
                "--recovery-token",
                self.fixture.recovery_token(),
            ).stdout
        )
        new_token = takeover["controller_token"]
        stale = self.fixture.controller(
            "bind",
            "--claim",
            claim,
            "--submission-id",
            "200002",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(stale.returncode, 0)
        status = json.loads(self.fixture.controller("controller-status").stdout)
        self.assertEqual(status["active_claim"]["phase"], "armed")
        evidence = self.fixture.repo / "artifacts" / "raw" / "xpuoj" / "reconcile-200002.txt"
        evidence.parent.mkdir(parents=True)
        evidence.write_text(
            "OJ submission history checked; no matching submission exists.\n",
            encoding="utf-8",
            newline="\n",
        )
        missing_evidence = self.fixture.controller(
            "abandon",
            "--claim",
            claim,
            "--reason",
            "OJ history confirms no submit occurred",
            "--confirmed-no-submit",
            "--controller-token",
            new_token,
            check=False,
        )
        self.assertNotEqual(missing_evidence.returncode, 0)
        wrong_source = self.fixture.controller(
            "abandon",
            "--claim",
            claim,
            "--reason",
            "OJ history confirms no submit occurred",
            "--confirmed-no-submit",
            "--oj-history-evidence",
            "artifacts/raw/xpuoj/reconcile-200002.txt",
            "--checked-at",
            "2099-08-22T12:00:00+00:00",
            "--expected-source-sha256",
            "0" * 64,
            "--controller-token",
            new_token,
            check=False,
        )
        self.assertNotEqual(wrong_source.returncode, 0)
        stale_check = self.fixture.controller(
            "abandon",
            "--claim",
            claim,
            "--reason",
            "OJ history confirms no submit occurred",
            "--confirmed-no-submit",
            "--oj-history-evidence",
            "artifacts/raw/xpuoj/reconcile-200002.txt",
            "--checked-at",
            "2000-01-01T00:00:00+00:00",
            "--expected-source-sha256",
            status["active_claim"]["source_sha256"],
            "--controller-token",
            new_token,
            check=False,
        )
        self.assertNotEqual(stale_check.returncode, 0)
        self.fixture.controller(
            "abandon",
            "--claim",
            claim,
            "--reason",
            "OJ history confirms no submit occurred",
            "--confirmed-no-submit",
            "--oj-history-evidence",
            "artifacts/raw/xpuoj/reconcile-200002.txt",
            "--checked-at",
            "2099-08-22T12:00:00+00:00",
            "--expected-source-sha256",
            status["active_claim"]["source_sha256"],
            "--controller-token",
            new_token,
        )
        connection = sqlite3.connect(self.fixture.db)
        try:
            payload = json.loads(
                connection.execute(
                    "SELECT payload_json FROM events WHERE event='abandon' ORDER BY seq DESC LIMIT 1"
                ).fetchone()[0]
            )
        finally:
            connection.close()
        self.assertEqual(
            payload["oj_history_evidence"],
            "artifacts/raw/xpuoj/reconcile-200002.txt",
        )
        self.assertEqual(payload["source_sha256"], status["active_claim"]["source_sha256"])
        self.assertEqual(len(payload["oj_history_evidence_sha256"]), 64)
        self.assertEqual(self.fixture.controller("check").returncode, 0)

    def test_takeover_works_after_finalize_dirties_tracked_mirror(self) -> None:
        token = self.fixture.acquire("main-before-crash")
        self.fixture.enqueue_and_promote(token)
        claim = json.loads(
            self.fixture.controller(
                "claim",
                "--candidate",
                "cand-exp-100-a",
                "--request-id",
                "req-dirty-takeover",
                "--controller-token",
                token,
            ).stdout
        )["claim_id"]
        self.fixture.controller(
            "arm",
            "--claim",
            claim,
            "--problem-url",
            "https://xpuoj.com/contest/12/problem/1",
            "--language",
            "CUDA Maca",
            "--controller-token",
            token,
        )
        self.fixture.controller(
            "bind",
            "--claim",
            claim,
            "--submission-id",
            "200010",
            "--controller-token",
            token,
        )
        self.fixture.controller(
            "finalize",
            "--claim",
            claim,
            "--operator",
            "Fused MoE i8 tn",
            "--status",
            "Accepted",
            "--score",
            "82.00",
            "--previous-score",
            "82.25",
            "--rank",
            "20",
            "--evidence",
            "artifacts/raw/xpuoj/200010",
            "--controller-token",
            token,
        )
        self.assertNotEqual(
            run(["git", "status", "--porcelain"], cwd=self.fixture.repo).stdout,
            "",
        )
        takeover = json.loads(
            self.fixture.controller(
                "controller-takeover",
                "--owner",
                "main-after-crash",
                "--reason",
                "resume with deferred user report",
                "--expected-owner",
                "main-before-crash",
                "--expected-epoch",
                "1",
                "--recovery-token",
                self.fixture.recovery_token(),
            ).stdout
        )
        self.fixture.controller(
            "report",
            "--submission-id",
            "200010",
            "--summary",
            "Accepted at 82.00",
            "--message-ref",
            "user-update-after-takeover",
            "--controller-token",
            takeover["controller_token"],
        )
        self.assertEqual(self.fixture.controller("check").returncode, 0)

    def test_claim_request_is_permanent_and_bound_to_one_candidate(self) -> None:
        token = self.fixture.acquire()
        self.fixture.enqueue_and_promote(token)
        self.fixture.controller(
            *self.fixture.enqueue_arguments(
                "cand-exp-101-b", "exp-20260822-101", "subagent-b"
            )
        )
        self.fixture.controller(
            "candidate-promote",
            "--candidate",
            "cand-exp-101-b",
            "--commit",
            self.fixture.commit,
            "--controller-token",
            token,
        )
        mismatched = self.fixture.enqueue_arguments(
            "cand-mismatched-baseline", "exp-20260822-103"
        )
        baseline_index = mismatched.index("--performance-baseline-commit") + 1
        mismatched[baseline_index] = self.fixture.mismatched_performance_commit
        mismatch_result = self.fixture.controller(*mismatched, check=False)
        self.assertNotEqual(mismatch_result.returncode, 0)
        claimed = json.loads(
            self.fixture.controller(
                "claim",
                "--candidate",
                "cand-exp-100-a",
                "--request-id",
                "req-permanent-a",
                "--controller-token",
                token,
            ).stdout
        )
        wrong_candidate = self.fixture.controller(
            "claim",
            "--candidate",
            "cand-exp-101-b",
            "--request-id",
            "req-permanent-a",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(wrong_candidate.returncode, 0)
        self.fixture.controller(
            "abandon",
            "--claim",
            claimed["claim_id"],
            "--reason",
            "candidate deferred before arming",
            "--controller-token",
            token,
        )
        completed_retry = self.fixture.controller(
            "claim",
            "--candidate",
            "cand-exp-100-a",
            "--request-id",
            "req-permanent-a",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(completed_retry.returncode, 0)
        replacement = self.fixture.controller(
            "claim",
            "--candidate",
            "cand-exp-100-a",
            "--request-id",
            "req-permanent-b",
            "--controller-token",
            token,
        )
        self.assertEqual(json.loads(replacement.stdout)["candidate_id"], "cand-exp-100-a")

    def test_blank_terminal_status_and_missing_submission_row_are_blocked(self) -> None:
        token = self.fixture.acquire()
        self.fixture.enqueue_and_promote(token)
        claim = json.loads(
            self.fixture.controller(
                "claim",
                "--candidate",
                "cand-exp-100-a",
                "--request-id",
                "req-terminal-a",
                "--controller-token",
                token,
            ).stdout
        )["claim_id"]
        self.fixture.controller(
            "arm",
            "--claim",
            claim,
            "--problem-url",
            "https://xpuoj.com/contest/12/problem/1",
            "--language",
            "CUDA Maca",
            "--controller-token",
            token,
        )
        self.fixture.controller(
            "bind",
            "--claim",
            claim,
            "--submission-id",
            "200020",
            "--controller-token",
            token,
        )
        blank = self.fixture.controller(
            "finalize",
            "--claim",
            claim,
            "--operator",
            "Fused MoE i8 tn",
            "--status",
            "   ",
            "--evidence",
            "artifacts/raw/xpuoj/200020",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(blank.returncode, 0)
        waiting = self.fixture.controller(
            "finalize",
            "--claim",
            claim,
            "--operator",
            "Fused MoE i8 tn",
            "--status",
            "Waiting",
            "--evidence",
            "artifacts/raw/xpuoj/200020",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(waiting.returncode, 0)
        self.fixture.controller(
            "finalize",
            "--claim",
            claim,
            "--operator",
            "Fused MoE i8 tn",
            "--status",
            "Accepted",
            "--evidence",
            "artifacts/raw/xpuoj/200020",
            "--controller-token",
            token,
        )
        database = sqlite3.connect(self.fixture.db)
        try:
            database.execute("DELETE FROM submissions WHERE id='200020'")
            database.commit()
        finally:
            database.close()
        missing = self.fixture.controller(
            "report",
            "--submission-id",
            "200020",
            "--summary",
            "must not release",
            "--message-ref",
            "missing-row-test",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(missing.returncode, 0)
        status = json.loads(self.fixture.controller("controller-status").stdout)
        self.assertIsNone(status["active_claim"])

    def test_canonical_paths_and_mirror_drift_gate(self) -> None:
        self.fixture.controller("init")
        token = self.fixture.acquire()
        alternate_db = self.fixture.root / "split-brain.sqlite3"
        split_brain = run(
            [
                sys.executable,
                str(CONTROLLER),
                "--repo",
                str(self.fixture.repo),
                "--db",
                str(alternate_db),
                "candidate-list",
            ],
            cwd=self.fixture.repo,
            check=False,
        )
        self.assertNotEqual(split_brain.returncode, 0)
        self.assertFalse(alternate_db.exists())

        linked = self.fixture.root / "linked"
        run(["git", "worktree", "add", "--detach", str(linked), self.fixture.commit], cwd=self.fixture.repo)
        linked_list = run(
            [
                sys.executable,
                str(CONTROLLER),
                "--repo",
                str(linked),
                "candidate-list",
            ],
            cwd=linked,
        )
        self.assertEqual(json.loads(linked_list.stdout), [])
        linked_mirror = run(
            [
                sys.executable,
                str(CONTROLLER),
                "--repo",
                str(linked),
                "--mirror",
                str(linked / "state" / "submission-state.json"),
                "candidate-list",
            ],
            cwd=linked,
            check=False,
        )
        self.assertNotEqual(linked_mirror.returncode, 0)

        enqueue = self.fixture.enqueue_arguments(
            "cand-drift-a", "exp-20260822-102"
        )
        self.fixture.mirror.write_text("{}\n", encoding="utf-8", newline="\n")
        blocked = self.fixture.controller(
            *enqueue,
            check=False,
        )
        self.assertNotEqual(blocked.returncode, 0)
        doctor = self.fixture.controller("doctor", check=False)
        self.assertEqual(doctor.returncode, 1)
        doctor_value = json.loads(doctor.stdout)
        self.assertFalse(doctor_value["mirror_consistent"])
        self.assertEqual(self.fixture.controller("check", check=False).returncode, 2)
        unconfirmed_export = self.fixture.controller(
            "export-ledger", "--controller-token", token, check=False
        )
        self.assertNotEqual(unconfirmed_export.returncode, 0)
        self.fixture.controller(
            "export-ledger",
            "--reconcile-drift",
            "--expected-current-sha256",
            doctor_value["mirror_actual_sha256"],
            "--controller-token",
            token,
        )
        repaired = json.loads(self.fixture.controller("doctor").stdout)
        self.assertTrue(repaired["mirror_consistent"])

        database = sqlite3.connect(self.fixture.db)
        try:
            database.execute("UPDATE meta SET value='1' WHERE key='mirror_dirty'")
            database.commit()
        finally:
            database.close()
        dirty_blocked = self.fixture.controller(*enqueue, check=False)
        self.assertNotEqual(dirty_blocked.returncode, 0)
        self.assertTrue(
            json.loads(self.fixture.controller("doctor", check=False).stdout)["mirror_dirty"]
        )
        self.fixture.controller("export-ledger", "--controller-token", token)
        self.fixture.controller(*enqueue)

    def test_arm_revalidates_claimed_source_at_main_head(self) -> None:
        token = self.fixture.acquire()
        self.fixture.enqueue_and_promote(token)
        claim = json.loads(
            self.fixture.controller(
                "claim",
                "--candidate",
                "cand-exp-100-a",
                "--request-id",
                "req-source-toctou",
                "--controller-token",
                token,
            ).stdout
        )["claim_id"]
        wrong_problem = self.fixture.controller(
            "arm",
            "--claim",
            claim,
            "--problem-url",
            "https://xpuoj.com/contest/12/problem/2",
            "--language",
            "CUDA Maca",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(wrong_problem.returncode, 0)
        (self.fixture.repo / "operators" / "submission.cu").write_text(
            "extern \"C\" void run_kernel() { /* changed after claim */ }\n",
            encoding="utf-8",
            newline="\n",
        )
        run(["git", "add", "operators/submission.cu"], cwd=self.fixture.repo)
        run(["git", "commit", "-m", "change source after claim"], cwd=self.fixture.repo)
        armed = self.fixture.controller(
            "arm",
            "--claim",
            claim,
            "--problem-url",
            "https://xpuoj.com/contest/12/problem/1",
            "--language",
            "CUDA Maca",
            "--controller-token",
            token,
            check=False,
        )
        self.assertNotEqual(armed.returncode, 0)
        status = json.loads(self.fixture.controller("controller-status").stdout)
        self.assertEqual(status["active_claim"]["phase"], "claimed")

    def test_parallel_enqueue_has_no_lost_updates(self) -> None:
        self.fixture.controller("init")
        processes: list[subprocess.Popen[str]] = []
        for number in range(100, 108):
            experiment = f"exp-20260822-{number:03d}"
            command = [
                sys.executable,
                str(CONTROLLER),
                "--repo",
                str(self.fixture.repo),
                "--db",
                str(self.fixture.db),
                "--mirror",
                str(self.fixture.mirror),
                *self.fixture.enqueue_arguments(
                    f"cand-parallel-{number}", experiment, f"subagent-{number}"
                ),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=self.fixture.repo,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        outcomes = [process.communicate(timeout=30) + (process.returncode,) for process in processes]
        self.assertTrue(all(item[2] == 0 for item in outcomes), outcomes)
        candidates = json.loads(self.fixture.controller("candidate-list").stdout)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(
            run(["git", "status", "--porcelain"], cwd=self.fixture.repo).stdout,
            "",
        )

    def test_legacy_mutations_are_disabled_after_controller_initialization(self) -> None:
        self.fixture.controller("init")
        legacy_check = run(
            [sys.executable, str(LEGACY_LEDGER), "check", "--state", str(self.fixture.mirror)],
            cwd=self.fixture.repo,
        )
        self.assertIn("centralized XPU-OJ submit slot is clear", legacy_check.stdout)
        legacy_record = run(
            [
                sys.executable,
                str(LEGACY_LEDGER),
                "record",
                "--state",
                str(self.fixture.mirror),
                "--id",
                "999999",
                "--operator",
                "Fused MoE i8 tn",
                "--language",
                "CUDA Maca",
                "--commit",
                self.fixture.commit,
                "--status",
                "Accepted",
                "--evidence",
                "artifacts/raw/xpuoj/999999",
                "--experiment",
                "exp-20260822-115",
            ],
            cwd=self.fixture.repo,
            check=False,
        )
        self.assertNotEqual(legacy_record.returncode, 0)
        self.assertIn("centralized submission controller is active", legacy_record.stderr)
        external_state = self.fixture.root / "outside-repo-state.json"
        external_record = run(
            [
                sys.executable,
                str(LEGACY_LEDGER),
                "record",
                "--state",
                str(external_state),
                "--id",
                "999998",
                "--operator",
                "Fused MoE i8 tn",
                "--language",
                "CUDA Maca",
                "--commit",
                self.fixture.commit,
                "--status",
                "Accepted",
                "--evidence",
                "artifacts/raw/xpuoj/999998",
                "--experiment",
                "exp-20260822-114",
            ],
            cwd=self.fixture.repo,
            check=False,
        )
        self.assertNotEqual(external_record.returncode, 0)
        self.assertFalse(external_state.exists())


class ParallelWorktreeTests(unittest.TestCase):
    def test_create_requires_explicit_base_and_isolates_branch(self) -> None:
        fixture = RepositoryFixture()
        try:
            historical_tooling_base = run(
                [
                    sys.executable,
                    str(WORKTREE),
                    "--repo",
                    str(fixture.repo),
                    "create",
                    "--batch",
                    "batch-20260822-01",
                    "--lane",
                    "g2s-pipeline",
                    "--experiment",
                    "exp-20260822-999",
                    "--worktree-base-commit",
                    fixture.performance_commit,
                    "--performance-baseline-commit",
                    fixture.performance_commit,
                    "--source",
                    "operators/submission.cu",
                ],
                cwd=fixture.repo,
                check=False,
            )
            self.assertNotEqual(historical_tooling_base.returncode, 0)
            mismatched_performance_base = run(
                [
                    sys.executable,
                    str(WORKTREE),
                    "--repo",
                    str(fixture.repo),
                    "create",
                    "--batch",
                    "batch-20260822-01",
                    "--lane",
                    "g2s-pipeline",
                    "--experiment",
                    "exp-20260822-999",
                    "--worktree-base-commit",
                    fixture.commit,
                    "--performance-baseline-commit",
                    fixture.mismatched_performance_commit,
                    "--source",
                    "operators/submission.cu",
                ],
                cwd=fixture.repo,
                check=False,
            )
            self.assertNotEqual(mismatched_performance_base.returncode, 0)
            result = run(
                [
                    sys.executable,
                    str(WORKTREE),
                    "--repo",
                    str(fixture.repo),
                    "create",
                    "--batch",
                    "batch-20260822-01",
                    "--lane",
                    "g2s-pipeline",
                    "--experiment",
                    "exp-20260822-999",
                    "--worktree-base-commit",
                    fixture.commit,
                    "--performance-baseline-commit",
                    fixture.performance_commit,
                    "--source",
                    "operators/submission.cu",
                ],
                cwd=fixture.repo,
            )
            value = json.loads(result.stdout)
            worktree = Path(value["worktree"])
            self.assertTrue(worktree.is_dir())
            self.assertEqual(
                run(["git", "branch", "--show-current"], cwd=worktree).stdout.strip(),
                "candidate/batch-20260822-01/g2s-pipeline-exp-20260822-999",
            )
            self.assertEqual(
                run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip(),
                fixture.commit,
            )
            self.assertEqual(
                run(["git", "branch", "--show-current"], cwd=fixture.repo).stdout.strip(),
                "main",
            )
            audit_result = run(
                [
                    sys.executable,
                    str(WORKTREE),
                    "--repo",
                    str(fixture.repo),
                    "audit-create",
                    "--batch",
                    "batch-20260822-01",
                    "--lane",
                    "g2s-pipeline",
                    "--experiment",
                    "exp-20260822-999",
                    "--candidate-commit",
                    fixture.commit,
                    "--performance-baseline-commit",
                    fixture.performance_commit,
                    "--source",
                    "operators/submission.cu",
                ],
                cwd=fixture.repo,
            )
            audit_worktree = Path(json.loads(audit_result.stdout)["worktree"])
            self.assertEqual(
                run(["git", "branch", "--show-current"], cwd=audit_worktree).stdout.strip(),
                "",
            )
            self.assertEqual(
                run(["git", "rev-parse", "HEAD"], cwd=audit_worktree).stdout.strip(),
                fixture.commit,
            )
            wrong_lane_audit = run(
                [
                    sys.executable,
                    str(WORKTREE),
                    "--repo",
                    str(fixture.repo),
                    "audit-create",
                    "--batch",
                    "batch-20260822-01",
                    "--lane",
                    "unallocated-lane",
                    "--experiment",
                    "exp-20260822-999",
                    "--candidate-commit",
                    fixture.commit,
                    "--performance-baseline-commit",
                    fixture.performance_commit,
                    "--source",
                    "operators/submission.cu",
                ],
                cwd=fixture.repo,
                check=False,
            )
            self.assertNotEqual(wrong_lane_audit.returncode, 0)

            common = [
                sys.executable,
                str(WORKTREE),
                "--repo",
                str(fixture.repo),
                "create",
                "--batch",
                "batch-20260822-02",
                "--experiment",
                "exp-20260822-998",
                "--worktree-base-commit",
                fixture.commit,
                "--performance-baseline-commit",
                fixture.performance_commit,
                "--source",
                "operators/submission.cu",
            ]
            first = subprocess.Popen(
                [*common, "--lane", "lane-a"],
                cwd=fixture.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            second = subprocess.Popen(
                [*common, "--lane", "lane-b"],
                cwd=fixture.repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            outcomes = [
                first.communicate(timeout=30) + (first.returncode,),
                second.communicate(timeout=30) + (second.returncode,),
            ]
            self.assertEqual(sum(item[2] == 0 for item in outcomes), 1, outcomes)
            reservation = run(
                [
                    "git",
                    "rev-parse",
                    "refs/xh-202628/experiments/exp-20260822-998",
                ],
                cwd=fixture.repo,
            ).stdout.strip()
            self.assertEqual(reservation, fixture.commit)
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
