#!/usr/bin/env python3
"""Coordinate parallel candidates through one crash-safe XPU-OJ submit slot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sqlite3
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence


SCRIPT_REPO = Path(__file__).resolve().parents[3]
EXPERIMENT_RE = re.compile(r"^exp-[0-9]{8}-[0-9]{3}$")
BATCH_RE = re.compile(r"^batch-[0-9]{8}-[0-9]{2}$")
LANE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")
CANDIDATE_RE = re.compile(r"^cand-[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUBMISSION_RE = re.compile(r"^[0-9]+$")
REQUEST_RE = re.compile(r"^req-[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")
TARGET_PROBLEM_URL = "https://xpuoj.com/contest/12/problem/1"
NON_TERMINAL_WORDS = ("pending", "judging", "compiling", "queued", "running")
TERMINAL_STATUSES = {
    "accepted",
    "canceled",
    "cancelled",
    "compilation error",
    "compile error",
    "failed",
    "internal error",
    "judgement failed",
    "judgment failed",
    "memory limit exceeded",
    "output limit exceeded",
    "presentation error",
    "runtime error",
    "system error",
    "time limit exceeded",
    "timeout",
    "wrong answer",
}
SCHEMA_VERSION = "6"
MIRROR_DRIFT_SAFE_COMMANDS = {
    "candidate-list",
    "check",
    "controller-acquire",
    "controller-status",
    "controller-takeover",
    "doctor",
    "export-ledger",
    "unreported-list",
}
MIRROR_DIRTY_SAFE_COMMANDS = MIRROR_DRIFT_SAFE_COMMANDS | {"finalize", "report"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_git(repo: Path, args: Sequence[str], *, text: bool = True) -> str | bytes:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip() if text else process.stderr.decode(errors="replace").strip()
        raise SystemExit(f"git {' '.join(args)} failed: {stderr}")
    return process.stdout.strip() if text else process.stdout


def optional_ref(repo: Path, ref: str) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode == 1:
        return None
    if process.returncode != 0:
        raise SystemExit(
            f"git show-ref --verify --quiet {ref} failed: {process.stderr.strip()}"
        )
    return str(run_git(repo, ["rev-parse", "--verify", ref]))


def git_object_exists(repo: Path, revision: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", revision],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def resolve_repo(value: Path) -> Path:
    repo = value.resolve()
    top = Path(str(run_git(repo, ["rev-parse", "--show-toplevel"]))).resolve()
    if top != repo:
        raise SystemExit(f"--repo must be a Git worktree root: {repo}")
    return repo


def git_common_dir(repo: Path) -> Path:
    return Path(
        str(run_git(repo, ["rev-parse", "--path-format=absolute", "--git-common-dir"]))
    ).resolve()


def primary_repo(repo: Path) -> Path:
    primary = git_common_dir(repo).parent.resolve()
    top = Path(str(run_git(primary, ["rev-parse", "--show-toplevel"]))).resolve()
    if top != primary:
        raise SystemExit("Could not resolve the primary worktree from Git common directory.")
    return primary


def default_db(repo: Path) -> Path:
    return git_common_dir(repo) / "xh-202628" / "submission-control.sqlite3"


def default_mirror(repo: Path) -> Path:
    return primary_repo(repo) / "state" / "submission-state.json"


def resolve_control_paths(args: argparse.Namespace) -> None:
    canonical_db = default_db(args.repo).resolve()
    canonical_mirror = default_mirror(args.repo).resolve()
    if args.db is not None and args.db.resolve() != canonical_db:
        raise SystemExit(
            f"--db must be the shared Git-common controller database: {canonical_db}"
        )
    if args.mirror is not None and args.mirror.resolve() != canonical_mirror:
        raise SystemExit(
            f"--mirror must be the primary tracked submission mirror: {canonical_mirror}"
        )
    args.db = canonical_db
    args.mirror = canonical_mirror


def file_digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_source_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise SystemExit(f"Unsafe repository-relative source path: {value}")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", normalized):
        raise SystemExit(f"Unsupported characters in source path: {value}")
    return normalized


def aware_timestamp(value: str, label: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SystemExit(f"{label} must be an ISO-8601 timestamp with a timezone.") from exc
    if parsed.tzinfo is None:
        raise SystemExit(f"{label} must include a timezone offset.")
    return parsed.astimezone(timezone.utc)


def oj_history_evidence(repo: Path, value: str) -> tuple[str, str]:
    relative = safe_source_path(value)
    if not relative.startswith("artifacts/raw/xpuoj/"):
        raise SystemExit(
            "OJ history evidence must be a file below artifacts/raw/xpuoj/."
        )
    path = (repo / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError as exc:
        raise SystemExit("OJ history evidence escapes the primary repository.") from exc
    if not path.is_file():
        raise SystemExit(f"OJ history evidence file does not exist: {relative}")
    return relative, file_digest(path)


def git_blob(repo: Path, commit: str, source: str) -> bytes:
    if not COMMIT_RE.fullmatch(commit):
        raise SystemExit(f"Expected a full 40-character commit: {commit}")
    run_git(repo, ["cat-file", "-e", f"{commit}^{{commit}}"])
    return bytes(run_git(repo, ["show", f"{commit}:{source}"], text=False))


def blob_sha256(repo: Path, commit: str, source: str) -> str:
    return hashlib.sha256(git_blob(repo, commit, source)).hexdigest()


def require_main(repo: Path, *, clean: bool) -> None:
    branch = str(run_git(repo, ["branch", "--show-current"]))
    if branch != "main":
        raise SystemExit(f"Coordinator operation requires branch main, found {branch!r}.")
    git_dir = Path(
        str(run_git(repo, ["rev-parse", "--path-format=absolute", "--git-dir"]))
    ).resolve()
    common_dir = Path(
        str(run_git(repo, ["rev-parse", "--path-format=absolute", "--git-common-dir"]))
    ).resolve()
    if git_dir != common_dir:
        raise SystemExit("Coordinator operation requires the primary main worktree.")
    if clean:
        status = str(run_git(repo, ["status", "--porcelain=v1", "--untracked-files=all"]))
        if status:
            raise SystemExit(f"Coordinator operation requires a clean main worktree:\n{status}")


def is_ancestor(repo: Path, commit: str, descendant: str = "HEAD") -> bool:
    process = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, descendant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.returncode == 0


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_controller_token() -> str:
    return "ctl_" + secrets.token_urlsafe(32)


def recovery_key_path(db: Path) -> Path:
    return db.with_name("controller-recovery.key")


def write_private_token(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(token + "\n")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def immediate(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS controller (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            owner TEXT NOT NULL,
            token_hash TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            acquired_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            producer TEXT NOT NULL,
            experiment TEXT NOT NULL,
            batch TEXT NOT NULL,
            lane TEXT NOT NULL,
            base_commit TEXT NOT NULL,
            performance_baseline_commit TEXT NOT NULL,
            producer_commit TEXT NOT NULL,
            integrated_commit TEXT,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            language TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            evidence TEXT NOT NULL,
            priority INTEGER NOT NULL,
            status TEXT NOT NULL,
            rejection_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS active_claim (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            claim_id TEXT NOT NULL UNIQUE,
            claim_request_id TEXT NOT NULL UNIQUE,
            candidate_id TEXT NOT NULL UNIQUE REFERENCES candidates(candidate_id),
            controller_epoch INTEGER NOT NULL,
            phase TEXT NOT NULL,
            problem_url TEXT,
            language TEXT,
            source_sha256 TEXT NOT NULL,
            submission_id TEXT UNIQUE,
            submitted_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS claim_requests (
            request_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
            claim_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS submissions (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            candidate_id TEXT,
            claim_id TEXT,
            submitted_at TEXT NOT NULL,
            operator TEXT NOT NULL,
            language TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            experiment TEXT NOT NULL,
            status TEXT NOT NULL,
            score TEXT NOT NULL,
            previous_score TEXT NOT NULL,
            rank TEXT NOT NULL,
            url TEXT NOT NULL,
            evidence TEXT NOT NULL,
            notes TEXT NOT NULL,
            reported_to_user INTEGER NOT NULL,
            reported_at TEXT,
            report_summary TEXT,
            message_ref TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            occurred_at TEXT NOT NULL,
            actor TEXT NOT NULL,
            event TEXT NOT NULL,
            entity TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )
    with immediate(connection):
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(candidates)")
        }
        if "base_commit" not in columns:
            connection.execute("ALTER TABLE candidates ADD COLUMN base_commit TEXT")
        if "performance_baseline_commit" not in columns:
            connection.execute(
                "ALTER TABLE candidates ADD COLUMN performance_baseline_commit TEXT"
            )
            connection.execute(
                "UPDATE candidates SET performance_baseline_commit=base_commit "
                "WHERE performance_baseline_commit IS NULL"
            )
        if "batch" not in columns:
            connection.execute("ALTER TABLE candidates ADD COLUMN batch TEXT")
        if "lane" not in columns:
            connection.execute("ALTER TABLE candidates ADD COLUMN lane TEXT")
        active_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(active_claim)")
        }
        if "claim_request_id" not in active_columns:
            connection.execute("ALTER TABLE active_claim ADD COLUMN claim_request_id TEXT")
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS active_claim_request_unique
            ON active_claim(claim_request_id) WHERE claim_request_id IS NOT NULL
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        connection.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('mirror_dirty', '0')"
        )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS candidates_experiment_unique "
            "ON candidates(experiment)"
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO claim_requests(request_id,candidate_id,claim_id,created_at)
            SELECT claim_request_id,candidate_id,claim_id,created_at
            FROM active_claim
            WHERE claim_request_id IS NOT NULL
            """
        )


def ensure_recovery_capability(connection: sqlite3.Connection, db: Path) -> Path:
    path = recovery_key_path(db)
    with immediate(connection):
        existing = connection.execute(
            "SELECT value FROM meta WHERE key='recovery_token_hash'"
        ).fetchone()
        if existing is not None:
            if not path.exists():
                raise SystemExit(
                    f"Controller recovery key is missing: {path}. Do not rotate it "
                    "implicitly; reconcile the controller database manually."
                )
            token = path.read_text(encoding="utf-8").strip()
            if not token.startswith("rec_") or not secrets.compare_digest(
                existing["value"], token_digest(token)
            ):
                raise SystemExit(
                    "Controller recovery key does not match the centralized database."
                )
            return path
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if not token.startswith("rec_"):
                raise SystemExit("Invalid controller recovery key file.")
        else:
            token = "rec_" + secrets.token_urlsafe(48)
            write_private_token(path, token)
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('recovery_token_hash',?)",
            (token_digest(token),),
        )
    return path


def require_recovery_capability(
    connection: sqlite3.Connection, token: str | None
) -> None:
    if not token:
        raise SystemExit(
            "A recovery token is required for takeover; read the primary controller "
            "recovery key only from the Main Agent."
        )
    expected = connection.execute(
        "SELECT value FROM meta WHERE key='recovery_token_hash'"
    ).fetchone()
    if expected is None or not secrets.compare_digest(
        expected["value"], token_digest(token)
    ):
        raise SystemExit("Controller recovery token is invalid.")


def event(
    connection: sqlite3.Connection,
    actor: str,
    name: str,
    entity: str,
    payload: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT INTO events(occurred_at, actor, event, entity, payload_json) VALUES(?,?,?,?,?)",
        (utc_now(), actor, name, entity, json.dumps(payload or {}, sort_keys=True)),
    )


def import_mirror(connection: sqlite3.Connection, mirror: Path) -> None:
    with immediate(connection):
        imported = connection.execute(
            "SELECT value FROM meta WHERE key='mirror_imported'"
        ).fetchone()
        if imported:
            return
        if not mirror.exists():
            connection.execute(
                "INSERT INTO meta(key,value) VALUES('mirror_imported','empty')"
            )
            connection.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('mirror_sha256',?)",
                (file_digest(mirror),),
            )
            return
        try:
            data = json.loads(mirror.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid submission mirror {mirror}: {exc}") from exc
        submissions = data.get("submissions")
        if not isinstance(submissions, list):
            raise SystemExit(f"Invalid submission mirror schema: {mirror}")
        pending = data.get("pending_report")
        for item in submissions:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise SystemExit(f"Invalid submission entry in {mirror}")
            reported = bool(item.get("reported_to_user", item.get("id") != pending))
            connection.execute(
                """
                INSERT OR IGNORE INTO submissions(
                    id,candidate_id,claim_id,submitted_at,operator,language,commit_sha,
                    experiment,status,score,previous_score,rank,url,evidence,notes,
                    reported_to_user,reported_at,report_summary,message_ref
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    item["id"], item.get("candidate_id"), item.get("claim_id"),
                    item.get("submitted_at", utc_now()), item.get("operator", "unknown"),
                    item.get("language", "unknown"), item.get("commit", "unknown"),
                    item.get("experiment", "unknown"), item.get("status", "unknown"),
                    str(item.get("score", "n/a")), str(item.get("previous_score", "n/a")),
                    str(item.get("rank", "n/a")), item.get("url", ""),
                    item.get("evidence", ""), item.get("notes", ""), int(reported),
                    item.get("reported_at"), item.get("report_summary"), item.get("message_ref"),
                ),
            )
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('mirror_imported',?)",
            (utc_now(),),
        )
        event(connection, "migration", "mirror-import", str(mirror), {"count": len(submissions)})
        connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('mirror_sha256',?)",
            (file_digest(mirror),),
        )


def ensure_mirror_digest_metadata(
    connection: sqlite3.Connection, mirror: Path
) -> None:
    with immediate(connection):
        existing = connection.execute(
            "SELECT value FROM meta WHERE key='mirror_sha256'"
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO meta(key,value) VALUES('mirror_sha256',?)",
                (file_digest(mirror),),
            )


def mirror_digests(
    connection: sqlite3.Connection, mirror: Path
) -> tuple[str, str]:
    expected = connection.execute(
        "SELECT value FROM meta WHERE key='mirror_sha256'"
    ).fetchone()
    return ("missing-meta" if expected is None else expected["value"], file_digest(mirror))


def require_mirror_consistent(connection: sqlite3.Connection, mirror: Path) -> None:
    expected, actual = mirror_digests(connection, mirror)
    if expected != actual:
        raise SystemExit(
            "Tracked submission mirror differs from centralized state; privileged work "
            "is blocked. Inspect doctor/controller-status, then use export-ledger with "
            "the active controller after reconciling the Git change."
        )


def mirror_is_dirty(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        "SELECT value FROM meta WHERE key='mirror_dirty'"
    ).fetchone()
    return row is None or row["value"] != "0"


def require_mirror_clean(connection: sqlite3.Connection) -> None:
    if mirror_is_dirty(connection):
        raise SystemExit(
            "Centralized state has not been exported to the tracked submission mirror. "
            "Retry finalize/report or use export-ledger before continuing."
        )


def initialize(db: Path, mirror: Path) -> sqlite3.Connection:
    connection = connect(db)
    try:
        create_schema(connection)
        import_mirror(connection, mirror)
        ensure_mirror_digest_metadata(connection, mirror)
        ensure_recovery_capability(connection, db)
        return connection
    except BaseException:
        connection.close()
        raise


def controller_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM controller WHERE singleton=1").fetchone()


def active_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM active_claim WHERE singleton=1").fetchone()


def require_controller(connection: sqlite3.Connection, token: str | None) -> sqlite3.Row:
    if not token:
        raise SystemExit("A controller token is required; use --controller-token or XPUOJ_CONTROLLER_TOKEN.")
    row = controller_row(connection)
    if row is None or not secrets.compare_digest(row["token_hash"], token_digest(token)):
        raise SystemExit("Controller token is absent, stale, or invalid.")
    connection.execute(
        "UPDATE controller SET heartbeat_at=? WHERE singleton=1", (utc_now(),)
    )
    return row


def candidate_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def command_init(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    count = connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    print(json.dumps({
        "database": str(args.db),
        "imported_submissions": count,
        "recovery_key": str(recovery_key_path(args.db)),
    }, sort_keys=True))
    return 0


def command_controller_acquire(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    token = new_controller_token()
    now = utc_now()
    with immediate(connection):
        existing = controller_row(connection)
        if existing:
            raise SystemExit(
                f"Controller already held by {existing['owner']} at epoch {existing['epoch']}."
            )
        if active_row(connection) is not None:
            raise SystemExit(
                "An active claim exists without a controller; stop and reconcile the "
                "centralized database before acquiring a new lease."
            )
        epoch_row = connection.execute(
            "SELECT value FROM meta WHERE key='last_controller_epoch'"
        ).fetchone()
        epoch = int(epoch_row[0]) + 1 if epoch_row else 1
        connection.execute(
            "INSERT INTO controller VALUES(1,?,?,?,?,?)",
            (args.owner, token_digest(token), epoch, now, now),
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('last_controller_epoch',?)",
            (str(epoch),),
        )
        event(connection, args.owner, "controller-acquire", str(epoch))
    print(json.dumps({"owner": args.owner, "epoch": epoch, "controller_token": token}, sort_keys=True))
    return 0


def command_controller_status(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    controller = controller_row(connection)
    active = active_row(connection)
    mirror_expected, mirror_actual = mirror_digests(connection, args.mirror)
    result = {
        "controller": None if controller is None else {
            "owner": controller["owner"], "epoch": controller["epoch"],
            "acquired_at": controller["acquired_at"], "heartbeat_at": controller["heartbeat_at"],
        },
        "active_claim": None if active is None else {
            key: active[key] for key in (
                "claim_id", "candidate_id", "controller_epoch", "phase",
                "claim_request_id", "problem_url", "language", "source_sha256",
                "submission_id", "created_at", "updated_at"
            )
        },
        "mirror_expected_sha256": mirror_expected,
        "mirror_actual_sha256": mirror_actual,
        "mirror_consistent": mirror_expected == mirror_actual,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_controller_takeover(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    token = new_controller_token()
    now = utc_now()
    with immediate(connection):
        require_recovery_capability(connection, args.recovery_token)
        existing = controller_row(connection)
        if existing is None:
            raise SystemExit("No controller exists; use controller-acquire instead of takeover.")
        if existing["owner"] != args.expected_owner or int(existing["epoch"]) != args.expected_epoch:
            raise SystemExit(
                "Controller takeover precondition changed: expected "
                f"{args.expected_owner!r}/epoch {args.expected_epoch}, found "
                f"{existing['owner']!r}/epoch {existing['epoch']}."
            )
        previous_epoch = int(existing["epoch"])
        epoch_row = connection.execute(
            "SELECT value FROM meta WHERE key='last_controller_epoch'"
        ).fetchone()
        epoch = max(previous_epoch, int(epoch_row[0]) if epoch_row else 0) + 1
        connection.execute("DELETE FROM controller WHERE singleton=1")
        connection.execute(
            "INSERT INTO controller VALUES(1,?,?,?,?,?)",
            (args.owner, token_digest(token), epoch, now, now),
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('last_controller_epoch',?)",
            (str(epoch),),
        )
        connection.execute(
            "UPDATE active_claim SET controller_epoch=?,updated_at=? WHERE singleton=1",
            (epoch, now),
        )
        event(
            connection, args.owner, "controller-takeover", str(epoch),
            {
                "reason": args.reason,
                "previous_owner": existing["owner"],
                "previous_epoch": previous_epoch,
                "active_claim_preserved": active_row(connection) is not None,
            },
        )
    print(json.dumps({"owner": args.owner, "epoch": epoch, "controller_token": token}, sort_keys=True))
    return 0


def command_controller_release(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    with immediate(connection):
        owner = require_controller(connection, args.controller_token)
        if active_row(connection):
            raise SystemExit("Cannot release the controller while a claim is active.")
        connection.execute("DELETE FROM controller WHERE singleton=1")
        event(connection, owner["owner"], "controller-release", str(owner["epoch"]))
    print("Controller released.")
    return 0


def command_candidate_enqueue(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    if not CANDIDATE_RE.fullmatch(args.candidate):
        raise SystemExit("Candidate ID must match cand-[A-Za-z0-9._-]{3,96}.")
    if not EXPERIMENT_RE.fullmatch(args.experiment):
        raise SystemExit(f"Invalid experiment ID: {args.experiment}")
    if not BATCH_RE.fullmatch(args.batch):
        raise SystemExit(f"Invalid batch ID: {args.batch}")
    if not LANE_RE.fullmatch(args.lane):
        raise SystemExit(f"Invalid lane name: {args.lane}")
    if not COMMIT_RE.fullmatch(args.worktree_base_commit):
        raise SystemExit(
            "--worktree-base-commit must be a full 40-character commit."
        )
    if not COMMIT_RE.fullmatch(args.performance_baseline_commit):
        raise SystemExit(
            "--performance-baseline-commit must be a full 40-character commit."
        )
    source = safe_source_path(args.source)
    evidence = safe_source_path(args.evidence)
    source_hash = blob_sha256(args.repo, args.commit, source)
    run_git(
        args.repo,
        ["cat-file", "-e", f"{args.worktree_base_commit}^{{commit}}"],
    )
    run_git(
        args.repo,
        ["cat-file", "-e", f"{args.performance_baseline_commit}^{{commit}}"],
    )
    if not is_ancestor(args.repo, args.worktree_base_commit, args.commit):
        raise SystemExit(
            "Candidate commit must descend from its assigned worktree base commit."
        )
    worktree_base_hash = blob_sha256(
        args.repo, args.worktree_base_commit, source
    )
    performance_baseline_hash = blob_sha256(
        args.repo, args.performance_baseline_commit, source
    )
    if worktree_base_hash != performance_baseline_hash:
        raise SystemExit(
            "Worktree base source differs from the explicit performance baseline."
        )
    reservation = str(run_git(
        args.repo,
        ["rev-parse", "--verify", f"refs/xh-202628/experiments/{args.experiment}"],
    ))
    if reservation != args.worktree_base_commit:
        raise SystemExit(
            "Experiment reservation does not match the assigned worktree base commit."
        )
    baseline_ref = f"refs/xh-202628/baselines/{args.experiment}"
    baseline_reservation = optional_ref(args.repo, baseline_ref)
    c500_allocation = git_object_exists(
        args.repo, f"{args.worktree_base_commit}:state/c500-execution.json"
    )
    if c500_allocation and baseline_reservation is None:
        raise SystemExit(
            "C500 experiment is missing its immutable performance-baseline reservation."
        )
    if (
        baseline_reservation is not None
        and baseline_reservation != args.performance_baseline_commit
    ):
        raise SystemExit(
            "Performance baseline does not match the experiment baseline reservation."
        )
    expected_branch = f"candidate/{args.batch}/{args.lane}-{args.experiment}"
    branch_tip = str(run_git(
        args.repo, ["rev-parse", "--verify", f"refs/heads/{expected_branch}"]
    ))
    if branch_tip != args.commit:
        raise SystemExit(
            "Candidate commit must be the exact tip of its allocated candidate branch."
        )
    expected_job = f"remote-jobs/{args.experiment}.sh"
    expected_handoff = f"handoffs/{args.experiment}.md"
    if evidence != expected_handoff:
        raise SystemExit(
            f"Candidate evidence must be its committed handoff: {expected_handoff}"
        )
    run_git(args.repo, ["cat-file", "-e", f"{args.commit}:{expected_job}"])
    run_git(args.repo, ["cat-file", "-e", f"{args.commit}:{evidence}"])
    if args.source_sha256 and args.source_sha256 != source_hash:
        raise SystemExit(
            f"Source SHA-256 mismatch: supplied {args.source_sha256}, commit contains {source_hash}."
        )
    values = (
        args.candidate, args.producer, args.experiment, args.batch, args.lane,
        args.worktree_base_commit, args.performance_baseline_commit, args.commit,
        None, source, source_hash, args.language, args.hypothesis, evidence, args.priority,
        "queued", None, utc_now(), utc_now(),
    )
    with immediate(connection):
        existing = connection.execute(
            "SELECT * FROM candidates WHERE candidate_id=?", (args.candidate,)
        ).fetchone()
        if existing:
            comparable = (
                existing["producer"], existing["experiment"], existing["batch"],
                existing["lane"], existing["base_commit"],
                existing["performance_baseline_commit"], existing["producer_commit"],
                existing["source_path"],
                existing["source_sha256"], existing["language"],
                existing["hypothesis"], existing["evidence"], existing["priority"],
            )
            requested = (
                args.producer, args.experiment, args.batch, args.lane,
                args.worktree_base_commit, args.performance_baseline_commit,
                args.commit, source, source_hash, args.language,
                args.hypothesis, evidence, args.priority,
            )
            if comparable != requested:
                raise SystemExit(f"Candidate ID already exists with different data: {args.candidate}")
            print(json.dumps(candidate_dict(existing), ensure_ascii=False, sort_keys=True))
            return 0
        duplicate_experiment = connection.execute(
            "SELECT candidate_id FROM candidates WHERE experiment=?", (args.experiment,)
        ).fetchone()
        if duplicate_experiment:
            raise SystemExit(
                f"Experiment {args.experiment} already belongs to "
                f"{duplicate_experiment['candidate_id']}."
            )
        connection.execute(
            """
            INSERT INTO candidates(
                candidate_id,producer,experiment,batch,lane,base_commit,
                performance_baseline_commit,producer_commit,integrated_commit,
                source_path,source_sha256,language,hypothesis,evidence,priority,
                status,rejection_reason,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            values,
        )
        event(connection, args.producer, "candidate-enqueue", args.candidate, {"source_sha256": source_hash})
    print(json.dumps({
        "candidate_id": args.candidate,
        "batch": args.batch,
        "lane": args.lane,
        "worktree_base_commit": args.worktree_base_commit,
        "performance_baseline_commit": args.performance_baseline_commit,
        "status": "queued",
        "source_sha256": source_hash,
    }, sort_keys=True))
    return 0


def command_candidate_list(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    query = "SELECT * FROM candidates"
    values: tuple[Any, ...] = ()
    if args.status:
        query += " WHERE status=?"
        values = (args.status,)
    query += " ORDER BY priority DESC, created_at, candidate_id"
    rows = [candidate_dict(row) for row in connection.execute(query, values)]
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_candidate_promote(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=True)
    if not COMMIT_RE.fullmatch(args.commit):
        raise SystemExit("--commit must be a full 40-character commit.")
    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        candidate = connection.execute(
            "SELECT * FROM candidates WHERE candidate_id=?", (args.candidate,)
        ).fetchone()
        if candidate is None:
            raise SystemExit(f"Unknown candidate: {args.candidate}")
        if candidate["status"] == "ready" and candidate["integrated_commit"] == args.commit:
            print(json.dumps(candidate_dict(candidate), ensure_ascii=False, sort_keys=True))
            return 0
        if candidate["status"] != "queued":
            raise SystemExit(f"Candidate {args.candidate} is {candidate['status']}, not queued.")
        if not is_ancestor(args.repo, args.commit):
            raise SystemExit("Integrated commit must be an ancestor of main HEAD.")
        if not candidate["base_commit"] or not is_ancestor(
            args.repo, candidate["base_commit"], args.commit
        ):
            raise SystemExit("Integrated commit must descend from the assigned base commit.")
        integrated_hash = blob_sha256(args.repo, args.commit, candidate["source_path"])
        if integrated_hash != candidate["source_sha256"]:
            raise SystemExit(
                "Integrated source differs from the producer candidate; enqueue reviewed changes as a new candidate."
            )
        connection.execute(
            "UPDATE candidates SET integrated_commit=?,status='ready',updated_at=? WHERE candidate_id=?",
            (args.commit, utc_now(), args.candidate),
        )
        event(connection, controller["owner"], "candidate-promote", args.candidate, {"commit": args.commit})
    print(json.dumps({"candidate_id": args.candidate, "status": "ready", "integrated_commit": args.commit}, sort_keys=True))
    return 0


def command_candidate_reject(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        candidate = connection.execute(
            "SELECT * FROM candidates WHERE candidate_id=?", (args.candidate,)
        ).fetchone()
        if candidate is None:
            raise SystemExit(f"Unknown candidate: {args.candidate}")
        if candidate["status"] not in ("queued", "ready"):
            raise SystemExit(f"Cannot reject candidate in state {candidate['status']}.")
        connection.execute(
            "UPDATE candidates SET status='rejected',rejection_reason=?,updated_at=? WHERE candidate_id=?",
            (args.reason, utc_now(), args.candidate),
        )
        event(connection, controller["owner"], "candidate-reject", args.candidate, {"reason": args.reason})
    print(f"Rejected {args.candidate}.")
    return 0


def command_claim(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=True)
    if not REQUEST_RE.fullmatch(args.request_id):
        raise SystemExit("Claim request ID must match req-[A-Za-z0-9._-]{3,96}.")
    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        active = active_row(connection)
        request = connection.execute(
            "SELECT * FROM claim_requests WHERE request_id=?", (args.request_id,)
        ).fetchone()
        if request is not None:
            if request["candidate_id"] != args.candidate:
                raise SystemExit(
                    f"Claim request {args.request_id} is permanently bound to "
                    f"candidate {request['candidate_id']}."
                )
            if active is None or active["claim_id"] != request["claim_id"]:
                raise SystemExit(
                    f"Claim request {args.request_id} was already completed and cannot be reused."
                )
            candidate = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (active["candidate_id"],)
            ).fetchone()
            print(json.dumps({
                "claim_id": active["claim_id"],
                "request_id": active["claim_request_id"],
                "candidate_id": active["candidate_id"],
                "phase": active["phase"],
                "commit": candidate["integrated_commit"],
                "source": candidate["source_path"],
                "source_sha256": candidate["source_sha256"],
            }, sort_keys=True))
            return 0
        if active:
            raise SystemExit("Another XPU-OJ claim is already active.")
        candidate = connection.execute(
            "SELECT * FROM candidates WHERE candidate_id=?", (args.candidate,)
        ).fetchone()
        if candidate is None or candidate["status"] != "ready":
            state = None if candidate is None else candidate["status"]
            raise SystemExit(f"Candidate {args.candidate} is not ready (state={state!r}).")
        commit = candidate["integrated_commit"]
        if not commit or not is_ancestor(args.repo, commit):
            raise SystemExit("Candidate integrated commit is not on main history.")
        head_hash = blob_sha256(args.repo, str(run_git(args.repo, ["rev-parse", "HEAD"])), candidate["source_path"])
        if head_hash != candidate["source_sha256"]:
            raise SystemExit("main HEAD does not contain the candidate's exact source blob.")
        claim_id = "claim-" + uuid.uuid4().hex
        now = utc_now()
        connection.execute(
            """
            INSERT INTO active_claim(
                singleton,claim_id,claim_request_id,candidate_id,controller_epoch,
                phase,problem_url,language,source_sha256,submission_id,submitted_at,
                created_at,updated_at
            ) VALUES(1,?,?,?,?,?,NULL,NULL,?,NULL,NULL,?,?)
            """,
            (
                claim_id, args.request_id, args.candidate, controller["epoch"],
                "claimed", candidate["source_sha256"], now, now,
            ),
        )
        connection.execute(
            "INSERT INTO claim_requests(request_id,candidate_id,claim_id,created_at) VALUES(?,?,?,?)",
            (args.request_id, args.candidate, claim_id, now),
        )
        connection.execute(
            "UPDATE candidates SET status='claimed',updated_at=? WHERE candidate_id=?",
            (now, args.candidate),
        )
        event(connection, controller["owner"], "claim", claim_id, {"candidate": args.candidate})
    print(json.dumps({
        "claim_id": claim_id,
        "request_id": args.request_id,
        "candidate_id": args.candidate,
        "commit": commit,
        "source": candidate["source_path"],
        "source_sha256": candidate["source_sha256"],
    }, sort_keys=True))
    return 0


def get_claim(connection: sqlite3.Connection, claim_id: str) -> sqlite3.Row:
    claim = active_row(connection)
    if claim is None or claim["claim_id"] != claim_id:
        raise SystemExit(f"Claim is absent or not active: {claim_id}")
    return claim


def require_claim_source_at_head(
    repo: Path, connection: sqlite3.Connection, claim: sqlite3.Row
) -> sqlite3.Row:
    candidate = connection.execute(
        "SELECT * FROM candidates WHERE candidate_id=?", (claim["candidate_id"],)
    ).fetchone()
    if candidate is None or not candidate["integrated_commit"]:
        raise SystemExit("Claim candidate is missing its integrated commit.")
    if not is_ancestor(repo, candidate["integrated_commit"]):
        raise SystemExit("Claimed candidate commit is no longer on main history.")
    head = str(run_git(repo, ["rev-parse", "HEAD"]))
    head_hash = blob_sha256(repo, head, candidate["source_path"])
    if head_hash != claim["source_sha256"]:
        raise SystemExit(
            "main HEAD no longer contains the exact source claimed for XPU-OJ. "
            "Restore the claimed source or reconcile the claim before arming."
        )
    return candidate


def command_arm(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=True)
    if args.problem_url != TARGET_PROBLEM_URL:
        raise SystemExit(
            f"This controller is bound to the exact problem URL {TARGET_PROBLEM_URL}."
        )
    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        claim = get_claim(connection, args.claim)
        if claim["controller_epoch"] != controller["epoch"]:
            connection.execute(
                "UPDATE active_claim SET controller_epoch=? WHERE singleton=1",
                (controller["epoch"],),
            )
        candidate = require_claim_source_at_head(args.repo, connection, claim)
        if claim["phase"] == "armed":
            if claim["problem_url"] != args.problem_url or claim["language"] != args.language:
                raise SystemExit("Claim is already armed with different metadata.")
            print(json.dumps({"claim_id": args.claim, "phase": "armed"}, sort_keys=True))
            return 0
        if claim["phase"] != "claimed":
            raise SystemExit(f"Cannot arm a claim in phase {claim['phase']}.")
        if args.language != candidate["language"]:
            raise SystemExit("Armed language differs from the queued candidate language.")
        now = utc_now()
        connection.execute(
            "UPDATE active_claim SET phase='armed',problem_url=?,language=?,updated_at=? WHERE singleton=1",
            (args.problem_url, args.language, now),
        )
        event(connection, controller["owner"], "arm", args.claim, {"problem_url": args.problem_url})
    print(json.dumps({"claim_id": args.claim, "phase": "armed", "source_sha256": claim["source_sha256"]}, sort_keys=True))
    return 0


def command_bind(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=True)
    if not SUBMISSION_RE.fullmatch(args.submission_id):
        raise SystemExit("XPU-OJ submission ID must be numeric.")
    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        claim = get_claim(connection, args.claim)
        if claim["phase"] == "judging":
            if claim["submission_id"] != args.submission_id:
                raise SystemExit("Claim is already bound to a different submission ID.")
            if args.timestamp and claim["submitted_at"] != args.timestamp:
                raise SystemExit("Claim is already bound with a different timestamp.")
            print(json.dumps({"claim_id": args.claim, "phase": "judging", "submission_id": args.submission_id}, sort_keys=True))
            return 0
        if claim["phase"] != "armed":
            raise SystemExit(f"Cannot bind a claim in phase {claim['phase']}.")
        duplicate = connection.execute(
            "SELECT id FROM submissions WHERE id=?", (args.submission_id,)
        ).fetchone()
        if duplicate:
            raise SystemExit(f"Submission ID already exists: {args.submission_id}")
        now = utc_now()
        connection.execute(
            "UPDATE active_claim SET phase='judging',submission_id=?,submitted_at=?,updated_at=? WHERE singleton=1",
            (args.submission_id, args.timestamp or now, now),
        )
        connection.execute(
            "UPDATE candidates SET status='submitted',updated_at=? WHERE candidate_id=?",
            (now, claim["candidate_id"]),
        )
        event(connection, controller["owner"], "bind", args.claim, {"submission_id": args.submission_id})
    print(json.dumps({"claim_id": args.claim, "phase": "judging", "submission_id": args.submission_id}, sort_keys=True))
    return 0


def terminal_status(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise SystemExit("Terminal status cannot be empty.")
    if any(word in normalized.lower() for word in NON_TERMINAL_WORDS):
        raise SystemExit(f"Status is not terminal: {normalized}")
    if normalized.casefold() not in TERMINAL_STATUSES:
        raise SystemExit(
            f"Unrecognized terminal status: {normalized}. Update the controller's "
            "explicit terminal-status set only after verifying the live OJ state."
        )
    return normalized


def command_finalize(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    status = terminal_status(args.status)
    submission_id: str
    expected = (
        args.operator, status, args.score, args.previous_score, args.rank,
        args.url, args.evidence, args.notes,
    )

    def recorded_values(row: sqlite3.Row | None) -> tuple[Any, ...] | None:
        if row is None:
            return None
        return (
            row["operator"], row["status"], row["score"], row["previous_score"],
            row["rank"], row["url"], row["evidence"], row["notes"],
        )

    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        claim = active_row(connection)
        if claim is None or claim["claim_id"] != args.claim:
            existing = connection.execute(
                "SELECT * FROM submissions WHERE claim_id=?", (args.claim,)
            ).fetchone()
            if recorded_values(existing) == expected:
                submission_id = existing["id"]
            elif existing is None:
                raise SystemExit(f"Claim is absent and has no finalized result: {args.claim}")
            else:
                raise SystemExit("Finalization conflicts with the existing terminal result.")
        elif claim["phase"] == "awaiting_report":
            existing = connection.execute(
                "SELECT * FROM submissions WHERE id=?", (claim["submission_id"],)
            ).fetchone()
            if recorded_values(existing) != expected:
                raise SystemExit("Finalization conflicts with the existing terminal result.")
            submission_id = existing["id"]
            now = utc_now()
            connection.execute(
                "UPDATE candidates SET status='finalized',updated_at=? WHERE candidate_id=?",
                (now, claim["candidate_id"]),
            )
            connection.execute("DELETE FROM active_claim WHERE singleton=1")
            connection.execute("UPDATE meta SET value='1' WHERE key='mirror_dirty'")
            event(
                connection, controller["owner"], "finalize-release", submission_id,
                {"previous_phase": "awaiting_report"},
            )
        else:
            if claim["phase"] != "judging":
                raise SystemExit(f"Cannot finalize a claim in phase {claim['phase']}.")
            require_main(args.repo, clean=True)
            candidate = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id=?", (claim["candidate_id"],)
            ).fetchone()
            now = utc_now()
            submission_id = claim["submission_id"]
            connection.execute(
                """
                INSERT INTO submissions(
                    id,candidate_id,claim_id,submitted_at,operator,language,commit_sha,
                    experiment,status,score,previous_score,rank,url,evidence,notes,
                    reported_to_user,reported_at,report_summary,message_ref
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    submission_id, candidate["candidate_id"], claim["claim_id"],
                    claim["submitted_at"] or now, args.operator, candidate["language"],
                    candidate["integrated_commit"], candidate["experiment"], status,
                    args.score, args.previous_score, args.rank, args.url, args.evidence,
                    args.notes, 0, None, None, None,
                ),
            )
            connection.execute(
                "UPDATE candidates SET status='finalized',updated_at=? WHERE candidate_id=?",
                (now, candidate["candidate_id"]),
            )
            connection.execute("DELETE FROM active_claim WHERE singleton=1")
            connection.execute("UPDATE meta SET value='1' WHERE key='mirror_dirty'")
            event(
                connection, controller["owner"], "finalize", submission_id,
                {"status": status, "score": args.score},
            )
    export_mirror_after_commit(
        connection, args.mirror, args.controller_token
    )
    print(json.dumps({"submission_id": submission_id, "phase": "finalized"}, sort_keys=True))
    return 0


def command_report(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    already_reported = False
    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        claim = active_row(connection)
        existing = connection.execute(
            "SELECT * FROM submissions WHERE id=?", (args.submission_id,)
        ).fetchone()
        if existing is None:
            raise SystemExit(f"Submission row is missing: {args.submission_id}")
        if existing["reported_to_user"]:
            if (
                existing["report_summary"] != args.summary
                or existing["message_ref"] != args.message_ref
                or (args.timestamp and existing["reported_at"] != args.timestamp)
            ):
                raise SystemExit("Report retry conflicts with the recorded report metadata.")
            already_reported = True
        else:
            now = args.timestamp or utc_now()
            connection.execute(
                "UPDATE submissions SET reported_to_user=1,reported_at=?,report_summary=?,message_ref=? WHERE id=?",
                (now, args.summary, args.message_ref, args.submission_id),
            )
            connection.execute(
                "UPDATE candidates SET status='reported',updated_at=? WHERE candidate_id=?",
                (now, existing["candidate_id"]),
            )
            if (
                claim is not None
                and claim["phase"] == "awaiting_report"
                and claim["submission_id"] == args.submission_id
            ):
                connection.execute("DELETE FROM active_claim WHERE singleton=1")
            connection.execute("UPDATE meta SET value='1' WHERE key='mirror_dirty'")
            event(connection, controller["owner"], "report", args.submission_id, {"message_ref": args.message_ref})
    export_mirror_after_commit(
        connection, args.mirror, args.controller_token
    )
    if already_reported:
        print(f"Submission {args.submission_id} was already reported; mirror verified.")
    else:
        print(f"Marked deferred submission {args.submission_id} reported.")
    return 0


def command_abandon(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    with immediate(connection):
        controller = require_controller(connection, args.controller_token)
        claim = get_claim(connection, args.claim)
        if claim["phase"] == "judging" or claim["phase"] == "awaiting_report":
            raise SystemExit("A submitted or finalized claim cannot be abandoned; reconcile it.")
        payload: dict[str, Any] = {"reason": args.reason, "phase": claim["phase"]}
        if claim["phase"] == "armed":
            if not args.confirmed_no_submit:
                raise SystemExit(
                    "Armed claims require --confirmed-no-submit after checking OJ history."
                )
            if not args.oj_history_evidence or not args.checked_at or not args.expected_source_sha256:
                raise SystemExit(
                    "Armed claims also require --oj-history-evidence, --checked-at, "
                    "and --expected-source-sha256."
                )
            if not SHA256_RE.fullmatch(args.expected_source_sha256):
                raise SystemExit("--expected-source-sha256 must be 64 lowercase hex characters.")
            if args.expected_source_sha256 != claim["source_sha256"]:
                raise SystemExit("Recovery source hash does not match the armed claim.")
            checked_at = aware_timestamp(args.checked_at, "--checked-at")
            armed_at = aware_timestamp(claim["updated_at"], "armed claim timestamp")
            if checked_at < armed_at:
                raise SystemExit("OJ history check predates the armed claim.")
            evidence_path, evidence_sha256 = oj_history_evidence(
                args.repo, args.oj_history_evidence
            )
            payload.update({
                "confirmed_no_submit": True,
                "oj_history_evidence": evidence_path,
                "oj_history_evidence_sha256": evidence_sha256,
                "checked_at": args.checked_at,
                "source_sha256": claim["source_sha256"],
                "problem_url": claim["problem_url"],
                "language": claim["language"],
            })
        connection.execute(
            "UPDATE candidates SET status='ready',updated_at=? WHERE candidate_id=?",
            (utc_now(), claim["candidate_id"]),
        )
        connection.execute("DELETE FROM active_claim WHERE singleton=1")
        event(connection, controller["owner"], "abandon", args.claim, payload)
    print(f"Abandoned {args.claim}; candidate returned to ready.")
    return 0


def command_check(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    active = active_row(connection)
    unreported = connection.execute(
        "SELECT COUNT(*) FROM submissions WHERE reported_to_user=0"
    ).fetchone()[0]
    dirty = connection.execute(
        "SELECT value FROM meta WHERE key='mirror_dirty'"
    ).fetchone()[0]
    blockers: list[str] = []
    if active:
        blockers.append(f"active claim {active['claim_id']} is {active['phase']}")
    if dirty == "1":
        blockers.append("tracked submission mirror needs export/repair")
    mirror_expected, mirror_actual = mirror_digests(connection, args.mirror)
    if mirror_expected != mirror_actual:
        blockers.append("tracked submission mirror digest differs from centralized state")
    if blockers:
        print("BLOCKED: " + "; ".join(blockers), file=os.sys.stderr)
        return 2
    print(
        "PASS: centralized XPU-OJ submit slot is clear; "
        f"deferred user reports={unreported}."
    )
    return 0


def command_unreported_list(
    args: argparse.Namespace, connection: sqlite3.Connection
) -> int:
    rows = connection.execute(
        "SELECT * FROM submissions WHERE reported_to_user=0 ORDER BY seq"
    ).fetchall()
    result = [
        {
            "id": row["id"],
            "submitted_at": row["submitted_at"],
            "operator": row["operator"],
            "language": row["language"],
            "commit": row["commit_sha"],
            "experiment": row["experiment"],
            "status": row["status"],
            "score": row["score"],
            "previous_score": row["previous_score"],
            "rank": row["rank"],
            "url": row["url"],
            "evidence": row["evidence"],
            "notes": row["notes"],
        }
        for row in rows
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def export_mirror(connection: sqlite3.Connection, mirror: Path) -> None:
    rows = connection.execute("SELECT * FROM submissions ORDER BY seq").fetchall()
    pending_row = connection.execute(
        "SELECT id FROM submissions WHERE reported_to_user=0 ORDER BY seq LIMIT 1"
    ).fetchone()
    submissions: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "id": row["id"], "submitted_at": row["submitted_at"],
            "operator": row["operator"], "language": row["language"],
            "commit": row["commit_sha"], "experiment": row["experiment"],
            "status": row["status"], "score": row["score"],
            "previous_score": row["previous_score"], "rank": row["rank"],
            "url": row["url"], "evidence": row["evidence"], "notes": row["notes"],
            "reported_to_user": bool(row["reported_to_user"]),
            "reported_at": row["reported_at"], "report_summary": row["report_summary"],
        }
        for key in ("candidate_id", "claim_id", "message_ref"):
            if row[key] is not None:
                item[key] = row[key]
        submissions.append(item)
    data = {
        "version": 2,
        "pending_report": None if pending_row is None else pending_row["id"],
        "submissions": submissions,
    }
    mirror.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=mirror.name + ".", suffix=".tmp", dir=mirror.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, mirror)
        connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('mirror_sha256',?)",
            (file_digest(mirror),),
        )
        connection.execute("UPDATE meta SET value='0' WHERE key='mirror_dirty'")
    finally:
        if temporary.exists():
            temporary.unlink()


def export_mirror_after_commit(
    connection: sqlite3.Connection, mirror: Path, controller_token: str | None
) -> None:
    with immediate(connection):
        require_controller(connection, controller_token)
        require_mirror_consistent(connection, mirror)
        export_mirror(connection, mirror)


def command_export(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    require_main(args.repo, clean=False)
    with immediate(connection):
        require_controller(connection, args.controller_token)
        expected, actual = mirror_digests(connection, args.mirror)
        if args.expected_current_sha256 and args.expected_current_sha256 != actual:
            raise SystemExit(
                "Tracked mirror changed after inspection: expected current SHA-256 "
                f"{args.expected_current_sha256}, found {actual}."
            )
        if expected != actual and (
            not args.reconcile_drift or args.expected_current_sha256 != actual
        ):
            raise SystemExit(
                "Refusing to overwrite a drifted tracked mirror. Inspect doctor and Git "
                "diff, then repeat with --reconcile-drift and "
                f"--expected-current-sha256 {actual}."
            )
        export_mirror(connection, args.mirror)
    print(f"Exported submission history to {args.mirror}.")
    return 0


def command_doctor(args: argparse.Namespace, connection: sqlite3.Connection) -> int:
    controller = controller_row(connection)
    active = active_row(connection)
    counts = {
        row["status"]: row["count"]
        for row in connection.execute(
            "SELECT status,COUNT(*) AS count FROM candidates GROUP BY status"
        )
    }
    unreported = connection.execute(
        "SELECT COUNT(*) FROM submissions WHERE reported_to_user=0"
    ).fetchone()[0]
    dirty = connection.execute(
        "SELECT value FROM meta WHERE key='mirror_dirty'"
    ).fetchone()[0]
    mirror_expected, mirror_actual = mirror_digests(connection, args.mirror)
    result = {
        "database": str(args.db), "mirror": str(args.mirror),
        "controller_owner": None if controller is None else controller["owner"],
        "controller_epoch": None if controller is None else controller["epoch"],
        "active_phase": None if active is None else active["phase"],
        "active_claim": None if active is None else active["claim_id"],
        "candidate_counts": counts, "unreported_submissions": unreported,
        "mirror_dirty": dirty == "1",
        "mirror_expected_sha256": mirror_expected,
        "mirror_actual_sha256": mirror_actual,
        "mirror_consistent": mirror_expected == mirror_actual,
        "recovery_key_present": recovery_key_path(args.db).is_file(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if dirty == "1" or mirror_expected != mirror_actual else 0


def add_controller_token(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--controller-token", default=os.environ.get("XPUOJ_CONTROLLER_TOKEN")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=SCRIPT_REPO)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--mirror", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init")
    acquire = commands.add_parser("controller-acquire")
    acquire.add_argument("--owner", required=True)
    commands.add_parser("controller-status")
    takeover = commands.add_parser("controller-takeover")
    takeover.add_argument("--owner", required=True)
    takeover.add_argument("--reason", required=True)
    takeover.add_argument("--expected-owner", required=True)
    takeover.add_argument("--expected-epoch", required=True, type=int)
    takeover.add_argument(
        "--recovery-token", default=os.environ.get("XPUOJ_RECOVERY_TOKEN")
    )
    release = commands.add_parser("controller-release")
    add_controller_token(release)

    enqueue = commands.add_parser("candidate-enqueue")
    enqueue.add_argument("--candidate", required=True)
    enqueue.add_argument("--producer", required=True)
    enqueue.add_argument("--experiment", required=True)
    enqueue.add_argument("--batch", required=True)
    enqueue.add_argument("--lane", required=True)
    enqueue.add_argument("--worktree-base-commit", required=True)
    enqueue.add_argument("--performance-baseline-commit", required=True)
    enqueue.add_argument("--commit", required=True)
    enqueue.add_argument("--source", required=True)
    enqueue.add_argument("--source-sha256")
    enqueue.add_argument("--language", required=True)
    enqueue.add_argument("--hypothesis", required=True)
    enqueue.add_argument("--evidence", required=True)
    enqueue.add_argument("--priority", type=int, default=0)
    candidate_list = commands.add_parser("candidate-list")
    candidate_list.add_argument("--status")
    promote = commands.add_parser("candidate-promote")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--commit", required=True)
    add_controller_token(promote)
    reject = commands.add_parser("candidate-reject")
    reject.add_argument("--candidate", required=True)
    reject.add_argument("--reason", required=True)
    add_controller_token(reject)

    claim = commands.add_parser("claim")
    claim.add_argument("--candidate", required=True)
    claim.add_argument("--request-id", required=True)
    add_controller_token(claim)
    arm = commands.add_parser("arm")
    arm.add_argument("--claim", required=True)
    arm.add_argument("--problem-url", required=True)
    arm.add_argument("--language", required=True)
    add_controller_token(arm)
    bind = commands.add_parser("bind")
    bind.add_argument("--claim", required=True)
    bind.add_argument("--submission-id", required=True)
    bind.add_argument("--timestamp")
    add_controller_token(bind)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--claim", required=True)
    finalize.add_argument("--operator", required=True)
    finalize.add_argument("--status", required=True)
    finalize.add_argument("--score", default="n/a")
    finalize.add_argument("--previous-score", default="n/a")
    finalize.add_argument("--rank", default="n/a")
    finalize.add_argument("--url", default="")
    finalize.add_argument("--evidence", required=True)
    finalize.add_argument("--notes", default="")
    add_controller_token(finalize)
    report = commands.add_parser("report")
    report.add_argument("--submission-id", required=True)
    report.add_argument("--summary", required=True)
    report.add_argument("--message-ref", required=True)
    report.add_argument("--timestamp")
    add_controller_token(report)
    abandon = commands.add_parser("abandon")
    abandon.add_argument("--claim", required=True)
    abandon.add_argument("--reason", required=True)
    abandon.add_argument("--confirmed-no-submit", action="store_true")
    abandon.add_argument("--oj-history-evidence")
    abandon.add_argument("--checked-at")
    abandon.add_argument("--expected-source-sha256")
    add_controller_token(abandon)

    commands.add_parser("check")
    commands.add_parser("unreported-list")
    export = commands.add_parser("export-ledger")
    export.add_argument("--reconcile-drift", action="store_true")
    export.add_argument("--expected-current-sha256")
    add_controller_token(export)
    commands.add_parser("doctor")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.repo = resolve_repo(args.repo)
    resolve_control_paths(args)
    if not args.db.exists():
        require_main(args.repo, clean=False)
    connection = initialize(args.db, args.mirror)
    try:
        if args.command not in MIRROR_DRIFT_SAFE_COMMANDS:
            require_mirror_consistent(connection, args.mirror)
        if args.command not in MIRROR_DIRTY_SAFE_COMMANDS:
            require_mirror_clean(connection)
        handlers = {
            "init": command_init,
            "controller-acquire": command_controller_acquire,
            "controller-status": command_controller_status,
            "controller-takeover": command_controller_takeover,
            "controller-release": command_controller_release,
            "candidate-enqueue": command_candidate_enqueue,
            "candidate-list": command_candidate_list,
            "candidate-promote": command_candidate_promote,
            "candidate-reject": command_candidate_reject,
            "claim": command_claim,
            "arm": command_arm,
            "bind": command_bind,
            "finalize": command_finalize,
            "report": command_report,
            "abandon": command_abandon,
            "check": command_check,
            "unreported-list": command_unreported_list,
            "export-ledger": command_export,
            "doctor": command_doctor,
        }
        return handlers[args.command](args, connection)
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
