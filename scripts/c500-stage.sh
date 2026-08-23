#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 14 ]]; then
  printf '%s\n' \
    'usage: c500-stage.sh BASE RUN_ID CANDIDATE_COMMIT BASELINE_COMMIT WORKFLOW_COMMIT SOURCE_PATH ENTRYPOINT MARKER_VALUE WORKFLOW_ARCHIVE_SHA CANDIDATE_ARCHIVE_SHA BASELINE_ARCHIVE_SHA STAGE_SHA CANDIDATE_SOURCE_SHA BASELINE_SOURCE_SHA' >&2
  exit 64
fi

base=$1
run_id=$2
candidate_commit=$3
baseline_commit=$4
workflow_commit=$5
source_path=$6
entrypoint=$7
marker_value=$8
workflow_archive_sha=$9
candidate_archive_sha=${10}
baseline_archive_sha=${11}
stage_sha=${12}
candidate_source_sha=${13}
baseline_source_sha=${14}

[[ "$base" =~ ^/[A-Za-z0-9._/-]+$ ]] || { printf 'unsafe base path\n' >&2; exit 64; }
[[ "$base" != *"/../"* && "$base" != *"/./"* && "$base" != */.. && "$base" != */. ]] \
  || { printf 'unsafe base path\n' >&2; exit 64; }
[[ "$run_id" =~ ^exp-[0-9]{8}-[0-9]{3}-[0-9a-f]{12}-a[0-9]{2}$ ]] \
  || { printf 'unsafe run id\n' >&2; exit 64; }
for commit in "$candidate_commit" "$baseline_commit" "$workflow_commit"; do
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { printf 'unsafe commit\n' >&2; exit 64; }
done
[[ "${candidate_commit:0:12}" == "${run_id:17:12}" ]] \
  || { printf 'run id does not match candidate commit\n' >&2; exit 64; }
for path in "$source_path" "$entrypoint"; do
  [[ "$path" =~ ^[A-Za-z0-9._/-]+$ ]] || { printf 'unsafe repository path\n' >&2; exit 64; }
  [[ "$path" != /* && "$path" != *"../"* && "$path" != *"/./"* && "$path" != */.. && "$path" != */. ]] \
    || { printf 'unsafe repository path\n' >&2; exit 64; }
done
[[ "$entrypoint" =~ ^remote-jobs/[A-Za-z0-9._/-]+[.]sh$ ]] \
  || { printf 'entrypoint must be a committed remote job\n' >&2; exit 64; }
[[ "$marker_value" == "xh-202628-c500-execution-mirror-v1" ]] \
  || { printf 'unexpected marker value\n' >&2; exit 64; }
for digest in \
  "$workflow_archive_sha" "$candidate_archive_sha" "$baseline_archive_sha" \
  "$stage_sha" "$candidate_source_sha" "$baseline_source_sha"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { printf 'unsafe SHA-256 digest\n' >&2; exit 64; }
done

marker="$base/.xh-202628-c500-execution-mirror"
workflow_archive="$base/incoming/$run_id.workflow.tar"
candidate_archive="$base/incoming/$run_id.candidate-source.tar"
baseline_archive="$base/incoming/$run_id.baseline-source.tar"
stage_script="$base/incoming/$run_id.stage.sh"
run_dir="$base/runs/$run_id"
staging_dir="$base/runs/.staging-$run_id"

grep -qx "$marker_value" "$marker"
for incoming in "$workflow_archive" "$candidate_archive" "$baseline_archive" "$stage_script"; do
  [[ -f "$incoming" && ! -L "$incoming" ]] || {
    printf 'required incoming file is missing or is a symlink: %s\n' "$incoming" >&2
    exit 66
  }
done
[[ ! -e "$run_dir" && ! -e "$staging_dir" ]] \
  || { printf 'run or staging directory already exists\n' >&2; exit 73; }

umask 077
stage_complete=0

write_status() {
  local state=$1
  local exit_code=$2
  local temporary="$staging_dir/results/status.json.tmp.$$"
  mkdir -p -m 700 -- "$staging_dir/results"
  printf '{\n  "schema_version": 2,\n  "run_id": "%s",\n  "commit": "%s",\n  "baseline_commit": "%s",\n  "workflow_commit": "%s",\n  "entrypoint": "%s",\n  "submission_source": "%s",\n  "candidate_source_sha256": "%s",\n  "baseline_source_sha256": "%s",\n  "workflow_archive_sha256": "%s",\n  "candidate_archive_sha256": "%s",\n  "baseline_archive_sha256": "%s",\n  "stage_sha256": "%s",\n  "device_class": "c500-local",\n  "state": "%s",\n  "exit_code": %s\n}\n' \
    "$run_id" "$candidate_commit" "$baseline_commit" "$workflow_commit" \
    "$entrypoint" "$source_path" "$candidate_source_sha" "$baseline_source_sha" \
    "$workflow_archive_sha" "$candidate_archive_sha" "$baseline_archive_sha" \
    "$stage_sha" "$state" "$exit_code" > "$temporary"
  mv -f -- "$temporary" "$staging_dir/results/status.json"
}

on_stage_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "$stage_complete" -eq 0 && -d "$staging_dir" && ! -L "$staging_dir" ]]; then
    set +e
    (( exit_code != 0 )) || exit_code=70
    mkdir -p -m 700 -- "$staging_dir/results"
    printf 'staging failed with exit code %s\n' "$exit_code" \
      > "$staging_dir/results/staging-error.txt"
    write_status "staging-failed" "$exit_code"
    if [[ ! -e "$run_dir" ]]; then
      mv -T -- "$staging_dir" "$run_dir"
    fi
  fi
  exit "$exit_code"
}
trap on_stage_exit EXIT

mkdir -m 700 -- "$staging_dir"
mkdir -m 700 -- "$staging_dir/source" "$staging_dir/baseline" "$staging_dir/results"
write_status "staging" 0

verify_digest() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum -- "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    printf 'SHA-256 mismatch for %s\n' "$path" >&2
    return 1
  }
}

verify_digest "$workflow_archive" "$workflow_archive_sha"
verify_digest "$candidate_archive" "$candidate_archive_sha"
verify_digest "$baseline_archive" "$baseline_archive_sha"
verify_digest "$stage_script" "$stage_sha"

validate_archive() {
  python3 - "$1" "$2" "$source_path" "$entrypoint" <<'PY'
import pathlib
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
source = pathlib.PurePosixPath(sys.argv[3])
entrypoint = pathlib.PurePosixPath(sys.argv[4])
allowed_source = {source, *source.parents}
allowed_candidate = {source, entrypoint, *source.parents, *entrypoint.parents}
source_entries = 0
entrypoint_entries = 0

with tarfile.open(archive, "r:") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit(f"empty tar archive: {archive}")
    for member in members:
        path = pathlib.PurePosixPath(member.name.rstrip("/"))
        if path == pathlib.PurePosixPath(".") and member.isdir() and mode == "workflow":
            continue
        if not member.name or path.is_absolute() or path == pathlib.PurePosixPath(".") or ".." in path.parts:
            raise SystemExit(f"unsafe tar member path: {member.name!r}")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported tar member type: {member.name!r}")
        allowed = allowed_candidate if mode == "candidate" else allowed_source
        if mode in {"candidate", "baseline"} and path not in allowed:
            raise SystemExit(f"overlay archive contains unexpected path: {member.name!r}")
        if path == source:
            if not member.isfile():
                raise SystemExit("submission source is not a regular file")
            source_entries += 1
        if path == entrypoint:
            if not member.isfile():
                raise SystemExit("entrypoint is not a regular file")
            entrypoint_entries += 1
    if mode in {"candidate", "baseline"} and source_entries != 1:
        raise SystemExit("overlay archive must contain exactly one submission source")
    if mode == "candidate" and entrypoint_entries != 1:
        raise SystemExit("candidate archive must contain exactly one trusted entrypoint")
PY
}

validate_archive "$workflow_archive" workflow
validate_archive "$candidate_archive" candidate
validate_archive "$baseline_archive" baseline

tar --no-same-owner --no-same-permissions -xf "$workflow_archive" -C "$staging_dir/source"
tar --no-same-owner --no-same-permissions -xf "$workflow_archive" -C "$staging_dir/baseline"
tar --no-same-owner --no-same-permissions -xf "$candidate_archive" -C "$staging_dir/source"
tar --no-same-owner --no-same-permissions -xf "$candidate_archive" -C "$staging_dir/baseline"
tar --no-same-owner --no-same-permissions -xf "$baseline_archive" -C "$staging_dir/baseline"

for required in \
  "$staging_dir/source/$source_path" "$staging_dir/baseline/$source_path" \
  "$staging_dir/source/$entrypoint" "$staging_dir/source/scripts/c500-runner.sh" \
  "$staging_dir/source/scripts/run-c500-fused-moe-paired.sh" \
  "$staging_dir/source/scripts/summarize-c500-abba.py"; do
  [[ -f "$required" && ! -L "$required" ]] || {
    printf 'required staged file is missing or is a symlink: %s\n' "$required" >&2
    exit 66
  }
done

verify_digest "$staging_dir/source/$source_path" "$candidate_source_sha"
verify_digest "$staging_dir/baseline/$source_path" "$baseline_source_sha"

(
  cd -- "$staging_dir/source"
  find . -type f ! -path "./$source_path" ! -path './build/*' -print0 \
    | sort -z | xargs -0 sha256sum
) > "$staging_dir/workflow-files.sha256"
[[ -s "$staging_dir/workflow-files.sha256" ]]

printf '%s\n' "$candidate_commit" > "$staging_dir/source-commit.txt"
printf '%s\n' "$baseline_commit" > "$staging_dir/baseline-commit.txt"
printf '%s\n' "$workflow_commit" > "$staging_dir/workflow-commit.txt"
printf '%s\n' "$source_path" > "$staging_dir/submission-source-path.txt"
printf '%s\n' "$entrypoint" > "$staging_dir/entrypoint.txt"
printf '%s\n' "$candidate_source_sha" > "$staging_dir/candidate-source-sha256.txt"
printf '%s\n' "$baseline_source_sha" > "$staging_dir/baseline-source-sha256.txt"
printf '%s\n' "$workflow_archive_sha" > "$staging_dir/workflow-archive-sha256.txt"
printf '%s\n' "$candidate_archive_sha" > "$staging_dir/candidate-archive-sha256.txt"
printf '%s\n' "$baseline_archive_sha" > "$staging_dir/baseline-archive-sha256.txt"
printf '%s\n' "$stage_sha" > "$staging_dir/stage-sha256.txt"
printf '%s\n' "$run_id" > "$staging_dir/run-id.txt"
date --iso-8601=seconds > "$staging_dir/staged-at.txt"

mv -- "$workflow_archive" "$staging_dir/workflow-source.tar"
mv -- "$candidate_archive" "$staging_dir/candidate-source-overlay.tar"
mv -- "$baseline_archive" "$staging_dir/baseline-source-overlay.tar"
mv -- "$stage_script" "$staging_dir/stage.sh"
write_status "staged" 0

mv -T -- "$staging_dir" "$run_dir"
stage_complete=1
printf '%s\n' "$run_dir"
