#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 12 ]]; then
  printf '%s\n' \
    'usage: c500-runner.sh BASE RUN_ID ENTRYPOINT CANDIDATE_COMMIT BASELINE_COMMIT WORKFLOW_COMMIT MAX_UTIL MAX_VRAM MAX_SECONDS EXPECTED_COMPUTE EXPECTED_VRAM MACA_PATH' >&2
  exit 64
fi

base=$1
run_id=$2
entrypoint=$3
candidate_commit=$4
baseline_commit=$5
workflow_commit=$6
max_util=$7
max_vram=$8
max_seconds=$9
expected_compute=${10}
expected_vram=${11}
maca_path=${12}

[[ "$base" =~ ^/[A-Za-z0-9._/-]+$ ]] || exit 64
[[ "$base" != *"/../"* && "$base" != *"/./"* && "$base" != */.. && "$base" != */. ]] || exit 64
[[ "$run_id" =~ ^exp-[0-9]{8}-[0-9]{3}-[0-9a-f]{12}-a[0-9]{2}$ ]] || exit 64
[[ "$entrypoint" =~ ^remote-jobs/[A-Za-z0-9._/-]+[.]sh$ ]] || exit 64
[[ "$entrypoint" != *"../"* && "$entrypoint" != *"/./"* ]] || exit 64
for commit in "$candidate_commit" "$baseline_commit" "$workflow_commit"; do
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 64
done
[[ "${candidate_commit:0:12}" == "${run_id:17:12}" ]] || exit 64
for numeric in "$max_util" "$max_vram" "$max_seconds" "$expected_compute" "$expected_vram"; do
  [[ "$numeric" =~ ^[0-9]+$ ]] || exit 64
done
(( max_seconds > 0 )) || exit 64
[[ "$maca_path" =~ ^/[A-Za-z0-9._/-]+$ ]] || exit 64
[[ "$maca_path" != *"/../"* && "$maca_path" != *"/./"* && "$maca_path" != */.. && "$maca_path" != */. ]] || exit 64

run_dir="$base/runs/$run_id"
source_dir="$run_dir/source"
baseline_dir="$run_dir/baseline"
results_dir="$run_dir/results"
status_file="$results_dir/status.json"
entrypoint_path="$source_dir/$entrypoint"
cucc="$maca_path/tools/cu-bridge/bin/cucc"
mxcc="$maca_path/mxgpu_llvm/bin/mxcc"
lock_dir="$base/locks/c500-run-slot.lock"
runner_claim_dir="$run_dir/.runner-claim"

mkdir -p -- "$results_dir"
mkdir -- "$runner_claim_dir" 2>/dev/null || {
  printf 'run already has a runner claim and will not be executed twice: %s\n' "$run_id" >&2
  exit 75
}
date --iso-8601=seconds > "$runner_claim_dir/claimed-at.txt"

read_metadata() {
  local name=$1
  local pattern=$2
  local value
  [[ -f "$run_dir/$name" && ! -L "$run_dir/$name" ]] || return 1
  value=$(<"$run_dir/$name")
  [[ "$value" =~ $pattern ]] || return 1
  printf '%s' "$value"
}

fail_metadata() {
  local message=$1
  printf '%s\n' "$message" > "$results_dir/preflight-error.txt"
  python3 - "$status_file" <<'PY' || true
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["state"] = "preflight-failed"
payload["exit_code"] = 125
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
  exit 125
}

candidate_source_sha=$(read_metadata candidate-source-sha256.txt '^[0-9a-f]{64}$') \
  || fail_metadata "candidate source hash metadata is missing or invalid"
baseline_source_sha=$(read_metadata baseline-source-sha256.txt '^[0-9a-f]{64}$') \
  || fail_metadata "baseline source hash metadata is missing or invalid"
workflow_archive_sha=$(read_metadata workflow-archive-sha256.txt '^[0-9a-f]{64}$') \
  || fail_metadata "workflow archive hash metadata is missing or invalid"
candidate_archive_sha=$(read_metadata candidate-archive-sha256.txt '^[0-9a-f]{64}$') \
  || fail_metadata "candidate archive hash metadata is missing or invalid"
baseline_archive_sha=$(read_metadata baseline-archive-sha256.txt '^[0-9a-f]{64}$') \
  || fail_metadata "baseline archive hash metadata is missing or invalid"
stage_sha=$(read_metadata stage-sha256.txt '^[0-9a-f]{64}$') \
  || fail_metadata "stage hash metadata is missing or invalid"
submission_source=$(read_metadata submission-source-path.txt '^[A-Za-z0-9._/-]+$') \
  || fail_metadata "submission source metadata is missing or invalid"

write_status() {
  local state=$1
  local exit_code=$2
  local temporary="$results_dir/status.json.tmp.$$"
  printf '{\n  "schema_version": 2,\n  "run_id": "%s",\n  "commit": "%s",\n  "baseline_commit": "%s",\n  "workflow_commit": "%s",\n  "entrypoint": "%s",\n  "submission_source": "%s",\n  "candidate_source_sha256": "%s",\n  "baseline_source_sha256": "%s",\n  "workflow_archive_sha256": "%s",\n  "candidate_archive_sha256": "%s",\n  "baseline_archive_sha256": "%s",\n  "stage_sha256": "%s",\n  "device_class": "c500-local",\n  "state": "%s",\n  "exit_code": %s\n}\n' \
    "$run_id" "$candidate_commit" "$baseline_commit" "$workflow_commit" \
    "$entrypoint" "$submission_source" "$candidate_source_sha" "$baseline_source_sha" \
    "$workflow_archive_sha" "$candidate_archive_sha" "$baseline_archive_sha" \
    "$stage_sha" "$state" "$exit_code" > "$temporary"
  mv -f -- "$temporary" "$status_file"
}

terminal_written=0
lock_acquired=0
run_started=0
child_pid=''
post_captured=0

capture_after() {
  local capture_status=0
  mx-smi > "$results_dir/mx-smi-after.txt" 2>&1 || capture_status=1
  mx-smi -j > "$results_dir/mx-smi-after.json.txt" 2>&1 || capture_status=1
  mx-smi --show-process > "$results_dir/mx-smi-process-after.txt" 2>&1 || capture_status=1
  mx-smi --show-clocks xcore > "$results_dir/mx-smi-clock-after.txt" 2>&1 || capture_status=1
  post_captured=1
  return "$capture_status"
}

cleanup_runner() {
  local exit_code=$?
  trap - EXIT
  set +e
  if [[ "$terminal_written" -eq 0 ]]; then
    if [[ "$run_started" -eq 1 ]]; then
      (( exit_code != 0 )) || exit_code=70
      [[ "$post_captured" -eq 1 ]] || capture_after
      write_status "failed" "$exit_code"
    else
      exit_code=125
      write_status "preflight-failed" "$exit_code"
    fi
  fi
  if [[ "$lock_acquired" -eq 1 ]]; then
    rmdir -- "$lock_dir" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup_runner EXIT

fail_preflight() {
  local message=$1
  printf '%s\n' "$message" > "$results_dir/preflight-error.txt"
  write_status "preflight-failed" 125
  terminal_written=1
  exit 125
}

handle_signal() {
  local signal_name=$1
  local exit_code=$2
  trap - HUP INT TERM
  set +e
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
    child_pid=''
  fi
  printf '%s\n' "$signal_name" > "$results_dir/interrupted-signal.txt"
  [[ "$post_captured" -eq 1 ]] || capture_after
  date --iso-8601=seconds > "$results_dir/finished-at.txt"
  write_status "interrupted" "$exit_code"
  terminal_written=1
  exit "$exit_code"
}
trap 'handle_signal HUP 129' HUP
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

grep -qx 'xh-202628-c500-execution-mirror-v1' "$base/.xh-202628-c500-execution-mirror" \
  || fail_preflight "C500 execution mirror marker is missing"
for required in \
  "$entrypoint_path" "$source_dir/scripts/run-c500-fused-moe-paired.sh" \
  "$source_dir/scripts/summarize-c500-abba.py" "$run_dir/workflow-files.sha256"; do
  [[ -f "$required" && ! -L "$required" ]] \
    || fail_preflight "trusted workflow file is missing or is a symlink: $required"
done
[[ "$(<"$run_dir/source-commit.txt")" == "$candidate_commit" ]] \
  || fail_preflight "candidate commit metadata does not match"
[[ "$(<"$run_dir/baseline-commit.txt")" == "$baseline_commit" ]] \
  || fail_preflight "baseline commit metadata does not match"
[[ "$(<"$run_dir/workflow-commit.txt")" == "$workflow_commit" ]] \
  || fail_preflight "workflow commit metadata does not match"
[[ "$(<"$run_dir/entrypoint.txt")" == "$entrypoint" ]] \
  || fail_preflight "entrypoint metadata does not match"

python3 - "$status_file" "$run_id" "$candidate_commit" "$baseline_commit" "$workflow_commit" <<'PY' \
  || fail_preflight "staged status metadata is invalid"
import json
import pathlib
import sys

status = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": 2,
    "run_id": sys.argv[2],
    "commit": sys.argv[3],
    "baseline_commit": sys.argv[4],
    "workflow_commit": sys.argv[5],
    "state": "staged",
    "exit_code": 0,
    "device_class": "c500-local",
}
if any(status.get(key) != value for key, value in expected.items()):
    raise SystemExit("staged status mismatch")
PY

command -v mx-smi >/dev/null 2>&1 || fail_preflight "mx-smi is unavailable"
command -v timeout >/dev/null 2>&1 || fail_preflight "timeout is unavailable"
command -v python3 >/dev/null 2>&1 || fail_preflight "python3 is unavailable"
command -v sha256sum >/dev/null 2>&1 || fail_preflight "sha256sum is unavailable"
[[ -x "$cucc" ]] || fail_preflight "MACA cu-bridge compiler is unavailable"
[[ -x "$mxcc" ]] || fail_preflight "MXCC is unavailable"

(
  cd -- "$source_dir"
  sha256sum -c "$run_dir/workflow-files.sha256"
) > "$results_dir/source-integrity-before.log" 2>&1 \
  || fail_preflight "candidate workflow tree failed its pre-run integrity check"
(
  cd -- "$baseline_dir"
  sha256sum -c "$run_dir/workflow-files.sha256"
) > "$results_dir/baseline-integrity-before.log" 2>&1 \
  || fail_preflight "baseline workflow tree failed its pre-run integrity check"
[[ "$(sha256sum -- "$source_dir/$submission_source" | awk '{print $1}')" == "$candidate_source_sha" ]] \
  || fail_preflight "candidate source hash changed before execution"
[[ "$(sha256sum -- "$baseline_dir/$submission_source" | awk '{print $1}')" == "$baseline_source_sha" ]] \
  || fail_preflight "baseline source hash changed before execution"

mkdir -- "$lock_dir" 2>/dev/null \
  || fail_preflight "the C500 project run slot is active or a stale lock needs review"
lock_acquired=1

mx-smi > "$results_dir/mx-smi-before.txt" 2>&1 \
  || fail_preflight "could not capture mx-smi state"
mx-smi -j > "$results_dir/mx-smi-before.json.txt" 2>&1 \
  || fail_preflight "could not capture mx-smi JSON state"
mx-smi --show-process > "$results_dir/mx-smi-process-before.txt" 2>&1 \
  || fail_preflight "could not inspect C500 processes"
grep -q 'no process found' "$results_dir/mx-smi-process-before.txt" \
  || fail_preflight "the C500 already has a device process"

device_state=$(python3 - "$results_dir/mx-smi-before.json.txt" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
start = text.find("{")
if start < 0:
    raise SystemExit("mx-smi JSON payload is missing")
payload = json.loads(text[start:])
if payload.get("attached_gpus") != 1:
    raise SystemExit("expected exactly one attached GPU")
devices = [value for value in payload.values() if isinstance(value, dict) and "device_id" in value]
if len(devices) != 1 or devices[0].get("device_id") != 0:
    raise SystemExit("expected device 0 to be the only visible device")
device = devices[0]
sgpus = device.get("sgpu_info", [])
if len(sgpus) != 1:
    raise SystemExit("expected exactly one visible sGPU slice")
sgpu = sgpus[0]
utilization = int(str(sgpu["utilization"]).rstrip("%"))
print(device["utilization"], utilization, sgpu["vram_used"], sgpu["compute_quota"], sgpu["vram_quota"])
PY
) || fail_preflight "could not parse C500 slice state"
read -r physical_util slice_util vram_used compute_quota vram_quota <<< "$device_state"
for numeric in "$physical_util" "$slice_util" "$vram_used" "$compute_quota" "$vram_quota"; do
  [[ "$numeric" =~ ^[0-9]+$ ]] || fail_preflight "C500 slice state contains a non-numeric value"
done
(( physical_util <= max_util && slice_util <= max_util )) \
  || fail_preflight "C500 utilization exceeds the approved start threshold"
(( vram_used <= max_vram )) \
  || fail_preflight "C500 slice memory use exceeds the approved start threshold"
(( compute_quota == expected_compute )) \
  || fail_preflight "C500 compute quota differs from the recorded environment"
(( vram_quota == expected_vram )) \
  || fail_preflight "C500 VRAM quota differs from the recorded environment"

set +e
(
  set -e
  printf 'run_id=%s\ncandidate_commit=%s\nbaseline_commit=%s\nworkflow_commit=%s\nentrypoint=%s\n' \
    "$run_id" "$candidate_commit" "$baseline_commit" "$workflow_commit" "$entrypoint"
  printf 'candidate_source_sha256=%s\nbaseline_source_sha256=%s\n' \
    "$candidate_source_sha" "$baseline_source_sha"
  date --iso-8601=seconds
  uname -a
  lscpu
  free -h
  mx-smi --show-version
  mx-smi --show-hwinfo
  mx-smi --show-clocks xcore
  "$mxcc" --version
  "$cucc" -V
  python3 --version
  if command -v mcProfiler >/dev/null 2>&1; then
    mcProfiler version 2>&1 || true
  fi
) > "$results_dir/environment.txt" 2>&1
environment_status=$?
set -e
if [[ "$environment_status" -ne 0 ]]; then
  fail_preflight "could not capture the complete C500 environment"
fi

printf '%s\n' "bash $entrypoint" > "$results_dir/remote-command.txt"
date --iso-8601=seconds > "$results_dir/started-at.txt"
start_ns=$(date +%s%N)
write_status "running" 0
run_started=1

runtime_home="$run_dir/runtime-home"
runtime_tmp="$run_dir/runtime-tmp"
mkdir -m 700 -- "$runtime_home" "$runtime_tmp"
clean_path="$maca_path/mxgpu_llvm/bin:$maca_path/tools/cu-bridge/bin:$maca_path/ompi/bin:$maca_path/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
clean_ld_library_path="$maca_path/lib:$maca_path/ompi/lib:$maca_path/ucx/lib:/opt/mxdriver/lib"
clean_library_path="/opt/mxdriver/lib"

set +e
(
  cd -- "$source_dir"
  exec env -i \
    HOME="$runtime_home" TMPDIR="$runtime_tmp" USER=root LOGNAME=root SHELL=/bin/bash \
    LC_ALL=C PATH="$clean_path" LD_LIBRARY_PATH="$clean_ld_library_path" \
    LIBRARY_PATH="$clean_library_path" MACA_PATH="$maca_path" \
    CUDA_PATH="$maca_path/tools/cu-bridge" CUCC="$cucc" CUDA_VISIBLE_DEVICES=0 \
    C500_CANDIDATE_SOURCE="$source_dir" C500_BASELINE_SOURCE="$baseline_dir" \
    C500_RESULTS_DIR="$results_dir" C500_CANDIDATE_COMMIT="$candidate_commit" \
    C500_BASELINE_COMMIT="$baseline_commit" C500_WORKFLOW_COMMIT="$workflow_commit" \
    C500_SUBMISSION_SOURCE="$submission_source" \
    C500_CANDIDATE_SOURCE_SHA256="$candidate_source_sha" \
    C500_BASELINE_SOURCE_SHA256="$baseline_source_sha" \
    timeout --signal=TERM --kill-after=30s "${max_seconds}s" bash "$entrypoint_path"
) > "$results_dir/stdout.log" 2> "$results_dir/stderr.log" &
child_pid=$!
wait "$child_pid"
entrypoint_status=$?
child_pid=''
set -e

post_capture_status=0
capture_after || post_capture_status=1

(
  cd -- "$source_dir"
  sha256sum -c "$run_dir/workflow-files.sha256"
) > "$results_dir/source-integrity-after.log" 2>&1 || entrypoint_status=66
(
  cd -- "$baseline_dir"
  sha256sum -c "$run_dir/workflow-files.sha256"
) > "$results_dir/baseline-integrity-after.log" 2>&1 || entrypoint_status=66
[[ "$(sha256sum -- "$source_dir/$submission_source" | awk '{print $1}')" == "$candidate_source_sha" ]] \
  || entrypoint_status=66
[[ "$(sha256sum -- "$baseline_dir/$submission_source" | awk '{print $1}')" == "$baseline_source_sha" ]] \
  || entrypoint_status=66
(( post_capture_status == 0 )) || entrypoint_status=66

if [[ "$entrypoint_status" -eq 0 ]]; then
  required_nonempty=(
    correctness.log regression.log commands.txt warmup-baseline.log warmup-candidate.log
    benchmark-baseline-a.log benchmark-candidate-a.log benchmark-candidate-b.log
    benchmark-baseline-b.log paired-benchmark.json environment.txt
  )
  required_present=(build-candidate.log build-baseline.log stdout.log stderr.log)
  for result in "${required_nonempty[@]}"; do
    if [[ ! -s "$results_dir/$result" ]]; then
      printf 'entrypoint did not produce required non-empty result: %s\n' "$result" \
        >> "$results_dir/stderr.log"
      entrypoint_status=66
    fi
  done
  for result in "${required_present[@]}"; do
    if [[ ! -f "$results_dir/$result" || -L "$results_dir/$result" ]]; then
      printf 'entrypoint did not produce required result: %s\n' "$result" \
        >> "$results_dir/stderr.log"
      entrypoint_status=66
    fi
  done
fi

if [[ "$entrypoint_status" -eq 0 ]]; then
  python3 - "$results_dir/paired-benchmark.json" \
    "$candidate_commit" "$baseline_commit" "$submission_source" \
    "$candidate_source_sha" "$baseline_source_sha" <<'PY' \
    || entrypoint_status=66
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": 1,
    "device_class": "c500-local",
    "benchmark_design": "ABBA",
    "interpretation": "paired-relative-only",
    "candidate_commit": sys.argv[2],
    "baseline_commit": sys.argv[3],
    "submission_source": sys.argv[4],
    "candidate_source_sha256": sys.argv[5],
    "baseline_source_sha256": sys.argv[6],
}
if any(payload.get(key) != value for key, value in expected.items()):
    raise SystemExit("paired benchmark metadata mismatch")
required_cases = {"decode-gate-up", "prefill-gate-up", "decode-down", "prefill-down"}
if set(payload.get("cases", {})) != required_cases:
    raise SystemExit("paired benchmark case set mismatch")
PY
fi

end_ns=$(date +%s%N)
date --iso-8601=seconds > "$results_dir/finished-at.txt"
printf '%s\n' "$((end_ns - start_ns))" > "$results_dir/duration-ns.txt"

manifest_temporary="$run_dir/result-manifest.sha256.tmp"
(
  cd -- "$results_dir"
  find . -maxdepth 1 -type f ! -name status.json -printf '%P\0' \
    | sort -z | xargs -0 sha256sum
) > "$manifest_temporary"
mv -- "$manifest_temporary" "$results_dir/result-manifest.sha256"

if [[ "$entrypoint_status" -eq 0 ]]; then
  write_status "succeeded" 0
else
  write_status "failed" "$entrypoint_status"
fi
terminal_written=1
exit "$entrypoint_status"
