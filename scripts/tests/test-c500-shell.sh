#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
temp_root=$(mktemp -d /tmp/xh-202628-c500-test.XXXXXXXX)

cleanup() {
  case "$temp_root" in
    /tmp/xh-202628-c500-test.*) rm -rf -- "$temp_root" ;;
    *) printf 'refusing to remove unexpected test path: %s\n' "$temp_root" >&2 ;;
  esac
}
trap cleanup EXIT

base="$temp_root/mirror"
workflow_fixture="$temp_root/workflow-fixture"
candidate_fixture="$temp_root/candidate-fixture"
baseline_fixture="$temp_root/baseline-fixture"
fake_bin="$temp_root/fake-bin"
fault_bin="$temp_root/fault-bin"
maca_path="$temp_root/maca"
candidate_commit=0123456789abcdef0123456789abcdef01234567
baseline_commit=89abcdef0123456789abcdef0123456789abcdef
workflow_commit=abcdef0123456789abcdef0123456789abcdef01
source_path=operators/example/submission.cu
entrypoint=remote-jobs/c500-fixture.sh

mkdir -p -- \
  "$base/incoming" "$base/locks" "$base/runs" \
  "$workflow_fixture/scripts" "$workflow_fixture/remote-jobs" \
  "$workflow_fixture/operators/example" "$candidate_fixture/operators/example" \
  "$baseline_fixture/operators/example" "$fake_bin" "$fault_bin" \
  "$maca_path/tools/cu-bridge/bin" "$maca_path/mxgpu_llvm/bin"
printf 'xh-202628-c500-execution-mirror-v1\n' > "$base/.xh-202628-c500-execution-mirror"
cp -- \
  "$repo_root/scripts/c500-runner.sh" \
  "$repo_root/scripts/run-c500-fused-moe-paired.sh" \
  "$repo_root/scripts/summarize-c500-abba.py" \
  "$workflow_fixture/scripts/"
printf 'workflow placeholder\n' > "$workflow_fixture/$source_path"
printf 'candidate\n' > "$candidate_fixture/$source_path"
printf 'baseline\n' > "$baseline_fixture/$source_path"

cat > "$workflow_fixture/$entrypoint" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

[[ -z "${C500_SECRET_SENTINEL+x}" ]]

emit_benchmark() {
  local value=$1
  for case_name in decode-gate-up prefill-gate-up decode-down prefill-down; do
    printf 'BENCHMARK device=c500-local case=%s samples_ms=%s,%s,%s,%s,%s median_ms=%s effective_TOPS=1.000 sampled_correctness=PASS\n' \
      "$case_name" "$value" "$value" "$value" "$value" "$value" "$value"
  done
}

: > "$C500_RESULTS_DIR/build-candidate.log"
: > "$C500_RESULTS_DIR/build-baseline.log"
printf 'fixture commands\n' > "$C500_RESULTS_DIR/commands.txt"
printf 'CORRECTNESS PASS\n' > "$C500_RESULTS_DIR/correctness.log"
printf 'REGRESSION PASS\n' > "$C500_RESULTS_DIR/regression.log"
emit_benchmark 10.000 > "$C500_RESULTS_DIR/warmup-baseline.log"
emit_benchmark 9.000 > "$C500_RESULTS_DIR/warmup-candidate.log"
emit_benchmark 10.000 > "$C500_RESULTS_DIR/benchmark-baseline-a.log"
emit_benchmark 9.000 > "$C500_RESULTS_DIR/benchmark-candidate-a.log"
emit_benchmark 9.100 > "$C500_RESULTS_DIR/benchmark-candidate-b.log"
emit_benchmark 10.100 > "$C500_RESULTS_DIR/benchmark-baseline-b.log"
python3 "$C500_CANDIDATE_SOURCE/scripts/summarize-c500-abba.py" \
  --baseline-a "$C500_RESULTS_DIR/benchmark-baseline-a.log" \
  --candidate-a "$C500_RESULTS_DIR/benchmark-candidate-a.log" \
  --candidate-b "$C500_RESULTS_DIR/benchmark-candidate-b.log" \
  --baseline-b "$C500_RESULTS_DIR/benchmark-baseline-b.log" \
  --output "$C500_RESULTS_DIR/paired-benchmark.json"
printf 'fixture C500 benchmark succeeded\n'
SH
chmod 700 -- "$workflow_fixture/$entrypoint"

cat > "$fake_bin/mx-smi" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case " $* " in
  *" -j "*)
    compute=${FAKE_COMPUTE_QUOTA:-25}
    printf 'mx-smi version: test\n'
    printf '{"attached_gpus":1,"0000:00:00.0":{"device_id":0,"utilization":0,"sgpu_info":[{"utilization":"0%%","vram_used":"0","compute_quota":%s,"vram_quota":16000}]}}\n' "$compute"
    ;;
  *" --show-process "*)
    if [[ "${FAKE_BUSY:-0}" == 1 ]]; then
      printf '| 0 4242 busy-process 128 |\n'
    else
      printf '| no process found |\n'
    fi
    ;;
  *) printf 'fake mx-smi %s\n' "$*" ;;
esac
SH
chmod 700 -- "$fake_bin/mx-smi"
cp -- "$fake_bin/mx-smi" "$maca_path/mxgpu_llvm/bin/mx-smi"

cat > "$fault_bin/mkdir" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
count=0
[[ ! -f "$FAKE_MKDIR_COUNT" ]] || count=$(<"$FAKE_MKDIR_COUNT")
count=$((count + 1))
printf '%s\n' "$count" > "$FAKE_MKDIR_COUNT"
[[ "$count" -ne 2 ]] || exit 70
exec /usr/bin/mkdir "$@"
SH
chmod 700 -- "$fault_bin/mkdir"

cat > "$maca_path/tools/cu-bridge/bin/cucc" <<'SH'
#!/usr/bin/env bash
printf 'fake cucc\n'
SH
cat > "$maca_path/mxgpu_llvm/bin/mxcc" <<'SH'
#!/usr/bin/env bash
printf 'fake mxcc\n'
SH
chmod 700 -- "$maca_path/tools/cu-bridge/bin/cucc" "$maca_path/mxgpu_llvm/bin/mxcc"

stage_run() {
  local run_id=$1
  local workflow_source=${2:-$workflow_fixture}
  local mkdir_mode=${3:-normal}
  local workflow_archive="$base/incoming/$run_id.workflow.tar"
  local candidate_archive="$base/incoming/$run_id.candidate-source.tar"
  local baseline_archive="$base/incoming/$run_id.baseline-source.tar"
  local stage_script="$base/incoming/$run_id.stage.sh"
  local workflow_sha candidate_archive_sha baseline_archive_sha stage_sha candidate_sha baseline_sha

  tar -cf "$workflow_archive" -C "$workflow_source" .
  mkdir -p -- "$candidate_fixture/remote-jobs"
  cp -- "$workflow_fixture/$entrypoint" "$candidate_fixture/$entrypoint"
  tar -cf "$candidate_archive" -C "$candidate_fixture" "$source_path" "$entrypoint"
  tar -cf "$baseline_archive" -C "$baseline_fixture" "$source_path"
  cp -- "$repo_root/scripts/c500-stage.sh" "$stage_script"
  workflow_sha=$(sha256sum "$workflow_archive" | awk '{print $1}')
  candidate_archive_sha=$(sha256sum "$candidate_archive" | awk '{print $1}')
  baseline_archive_sha=$(sha256sum "$baseline_archive" | awk '{print $1}')
  stage_sha=$(sha256sum "$stage_script" | awk '{print $1}')
  candidate_sha=$(sha256sum "$candidate_fixture/$source_path" | awk '{print $1}')
  baseline_sha=$(sha256sum "$baseline_fixture/$source_path" | awk '{print $1}')
  stage_args=(
    "$base" "$run_id" "$candidate_commit" "$baseline_commit" "$workflow_commit"
    "$source_path" "$entrypoint" xh-202628-c500-execution-mirror-v1
    "$workflow_sha" "$candidate_archive_sha" "$baseline_archive_sha" "$stage_sha"
    "$candidate_sha" "$baseline_sha"
  )
  if [[ "$mkdir_mode" == fail-second ]]; then
    env FAKE_MKDIR_COUNT="$temp_root/mkdir-count-$run_id" PATH="$fault_bin:$PATH" \
      bash "$stage_script" "${stage_args[@]}" >/dev/null
  else
    bash "$stage_script" "${stage_args[@]}" >/dev/null
  fi
}

run_fixture() {
  local run_id=$1
  shift
  env "$@" C500_SECRET_SENTINEL=must-not-reach-candidate PATH="$fake_bin:$PATH" \
    bash "$base/runs/$run_id/source/scripts/c500-runner.sh" \
      "$base" "$run_id" "$entrypoint" "$candidate_commit" "$baseline_commit" \
      "$workflow_commit" 5 512 30 25 16000 "$maca_path"
}

success_id=exp-20260823-999-0123456789ab-a01
stage_run "$success_id"
run_fixture "$success_id"

grep -q '"state": "succeeded"' "$base/runs/$success_id/results/status.json"
grep -q 'fixture C500 benchmark succeeded' "$base/runs/$success_id/results/stdout.log"
grep -q 'candidate' "$base/runs/$success_id/source/$source_path"
grep -q 'baseline' "$base/runs/$success_id/baseline/$source_path"
grep -q '"prefill-gate-up"' "$base/runs/$success_id/results/paired-benchmark.json"
[[ -f "$base/runs/$success_id/workflow-source.tar" ]]
[[ -f "$base/runs/$success_id/candidate-source-overlay.tar" ]]
[[ -f "$base/runs/$success_id/baseline-source-overlay.tar" ]]
[[ -s "$base/runs/$success_id/results/result-manifest.sha256" ]]
[[ ! -e "$base/locks/c500-run-slot.lock" ]]

set +e
run_fixture "$success_id"
duplicate_status=$?
set -e
[[ "$duplicate_status" -eq 75 ]]
grep -q '"state": "succeeded"' "$base/runs/$success_id/results/status.json"

busy_id=exp-20260823-998-0123456789ab-a01
stage_run "$busy_id"
set +e
run_fixture "$busy_id" FAKE_BUSY=1
busy_status=$?
set -e

[[ "$busy_status" -eq 125 ]]
grep -q '"state": "preflight-failed"' "$base/runs/$busy_id/results/status.json"
grep -q 'already has a device process' "$base/runs/$busy_id/results/preflight-error.txt"
[[ ! -e "$base/locks/c500-run-slot.lock" ]]

quota_id=exp-20260823-997-0123456789ab-a01
stage_run "$quota_id"
set +e
run_fixture "$quota_id" FAKE_COMPUTE_QUOTA=50
quota_status=$?
set -e

[[ "$quota_status" -eq 125 ]]
grep -q 'compute quota differs' "$base/runs/$quota_id/results/preflight-error.txt"
[[ ! -e "$base/locks/c500-run-slot.lock" ]]
grep -q -- '-resource-usage' "$repo_root/scripts/run-c500-fused-moe-paired.sh"

unsafe_workflow="$temp_root/unsafe-workflow"
cp -a -- "$workflow_fixture" "$unsafe_workflow"
ln -s /tmp "$unsafe_workflow/build"
unsafe_id=exp-20260823-996-0123456789ab-a01
set +e
stage_run "$unsafe_id" "$unsafe_workflow"
unsafe_status=$?
set -e

[[ "$unsafe_status" -ne 0 ]]
[[ ! -e "$base/runs/.staging-$unsafe_id" ]]
grep -q '"state": "staging-failed"' "$base/runs/$unsafe_id/results/status.json"

early_mkdir_id=exp-20260823-995-0123456789ab-a01
set +e
stage_run "$early_mkdir_id" "$workflow_fixture" fail-second
early_mkdir_status=$?
set -e

[[ "$early_mkdir_status" -eq 70 ]]
[[ ! -e "$base/runs/.staging-$early_mkdir_id" ]]
grep -q '"state": "staging-failed"' "$base/runs/$early_mkdir_id/results/status.json"

printf 'PASS: C500 atomic staging, integrity, runner and evidence tests\n'
