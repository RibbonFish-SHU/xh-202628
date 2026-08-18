#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
temp_root=$(mktemp -d /tmp/xh-202628-remote-test.XXXXXXXX)

cleanup() {
  case "$temp_root" in
    /tmp/xh-202628-remote-test.*) rm -rf -- "$temp_root" ;;
    *) printf 'refusing to remove unexpected test path: %s\n' "$temp_root" >&2 ;;
  esac
}
trap cleanup EXIT

base="$temp_root/mirror"
fixture="$temp_root/fixture"
fake_bin="$temp_root/fake-bin"
cuda_home="$temp_root/cuda"
commit=0123456789abcdef0123456789abcdef01234567

mkdir -p -- "$base/incoming" "$base/locks" "$base/runs" "$fixture/scripts" "$fixture/remote-jobs" "$fake_bin" "$cuda_home/bin"
printf 'xh-202628-execution-mirror-v1\n' > "$base/.xh-202628-execution-mirror"
cp -- "$repo_root/scripts/remote-runner.sh" "$fixture/scripts/remote-runner.sh"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'printf "fixture benchmark succeeded\\n"' \
  > "$fixture/remote-jobs/exp-20260818-999.sh"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'case " $* " in' \
  '  *"--query-compute-apps=pid"*) [[ "${FAKE_BUSY:-0}" == 1 ]] && printf "4242\\n" ;;' \
  '  *"--query-gpu="*) printf "0, NVIDIA RTX A5000, GPU-test, 0, 0, 24564\\n" ;;' \
  '  *) printf "fake nvidia-smi\\n" ;;' \
  'esac' \
  > "$fake_bin/nvidia-smi"
chmod 700 -- "$fake_bin/nvidia-smi"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'printf "Cuda compilation tools, release 12.2, V12.2.0\\n"' \
  > "$cuda_home/bin/nvcc"
chmod 700 -- "$cuda_home/bin/nvcc"

stage_run() {
  local run_id=$1
  tar -cf "$base/incoming/$run_id.tar" -C "$fixture" .
  cp -- "$repo_root/scripts/remote-stage.sh" "$base/incoming/$run_id.stage.sh"
  bash "$base/incoming/$run_id.stage.sh" "$base" "$run_id" "$commit" >/dev/null
}

success_id=exp-20260818-999-0123456789ab
stage_run "$success_id"
mkdir -- "$base/locks/run-slot-1.lock"
PATH="$fake_bin:$PATH" bash "$base/runs/$success_id/source/scripts/remote-runner.sh" \
  "$base" "$success_id" remote-jobs/exp-20260818-999.sh "$commit" 0 10 512 30 4 "$cuda_home"

grep -q '"state": "succeeded"' "$base/runs/$success_id/results/status.json"
grep -q 'fixture benchmark succeeded' "$base/runs/$success_id/results/stdout.log"
[[ -f "$base/runs/$success_id/source.tar" ]]
[[ -f "$base/runs/$success_id/stage.sh" ]]
[[ -d "$base/locks/run-slot-1.lock" ]]
[[ ! -e "$base/locks/run-slot-2.lock" ]]
[[ ! -e "$base/locks/gpu-0.lock" ]]
rmdir -- "$base/locks/run-slot-1.lock"

busy_id=exp-20260818-998-0123456789ab
stage_run "$busy_id"
set +e
FAKE_BUSY=1 PATH="$fake_bin:$PATH" bash "$base/runs/$busy_id/source/scripts/remote-runner.sh" \
  "$base" "$busy_id" remote-jobs/exp-20260818-999.sh "$commit" 0 10 512 30 4 "$cuda_home"
busy_status=$?
set -e

[[ "$busy_status" -eq 125 ]]
grep -q '"state": "preflight-failed"' "$base/runs/$busy_id/results/status.json"
grep -q 'already has a compute process' "$base/runs/$busy_id/results/preflight-error.txt"
[[ ! -e "$base/locks/run-slot-1.lock" ]]
[[ ! -e "$base/locks/gpu-0.lock" ]]

slots_id=exp-20260818-997-0123456789ab
stage_run "$slots_id"
for slot in 1 2 3 4; do
  mkdir -- "$base/locks/run-slot-$slot.lock"
done
set +e
PATH="$fake_bin:$PATH" bash "$base/runs/$slots_id/source/scripts/remote-runner.sh" \
  "$base" "$slots_id" remote-jobs/exp-20260818-999.sh "$commit" 0 10 512 30 4 "$cuda_home"
slots_status=$?
set -e

[[ "$slots_status" -eq 125 ]]
grep -q 'all approved project run slots are active' "$base/runs/$slots_id/results/preflight-error.txt"
for slot in 1 2 3 4; do
  rmdir -- "$base/locks/run-slot-$slot.lock"
done

printf 'PASS: remote shell staging and runner tests\n'
