#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 10 ]]; then
  printf 'usage: remote-runner.sh BASE RUN_ID ENTRYPOINT COMMIT GPU_IDS MAX_UTIL MAX_MEM MAX_SECONDS MAX_PARALLEL CUDA_HOME\n' >&2
  exit 64
fi

base=$1
run_id=$2
entrypoint=$3
commit=$4
gpu_ids=$5
max_util=$6
max_mem=$7
max_seconds=$8
max_parallel=$9
cuda_home=${10}

[[ "$base" =~ ^/[A-Za-z0-9._/-]+$ ]] || exit 64
[[ "$base" != *"/../"* && "$base" != *"/./"* ]] || exit 64
[[ "$run_id" =~ ^exp-[0-9]{8}-[0-9]{3}-[0-9a-f]{12}$ ]] || exit 64
[[ "$entrypoint" =~ ^remote-jobs/[A-Za-z0-9._/-]+[.]sh$ ]] || exit 64
[[ "$entrypoint" != *"../"* ]] || exit 64
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 64
[[ "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 64
[[ "$max_util" =~ ^[0-9]+$ ]] || exit 64
[[ "$max_mem" =~ ^[0-9]+$ ]] || exit 64
[[ "$max_seconds" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "$max_parallel" =~ ^[1-8]$ ]] || exit 64
[[ "$cuda_home" =~ ^/[A-Za-z0-9._/-]+$ ]] || exit 64

run_dir="$base/runs/$run_id"
source_dir="$run_dir/source"
results_dir="$run_dir/results"
status_file="$results_dir/status.json"
entrypoint_path="$source_dir/$entrypoint"

mkdir -p -- "$results_dir"

write_status() {
  local state=$1
  local exit_code=$2
  printf '{\n  "schema_version": 1,\n  "run_id": "%s",\n  "commit": "%s",\n  "state": "%s",\n  "exit_code": %s\n}\n' \
    "$run_id" "$commit" "$state" "$exit_code" > "$status_file"
}

fail_preflight() {
  local message=$1
  printf '%s\n' "$message" > "$results_dir/preflight-error.txt"
  write_status "preflight-failed" 125
  exit 125
}

grep -qx 'xh-202628-execution-mirror-v1' "$base/.xh-202628-execution-mirror" || fail_preflight "execution mirror marker is missing"
[[ -f "$entrypoint_path" && ! -L "$entrypoint_path" ]] || fail_preflight "committed entrypoint is missing or is a symlink"
[[ "$(<"$run_dir/source-commit.txt")" == "$commit" ]] || fail_preflight "staged commit metadata does not match"
command -v nvidia-smi >/dev/null 2>&1 || fail_preflight "nvidia-smi is unavailable"
command -v timeout >/dev/null 2>&1 || fail_preflight "timeout is unavailable"
[[ -x "$cuda_home/bin/nvcc" ]] || fail_preflight "approved CUDA nvcc is unavailable"

declare -a acquired_locks=()
cleanup_locks() {
  local lock
  for lock in "${acquired_locks[@]}"; do
    rmdir -- "$lock" 2>/dev/null || true
  done
}
trap cleanup_locks EXIT

run_slot=
for ((slot = 1; slot <= max_parallel; slot++)); do
  candidate="$base/locks/run-slot-$slot.lock"
  if mkdir -- "$candidate" 2>/dev/null; then
    run_slot=$candidate
    acquired_locks+=("$candidate")
    break
  fi
done
[[ -n "$run_slot" ]] || fail_preflight "all approved project run slots are active or stale locks need review"

printf 'index,name,uuid,utilization.gpu,memory.used,memory.total\n' > "$results_dir/gpu-before.csv"
IFS=',' read -r -a gpu_array <<< "$gpu_ids"
for gpu_id in "${gpu_array[@]}"; do
  gpu_lock="$base/locks/gpu-$gpu_id.lock"
  mkdir -- "$gpu_lock" 2>/dev/null || fail_preflight "GPU $gpu_id is locked by another project run"
  acquired_locks+=("$gpu_lock")

  snapshot=$(nvidia-smi -i "$gpu_id" --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>>"$results_dir/preflight-stderr.log") \
    || fail_preflight "could not inspect GPU $gpu_id"
  printf '%s\n' "$snapshot" >> "$results_dir/gpu-before.csv"
  IFS=',' read -r _index _name _uuid utilization memory_used _memory_total <<< "$snapshot"
  utilization=$(printf '%s' "$utilization" | tr -d '[:space:]')
  memory_used=$(printf '%s' "$memory_used" | tr -d '[:space:]')
  [[ "$utilization" =~ ^[0-9]+$ && "$memory_used" =~ ^[0-9]+$ ]] \
    || fail_preflight "could not parse utilization for GPU $gpu_id"
  (( utilization <= max_util )) || fail_preflight "GPU $gpu_id utilization exceeds the approved start threshold"
  (( memory_used <= max_mem )) || fail_preflight "GPU $gpu_id memory use exceeds the approved start threshold"

  process_ids=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
  [[ ! "$process_ids" =~ [0-9] ]] || fail_preflight "GPU $gpu_id already has a compute process"
done

{
  printf 'run_id=%s\ncommit=%s\nentrypoint=%s\ngpu_ids=%s\n' "$run_id" "$commit" "$entrypoint" "$gpu_ids"
  date --iso-8601=seconds
  uname -a
  lscpu
  free -h
  for gpu_id in "${gpu_array[@]}"; do
    nvidia-smi -i "$gpu_id"
  done
  "$cuda_home/bin/nvcc" --version
  python3 --version
  git --version
  cmake --version
} > "$results_dir/environment.txt" 2>&1

printf '%s\n' "bash $entrypoint" > "$results_dir/remote-command.txt"
date --iso-8601=seconds > "$results_dir/started-at.txt"
start_ns=$(date +%s%N)
write_status "running" 0

export CUDA_VISIBLE_DEVICES=$gpu_ids
export CUDA_HOME=$cuda_home
export PATH="$cuda_home/bin:$PATH"

(
  cd -- "$source_dir"
  timeout --signal=TERM --kill-after=30s "${max_seconds}s" bash "$entrypoint_path"
) > "$results_dir/stdout.log" 2> "$results_dir/stderr.log"
entrypoint_status=$?

end_ns=$(date +%s%N)
date --iso-8601=seconds > "$results_dir/finished-at.txt"
printf '%s\n' "$((end_ns - start_ns))" > "$results_dir/duration-ns.txt"
printf 'index,name,uuid,utilization.gpu,memory.used,memory.total\n' > "$results_dir/gpu-after.csv"
for gpu_id in "${gpu_array[@]}"; do
  nvidia-smi -i "$gpu_id" --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
    >> "$results_dir/gpu-after.csv" 2>&1 || true
done

if [[ "$entrypoint_status" -eq 0 ]]; then
  write_status "succeeded" 0
else
  write_status "failed" "$entrypoint_status"
fi
exit "$entrypoint_status"
