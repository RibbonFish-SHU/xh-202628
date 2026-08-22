#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  printf 'usage: remote-stage.sh BASE RUN_ID COMMIT\n' >&2
  exit 64
fi

base=$1
run_id=$2
commit=$3

[[ "$base" =~ ^/[A-Za-z0-9._/-]+$ ]] || { printf 'unsafe base path\n' >&2; exit 64; }
[[ "$base" != *"/../"* && "$base" != *"/./"* ]] || { printf 'unsafe base path\n' >&2; exit 64; }
[[ "$run_id" =~ ^exp-[0-9]{8}-[0-9]{3}-[0-9a-f]{12}(-a[0-9]{2})?$ ]] || { printf 'unsafe run id\n' >&2; exit 64; }
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { printf 'unsafe commit\n' >&2; exit 64; }

marker="$base/.xh-202628-execution-mirror"
archive="$base/incoming/$run_id.tar"
stage_script="$base/incoming/$run_id.stage.sh"
run_dir="$base/runs/$run_id"

grep -qx 'xh-202628-execution-mirror-v1' "$marker"
[[ -f "$archive" ]] || { printf 'archive is missing\n' >&2; exit 66; }
[[ -f "$stage_script" ]] || { printf 'stage script is missing\n' >&2; exit 66; }
[[ ! -e "$run_dir" ]] || { printf 'run directory already exists\n' >&2; exit 73; }

umask 077
mkdir -m 700 -- "$run_dir"
mkdir -m 700 -- "$run_dir/source" "$run_dir/results"
if tar -tf "$archive" | grep -Eq '(^/|(^|/)[.][.](/|$))'; then
  printf 'archive contains an unsafe path\n' >&2
  exit 65
fi
tar --no-same-owner --no-same-permissions -xf "$archive" -C "$run_dir/source"

printf '%s\n' "$commit" > "$run_dir/source-commit.txt"
printf '%s\n' "$run_id" > "$run_dir/run-id.txt"
date --iso-8601=seconds > "$run_dir/staged-at.txt"

mv -- "$archive" "$run_dir/source.tar"
mv -- "$stage_script" "$run_dir/stage.sh"
printf '%s\n' "$run_dir"
