#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Copy this file byte-for-byte to remote-jobs/exp-YYYYMMDD-NNN.sh; do not edit
# it in a candidate lane. The trusted shared driver builds candidate and paired
# baseline trees, then runs correctness, regression, warmup and an ABBA benchmark.
exec bash "$repo_root/scripts/run-c500-fused-moe-paired.sh"
