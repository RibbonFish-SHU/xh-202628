#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"

# Copy this file to remote-jobs/exp-YYYYMMDD-NNN.sh and replace the command
# below with the exact build, correctness, benchmark, and regression sequence.
printf 'remote job template has not been configured\n' >&2
exit 64
