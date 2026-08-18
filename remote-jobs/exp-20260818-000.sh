#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/remote-smoke"
mkdir -p -- "$build_dir"

"$CUDA_HOME/bin/nvcc" \
  -O2 \
  -std=c++14 \
  -arch=sm_86 \
  "$repo_root/smoke/remote_cuda_smoke.cu" \
  -o "$build_dir/remote-cuda-smoke"

"$build_dir/remote-cuda-smoke"
