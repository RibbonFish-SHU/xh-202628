#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260819-008"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -q '__builtin_mxc_ldg_b128_bsm' "$submission_file"
grep -Fq 'constexpr int kWideMmaTileN = 256;' "$submission_file"
grep -Fq 'if (use_wide_prefill_gate_up(config)) {' "$submission_file"
grep -Fq 'wide_mma_load_b<7>' "$submission_file"
if grep -q '__dp4a' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: unsupported __dp4a remains\n' >&2
  exit 1
fi
printf 'SOURCE CHECK PASS: prefill gate-up uses the 128x256 G2S MMA path and __dp4a is absent\n'

printf 'BUILD exp-20260819-008 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" \
  -O3 \
  -std=c++14 \
  -arch=sm_86 \
  -lineinfo \
  -Xcompiler=-Wall \
  "$source_file" \
  -lcuda \
  -o "$binary"
printf 'BUILD PASS\n'

"$binary" --correctness
"$binary" --benchmark
"$binary" --regression
