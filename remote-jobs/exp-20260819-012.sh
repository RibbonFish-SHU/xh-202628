#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260819-012"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<32768, 4096, 7168><<<grid, block>>>' "$submission_file"
grep -Fq 'load_b[i] = __builtin_mxc_ldg_b128(' "$submission_file"
grep -Fq '__builtin_mxc_ldg_b32(' "$submission_file"
grep -Fq '__builtin_mxc_stg_b64_predicator(' "$submission_file"
if grep -Eq '__builtin_mxc_ldg_[a-zA-Z0-9_]*predicator' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: a redundant exact-tile load predicate remains\n' >&2
  exit 1
fi
if grep -Eq 'col_limit|k_head|output_col_mask' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: redundant exact-tile boundary bookkeeping remains\n' >&2
  exit 1
fi
if grep -q '__dp4a' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: unsupported __dp4a remains\n' >&2
  exit 1
fi
if grep -q 'wide_mma' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: rejected wide MMA path remains\n' >&2
  exit 1
fi
printf 'SOURCE CHECK PASS: remaining exact-tile loads and boundary checks are unpredicated\n'

printf 'BUILD exp-20260819-012 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
