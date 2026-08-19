#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260819-013"
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

# exp-20260819-013: the six hoistable next-tile loads (B0..B3, A0..A1) must all
# be issued at the top of the K loop, before the first MMA of the body, while
# A2/A3 stay behind their cross-iteration register rotation point.
python3 - "$submission_file" <<'PYEOF'
import re
import sys

text = open(sys.argv[1], encoding='utf-8').read()
loop = re.search(
    r'for \(uint32_t tile_k = 0; tile_k < loop_k_tiles; \+\+tile_k\) \{(.*?)\n\}',
    text,
    re.S,
)
if not loop:
    sys.exit('SOURCE CHECK FAIL: K loop not found')
body = loop.group(1)
first_mma = body.index('XH_MMA_STAGE_MNKX2(0, 0, 0)')
head = body[:first_mma]
for token in (
    'XH_LDG_B_STAGE_I(0)',
    'XH_LDG_B_STAGE_I(1)',
    'XH_LDG_B_STAGE_I(2)',
    'XH_LDG_B_STAGE_I(3)',
    'XH_LDG_A_STAGE_I(0)',
    'XH_LDG_A_STAGE_I(1)',
):
    if token not in head:
        sys.exit(f'SOURCE CHECK FAIL: {token} is not hoisted to the K loop head')
for token in ('XH_LDG_A_STAGE_I(2)', 'XH_LDG_A_STAGE_I(3)'):
    if token in head:
        sys.exit(f'SOURCE CHECK FAIL: {token} must stay behind its rotation point')
if body.count('XH_LDG_B_STAGE_I(') != 4 or body.count('XH_LDG_A_STAGE_I(') != 4:
    sys.exit('SOURCE CHECK FAIL: loop must keep exactly four A and four B staged loads')
print('SOURCE CHECK PASS: six hoistable next-tile loads issued at K loop head')
PYEOF

printf 'BUILD exp-20260819-013 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
