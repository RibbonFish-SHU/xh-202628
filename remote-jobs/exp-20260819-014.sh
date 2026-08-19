#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260819-014"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<32768, 4096, 7168><<<grid, block>>>' "$submission_file"
grep -Fq '__builtin_mxc_ldg_b128_bsm(' "$submission_file"
grep -Fq '__builtin_mxc_arrive(64 + 8)' "$submission_file"
grep -Fq '__builtin_mxc_barrier_inst()' "$submission_file"
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

# exp-20260819-014: G2S must use bsm async copies straight into a double-buffered
# 64 KiB shared region; the register-staged LDG/STS path must be gone, while the
# per-tile LDS/MMA interleave is preserved exactly once inside XH_MMA_CONSUME_TILE.
python3 - "$submission_file" <<'PYEOF'
import re
import sys

text = open(sys.argv[1], encoding='utf-8').read()

required = (
    '2 * kMmaSharedBytes',
    '__builtin_mxc_ldg_b128_bsm(',
    '__builtin_mxc_arrive(64 + 8);',
    '__builtin_mxc_arrive(64);',
    '__builtin_mxc_barrier_inst();',
    'XH_MMA_CONSUME_TILE();',
)
for token in required:
    if token not in text:
        sys.exit(f'SOURCE CHECK FAIL: missing {token}')

forbidden = (
    'XH_LDG_A_STAGE_I',
    'XH_LDG_B_STAGE_I',
    'XH_MMA_STS(',
    'load_a[',
    'load_b[',
    'MmaLoad128 load_a',
    'MmaLoad128 load_b',
)
for token in forbidden:
    if token in text:
        sys.exit(f'SOURCE CHECK FAIL: removed register-staging artifact remains: {token}')

if text.count('__builtin_mxc_ldg_b128_bsm(') != 2:
    sys.exit('SOURCE CHECK FAIL: expected exactly the A and B bsm copy macros')
if text.count('XH_MMA_CONSUME_TILE();') != 2:
    sys.exit('SOURCE CHECK FAIL: consume must run once in the K loop and once in the peel')
if text.count('XH_MMA_STAGE_MNKX2(') != 65:  # 1 macro definition + 64 per-tile stages
    sys.exit('SOURCE CHECK FAIL: per-tile MMA sequence must keep exactly 64 stages')
if text.count('XH_LDS_A_B128(') != 5 or text.count('XH_LDS_B_B128(') != 17:
    sys.exit('SOURCE CHECK FAIL: per-tile LDS sequence changed')
if text.count('__syncthreadshared()') != 1:
    sys.exit('SOURCE CHECK FAIL: double buffering keeps exactly one loop-end shared barrier')

loop = re.search(
    r'for \(uint32_t tile_k = 0; tile_k < loop_k_tiles; \+\+tile_k\) \{(.*?)\n    \}',
    text,
    re.S,
)
if not loop:
    sys.exit('SOURCE CHECK FAIL: K loop not found')
body = loop.group(1)
pos = -1
for token in (
    'XH_BSM_ISSUE_TILE(tile_k + 1);',
    '__builtin_mxc_arrive(64 + 8);',
    '__builtin_mxc_barrier_inst();',
    'XH_MMA_CONSUME_TILE();',
    '__syncthreadshared();',
):
    nxt = body.find(token, pos + 1)
    if nxt < 0:
        sys.exit(f'SOURCE CHECK FAIL: {token} missing or out of order in K loop')
    pos = nxt

tail = text[text.index('__builtin_mxc_arrive(64);'):]
if tail.index('__builtin_mxc_barrier_inst();') > tail.index('XH_MMA_CONSUME_TILE();'):
    sys.exit('SOURCE CHECK FAIL: peeled tail must arrive/barrier before final consume')

print('SOURCE CHECK PASS: bsm async G2S double buffering with preserved per-tile MMA order')
PYEOF

printf 'BUILD exp-20260819-014 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
