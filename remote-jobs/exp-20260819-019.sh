#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260819-019"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

# The 82.25-point 128x128 MACA kernel remains the path for three public cases.
grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'XH_LDG_A_STAGE_I' "$submission_file"
grep -Fq 'XH_LDG_B_STAGE_I' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<<<grid, block>>>' "$submission_file"
grep -Fq 'load_a[i] = __builtin_mxc_ldg_b128(' "$submission_file"
grep -Fq 'load_b[i] = __builtin_mxc_ldg_b128_predicator(' "$submission_file"
grep -Fq '__builtin_mxc_ldg_b32_predicator(' "$submission_file"
grep -Fq '__builtin_mxc_stg_b64_predicator(' "$submission_file"

# exp-20260819-019: case 2 dispatches to the 64x128 occupancy kernel with
# grid (2, n / 128, em / 128) = (2, 32, 256). X pairs the two halves
# of one expert group so their duplicated B tile can be reused by cache.
grep -q 'fused_moe_i8_tn_mma_kernel_64' "$submission_file"
grep -Fq 'const dim3 grid_64(' "$submission_file"
grep -Fq 'config.n / kMma64TileN' "$submission_file"
grep -Fq 'config.em / 128' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel_64<<<grid_64, block>>>' "$submission_file"
if grep -q '__dp4a' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: unsupported __dp4a remains\n' >&2
  exit 1
fi
if grep -q 'wide_mma' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: rejected wide MMA path remains\n' >&2
  exit 1
fi
if grep -q 'ldg_b128_bsm' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: unverified ldg_b128_bsm load form remains\n' >&2
  exit 1
fi
if grep -q '__builtin_mxc_arrive' "$submission_file"; then
  printf 'SOURCE CHECK FAIL: unverified arrive barrier remains\n' >&2
  exit 1
fi

# The new kernel must use 64x128x128 tiles with exactly 24 KiB of shared
# memory. Its register-staged K loop keeps consume -> sync -> stage -> sync
# ordering with 32 MMA stage groups per tile.
python3 - "$submission_file" <<'PYEOF'
import hashlib
import re
import sys

text = open(sys.argv[1], encoding='utf-8').read()

baseline_start = text.index('__global__ void fused_moe_i8_tn_mma_kernel(')
baseline_end = text.index('// exp-20260819-019: case-2 occupancy variant.')
baseline_kernel = text[baseline_start:baseline_end].replace('\r\n', '\n').rstrip()
baseline_sha256 = hashlib.sha256(baseline_kernel.encode()).hexdigest()
if baseline_sha256 != 'eab0c946da8403f461122f38cd072b3f95db73e22652545277ee0fb373a58d5f':
    sys.exit(
        'SOURCE CHECK FAIL: non-case-2 kernel diverged from OJ #117114 baseline: '
        + baseline_sha256)


def read_const(name):
    match = re.search(r'constexpr int %s = (\d+);' % name, text)
    if not match:
        sys.exit(f'SOURCE CHECK FAIL: {name} definition not found')
    return int(match.group(1))


tile_m = read_const('kMma64TileM')
tile_n = read_const('kMma64TileN')
tile_k = read_const('kMma64TileK')
if (tile_m, tile_n, tile_k) != (64, 128, 128):
    sys.exit('SOURCE CHECK FAIL: occupancy kernel must use 64x128x128 tiles')
if 'constexpr int kMma64SharedABytes = kMma64TileM * kMma64TileK;' not in text:
    sys.exit('SOURCE CHECK FAIL: kMma64SharedABytes formula missing')
if 'constexpr int kMma64SharedBBytes = kMma64TileN * kMma64TileK;' not in text:
    sys.exit('SOURCE CHECK FAIL: kMma64SharedBBytes formula missing')
if 'constexpr int kMma64SharedBytes = kMma64SharedABytes + kMma64SharedBBytes;' not in text:
    sys.exit('SOURCE CHECK FAIL: kMma64SharedBytes formula missing')
shared_bytes = tile_m * tile_k + tile_n * tile_k
if shared_bytes != 24576:
    sys.exit(f'SOURCE CHECK FAIL: 64x128 smem must be 24 KiB, got {shared_bytes}')
if '__shared__ int8_t shared_data[kMma64SharedBytes];' not in text:
    sys.exit('SOURCE CHECK FAIL: 64x128 kernel must allocate kMma64SharedBytes')

kernel = re.search(
    r'__global__ void fused_moe_i8_tn_mma_kernel_64\(', text)
if not kernel:
    sys.exit('SOURCE CHECK FAIL: fused_moe_i8_tn_mma_kernel_64 not found')
body = text[kernel.start():]
loop = re.search(
    r'for \(uint32_t tile_k = 0; tile_k < num_k_tiles; \+\+tile_k\) \{(.*?)\n    \}',
    body,
    re.S,
)
if not loop:
    sys.exit('SOURCE CHECK FAIL: 64x64 serial K loop not found')
loop_body = loop.group(1)
if loop_body.count('XH_MMA64_STAGE_MNKX2(') != 32:
    sys.exit('SOURCE CHECK FAIL: 64x128 tile must run exactly 32 MMA stage groups')
if loop_body.count('XH_MMA64_LDS_A_B128(') != 2:
    sys.exit('SOURCE CHECK FAIL: 64x64 tile must run exactly 2 A LDS loads')
if loop_body.count('XH_MMA64_LDS_B_B128(') != 16:
    sys.exit('SOURCE CHECK FAIL: 64x128 tile must run exactly 16 B LDS loads')
if loop_body.count('__syncthreadshared()') != 2:
    sys.exit('SOURCE CHECK FAIL: 64x64 stage handoff must keep exactly two syncs')
stage = re.search(
    r'if \(tile_k \+ 1 < num_k_tiles\) \{(.*?)\n        \}', loop_body, re.S)
if not stage:
    sys.exit('SOURCE CHECK FAIL: 64x64 next-tile staging block not found')
stage_body = stage.group(1)
first_sync = stage_body.index('__syncthreadshared()')
first_sts = stage_body.index('XH_MMA_STS(')
last_sts = stage_body.rindex('XH_MMA_STS(')
last_sync = stage_body.rindex('__syncthreadshared()')
if not (first_sync < first_sts < last_sts < last_sync):
    sys.exit('SOURCE CHECK FAIL: staging must be sync -> STS -> sync')
for token in (
    'XH_MMA64_LDG_A_STAGE_I(0)',
    'XH_MMA64_LDG_A_STAGE_I(1)',
    'XH_MMA64_LDG_B_STAGE_I(0, tile_k + 1)',
    'XH_MMA64_LDG_B_STAGE_I(1, tile_k + 1)',
    'XH_MMA64_LDG_B_STAGE_I(2, tile_k + 1)',
    'XH_MMA64_LDG_B_STAGE_I(3, tile_k + 1)',
):
    if token not in stage_body[:first_sync]:
        sys.exit(f'SOURCE CHECK FAIL: {token} must issue before the first sync')
if 'int8_t* a_stage = const_cast<int8_t*>(a_ptr);' not in body:
    sys.exit('SOURCE CHECK FAIL: A load pointer must remain mutable for MXMACA builtins')
if 'const int8_t* a_stage' in body:
    sys.exit('SOURCE CHECK FAIL: const A load pointer would repeat OJ #117017 CE')
print('SOURCE CHECK PASS: 64x128 kernel is 24 KiB, 32 MMA stages, mutable A loads')
PYEOF

printf 'BUILD exp-20260819-019 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
