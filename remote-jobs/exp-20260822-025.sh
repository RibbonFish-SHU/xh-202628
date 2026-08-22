#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260822-025"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'template <bool kPairAdjacentM>' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<true><<<grid, block>>>' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<false><<<grid, block>>>' "$submission_file"
grep -Fq 'shared-boundary=PASS barrier-count=1/2' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")
core_start = text.index("    Tensor matrix_b = make_tensor(")
core_end = text.index("    int output_row[8];", core_start)
core_sha256 = hashlib.sha256(text[core_start:core_end].encode()).hexdigest()
if core_sha256 != "62ca88a6c2d98c493d1d73d1cfa037db6198c1be41a0ed7cdd88b9ff59127f2e":
    sys.exit("SOURCE CHECK FAIL: best-baseline K pipeline changed: " + core_sha256)

required = (
    "return config.em == 32768 && config.n == 4096 && config.k == 7168;",
    "const int physical_tile_m = blockIdx.x + blockIdx.z * gridDim.x;",
    "mma_adjacent_m_is_follower(",
    "mma_adjacent_m_tile_count(",
    "for (int paired_tile = 0; paired_tile < tile_count; ++paired_tile)",
    "const int tile_m = physical_tile_m + paired_tile;",
    "const int expert = first_expert;",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: adjacent-M pair token missing: " + token)

barrier = """    if (mma_adjacent_m_needs_shared_barrier(paired_tile, tile_count)) {
        __syncthreadshared();
    }
"""
if text.count(barrier) != 1:
    sys.exit("SOURCE CHECK FAIL: expected one paired-loop shared barrier")
last_lds = text.rindex("    XH_MMA_STAGE_MNKX2(1, 7, 6);")
barrier_pos = text.index(barrier)
epilogue_pos = text.index("    MmaInt4 output[kMmaOutputVectors];")
if not last_lds < barrier_pos < epilogue_pos:
    sys.exit("SOURCE CHECK FAIL: paired-loop barrier is not after final LDS use")
if text.count("__syncthreadshared();") != 4:
    sys.exit("SOURCE CHECK FAIL: unexpected target CTA barrier count")

if text.count("__shared__ int8_t shared_data[kMmaSharedBytes];") != 1:
    sys.exit("SOURCE CHECK FAIL: candidate must retain one 32 KiB shared allocation")
if text.count("fused_moe_i8_tn_mma_kernel<true><<<grid, block>>>") != 1:
    sys.exit("SOURCE CHECK FAIL: case 2 must have one paired launch")
if text.count("fused_moe_i8_tn_mma_kernel<false><<<grid, block>>>") != 1:
    sys.exit("SOURCE CHECK FAIL: baseline shapes must have one original launch")

print("SOURCE CHECK PASS: race-free case-2 adjacent-M pair; baseline K pipeline exact")
PYEOF

printf 'BUILD exp-20260822-025 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
