#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260822-023"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'template <bool kCooperativeEpilogue>' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<true><<<grid, block>>>' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<false><<<grid, block>>>' "$submission_file"
grep -Fq 'scale-global-bytes=' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")
prefix_start = text.index("    const int wave = tid / kMmaWaveSize;")
prefix_end = text.index("    float weights[2][4];", prefix_start)
prefix_sha256 = hashlib.sha256(text[prefix_start:prefix_end].encode()).hexdigest()
if prefix_sha256 != "e9124c21fe506ac93878eab36d04e13867c74a1d265a9ff5478a941b63241216":
    sys.exit("SOURCE CHECK FAIL: best-baseline matrix core changed: " + prefix_sha256)

required = (
    "return config.em == 32768 && config.n == 4096 && config.k == 7168;",
    "constexpr int kMmaEpilogueScaleBytes =",
    "(kMmaTileM + kMmaTileN) * sizeof(float);",
    "float* shared_row_scale = reinterpret_cast<float*>(shared_data);",
    "float* shared_col_scale = shared_row_scale + kMmaTileM;",
    "float combined_row_scale = row_scale_value * row_weight;",
    "XH_MMA_STS(shared_row_scale[tid], combined_row_scale, float);",
    "XH_MMA_STS(shared_col_scale[tid], col_scale_value, float);",
    "shared_row_scale[output_row[0] - row_base]",
    "shared_row_scale[output_row[4] - row_base]",
    "shared_col_scale[output_col[0]]",
    "shared_col_scale[output_col[1]]",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: cooperative epilogue token missing: " + token)

if text.count("__shared__ int8_t shared_data[kMmaSharedBytes];") != 1:
    sys.exit("SOURCE CHECK FAIL: candidate must retain one 32 KiB shared allocation")
if text.count("fused_moe_i8_tn_mma_kernel<true><<<grid, block>>>") != 1:
    sys.exit("SOURCE CHECK FAIL: case 2 must have one cooperative launch")
if text.count("fused_moe_i8_tn_mma_kernel<false><<<grid, block>>>") != 1:
    sys.exit("SOURCE CHECK FAIL: baseline shapes must have one original launch")

block_start = text.index("    if (kCooperativeEpilogue) {")
block_end = text.index("    } else {", block_start)
block = text[block_start:block_end]
if block.count("__syncthreadshared();") != 2:
    sys.exit("SOURCE CHECK FAIL: LDS reuse requires exactly two CTA barriers")
if block.count("__builtin_mxc_ldg_b32(") != 3:
    sys.exit("SOURCE CHECK FAIL: cooperative path must issue three scalar loads per producer")
if block.count("XH_MMA_STS(") != 2 or block.count("XH_MMA_LDS(") != 4:
    sys.exit("SOURCE CHECK FAIL: cooperative LDS write/read structure changed")

print("SOURCE CHECK PASS: case-2 scales 24 KiB->1.5 KiB; baseline matrix core exact")
PYEOF

printf 'BUILD exp-20260822-023 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
