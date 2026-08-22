#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260822-021"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'template <bool kCase2L2Schedule>' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<true><<<l2_grid, block>>>' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<false><<<grid, block>>>' "$submission_file"
grep -Fq 'const int tile_m = kCase2L2Schedule' "$submission_file"
grep -Fq '? blockIdx.z * kCase2L2NGroup + blockIdx.x' "$submission_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")
core_start = text.index("    const int wave = tid / kMmaWaveSize;")
core_end = text.index("\n}\n\n#else", core_start)
core_sha256 = hashlib.sha256(text[core_start:core_end].encode()).hexdigest()
if core_sha256 != "9aa8a43d86a4b7690ee928a7bd8e2dfc078e03a8fae9a7aa20052ca8decbcc47":
    sys.exit("SOURCE CHECK FAIL: MMA body after tile mapping changed: " + core_sha256)

required = (
    "constexpr int kCase2L2NGroup = 2;",
    "return config.em == 32768 && config.n == 4096 && config.k == 7168;",
    "? blockIdx.y",
    ": blockIdx.x + blockIdx.z * gridDim.x;",
    "? blockIdx.z * kCase2L2NGroup + blockIdx.x",
    ": blockIdx.y;",
    "const dim3 l2_grid(",
    "kCase2L2NGroup,\n            grid_m,",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: case-2 L2 schedule token missing: " + token)

print("SOURCE CHECK PASS: baseline MMA body exact; only case 2 uses grid (2,256,16)")
PYEOF

printf 'BUILD exp-20260822-021 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
