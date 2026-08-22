#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260822-026"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel_n64' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel_n64<<<n64_grid, n64_block>>>' "$submission_file"
grep -Fq 'contiguous-b64-columns=PASS' "$source_file"
grep -Fq 'A-bytes=2x B-bytes=1x' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")

baseline_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
baseline_end = text.index("// exp-20260822-026: case-2 128x64 occupancy candidate.")
baseline_kernel = text[baseline_start:baseline_end].rstrip()
baseline_sha256 = hashlib.sha256(baseline_kernel.encode()).hexdigest()
if baseline_sha256 != "eab0c946da8403f461122f38cd072b3f95db73e22652545277ee0fb373a58d5f":
    sys.exit(
        "SOURCE CHECK FAIL: non-case-2 kernel diverged from OJ #117114 baseline: "
        + baseline_sha256)


def read_const(name):
    match = re.search(r"constexpr int %s = (\d+);" % name, text)
    if not match:
        sys.exit("SOURCE CHECK FAIL: constant not found: " + name)
    return int(match.group(1))


tile_m = read_const("kMmaN64TileM")
tile_n = read_const("kMmaN64TileN")
tile_k = read_const("kMmaN64TileK")
threads = read_const("kMmaN64Threads")
if (tile_m, tile_n, tile_k, threads) != (128, 64, 128, 256):
    sys.exit("SOURCE CHECK FAIL: candidate geometry must be 128x64x128/256")
if tile_m * tile_k + tile_n * tile_k != 24 * 1024:
    sys.exit("SOURCE CHECK FAIL: candidate shared memory must be 24 KiB")

required = (
    "return config.em == 32768 && config.n == 4096 && config.k == 7168;",
    "return 4 * (store_row % 16) + store_row / 16;",
    "MmaInt4 accum[kMmaN64Rows][kMmaN64Cols] = {0};",
    "__shared__ int8_t shared_data[kMmaN64SharedBytes];",
    "const dim3 n64_grid(",
    "config.n / kMmaN64TileN",
    "config.em / kMmaN64TileM",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: N64 token missing: " + token)

candidate_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel_n64(")
candidate_end = text.index("\n#else", candidate_start)
candidate = text[candidate_start:candidate_end]
loop_start = candidate.index(
    "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {")
tail_start = candidate.index(
    "\n    XH_N64_MMA_STAGE_MNKX2(0, 0, 0);", loop_start)
loop = candidate[loop_start:tail_start]

expected_counts = {
    "XH_N64_MMA_STAGE_MNKX2(": 32,
    "XH_N64_LDS_A_B128(": 4,
    "XH_N64_LDS_B_B128(": 8,
    "XH_N64_LDG_A_STAGE_I(": 4,
    "XH_N64_LDG_B_STAGE_I(": 2,
    "__syncthreadshared();": 2,
}
for token, expected in expected_counts.items():
    actual = loop.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: loop token count %s expected=%d actual=%d"
            % (token, expected, actual))

first_barrier = loop.index("__syncthreadshared();")
second_barrier = loop.index("__syncthreadshared();", first_barrier + 1)
last_current_lds = loop.index("XH_N64_LDS_A_B128(1, 0);")
first_next_store = loop.index(
    "XH_MMA_STS(shared_b_tensor(store_row_b[0], store_col), load_b[0], MmaLoad128);")
last_next_store = loop.index(
    "XH_MMA_STS(shared_a_tensor(store_row_a[1], store_col), load_a[1], MmaLoad128);")
next_tile_lds = loop.index("XH_N64_LDS_A_B128(0, 0);")
if not (
    last_current_lds < first_barrier < first_next_store
    < last_next_store < second_barrier < next_tile_lds
):
    sys.exit("SOURCE CHECK FAIL: N64 shared-memory handoff ordering is unsafe")

baseline_ctas = (32768 // 128) * (4096 // 128)
candidate_ctas = (32768 // 128) * (4096 // 64)
baseline_a = baseline_ctas * 128 * 7168
candidate_a = candidate_ctas * 128 * 7168
baseline_b = baseline_ctas * 128 * 7168
candidate_b = candidate_ctas * 64 * 7168
if not (
    candidate_ctas == 2 * baseline_ctas
    and candidate_a == 2 * baseline_a
    and candidate_b == baseline_b
):
    sys.exit("SOURCE CHECK FAIL: N64 traffic proof changed")

print(
    "SOURCE CHECK PASS: baseline hash exact; case2=128x64x128/256, "
    "24 KiB shared, half accumulators, B traffic preserved, barriers ordered")
PYEOF

printf 'BUILD exp-20260822-026 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
