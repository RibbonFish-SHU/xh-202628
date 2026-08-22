#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-028"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'REGRESSION maca-accum-epilogue-remap' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64'; do
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

old_materialization = """    MmaInt4 output[kMmaOutputVectors];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            output[i * 8 + 2 * j][0] = accum[i][0][j];
            output[i * 8 + 2 * j][1] = accum[i][2][j];
            output[i * 8 + 2 * j][2] = accum[i][4][j];
            output[i * 8 + 2 * j][3] = accum[i][6][j];
            output[i * 8 + 2 * j + 1][0] = accum[i][1][j];
            output[i * 8 + 2 * j + 1][1] = accum[i][3][j];
            output[i * 8 + 2 * j + 1][2] = accum[i][5][j];
            output[i * 8 + 2 * j + 1][3] = accum[i][7][j];
        }
    }

"""
old_values = """            values[0] = output[i * 8 + 2 * j][0];
            values[1] = output[i * 8 + 2 * j][1];
            values[2] = output[i * 8 + 2 * j][2];
            values[3] = output[i * 8 + 2 * j][3];
            values[4] = output[i * 8 + 2 * j + 1][0];
            values[5] = output[i * 8 + 2 * j + 1][1];
            values[6] = output[i * 8 + 2 * j + 1][2];
            values[7] = output[i * 8 + 2 * j + 1][3];
"""
direct_values = """            values[0] = accum[i][0][j];
            values[1] = accum[i][2][j];
            values[2] = accum[i][4][j];
            values[3] = accum[i][6][j];
            values[4] = accum[i][1][j];
            values[5] = accum[i][3][j];
            values[6] = accum[i][5][j];
            values[7] = accum[i][7][j];
"""

if "MmaInt4 output[kMmaOutputVectors];" in text or re.search(r"\boutput\s*\[", text):
    sys.exit("SOURCE CHECK FAIL: temporary output materialization remains")
if text.count(direct_values) != 1:
    sys.exit("SOURCE CHECK FAIL: direct accumulator mapping is not exact")

assignment_pattern = re.compile(
    r"^\s*values\[(\d)\] = accum\[i\]\[(\d)\]\[j\];$", re.MULTILINE)
assignments = [(int(dst), int(fragment)) for dst, fragment in assignment_pattern.findall(text)]
expected_assignments = [
    (0, 0), (1, 2), (2, 4), (3, 6),
    (4, 1), (5, 3), (6, 5), (7, 7),
]
if assignments != expected_assignments:
    sys.exit("SOURCE CHECK FAIL: direct values-to-accum mapping changed: " + repr(assignments))

anchor = "    int output_col[2];\n"
if text.count(anchor) != 1:
    sys.exit("SOURCE CHECK FAIL: epilogue anchor is not unique")
reconstructed = text.replace(anchor, old_materialization + anchor, 1)
reconstructed = reconstructed.replace(direct_values, old_values, 1)
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61"
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes extend beyond direct accumulator remap: "
        + baseline_sha256)

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
candidate_kernel = text[kernel_start:kernel_end]
baseline_kernel = reconstructed[kernel_start:reconstructed.index("\n#else", kernel_start)]
mma_pattern = re.compile(r"^\s*XH_MMA_STAGE_MNKX2\(([^)]*)\);$", re.MULTILINE)
candidate_mma = mma_pattern.findall(candidate_kernel)
baseline_mma = mma_pattern.findall(baseline_kernel)
if candidate_mma != baseline_mma or len(candidate_mma) != 128:
    sys.exit(
        "SOURCE CHECK FAIL: MMA invocation count/order changed: candidate=%d baseline=%d"
        % (len(candidate_mma), len(baseline_mma)))

store_pattern = re.compile(r"\s*__builtin_mxc_stg_b64_predicator\(.*?\n\s*MACA_ICMP_EQ\);", re.DOTALL)
candidate_stores = store_pattern.findall(candidate_kernel)
baseline_stores = store_pattern.findall(baseline_kernel)
if candidate_stores != baseline_stores or len(candidate_stores) != 2:
    sys.exit(
        "SOURCE CHECK FAIL: output store count/order changed: candidate=%d baseline=%d"
        % (len(candidate_stores), len(baseline_stores)))

print(
    "SOURCE CHECK PASS: baseline reconstructed exactly; output[16]/64 copies absent; "
    "direct-map=8 static/64 dynamic components; MMA-order=128x2 exact; stores=2 exact")
PYEOF

printf 'BUILD exp-20260823-028 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
