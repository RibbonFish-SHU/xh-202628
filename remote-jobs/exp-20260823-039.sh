#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-039"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'float row_factor[2][4];' "$submission_file"
grep -Fq 'row_factor[i][j] *= weight;' "$submission_file"
grep -Fq 'REGRESSION maca-row-factor-fusion' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' '__shfl_sync' \
                'mma_kernel_64' 'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' \
                'kCase2L2NGroup' 'kSerpentineN' 'kCooperativeEpilogue' \
                'mma_adjacent_m' 'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaEpilogueScaleBytes' \
                'kMmaSharedScaleBBytes' 'shared_scale_b' 'shared_row_scale' \
                'shared_col_scale' 'combined_row_scale' 'kMmaBFragmentCols'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected historical mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")

candidate_metadata = """    float row_factor[2][4];
    MmaFloat4 col_scale[2];

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const int row = output_row[i * 4 + j];
            float weight = 0.0f;
            *reinterpret_cast<MmaInt1*>(&weight) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(moe_weights_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
            *(reinterpret_cast<MmaInt1*>(&row_factor[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(scale_a_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
            row_factor[i][j] *= weight;
        }
    }
"""

baseline_metadata = """    float weights[2][4];
    float row_scale[2][4];
    MmaFloat4 col_scale[2];

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const int row = output_row[i * 4 + j];
            *(reinterpret_cast<MmaInt1*>(&weights[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(moe_weights_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
            *(reinterpret_cast<MmaInt1*>(&row_scale[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(scale_a_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
        }
    }
"""

candidate_consumer = (
    "            MmaFloat2 row_scale2 = "
    "{row_factor[i][j], row_factor[i][j]};")
baseline_consumer = """            row_scale[i][j] *= weights[i][j];
            MmaFloat2 row_scale2 = {row_scale[i][j], row_scale[i][j]};"""


def replace_unique(source, candidate, baseline, label):
    count = source.count(candidate)
    if count != 1:
        sys.exit(
            "SOURCE CHECK FAIL: %s candidate block count expected=1 actual=%d"
            % (label, count))
    return source.replace(candidate, baseline, 1)


reconstructed = replace_unique(
    text, candidate_metadata, baseline_metadata, "metadata")
reconstructed = replace_unique(
    reconstructed, candidate_consumer, baseline_consumer, "consumer")
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped assigned row-factor block: "
        + baseline_sha256)


def read_literal_const(name):
    match = re.search(r"constexpr int %s = (\d+);" % name, text)
    if not match:
        sys.exit("SOURCE CHECK FAIL: literal constant not found: " + name)
    return int(match.group(1))


tile_m = read_literal_const("kMmaTileM")
tile_n = read_literal_const("kMmaTileN")
tile_k = read_literal_const("kMmaTileK")
threads = read_literal_const("kMmaThreads")
wave_size = read_literal_const("kMmaWaveSize")
shared_bytes = tile_m * tile_k + tile_n * tile_k
if (tile_m, tile_n, tile_k, threads, wave_size, shared_bytes) != (
        128, 128, 128, 256, 64, 32768):
    sys.exit("SOURCE CHECK FAIL: CTA/tile/wave/shared geometry changed")

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
baseline_kernel = reconstructed[kernel_start:kernel_end]

required_exact = (
    "float row_factor[2][4];",
    "float weight = 0.0f;",
    "const_cast<float*>(moe_weights_ptr + row)",
    "const_cast<float*>(scale_a_ptr + row)",
    "row_factor[i][j] *= weight;",
    "MmaFloat2 row_scale2 = {row_factor[i][j], row_factor[i][j]};",
)
for token in required_exact:
    if kernel.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: required row-factor token is not exact: " + token)
for removed in ("float weights[2][4];", "float row_scale[2][4];"):
    if removed in kernel:
        sys.exit("SOURCE CHECK FAIL: long-lived metadata array remains: " + removed)

unchanged_counts = {
    "__builtin_mxc_ldg_b32_predicator(": 2,
    "__builtin_mxc_ldg_b128_predicator(": 2,
    "__builtin_mxc_stg_b64_predicator(": 2,
    "XH_MMA_STAGE_MNKX2(": 129,
    "XH_LDG_A_STAGE_I(": 5,
    "XH_LDG_B_STAGE_I(": 5,
    "XH_MMA_LDS(": 2,
    "XH_MMA_STS(": 13,
    "__syncthreadshared();": 3,
}
for token, expected in unchanged_counts.items():
    actual = kernel.count(token)
    baseline_actual = baseline_kernel.count(token)
    if actual != expected or baseline_actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: unchanged count %s expected=%d candidate=%d baseline=%d"
            % (token, expected, actual, baseline_actual))

# The predicated row address is unchanged and the public contract tiles EM by 128.
visits = [0] * tile_m
consumers = 0
for thread_id in range(threads):
    wave = thread_id // wave_size
    lane = thread_id % wave_size
    for i in range(2):
        for j in range(4):
            row = (
                ((lane // 16) % 2) * 4
                + wave * 8
                + (lane // 32) * 32
                + i * 64
                + j)
            if not 0 <= row < tile_m:
                sys.exit("SOURCE CHECK FAIL: row-factor address is out of range")
            visits[row] += 1
            consumers += 1
if consumers != 2048 or visits != [16] * tile_m:
    sys.exit("SOURCE CHECK FAIL: row-factor consumer coverage changed")

row_values = 2 * 4
metadata_loads = row_values * 2
multiplies = row_values
if (row_values, metadata_loads, multiplies) != (8, 16, 8):
    sys.exit("SOURCE CHECK FAIL: row-factor operation model changed")

public_shapes = (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
)
for em, n, k in public_shapes:
    if em % tile_m or n % tile_n or k % tile_k:
        sys.exit("SOURCE CHECK FAIL: public shape does not tile exactly")
    k_tiles = k // tile_k
    barriers = 2 * k_tiles - 1
    print(
        "SOURCE MODEL em=%d n=%d k=%d row-values/thread=%d "
        "row-b32/thread=%d row-mul/thread=%d MMA/thread=%d barriers=%d"
        % (em, n, k, row_values, metadata_loads, multiplies,
           128 * k_tiles, barriers))

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: exact baseline reconstructed; one 8-float row-factor "
    "array; scalar weight lifetime; row addresses/predicates exact; b32 loads="
    "16/thread; multiplies=8/thread; matrix/output/store protected; LDS=32768; "
    "barrier sites=3; source_sha256=" + source_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD exp-20260823-039 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
