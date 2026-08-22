#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-037"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq '__shfl_sync(' "$submission_file"
grep -Fq '0xffffffffu, owned_row_factor, row_factor_lane, 16' "$submission_file"
grep -Fq 'REGRESSION maca-row-factor-shuffle' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaEpilogueScaleBytes' \
                'kMmaSharedScaleBBytes' 'shared_scale_b' 'shared_row_scale' \
                'shared_col_scale' 'combined_row_scale'; do
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

candidate_row_factor = """    MmaFloat4 col_scale[2];
    const int row_group_16 = lane >> 4;
    const int row_group_lane = lane & 15;
    float owned_row_factor = 0.0f;

    if (row_group_lane < 8) {
        const int owned_row =
            row_base + wave * 8
            + (row_group_16 & 1) * 4
            + (row_group_16 >> 1) * 32
            + (row_group_lane >> 2) * 64
            + (row_group_lane & 3);
        float owned_weight = 0.0f;
        *reinterpret_cast<MmaInt1*>(&owned_weight) =
            __builtin_mxc_ldg_b32_predicator(
                const_cast<float*>(moe_weights_ptr + owned_row),
                0,
                true,
                true,
                false,
                false,
                owned_row,
                em,
                MACA_ICMP_SLT);
        *reinterpret_cast<MmaInt1*>(&owned_row_factor) =
            __builtin_mxc_ldg_b32_predicator(
                const_cast<float*>(scale_a_ptr + owned_row),
                0,
                true,
                true,
                false,
                false,
                owned_row,
                em,
                MACA_ICMP_SLT);
        owned_row_factor *= owned_weight;
    }
"""

baseline_row_factor = """    float weights[2][4];
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

candidate_consumer = """            const int row_factor_lane = i * 4 + j;
            const float row_factor = __shfl_sync(
                0xffffffffu, owned_row_factor, row_factor_lane, 16);
            MmaFloat2 row_scale2 = {row_factor, row_factor};"""

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
    text, candidate_row_factor, baseline_row_factor, "producer")
reconstructed = replace_unique(
    reconstructed, candidate_consumer, baseline_consumer, "consumer")
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped assigned row-factor blocks: "
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
if "constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;" not in text:
    sys.exit("SOURCE CHECK FAIL: baseline 32-KiB shared allocation changed")

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
baseline_kernel = reconstructed[kernel_start:kernel_end]
if kernel.count("__shfl_sync(") != 1:
    sys.exit("SOURCE CHECK FAIL: row factor requires exactly one shuffle call site")
if kernel.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: barrier-site count changed")
if baseline_kernel.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: reconstructed baseline barrier count changed")
if "float weights[2][4];" in kernel or "float row_scale[2][4];" in kernel:
    sys.exit("SOURCE CHECK FAIL: per-thread row metadata arrays remain")

unchanged_counts = {
    "__builtin_mxc_ldg_b32_predicator(": 2,
    "__builtin_mxc_ldg_b128_predicator(": 2,
    "__builtin_mxc_stg_b64_predicator(": 2,
    "XH_MMA_STAGE_MNKX2(": 129,
    "XH_LDG_A_STAGE_I(": 5,
    "XH_LDG_B_STAGE_I(": 5,
    "XH_MMA_LDS(": 2,
    "XH_MMA_STS(": 13,
}
for token, expected in unchanged_counts.items():
    actual = kernel.count(token)
    baseline_actual = baseline_kernel.count(token)
    if actual != expected or baseline_actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: unchanged count %s expected=%d candidate=%d baseline=%d"
            % (token, expected, actual, baseline_actual))

producer_row = [-1] * threads
row_visits = [0] * tile_m
producer_count = 0
for thread_id in range(threads):
    wave = thread_id // wave_size
    lane = thread_id % wave_size
    group = lane >> 4
    group_lane = lane & 15
    if group_lane < 8:
        row = (
            wave * 8
            + (group & 1) * 4
            + (group >> 1) * 32
            + (group_lane >> 2) * 64
            + (group_lane & 3))
        if not 0 <= row < tile_m:
            sys.exit("SOURCE CHECK FAIL: producer row is out of range")
        producer_row[thread_id] = row
        row_visits[row] += 1
        producer_count += 1
if producer_count != 128 or row_visits != [1] * tile_m:
    sys.exit("SOURCE CHECK FAIL: producers do not exactly cover rows 0..127")

consumer_count = 0
for thread_id in range(threads):
    wave = thread_id // wave_size
    lane = thread_id % wave_size
    group_base = (thread_id // 16) * 16
    for i in range(2):
        for j in range(4):
            source_lane = i * 4 + j
            source_thread = group_base + source_lane
            if not 0 <= source_lane < 8 or producer_row[source_thread] < 0:
                sys.exit("SOURCE CHECK FAIL: shuffle source lane is not initialized")
            expected_row = (
                ((lane // 16) % 2) * 4
                + wave * 8
                + (lane // 32) * 32
                + i * 64
                + j)
            if producer_row[source_thread] != expected_row:
                sys.exit("SOURCE CHECK FAIL: consumer row mapping changed")
            consumer_count += 1
if consumer_count != 2048:
    sys.exit("SOURCE CHECK FAIL: consumer count changed")

baseline_loads = threads * 2 * 4 * 2
candidate_loads = producer_count * 2
baseline_multiplies = threads * 2 * 4
candidate_multiplies = producer_count
candidate_shuffles = consumer_count
if (baseline_loads, candidate_loads, baseline_multiplies,
        candidate_multiplies, candidate_shuffles) != (4096, 256, 2048, 128, 2048):
    sys.exit("SOURCE CHECK FAIL: operation reduction model changed")

public_shapes = (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
)
for em, n, k in public_shapes:
    if em % tile_m or n % tile_n or k % tile_k:
        sys.exit("SOURCE CHECK FAIL: public shape does not tile exactly")
    for row_base in range(0, em, tile_m):
        for local_row in producer_row:
            if local_row >= 0 and not 0 <= row_base + local_row < em:
                sys.exit("SOURCE CHECK FAIL: public producer row is out of bounds")
    k_tiles = k // tile_k
    a_bytes = tile_m * k
    b_bytes = tile_n * k
    mma_instructions = 128 * k_tiles
    barriers = 2 * k_tiles - 1
    print(
        "SOURCE MODEL em=%d n=%d k=%d A-bytes/CTA=%d B-bytes/CTA=%d "
        "MMA=%d barriers=%d row-b32=%d->%d row-mul=%d->%d shuffles=%d"
        % (em, n, k, a_bytes, b_bytes, mma_instructions, barriers,
           baseline_loads, candidate_loads, baseline_multiplies,
           candidate_multiplies, candidate_shuffles))

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: exact baseline reconstructed; 128 producers exactly cover "
    "rows 0..127; 2048 width-16 consumers match baseline; shuffle sources active; "
    "row b32 loads 4096->256; row multiplies 2048->128; shuffles=2048; "
    "LDS=32768; three barrier sites; source_sha256=" + source_sha256)
PYEOF

printf 'BUILD exp-20260823-037 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
