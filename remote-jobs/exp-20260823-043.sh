#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-043"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'const bool scale_b_producer = scale_b_lane < 16;' "$submission_file"
grep -Fq '0xffffffffu, scale_value, scale_b_source_lane, 32' "$submission_file"
grep -Fq 'REGRESSION maca-scale-b-wave-broadcast' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaBFragmentCols' \
                'kMmaSharedScaleBBytes' 'shared_scale_b' 'kMmaEpilogueScaleBytes' \
                'shared_row_scale' 'shared_col_scale' 'combined_row_scale' \
                'owned_row_factor' 'row_factor_lane' 'XH_A_FRAG' 'shared_b_next'; do
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
candidate_sha256 = hashlib.sha256(text.encode()).hexdigest()
expected_candidate_sha256 = (
    "9c125db78326a3797a6686899390839577f5070afcfd7f08a99cbaf7fc58aa27")
if candidate_sha256 != expected_candidate_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate source hash changed: " + candidate_sha256)

candidate_block = """    const int scale_b_lane = lane & 31;
    const int scale_b_source_lane = lane & 15;
    const bool scale_b_producer = scale_b_lane < 16;
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        if (scale_b_producer) {
            const float* ptr =
                scale_b_ptr + static_cast<uint64_t>(expert) * n
                + tile_n * kMmaTileN + output_col[i];
            col_scale[i] = __builtin_mxc_ldg_b128_predicator(
                const_cast<float*>(ptr),
                0,
                true,
                true,
                false,
                false,
                output_col_mask[i],
                1,
                MACA_ICMP_EQ);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            float scale_value = 0.0f;
            if (scale_b_producer) {
                scale_value = reinterpret_cast<float*>(&col_scale[i])[j];
            }
            reinterpret_cast<float*>(&col_scale[i])[j] = __shfl_sync(
                0xffffffffu, scale_value, scale_b_source_lane, 32);
        }
    }"""

baseline_block = """#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const float* ptr =
            scale_b_ptr + static_cast<uint64_t>(expert) * n
            + tile_n * kMmaTileN + output_col[i];
        col_scale[i] = __builtin_mxc_ldg_b128_predicator(
            const_cast<float*>(ptr),
            0,
            true,
            true,
            false,
            false,
            output_col_mask[i],
            1,
            MACA_ICMP_EQ);
    }"""

if text.count(candidate_block) != 1:
    sys.exit("SOURCE CHECK FAIL: assigned scale-B block is not exact")
reconstructed = text.replace(candidate_block, baseline_block, 1)
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped assigned scale-B block: "
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
    sys.exit("SOURCE CHECK FAIL: scale-B broadcast requires one shuffle call site")
if "0xffffffffu, scale_value, scale_b_source_lane, 32" not in kernel:
    sys.exit("SOURCE CHECK FAIL: width-32 shuffle signature changed")
if kernel.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: barrier-site count changed")

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

subgroup_width = 32
producer_lanes = 16
column_groups = 2
words_per_load = 4
subgroups = threads // subgroup_width
producer_values = {}
vector_visits = [[0] * (tile_n // words_per_load) for _ in range(subgroups)]
producer_count = 0
for thread_id in range(threads):
    lane = thread_id % wave_size
    subgroup_lane = lane & (subgroup_width - 1)
    if subgroup_lane >= producer_lanes:
        continue
    producer_count += 1
    for group in range(column_groups):
        output_col = (thread_id % 16) * 4 + group * 64
        vector_visits[thread_id // subgroup_width][output_col // 4] += 1
        for word in range(words_per_load):
            producer_values[(thread_id, group, word)] = output_col + word

if producer_count != 128:
    sys.exit("SOURCE CHECK FAIL: producer count changed")
if any(visits != [1] * 32 for visits in vector_visits):
    sys.exit("SOURCE CHECK FAIL: a width-32 subgroup does not cover all 128 scale values")

peer_checks = 0
for col_limit in (1, 63, 64, 65, 127, 128):
    for thread_id in range(threads):
        lane = thread_id % wave_size
        source_lane = lane & (producer_lanes - 1)
        source_thread = (thread_id // subgroup_width) * subgroup_width + source_lane
        if source_thread // subgroup_width != thread_id // subgroup_width:
            sys.exit("SOURCE CHECK FAIL: shuffle source escapes width-32 subgroup")
        if source_thread % subgroup_width >= producer_lanes:
            sys.exit("SOURCE CHECK FAIL: shuffle source is not an initialized producer")
        for group in range(column_groups):
            output_col = (thread_id % 16) * 4 + group * 64
            source_col = (source_thread % 16) * 4 + group * 64
            if output_col != source_col or (output_col < col_limit) != (source_col < col_limit):
                sys.exit("SOURCE CHECK FAIL: peer address or predicate differs")
            for word in range(words_per_load):
                if producer_values.get((source_thread, group, word)) != output_col + word:
                    sys.exit("SOURCE CHECK FAIL: source value is absent or bit identity changed")
                peer_checks += 1

baseline_loads = threads * column_groups
candidate_loads = producer_count * column_groups
candidate_shuffles = threads * column_groups * words_per_load
if (baseline_loads, candidate_loads, candidate_shuffles) != (512, 256, 2048):
    sys.exit("SOURCE CHECK FAIL: scale-B load/shuffle model changed")
if peer_checks != 6 * candidate_shuffles:
    sys.exit("SOURCE CHECK FAIL: exhaustive peer cardinality changed")

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
    print(
        "SOURCE MODEL em=%d n=%d k=%d A-bytes/CTA=%d B-bytes/CTA=%d "
        "MMA=%d barriers=%d scaleB-bytes=%d->%d b128=%d->%d shuffles=%d"
        % (em, n, k, tile_m * k, tile_n * k, 128 * k_tiles,
           2 * k_tiles - 1, baseline_loads * 16, candidate_loads * 16,
           baseline_loads, candidate_loads, candidate_shuffles))

print(
    "SOURCE CHECK PASS: exact performance baseline reconstructed; all 256 threads "
    "retain scale-B address/predicate/float-bit identity; eight width-32 subgroups "
    "each use 16 initialized producers covering 128 values; b128 loads 512->256; "
    "scale-B bytes 8192->4096; scalar shuffles=2048; LDS=32768; barriers unchanged; "
    "source_sha256=" + candidate_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD exp-20260823-043 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
