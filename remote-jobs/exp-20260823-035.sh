#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-035"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'constexpr int kMmaSharedScaleBBytes = kMmaTileN * sizeof(float);' "$submission_file"
grep -Fq 'XH_MMA_STS(shared_scale_b[tid * 4], scale_b_stage, MmaLoad128);' "$submission_file"
grep -Fq 'shared_scale_b[output_col[i]]' "$submission_file"
grep -Fq 'REGRESSION maca-scale-b-preload' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaEpilogueScaleBytes' \
                'shared_row_scale' 'shared_col_scale' 'combined_row_scale'; do
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

candidate_constants = """constexpr int kMmaSharedScaleBBytes = kMmaTileN * sizeof(float);
constexpr int kMmaSharedBytes =
    kMmaSharedABytes + kMmaSharedBBytes + kMmaSharedScaleBBytes;"""
baseline_constants = (
    "constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;")

candidate_shared_pointer = """    float* shared_scale_b =
        reinterpret_cast<float*>(shared_b + kMmaSharedBBytes);
"""

candidate_global_pointer = """    const float* scale_b_tile =
        scale_b_ptr + static_cast<uint64_t>(expert) * n
        + tile_n * kMmaTileN;
"""

candidate_preload = """    if (tid < kMmaTileN / 4) {
        MmaLoad128 scale_b_stage = __builtin_mxc_ldg_b128(
            const_cast<float*>(scale_b_tile + tid * 4),
            0,
            -1,
            true,
            true,
            false,
            false);
        XH_MMA_STS(shared_scale_b[tid * 4], scale_b_stage, MmaLoad128);
    }
"""

candidate_epilogue = """#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        XH_MMA_LDS(
            col_scale[i],
            shared_scale_b[output_col[i]],
            MmaFloat4);
    }"""
baseline_epilogue = """#pragma unroll
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


def replace_unique(source, candidate, baseline, label):
    count = source.count(candidate)
    if count != 1:
        sys.exit(
            "SOURCE CHECK FAIL: %s candidate block count expected=1 actual=%d"
            % (label, count))
    return source.replace(candidate, baseline, 1)


reconstructed = text
reconstructed = replace_unique(
    reconstructed, candidate_constants, baseline_constants, "shared constants")
reconstructed = replace_unique(
    reconstructed, candidate_shared_pointer, "", "shared scale-B pointer")
reconstructed = replace_unique(
    reconstructed, candidate_global_pointer, "", "global scale-B tile pointer")
reconstructed = replace_unique(
    reconstructed, candidate_preload, "", "cooperative preload")
reconstructed = replace_unique(
    reconstructed, candidate_epilogue, baseline_epilogue, "epilogue LDS load")

baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped assigned scale-B preload blocks: "
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
wave_m = read_literal_const("kMmaWaveM")
if (tile_m, tile_n, tile_k, threads, wave_size, wave_m) != (
        128, 128, 128, 256, 64, 4):
    sys.exit("SOURCE CHECK FAIL: CTA/tile/wave geometry changed")

shared_a_bytes = tile_m * tile_k
shared_b_bytes = tile_n * tile_k
shared_scale_b_bytes = tile_n * 4
shared_ranges = (
    (0, shared_a_bytes),
    (shared_a_bytes, shared_a_bytes + shared_b_bytes),
    (shared_a_bytes + shared_b_bytes,
     shared_a_bytes + shared_b_bytes + shared_scale_b_bytes),
)
if shared_ranges != ((0, 16384), (16384, 32768), (32768, 33280)):
    sys.exit("SOURCE CHECK FAIL: shared-memory ranges changed: %r" % (shared_ranges,))
if any(begin % 16 or end % 16 for begin, end in shared_ranges):
    sys.exit("SOURCE CHECK FAIL: shared-memory range is not b128 aligned")
if any(shared_ranges[i][1] != shared_ranges[i + 1][0] for i in range(2)):
    sys.exit("SOURCE CHECK FAIL: shared-memory ranges overlap or contain a gap")
if "__shared__ int8_t shared_data[kMmaSharedBytes];" not in text:
    sys.exit("SOURCE CHECK FAIL: shared allocation declaration changed")
if text.count(candidate_constants) != 1:
    sys.exit("SOURCE CHECK FAIL: shared allocation is not exactly 33280 bytes")

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
initial_start = kernel.index("    int store_row_a[kMmaLoadsA];")
initial_barrier = kernel.index("    __syncthreadshared();", initial_start)
preload_start = kernel.index(candidate_preload, initial_start)
last_initial_ab_store = max(
    kernel.rindex("XH_MMA_STS(shared_a_tensor(", initial_start, initial_barrier),
    kernel.rindex("XH_MMA_STS(shared_b_tensor(", initial_start, initial_barrier),
)
if not last_initial_ab_store < preload_start < initial_barrier:
    sys.exit("SOURCE CHECK FAIL: scale-B preload is outside initial store/barrier window")
if kernel.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: source barrier-site count changed")
if reconstructed[kernel_start:kernel_end].count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: reconstructed baseline barrier count changed")

preload = kernel[preload_start:preload_start + len(candidate_preload)]
if preload.count("__builtin_mxc_ldg_b128(") != 1:
    sys.exit("SOURCE CHECK FAIL: producer must issue one global b128 load")
if preload.count("XH_MMA_STS(") != 1:
    sys.exit("SOURCE CHECK FAIL: producer must issue one shared b128 store")
if "__syncthreadshared" in preload or "__syncthreads" in preload:
    sys.exit("SOURCE CHECK FAIL: preload introduced a barrier")
if kernel.count("shared_scale_b[output_col[i]]") != 1:
    sys.exit("SOURCE CHECK FAIL: epilogue scale-B LDS site changed")
if candidate_epilogue.count("XH_MMA_LDS(") != 1:
    sys.exit("SOURCE CHECK FAIL: epilogue must have one unrolled LDS call site")

unchanged_counts = {
    "__builtin_mxc_ldg_b32_predicator(": 2,
    "__builtin_mxc_stg_b64_predicator(": 2,
    "XH_MMA_STAGE_MNKX2(": 129,
    "XH_LDG_A_STAGE_I(": 5,
    "XH_LDG_B_STAGE_I(": 5,
}
for token, expected in unchanged_counts.items():
    actual = kernel.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: unchanged operation-site count %s expected=%d actual=%d"
            % (token, expected, actual))

producer_visits = [0] * tile_n
producer_vectors = []
for thread_id in range(threads):
    if thread_id < tile_n // 4:
        first_col = thread_id * 4
        byte_address = shared_ranges[2][0] + first_col * 4
        if byte_address % 16:
            sys.exit("SOURCE CHECK FAIL: scale-B producer is not b128 aligned")
        producer_vectors.append((thread_id, first_col))
        for col in range(first_col, first_col + 4):
            if not 0 <= col < tile_n:
                sys.exit("SOURCE CHECK FAIL: scale-B producer is out of range")
            producer_visits[col] += 1
if len(producer_vectors) != 32 or producer_visits != [1] * tile_n:
    sys.exit("SOURCE CHECK FAIL: 32 producers do not exactly cover scale-B[0..127]")

shared_values = list(range(tile_n))
lds_reads = 0
for thread_id in range(threads):
    for output_group in range(2):
        output_col = (thread_id % 16) * 4 + output_group * 64
        if output_col % 4 or not 0 <= output_col <= tile_n - 4:
            sys.exit("SOURCE CHECK FAIL: epilogue scale-B vector is invalid")
        global_values = list(range(output_col, output_col + 4))
        lds_values = shared_values[output_col:output_col + 4]
        if lds_values != global_values:
            sys.exit("SOURCE CHECK FAIL: epilogue LDS/global scale-B values differ")
        lds_reads += 1
if lds_reads != 512:
    sys.exit("SOURCE CHECK FAIL: epilogue LDS read count changed")

public_shapes = (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
)
for em, n, k in public_shapes:
    if n % tile_n or k % tile_k or em % tile_m:
        sys.exit("SOURCE CHECK FAIL: public shape does not tile exactly")
    for expert in range(256):
        for n_tile in range(n // tile_n):
            tile_byte_offset = (expert * n + n_tile * tile_n) * 4
            if tile_byte_offset % 16:
                sys.exit("SOURCE CHECK FAIL: global scale-B tile base is misaligned")
    k_tiles = k // tile_k
    a_bytes = tile_m * k
    b_bytes = tile_n * k
    mma_instructions = 128 * k_tiles
    barriers = 1 + 2 * (k_tiles - 1)
    if barriers != 2 * k_tiles - 1:
        sys.exit("SOURCE CHECK FAIL: dynamic barrier model changed")
    print(
        "SOURCE MODEL em=%d n=%d k=%d A-bytes/CTA=%d B-bytes/CTA=%d "
        "MMA=%d barriers=%d scale-B-global-b128=32 scale-B-global-bytes=512 "
        "scale-B-STS-b128=32 scale-B-LDS-b128=512"
        % (em, n, k, a_bytes, b_bytes, mma_instructions, barriers))

baseline_scale_loads = threads * 2
candidate_scale_loads = len(producer_vectors)
if (baseline_scale_loads, baseline_scale_loads * 16,
        candidate_scale_loads, candidate_scale_loads * 16) != (512, 8192, 32, 512):
    sys.exit("SOURCE CHECK FAIL: scale-B traffic reduction model changed")

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: exact baseline reconstructed; LDS=[0,16384),"
    "[16384,32768),[32768,33280); producers=32 exact aligned cover; "
    "scale-B global traffic 512 b128/8192 bytes -> 32 b128/512 bytes; "
    "32 STS b128; 512 LDS b128; three barrier sites; source_sha256="
    + source_sha256)
PYEOF

printf 'BUILD exp-20260823-035 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
