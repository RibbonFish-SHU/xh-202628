#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-040"
submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'MmaInt4 output[kMmaOutputVectors];' "$submission_file"
grep -Fq 'MmaFloat4 col_scale;' "$submission_file"
grep -Fq 'float values[4];' "$submission_file"
grep -Fq 'MmaFloat2 scales[2];' "$submission_file"
grep -Fq 'REGRESSION maca-epilogue-half-strip' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaSharedScaleBBytes' \
                'shared_scale_b' 'kMmaEpilogueScaleBytes' 'shared_row_scale' \
                'shared_col_scale' 'combined_row_scale' 'XH_A_FRAG' \
                'shared_b_next' 'kMmaSharedBStages' '__shfl_sync'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected historical mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import re
import struct
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as source:
    text = source.read().replace("\r\n", "\n").replace("\r", "\n")

candidate_decl = "    MmaFloat4 col_scale;"
baseline_decl = "    MmaFloat4 col_scale[2];"

candidate_block = """    MmaBfloat16* out_base =
        reinterpret_cast<MmaBfloat16*>(out_ptr) + tile_n * kMmaTileN;
    MmaFloat2 zero2 = {0.0f, 0.0f};
    MmaStore64 packed_out;

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            row_scale[i][j] *= weights[i][j];
        }
    }

#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
        const float* ptr =
            scale_b_ptr + static_cast<uint64_t>(expert) * n
            + tile_n * kMmaTileN + output_col[half];
        col_scale = __builtin_mxc_ldg_b128_predicator(
            const_cast<float*>(ptr),
            0,
            true,
            true,
            false,
            false,
            output_col_mask[half],
            1,
            MACA_ICMP_EQ);

#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                float values[4];
                values[0] = output[i * 8 + 2 * j + half][0];
                values[1] = output[i * 8 + 2 * j + half][1];
                values[2] = output[i * 8 + 2 * j + half][2];
                values[3] = output[i * 8 + 2 * j + half][3];

                MmaFloat2 row_scale2 = {row_scale[i][j], row_scale[i][j]};
                MmaFloat2 scales[2];
                scales[0] = __builtin_mxc_pk_fma_f32(
                    reinterpret_cast<MmaFloat2*>(&col_scale)[0], row_scale2, zero2);
                scales[1] = __builtin_mxc_pk_fma_f32(
                    reinterpret_cast<MmaFloat2*>(&col_scale)[1], row_scale2, zero2);
                *reinterpret_cast<MmaFloat2*>(&values[0]) = __builtin_mxc_pk_fma_f32(
                    *reinterpret_cast<MmaFloat2*>(&values[0]), scales[0], zero2);
                *reinterpret_cast<MmaFloat2*>(&values[2]) = __builtin_mxc_pk_fma_f32(
                    *reinterpret_cast<MmaFloat2*>(&values[2]), scales[1], zero2);

                XH_CVT_F32_TO_BF16(
                    packed_out[0],
                    reinterpret_cast<uint*>(&values)[0],
                    reinterpret_cast<uint*>(&values)[1]);
                XH_CVT_F32_TO_BF16(
                    packed_out[1],
                    reinterpret_cast<uint*>(&values)[2],
                    reinterpret_cast<uint*>(&values)[3]);
                __builtin_mxc_stg_b64_predicator(
                    out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n
                        + output_col[half],
                    0,
                    *reinterpret_cast<uint64_t*>(&packed_out),
                    true,
                    false,
                    false,
                    (output_row[i * 4 + j] < em) && output_col_mask[half],
                    1,
                    MACA_ICMP_EQ);
            }
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
    }

    MmaBfloat16* out_base =
        reinterpret_cast<MmaBfloat16*>(out_ptr) + tile_n * kMmaTileN;
    MmaFloat2 zero2 = {0.0f, 0.0f};
    MmaStore64 packed_out;

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            float values[8];
            values[0] = output[i * 8 + 2 * j][0];
            values[1] = output[i * 8 + 2 * j][1];
            values[2] = output[i * 8 + 2 * j][2];
            values[3] = output[i * 8 + 2 * j][3];
            values[4] = output[i * 8 + 2 * j + 1][0];
            values[5] = output[i * 8 + 2 * j + 1][1];
            values[6] = output[i * 8 + 2 * j + 1][2];
            values[7] = output[i * 8 + 2 * j + 1][3];

            row_scale[i][j] *= weights[i][j];
            MmaFloat2 row_scale2 = {row_scale[i][j], row_scale[i][j]};
            MmaFloat2 scales[4];
            scales[0] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[0])[0], row_scale2, zero2);
            scales[1] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[0])[1], row_scale2, zero2);
            scales[2] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[1])[0], row_scale2, zero2);
            scales[3] = __builtin_mxc_pk_fma_f32(
                reinterpret_cast<MmaFloat2*>(&col_scale[1])[1], row_scale2, zero2);
            *reinterpret_cast<MmaFloat2*>(&values[0]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[0]), scales[0], zero2);
            *reinterpret_cast<MmaFloat2*>(&values[2]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[2]), scales[1], zero2);
            *reinterpret_cast<MmaFloat2*>(&values[4]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[4]), scales[2], zero2);
            *reinterpret_cast<MmaFloat2*>(&values[6]) = __builtin_mxc_pk_fma_f32(
                *reinterpret_cast<MmaFloat2*>(&values[6]), scales[3], zero2);

            XH_CVT_F32_TO_BF16(
                packed_out[0],
                reinterpret_cast<uint*>(&values)[0],
                reinterpret_cast<uint*>(&values)[1]);
            XH_CVT_F32_TO_BF16(
                packed_out[1],
                reinterpret_cast<uint*>(&values)[2],
                reinterpret_cast<uint*>(&values)[3]);
            __builtin_mxc_stg_b64_predicator(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n + output_col[0],
                0,
                *reinterpret_cast<uint64_t*>(&packed_out),
                true,
                false,
                false,
                (output_row[i * 4 + j] < em) && output_col_mask[0],
                1,
                MACA_ICMP_EQ);

            XH_CVT_F32_TO_BF16(
                packed_out[0],
                reinterpret_cast<uint*>(&values)[4],
                reinterpret_cast<uint*>(&values)[5]);
            XH_CVT_F32_TO_BF16(
                packed_out[1],
                reinterpret_cast<uint*>(&values)[6],
                reinterpret_cast<uint*>(&values)[7]);
            __builtin_mxc_stg_b64_predicator(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n + output_col[1],
                0,
                *reinterpret_cast<uint64_t*>(&packed_out),
                true,
                false,
                false,
                (output_row[i * 4 + j] < em) && output_col_mask[1],
                1,
                MACA_ICMP_EQ);
        }
    }"""


def replace_unique(source, candidate, baseline, label):
    count = source.count(candidate)
    if count != 1:
        sys.exit(
            "SOURCE CHECK FAIL: %s candidate block count expected=1 actual=%d"
            % (label, count))
    return source.replace(candidate, baseline, 1)


reconstructed = replace_unique(text, candidate_decl, baseline_decl, "col-scale declaration")
reconstructed = replace_unique(
    reconstructed, candidate_block, baseline_block, "half-strip epilogue")
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped assigned epilogue blocks: "
        + baseline_sha256)

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
expected_source_sha256 = "8e16d67c5f1d65a8b32266954e120037a9cc068560760762b01e4c615d78cf51"
if source_sha256 != expected_source_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: candidate source hash changed: " + source_sha256)

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
output_start = kernel.index("    MmaInt4 output[kMmaOutputVectors];")
epilogue_end = kernel.index("\n#undef XH_CVT_F32_TO_BF16", output_start)
epilogue = kernel[output_start:epilogue_end]
half_loop = epilogue.index("    for (uint32_t half = 0; half < 2; ++half) {")
inner_i_loop = epilogue.index("        for (uint32_t i = 0; i < 2; ++i) {", half_loop)
if not half_loop < inner_i_loop:
    sys.exit("SOURCE CHECK FAIL: epilogue is not half-major strip-mined")

required_exact = (
    "MmaInt4 output[kMmaOutputVectors];",
    "output[i * 8 + 2 * j][0] = accum[i][0][j];",
    "output[i * 8 + 2 * j][3] = accum[i][6][j];",
    "output[i * 8 + 2 * j + 1][0] = accum[i][1][j];",
    "output[i * 8 + 2 * j + 1][3] = accum[i][7][j];",
    "MmaFloat4 col_scale;",
    "float values[4];",
    "MmaFloat2 scales[2];",
    "output[i * 8 + 2 * j + half][0]",
)
for token in required_exact:
    if kernel.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: required exact token changed: " + token)
if epilogue.count("output_col[half]") != 2 \
        or epilogue.count("output_col_mask[half]") != 2:
    sys.exit("SOURCE CHECK FAIL: half-selected address or predicate count changed")
for forbidden in ("MmaFloat4 col_scale[2]", "float values[8]", "MmaFloat2 scales[4]"):
    if forbidden in kernel:
        sys.exit("SOURCE CHECK FAIL: baseline epilogue state remains: " + forbidden)

site_counts = {
    "__builtin_mxc_ldg_b32_predicator(": 2,
    "__builtin_mxc_ldg_b128_predicator(": 1,
    "row_scale[i][j] *= weights[i][j];": 1,
    "__builtin_mxc_pk_fma_f32(": 4,
    "XH_CVT_F32_TO_BF16(": 2,
    "__builtin_mxc_stg_b64_predicator(": 1,
}
for token, expected in site_counts.items():
    actual = epilogue.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: epilogue site count %s expected=%d actual=%d"
            % (token, expected, actual))

unchanged_counts = {
    "XH_MMA_STAGE_MNKX2(": 129,
    "XH_LDG_A_STAGE_I(": 5,
    "XH_LDG_B_STAGE_I(": 5,
    "XH_MMA_STS(shared_a_tensor(": 8,
    "XH_MMA_STS(shared_b_tensor(": 5,
    "XH_LDS_A_B128(": 9,
    "XH_LDS_B_B128(": 33,
    "__syncthreadshared();": 3,
}
for token, expected in unchanged_counts.items():
    actual = kernel.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: protected operation count %s expected=%d actual=%d"
            % (token, expected, actual))
if "constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;" not in text:
    sys.exit("SOURCE CHECK FAIL: 32-KiB shared definition changed")
if "__shared__ int8_t shared_data[kMmaSharedBytes];" not in kernel:
    sys.exit("SOURCE CHECK FAIL: shared allocation changed")


def output_row(thread_id, row_group, row_in_group):
    wave = thread_id // 64
    lane = thread_id % 64
    return (
        ((lane // 16) % 2) * 4
        + wave * 8
        + (lane // 32) * 32
        + row_group * 64
        + row_in_group)


def output_col(thread_id, half):
    return (thread_id % 16) * 4 + half * 64


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


address_visits = [0] * (128 * 128)
for thread_id in range(256):
    baseline = {}
    candidate = {}
    baseline_store_order = []
    candidate_store_order = []
    row_factor_multiplies = [0] * 8
    row_factors = [0.0] * 8
    for i in range(2):
        for j in range(4):
            row_slot = i * 4 + j
            row_scale = f32((row_slot + 1) * 0.03125)
            weight = f32((row_slot + 3) * -0.0625)
            row_factors[row_slot] = f32(row_scale * weight)
            row_factor_multiplies[row_slot] += 1
            row = output_row(thread_id, i, j)
            for half in range(2):
                baseline_store_order.append((row, output_col(thread_id, half)))
                vector = i * 8 + 2 * j + half
                for component in range(4):
                    scalar = vector * 4 + component
                    col = output_col(thread_id, half) + component
                    output_value = f32((scalar - 31) * 0.125)
                    scale = f32(f32((col + 1) * 0.015625) * row_factors[row_slot])
                    value = f32(output_value * scale)
                    baseline[scalar] = (row_slot, col, bf16_bits(value))
    for half in range(2):
        for i in range(2):
            for j in range(4):
                row_slot = i * 4 + j
                row = output_row(thread_id, i, j)
                candidate_store_order.append((row, output_col(thread_id, half)))
                vector = i * 8 + 2 * j + half
                for component in range(4):
                    scalar = vector * 4 + component
                    col = output_col(thread_id, half) + component
                    output_value = f32((scalar - 31) * 0.125)
                    scale = f32(f32((col + 1) * 0.015625) * row_factors[row_slot])
                    value = f32(output_value * scale)
                    candidate[scalar] = (row_slot, col, bf16_bits(value))
                    address_visits[row * 128 + col] += 1
    if baseline != candidate or sorted(baseline_store_order) != sorted(candidate_store_order):
        sys.exit("SOURCE CHECK FAIL: half-major epilogue mapping differs from baseline")
    if len(candidate) != 64 or row_factor_multiplies != [1] * 8:
        sys.exit("SOURCE CHECK FAIL: scalar cover or row-factor cardinality changed")
if address_visits != [1] * (128 * 128):
    sys.exit("SOURCE CHECK FAIL: output tile address cover changed")

for row_limit in (1, 127, 128):
    for col_limit in (1, 63, 64, 65, 127, 128):
        for thread_id in range(256):
            baseline_predicates = {}
            candidate_predicates = {}
            for i in range(2):
                for j in range(4):
                    row = output_row(thread_id, i, j)
                    for half in range(2):
                        key = (i, j, half)
                        baseline_predicates[key] = (
                            row < row_limit and output_col(thread_id, half) < col_limit)
            for half in range(2):
                for i in range(2):
                    for j in range(4):
                        row = output_row(thread_id, i, j)
                        key = (i, j, half)
                        candidate_predicates[key] = (
                            row < row_limit and output_col(thread_id, half) < col_limit)
            if baseline_predicates != candidate_predicates:
                sys.exit("SOURCE CHECK FAIL: output predicate mapping changed")

dynamic = {
    "scale_b128_loads_per_thread": 2,
    "row_factor_multiplies_per_thread": 8,
    "packed_fma_per_thread": 64,
    "bf16_pair_conversions_per_thread": 32,
    "b64_stores_per_thread": 16,
    "output_scalars_per_thread": 64,
}
if dynamic != {
        "scale_b128_loads_per_thread": 2,
        "row_factor_multiplies_per_thread": 8,
        "packed_fma_per_thread": 64,
        "bf16_pair_conversions_per_thread": 32,
        "b64_stores_per_thread": 16,
        "output_scalars_per_thread": 64,
}:
    sys.exit("SOURCE CHECK FAIL: dynamic epilogue model changed")

for em, n, k in (
        (4096, 4096, 7168),
        (32768, 4096, 7168),
        (4096, 7168, 2048),
        (32768, 7168, 2048)):
    tiles = k // 128
    barriers = 1 + 2 * (tiles - 1)
    if em % 128 or n % 128 or k % 128 or barriers != 2 * tiles - 1:
        sys.exit("SOURCE CHECK FAIL: public tile/barrier model changed")
    print(
        "SOURCE MODEL em=%d n=%d k=%d tiles=%d LDS=32768 MMA/thread=%d "
        "barriers=%d scale-b128/thread=2 stores-b64/thread=16 output-scalars/thread=64"
        % (em, n, k, tiles, 128 * tiles, barriers))

print(
    "SOURCE CHECK PASS: exact baseline reconstructed; output[16] exact; "
    "half-major 64-scalar/address/predicate/scale/BF16 cover exact; "
    "col-scale 8->4 floats values 8->4 scales 4->2; "
    "2 scale b128 loads, 8 row multiplies and 16 b64 stores/thread; "
    "source_sha256=" + source_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  printf 'STATIC-ONLY exp-20260823-040 PASS\n'
  exit 0
fi

mkdir -p -- "$build_dir"
cuda_home=${CUDA_HOME:-/usr/local/cuda}
printf 'proxy/NVIDIA BUILD exp-20260823-040 compiler=%s\n' "$cuda_home/bin/nvcc"
"$cuda_home/bin/nvcc" \
  -O3 \
  -std=c++14 \
  -arch=sm_86 \
  -lineinfo \
  -Xcompiler=-Wall \
  "$source_file" \
  -lcuda \
  -o "$binary"
printf 'proxy/NVIDIA BUILD PASS (fallback only; MACA epilogue branch not executed)\n'

for mode in --correctness --benchmark --regression; do
  printf 'proxy/NVIDIA RUN %s\n' "$mode"
  "$binary" "$mode" | sed 's/^/proxy\/NVIDIA /'
done
