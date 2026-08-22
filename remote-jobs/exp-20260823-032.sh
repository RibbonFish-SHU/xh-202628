#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-032"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'constexpr int kMmaWaveM = 2;' "$submission_file"
grep -Fq 'maca-wave-2x2-fragments' "$source_file"
grep -Fq 'cross-wave-A=' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' 'load_b_ptr'; do
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


def canonicalize_protected_source(source):
    start = source.index(
        "__host__ __device__ __forceinline__ int mma_output_row_local(")
    end = source.index("#if XH_FUSED_MOE_MACA", start)
    source = source[:start] + "<OUTPUT_MAPPING>\n\n" + source[end:]
    source, count = re.subn(
        r"constexpr int kMmaWaveM = \d+;",
        "<WAVE_DECOMPOSITION>",
        source,
        count=1,
    )
    if count != 1:
        sys.exit("SOURCE CHECK FAIL: wave-decomposition constant not unique")
    start = source.index("    const int wave = tid / kMmaWaveSize;")
    end = source.index("    const int row_base = tile_m * kMmaTileM;", start)
    end = source.index("\n", end) + 1
    source = source[:start] + "    <WAVE_INDEXING>\n" + source[end:]
    start = source.index("    int store_row_a[kMmaLoadsA];")
    end = source.index("#undef XH_CVT_F32_TO_BF16", start)
    return source[:start] + "    <WAVE_BODY>\n\n" + source[end:]


protected_sha256 = hashlib.sha256(
    canonicalize_protected_source(text).encode()).hexdigest()
if protected_sha256 != "1b2393a9f1bbde0b7559121d5c9e4fc72e14cc3357a458866132ab824e178503":
    sys.exit(
        "SOURCE CHECK FAIL: source outside wave decomposition changed: "
        + protected_sha256)


def read_const(name):
    match = re.search(r"constexpr int %s = (\d+);" % name, text)
    if not match:
        sys.exit("SOURCE CHECK FAIL: constant not found: " + name)
    return int(match.group(1))


tile_m = read_const("kMmaTileM")
tile_n = read_const("kMmaTileN")
tile_k = read_const("kMmaTileK")
threads = read_const("kMmaThreads")
wave_size = read_const("kMmaWaveSize")
wave_m_count = read_const("kMmaWaveM")
waves = threads // wave_size
wave_n_count = waves // wave_m_count
mma_rows = tile_m // 16 // wave_m_count
mma_cols = tile_n // 16 // wave_n_count
mma_depth = tile_k // 16
if (tile_m, tile_n, tile_k, threads, wave_size) != (128, 128, 128, 256, 64):
    sys.exit("SOURCE CHECK FAIL: CTA geometry changed")
if (wave_m_count, wave_n_count, mma_rows, mma_cols) != (2, 2, 4, 4):
    sys.exit("SOURCE CHECK FAIL: wave decomposition is not 2x2 with 4x4 fragments")
if tile_m * tile_k + tile_n * tile_k != 32 * 1024:
    sys.exit("SOURCE CHECK FAIL: shared-memory footprint changed")

required = (
    "const int wave_m = wave / kMmaWaveN;",
    "const int wave_n = wave % kMmaWaveN;",
    "lds_row_a[i] = (tid % 16) + wave_m * 64 + 16 * i;",
    "lds_row_b[i] = (tid % 16) + wave_n * 16 + 32 * i;",
    "int32_t a_frag[kMmaRows][kMmaDepth];",
    "int32_t b_frag[kMmaCols][kMmaDepth];",
    "MmaInt4 accum[kMmaRows][kMmaCols] = {0};",
    "output[i * 4 + j][0] = accum[i][0][j];",
    "output[i * 4 + j][3] = accum[i][3][j];",
    "const int output_col = mma_output_col_local(tid, 0);",
    "float weights[4][4];",
    "float row_scale[4][4];",
    "MmaFloat4 col_scale;",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: 2x2 token missing: " + token)

body_start = text.index("    int store_row_a[kMmaLoadsA];")
loop_start = text.index(
    "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {",
    body_start,
)
tail_marker = "\n    }\n\n    XH_MMA_STAGE_MNKX2(0, 0, 0);"
tail_marker_start = text.index(tail_marker, loop_start)
tail_start = tail_marker_start + len("\n    }\n\n")
output_start = text.index("    MmaInt4 output[kMmaOutputVectors];", tail_start)
body_end = text.index("#undef XH_CVT_F32_TO_BF16", output_start)
initial = text[body_start:loop_start]
loop = text[loop_start:tail_marker_start]
tail = text[tail_start:output_start]
epilogue = text[output_start:body_end]


def count(region, token, expected, label):
    actual = region.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: %s count %s expected=%d actual=%d"
            % (label, token, expected, actual))


for region, label, expected in (
    (initial, "initial", {
        "XH_MMA_STS(shared_a_tensor(": 4,
        "XH_MMA_STS(shared_b_tensor(": 1,
        "XH_LDS_A_B128(": 4,
        "XH_LDS_B_B128(": 4,
        "__syncthreadshared();": 1,
    }),
    (loop, "steady", {
        "XH_LDG_A_STAGE_I(": 4,
        "XH_LDG_B_STAGE_I(": 4,
        "XH_MMA_STAGE_MNKX2(": 64,
        "XH_LDS_A_B128(": 8,
        "XH_LDS_B_B128(": 8,
        "XH_MMA_STS(shared_a_tensor(": 4,
        "XH_MMA_STS(shared_b_tensor(": 4,
        "__syncthreadshared();": 2,
    }),
    (tail, "tail", {
        "XH_MMA_STAGE_MNKX2(": 64,
        "XH_LDS_A_B128(": 4,
        "XH_LDS_B_B128(": 4,
        "XH_MMA_STS(": 0,
        "XH_LDG_": 0,
        "__syncthreadshared();": 0,
    }),
):
    for token, expected_count in expected.items():
        count(region, token, expected_count, label)

if text.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: source must contain exactly three CTA barriers")

initial_barrier = initial.index("__syncthreadshared();")
last_initial_store = max(
    initial.rindex("XH_MMA_STS(shared_a_tensor("),
    initial.rindex("XH_MMA_STS(shared_b_tensor("),
)
first_initial_lds = min(
    initial.index("XH_LDS_A_B128("), initial.index("XH_LDS_B_B128("))
if not last_initial_store < initial_barrier < first_initial_lds:
    sys.exit("SOURCE CHECK FAIL: initial all-store/barrier/first-LDS order is unsafe")

ldg_order = re.findall(r"XH_LDG_([AB])_STAGE_I\((\d)\);", loop)
if ldg_order != [
    ("B", "0"), ("B", "1"), ("B", "2"), ("B", "3"),
    ("A", "0"), ("A", "1"), ("A", "2"), ("A", "3"),
]:
    sys.exit("SOURCE CHECK FAIL: steady LDG order changed or was hoisted")

first_barrier = loop.index("__syncthreadshared();")
second_barrier = loop.index("__syncthreadshared();", first_barrier + 1)
second_half_lds = [
    loop.index("XH_LDS_%s_B128(%d, 1);" % (operand, fragment))
    for operand in ("A", "B") for fragment in range(4)
]
next_stores = [match.start() for match in re.finditer(
    r"XH_MMA_STS\(shared_[ab]_tensor\(", loop)]
next_first_half_lds = [
    loop.index("XH_LDS_%s_B128(%d, 0);" % (operand, fragment), second_barrier)
    for operand in ("A", "B") for fragment in range(4)
]
if not (
    max(second_half_lds) < first_barrier
    and all(first_barrier < position < second_barrier for position in next_stores)
    and min(next_first_half_lds) > second_barrier
):
    sys.exit("SOURCE CHECK FAIL: consume/store/produce barrier ownership is unsafe")


def prove_mma_chains(region, label):
    chains = {(m, n): [] for m in range(4) for n in range(4)}
    for m, n, kk in re.findall(
            r"XH_MMA_STAGE_MNKX2\((\d), (\d), (\d)\);", region):
        m, n, kk = int(m), int(n), int(kk)
        if (m, n) not in chains:
            sys.exit("SOURCE CHECK FAIL: %s MMA chain out of range" % label)
        chains[m, n].extend((kk, kk + 1))
    for chain, depths in chains.items():
        if depths != list(range(8)):
            sys.exit(
                "SOURCE CHECK FAIL: %s MMA chain %s depth order is %s"
                % (label, chain, depths))


prove_mma_chains(loop, "steady")
prove_mma_chains(tail, "tail")

count(epilogue, "__builtin_mxc_ldg_b32_predicator(", 2, "epilogue")
count(epilogue, "__builtin_mxc_ldg_b128_predicator(", 1, "epilogue")
count(epilogue, "XH_CVT_F32_TO_BF16(", 2, "epilogue")
count(epilogue, "__builtin_mxc_stg_b64_predicator(", 1, "epilogue")

visits = [0] * (tile_m * tile_n)
for thread_id in range(threads):
    wave = thread_id // wave_size
    lane = thread_id % wave_size
    wave_m = wave // 2
    wave_n = wave % 2
    for mma_row in range(4):
        for row_in_vector in range(4):
            row = (
                ((lane // 16) % 2) * 4
                + wave_m * 16
                + (mma_row // 2) * 8
                + (lane // 32) * 32
                + (mma_row % 2) * 64
                + row_in_vector
            )
            for col_in_vector in range(4):
                col = (thread_id % 16) * 4 + wave_n * 64 + col_in_vector
                if not (0 <= row < tile_m and 0 <= col < tile_n):
                    sys.exit("SOURCE CHECK FAIL: output mapping is out of range")
                visits[row * tile_n + col] += 1
if any(value != 1 for value in visits):
    sys.exit("SOURCE CHECK FAIL: output mapping is not an exact cover")

a_fragments = mma_rows * mma_depth
b_fragments = mma_cols * mma_depth
accumulator_vectors = mma_rows * mma_cols
lds_b128 = 2 * (mma_rows + mma_cols)
mma_per_tile = mma_rows * mma_cols * mma_depth
output_bytes = threads * accumulator_vectors * 8
if not (
    accumulator_vectors == 16
    and a_fragments == 32
    and b_fragments == 32
    and a_fragments + b_fragments == 64
    and lds_b128 == 16
    and mma_per_tile == 128
    and output_bytes == tile_m * tile_n * 2
):
    sys.exit("SOURCE CHECK FAIL: fragment/resource model changed")

for em, n, k in (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
):
    k_tiles = k // tile_k
    a_bytes = 4 * k_tiles * 16 * threads
    b_bytes = 4 * k_tiles * 16 * threads
    mma_instructions = mma_per_tile * k_tiles
    barriers = 1 + 2 * (k_tiles - 1)
    if a_bytes != tile_m * k or b_bytes != tile_n * k:
        sys.exit("SOURCE CHECK FAIL: A/B global traffic changed")
    print(
        "SOURCE MODEL em=%d n=%d k=%d A-bytes/CTA=%d B-bytes/CTA=%d "
        "LDS-b128/thread=%d MMA=%d barriers=%d"
        % (em, n, k, a_bytes, b_bytes, lds_b128 * k_tiles,
           mma_instructions, barriers))

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: protected baseline exact; 2x2 waves; output exact-cover; "
    "A32+B32 fragments; 16 LDS b128/tile; 128 MMA/tile; barriers ordered; sha256="
    + source_sha256)
PYEOF

printf 'BUILD exp-20260823-032 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
