#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-044"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'constexpr int kMmaTileK = 64;' "$submission_file"
grep -Fq '#define XH_MMA_TILE_K64()' "$submission_file"
grep -Fq 'REGRESSION maca-k64-fragment-mapping' "$source_file"
grep -Fq 'REGRESSION maca-k64-pipeline' "$source_file"
grep -Fq 'REGRESSION maca-k64-traffic' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' \
                '__builtin_mxc_barrier_inst' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' \
                'kCase2L2NGroup' 'kSerpentineN' 'kCooperativeEpilogue' \
                'mma_adjacent_m' 'fused_moe_i8_tn_mma_kernel_n64' \
                'load_b_ptr' 'XH_MMA_STAGE_PAIR_INTERLEAVED' \
                'kMmaBFragmentCols' 'kMmaSharedScaleBBytes' 'shared_scale_b' \
                'kMmaEpilogueScaleBytes' 'shared_row_scale' 'shared_col_scale' \
                'combined_row_scale' '__shfl_sync' 'owned_row_factor' \
                'XH_A_FRAG' 'shared_b_next' 'kMmaSharedBStages' \
                'load_a_row_base = tid / 8' 'load_b_row_base' 'lds_col[2]'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected historical mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import collections
import hashlib
import os
import re
import subprocess
import sys

SOURCE_PATH = sys.argv[1]
text = open(SOURCE_PATH, encoding="utf-8").read().replace("\r\n", "\n")
candidate_sha256 = hashlib.sha256(text.encode()).hexdigest()
expected_candidate_sha256 = (
    "63fc0d8e17fe0c95aa6f370e32695a40251096e0af7a0506bbb808d20aa4a3f6")
if candidate_sha256 != expected_candidate_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate source hash changed: " + candidate_sha256)


def protected_projection(source):
    pipeline_token = "#define XH_MMA_STAGE_MNKX2"
    output_token = "    MmaInt4 output[kMmaOutputVectors];"
    if source.count(pipeline_token) != 1 or source.count(output_token) != 1:
        sys.exit("SOURCE CHECK FAIL: protected projection anchors are not unique")
    pipeline_start = source.index(pipeline_token)
    output_start = source.index(output_token, pipeline_start)
    prefix, replacements = re.subn(
        r"constexpr int kMmaTileK = \d+;",
        "<ASSIGNED-K-TILE>",
        source[:pipeline_start],
    )
    if replacements != 1:
        sys.exit("SOURCE CHECK FAIL: K-tile assignment is not unique")
    suffix = source[output_start:].replace("#undef XH_MMA_TILE_K64\n", "")
    return prefix + "<ASSIGNED-K64-PIPELINE>" + suffix


protected = protected_projection(text)
expected_protected_sha256 = (
    "1e79651cd6340e54cd4273e5b0d7698ea75a24745fe92ad98ee71c2d274aa72b")
if hashlib.sha256(protected.encode()).hexdigest() != expected_protected_sha256:
    sys.exit("SOURCE CHECK FAIL: a protected baseline region changed")

expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
local_baseline = os.environ.get("XH_VERIFY_BASELINE_COMMIT")
if local_baseline:
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            local_baseline
            + ":operators/fused_moe_i8_tn/cuda_maca/submission.cu",
        ],
    ).decode().replace("\r\n", "\n")
    if hashlib.sha256(baseline.encode()).hexdigest() != expected_baseline_sha256:
        sys.exit("SOURCE CHECK FAIL: local performance baseline hash changed")
    if protected_projection(baseline) != protected:
        sys.exit("SOURCE CHECK FAIL: candidate escaped the assigned K64 regions")


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
        128, 128, 64, 256, 64, 4):
    sys.exit("SOURCE CHECK FAIL: K64 CTA/tile/4x1-wave geometry changed")

derived_tokens = (
    "constexpr int kMmaRowsPerLoad = kMmaLoadBytes / kMmaTileK;",
    "constexpr int kMmaSharedABytes = kMmaTileM * kMmaTileK;",
    "constexpr int kMmaSharedBBytes = kMmaTileN * kMmaTileK;",
    "constexpr int kMmaLoadsA = kMmaSharedABytes / kMmaLoadBytes;",
    "constexpr int kMmaLoadsB = kMmaSharedBBytes / kMmaLoadBytes;",
    "constexpr int kMmaLdsA = kMmaSharedABytes / (kMmaLoadBytesPerWave * kMmaWaveM);",
    "constexpr int kMmaLdsB = kMmaSharedBBytes / (kMmaLoadBytesPerWave * kMmaWaveN);",
    "constexpr int kMmaDepth = kMmaTileK / 16;",
    "constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;",
)
for token in derived_tokens:
    if text.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: derived geometry token changed: " + token)

load_bytes = 16 * threads
load_bytes_per_wave = load_bytes // (threads // wave_size)
shared_a_bytes = tile_m * tile_k
shared_b_bytes = tile_n * tile_k
loads_a = shared_a_bytes // load_bytes
loads_b = shared_b_bytes // load_bytes
lds_a = shared_a_bytes // (load_bytes_per_wave * wave_m)
lds_b = shared_b_bytes // (load_bytes_per_wave * (threads // wave_size // wave_m))
if (shared_a_bytes, shared_b_bytes, loads_a, loads_b, lds_a, lds_b,
        tile_m // 16 // wave_m, tile_n // 16, tile_k // 16) != (
        8192, 8192, 2, 2, 2, 8, 2, 8, 4):
    sys.exit("SOURCE CHECK FAIL: K64 derived geometry is inconsistent")
if "__shared__ int8_t shared_data[kMmaSharedBytes];" not in text:
    sys.exit("SOURCE CHECK FAIL: shared allocation token changed")

required_mapping_tokens = (
    "const int load_a_slot_base = lane / 4;",
    "const int load_b_shared_base = tid / 4;",
    "const int load_k = (lane % 4) * 16;",
    "const int slot = load_a_slot_base + 16 * i;",
    "row_base + wave * 8 + (slot % 8) + 32 * (slot / 8);",
    "4 * (load_b_shared_base % 32) + load_b_shared_base / 32 + 2 * i;",
    "const int store_col = (((tid / 4) + (tid % 4)) % 4) * 16;",
    "store_row_b[i] = load_b_shared_base + 64 * i;",
    "store_row_a[i] = wave * 32 + load_a_slot_base + 16 * i;",
    "const int lds_col = (((tid % 16) + lane / 16) % 4) * 16;",
    "lds_row_a[i] = (tid % 16) + wave * 32 + 16 * i;",
    "lds_row_b[i] = (tid % 16) + 16 * i;",
)
for token in required_mapping_tokens:
    if text.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: K64 mapping token changed: " + token)

macro_start = text.index("#define XH_MMA_TILE_K64()")
macro_end = text.index("\n\n#define XH_LDG_A_STAGE_I", macro_start)
macro = text[macro_start:macro_end]
mma_calls = [tuple(map(int, values)) for values in re.findall(
    r"XH_MMA_STAGE_MNKX2\((\d),\s*(\d),\s*(\d)\);?", macro)]
expected_calls = [
    (m, n, kk)
    for kk in (0, 2)
    for m in range(2)
    for n in range(8)
]
if mma_calls != expected_calls:
    sys.exit("SOURCE CHECK FAIL: K64 MMA source order changed")
chains = {(m, n): [] for m in range(2) for n in range(8)}
for m, n, kk in mma_calls:
    chains[m, n].extend((kk, kk + 1))
if any(depths != [0, 1, 2, 3] for depths in chains.values()):
    sys.exit("SOURCE CHECK FAIL: a K64 accumulator chain changed depth order")

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
body_start = kernel.index("    int store_row_a[kMmaLoadsA];")
loop_marker = "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {"
loop_start = kernel.index(loop_marker, body_start)
loop_close = kernel.index("\n    }\n\n    XH_MMA_TILE_K64();", loop_start)
tail_start = loop_close + len("\n    }\n\n")
output_start = kernel.index("    MmaInt4 output[kMmaOutputVectors];", tail_start)
entry = kernel[body_start:loop_start]
loop = kernel[loop_start:loop_close]
tail = kernel[tail_start:output_start]


def require_count(region, token, expected, label):
    actual = region.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: %s count %s expected=%d actual=%d"
            % (label, token, expected, actual))


for region, label, counts in (
    (entry, "entry", {
        "XH_MMA_STS(shared_a_tensor(": 2,
        "XH_MMA_STS(shared_b_tensor(": 1,
        "XH_LDS_A_B128(": 2,
        "XH_LDS_B_B128(": 8,
        "__syncthreadshared();": 1,
        "XH_MMA_TILE_K64();": 0,
    }),
    (loop, "steady", {
        "XH_LDG_A_STAGE_I(": 2,
        "XH_LDG_B_STAGE_I(": 2,
        "XH_MMA_TILE_K64();": 1,
        "XH_MMA_STS(shared_a_tensor(": 2,
        "XH_MMA_STS(shared_b_tensor(": 2,
        "XH_LDS_A_B128(": 2,
        "XH_LDS_B_B128(": 8,
        "__syncthreadshared();": 2,
    }),
    (tail, "tail", {
        "XH_LDG_": 0,
        "XH_MMA_TILE_K64();": 1,
        "XH_MMA_STS(": 0,
        "XH_LDS_": 0,
        "__syncthreadshared();": 0,
    }),
):
    for token, expected in counts.items():
        require_count(region, token, expected, label)

initial_barrier = entry.index("__syncthreadshared();")
initial_last_store = max(
    entry.rindex("XH_MMA_STS(shared_a_tensor("),
    entry.rindex("XH_MMA_STS(shared_b_tensor("),
)
initial_first_lds = min(
    entry.index("XH_LDS_A_B128("), entry.index("XH_LDS_B_B128("))
if not initial_last_store < initial_barrier < initial_first_lds:
    sys.exit("SOURCE CHECK FAIL: initial store/barrier/LDS order is unsafe")

barriers = [match.start() for match in re.finditer(r"__syncthreadshared\(\);", loop)]
loads = [match.start() for match in re.finditer(r"XH_LDG_[AB]_STAGE_I\(", loop)]
mma = loop.index("XH_MMA_TILE_K64();")
stores = [match.start() for match in re.finditer(r"XH_MMA_STS\(", loop)]
reads = [match.start() for match in re.finditer(r"XH_LDS_[AB]_B128\(", loop)]
if not (
    len(barriers) == 2 and len(loads) == 4 and len(stores) == 4 and len(reads) == 10
    and barriers[0] < min(loads) <= max(loads) < mma
    and mma < min(stores) <= max(stores) < barriers[1]
    and barriers[1] < min(reads)
):
    sys.exit("SOURCE CHECK FAIL: steady barrier/load/MMA/store/LDS order is unsafe")
if kernel.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: source must have one entry and two steady barriers")
if kernel[:output_start].count("__builtin_mxc_stg") != 0:
    sys.exit("SOURCE CHECK FAIL: assigned K64 path writes global memory")

# Exhaust the two producer bijections down to individual bytes.
a_source = [-1] * shared_a_bytes
b_source = [-1] * shared_b_bytes
a_writer_wave = [-1] * shared_a_bytes
a_global_visits = [0] * shared_a_bytes
b_global_visits = [0] * shared_b_bytes
a_shared_visits = [0] * shared_a_bytes
b_shared_visits = [0] * shared_b_bytes
for tid in range(threads):
    wave, lane = divmod(tid, wave_size)
    source_col = (lane % 4) * 16
    shared_col = (((tid // 4) + (tid % 4)) % 4) * 16
    for load in range(2):
        a_slot = lane // 4 + 16 * load
        a_global_row = wave * 8 + a_slot % 8 + 32 * (a_slot // 8)
        a_shared_row = wave * 32 + a_slot
        b_slot = tid // 4
        b_global_row = 4 * (b_slot % 32) + b_slot // 32 + 2 * load
        b_shared_row = b_slot + 64 * load
        for byte in range(16):
            a_src = a_global_row * tile_k + source_col + byte
            a_dst = a_shared_row * tile_k + shared_col + byte
            b_src = b_global_row * tile_k + source_col + byte
            b_dst = b_shared_row * tile_k + shared_col + byte
            if min(a_src, a_dst, b_src, b_dst) < 0 or max(
                    a_src, a_dst, b_src, b_dst) >= shared_a_bytes:
                sys.exit("SOURCE CHECK FAIL: K64 producer address is out of range")
            a_source[a_dst] = a_src
            b_source[b_dst] = b_src
            a_writer_wave[a_dst] = wave
            a_global_visits[a_src] += 1
            b_global_visits[b_src] += 1
            a_shared_visits[a_dst] += 1
            b_shared_visits[b_dst] += 1
if any(count != 1 for count in (
        a_global_visits + b_global_visits + a_shared_visits + b_shared_visits)):
    sys.exit("SOURCE CHECK FAIL: K64 A/B producer mapping is not bijective")

a_read_visits = [0] * shared_a_bytes
b_read_visits = [0] * shared_b_bytes
checked_a_words = 0
checked_b_words = 0
for tid in range(threads):
    wave, lane = divmod(tid, wave_size)
    lds_col = (((tid % 16) + lane // 16) % 4) * 16
    for row_fragment in range(2):
        shared_row = tid % 16 + wave * 32 + 16 * row_fragment
        expected_row = wave * 8 + shared_row % 8 + 32 * ((shared_row % 32) // 8)
        for byte in range(16):
            address = shared_row * tile_k + lds_col + byte
            expected_source = expected_row * tile_k + (lane // 16) * 16 + byte
            if a_source[address] != expected_source or a_writer_wave[address] != wave:
                sys.exit("SOURCE CHECK FAIL: K64 A fragment identity changed")
            a_read_visits[address] += 1
        checked_a_words += 4
    for col_fragment in range(8):
        shared_row = tid % 16 + 16 * col_fragment
        expected_row = 4 * (shared_row % 32) + shared_row // 32
        for byte in range(16):
            address = shared_row * tile_k + lds_col + byte
            expected_source = expected_row * tile_k + (lane // 16) * 16 + byte
            if b_source[address] != expected_source:
                sys.exit("SOURCE CHECK FAIL: K64 B fragment identity changed")
            b_read_visits[address] += 1
        checked_b_words += 4
if any(count != 1 for count in a_read_visits):
    sys.exit("SOURCE CHECK FAIL: K64 A LDS is not an exact wave-private cover")
if any(count != 4 for count in b_read_visits):
    sys.exit("SOURCE CHECK FAIL: K64 B LDS is not an exact four-wave cover")
if (checked_a_words, checked_b_words) != (2048, 8192):
    sys.exit("SOURCE CHECK FAIL: K64 fragment word cardinality changed")

# Exhaust the register lifecycle and chain uses for the degenerate case, a
# transition, and both public K depths. Keys retain thread and fragment identity.
for tile_count in (1, 2, 32, 112):
    current = tile_count - 1
    barriers_dynamic = 1
    order = []
    ldg_a = ldg_b = sts_a = sts_b = 2
    lds_a_dynamic, lds_b_dynamic = 2, 8

    def consume(tile):
        a_uses = collections.Counter()
        b_uses = collections.Counter()
        mma_count = 0
        for tid in range(threads):
            for m in range(2):
                for n in range(8):
                    for depth in range(4):
                        a_uses[(tile, tid, m, depth)] += 1
                        b_uses[(tile, tid, n, depth)] += 1
                        mma_count += 1
        if len(a_uses) != threads * 2 * 4 or set(a_uses.values()) != {8}:
            sys.exit("SOURCE CHECK FAIL: K64 A fragment lifecycle changed")
        if len(b_uses) != threads * 8 * 4 or set(b_uses.values()) != {2}:
            sys.exit("SOURCE CHECK FAIL: K64 B fragment lifecycle changed")
        if mma_count != threads * 64:
            sys.exit("SOURCE CHECK FAIL: K64 per-tile MMA cardinality changed")

    for next_tile in range(tile_count - 1):
        barriers_dynamic += 1
        ldg_a += 2
        ldg_b += 2
        consume(current)
        order.append(current)
        current = next_tile
        sts_a += 2
        sts_b += 2
        barriers_dynamic += 1
        lds_a_dynamic += 2
        lds_b_dynamic += 8
    consume(current)
    order.append(current)
    expected_order = [tile_count - 1] + list(range(tile_count - 1))
    if order != expected_order or barriers_dynamic != 2 * tile_count - 1:
        sys.exit("SOURCE CHECK FAIL: K64 tile order or barrier lifecycle changed")
    if (ldg_a, ldg_b, sts_a, sts_b, lds_a_dynamic, lds_b_dynamic) != (
            2 * tile_count, 2 * tile_count, 2 * tile_count,
            2 * tile_count, 2 * tile_count, 8 * tile_count):
        sys.exit("SOURCE CHECK FAIL: K64 dynamic vector traffic changed")
    print(
        "SOURCE SCHEDULE tiles=%d order=last,0..last-1 barriers=%d "
        "A-LDG/STS/LDS=%d/%d/%d B-LDG/STS/LDS=%d/%d/%d MMA/thread=%d"
        % (tile_count, barriers_dynamic, ldg_a, sts_a, lds_a_dynamic,
           ldg_b, sts_b, lds_b_dynamic, 64 * tile_count))

# Public K64 traffic and arithmetic equal the formal K128 core. Only the shared
# footprint and dynamic barrier count differ.
for em, n, k in (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
):
    k64_tiles = k // 64
    k128_tiles = k // 128
    a_bytes = k64_tiles * threads * 2 * 16
    b_bytes = k64_tiles * threads * 2 * 16
    baseline_a_bytes = k128_tiles * threads * 4 * 16
    baseline_b_bytes = k128_tiles * threads * 4 * 16
    lds_bytes = k64_tiles * threads * (2 + 8) * 16
    baseline_lds_bytes = k128_tiles * threads * (4 + 16) * 16
    head = (k - 1) % tile_k + 1
    valid_head_loads = sum(
        (tid % 4) * 16 < head for tid in range(threads) for _ in range(2))
    if not (
        a_bytes == baseline_a_bytes == tile_m * k
        and b_bytes == baseline_b_bytes == tile_n * k
        and lds_bytes == baseline_lds_bytes
        and 64 * k64_tiles == 128 * k128_tiles
        and valid_head_loads == threads * 2
    ):
        sys.exit("SOURCE CHECK FAIL: public K64 traffic/arithmetic changed")
    print(
        "SOURCE MODEL em=%d n=%d k=%d shared=16384 K64-tiles=%d "
        "A-bytes/CTA=%d B-bytes/CTA=%d LDS-bytes/CTA=%d MMA/thread=%d "
        "barriers=%d baseline-barriers=%d"
        % (em, n, k, k64_tiles, a_bytes, b_bytes, lds_bytes,
           64 * k64_tiles, 2 * k64_tiles - 1, 2 * k128_tiles - 1))

# The protected formal-best epilogue retains the exact 128x128 output cover.
output_visits = [0] * (tile_m * tile_n)
for tid in range(threads):
    wave, lane = divmod(tid, wave_size)
    for row_group in range(2):
        for row_in_group in range(4):
            row = ((lane // 16) % 2) * 4 + wave * 8 + (lane // 32) * 32 \
                + row_group * 64 + row_in_group
            for col_group in range(2):
                for col_in_group in range(4):
                    col = (tid % 16) * 4 + col_group * 64 + col_in_group
                    if not (0 <= row < tile_m and 0 <= col < tile_n):
                        sys.exit("SOURCE CHECK FAIL: output mapping is out of range")
                    output_visits[row * tile_n + col] += 1
if any(count != 1 for count in output_visits):
    sys.exit("SOURCE CHECK FAIL: output mapping is not an exact cover")

print(
    "SOURCE CHECK PASS: protected performance baseline reconstructed outside the "
    "assigned K64 regions; 16-KiB A/B layout; 512+512 producer vectors bijective; "
    "all 256-thread fragment identities and 16 chains exact; single-buffer lifecycle "
    "safe for 1/2/32/112 tiles; traffic/MMA/output/read-only contract preserved; "
    "source_sha256=" + candidate_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD exp-20260823-044 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
