#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-033"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'constexpr int kMmaWaveM = 4;' "$submission_file"
grep -Fq 'constexpr int kMmaSharedB1Offset = kMmaSharedB0Offset + kMmaSharedBBytes;' "$submission_file"
grep -Fq 'shared_b_inactive_tensor' "$submission_file"
grep -Fq 'maca-b-pingpong-layout' "$source_file"
grep -Fq 'maca-b-pingpong-reuse' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' \
                '__builtin_mxc_barrier_inst' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' \
                'kCase2L2NGroup' 'kSerpentineN' 'kCooperativeEpilogue' \
                'mma_adjacent_m' 'fused_moe_i8_tn_mma_kernel_n64' \
                'load_b_ptr' 'XH_MMA_STAGE_PAIR_INTERLEAVED' \
                'constexpr int kMmaWaveM = 2;'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import collections
import hashlib
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read().replace("\r\n", "\n")


def replace_region(source, start_token, end_token, replacement, label):
    if source.count(start_token) != 1:
        sys.exit("SOURCE CHECK FAIL: %s start token is not unique" % label)
    start = source.index(start_token)
    end = source.index(end_token, start)
    return source[:start] + replacement + source[end:]


def replace_exact(source, old, new, expected_count, label):
    actual = source.count(old)
    if actual != expected_count:
        sys.exit(
            "SOURCE CHECK FAIL: %s expected=%d actual=%d"
            % (label, expected_count, actual))
    return source.replace(old, new)


reconstructed = text
reconstructed = replace_region(
    reconstructed,
    "constexpr int kMmaSharedAOffset = 0;",
    "\n\n#define XH_MMA_FENCE",
    "constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;",
    "shared constants",
)

macro_start = reconstructed.index("#define XH_LDS_B_B128(rowi, coli)")
macro_header_end = reconstructed.index("\n", macro_start) + 1
macro_header = reconstructed[macro_start:macro_header_end]
reconstructed = replace_region(
    reconstructed,
    "#define XH_LDS_B_B128(rowi, coli)",
    "#define XH_CVT_F32_TO_BF16",
    macro_header
    + "    XH_MMA_LDS(b_frag[rowi][coli * 4], shared_b_tensor(lds_row_b[rowi], lds_col[coli]), MmaLoad128)\n\n",
    "active-B LDS macro",
)

reconstructed = replace_region(
    reconstructed,
    "    int8_t* shared_a = shared_data + kMmaSharedAOffset;",
    "\n\n    const int expert =",
    "    int8_t* shared_a = shared_data;\n"
    "    int8_t* shared_b = shared_a + kMmaSharedABytes;",
    "shared pointers",
)

reconstructed = replace_region(
    reconstructed,
    "    Tensor shared_b_active_tensor = make_tensor(",
    "\n\n    int store_row_a[kMmaLoadsA];",
    "    Tensor shared_b_tensor = make_tensor(\n"
    "        make_smem_ptr(shared_b),\n"
    "        make_shape(Int<kMmaTileN>{}, Int<kMmaTileK>{}),\n"
    "        make_stride(Int<kMmaTileK>{}, Int<1>{}));",
    "shared B tensors",
)

reconstructed = replace_exact(
    reconstructed,
    "shared_b_active_tensor(store_row_b[i], store_col)",
    "shared_b_tensor(store_row_b[i], store_col)",
    1,
    "initial active-B store",
)
reconstructed = replace_exact(
    reconstructed,
    "shared_b_inactive_tensor(store_row_b[",
    "shared_b_tensor(store_row_b[",
    4,
    "steady inactive-B stores",
)
reconstructed = replace_exact(
    reconstructed,
    "        Tensor shared_b_swap_tensor = shared_b_active_tensor;\n"
    "        shared_b_active_tensor = shared_b_inactive_tensor;\n"
    "        shared_b_inactive_tensor = shared_b_swap_tensor;\n",
    "",
    1,
    "B tensor swap",
)
reconstructed = replace_exact(
    reconstructed,
    "        XH_MMA_STAGE_MNKX2(0, 7, 6);\n\n"
    "        XH_MMA_STAGE_MNKX2(1, 0, 0);",
    "        XH_MMA_STAGE_MNKX2(0, 7, 6);\n\n"
    "        __syncthreadshared();\n"
    "        XH_MMA_STAGE_MNKX2(1, 0, 0);",
    1,
    "removed steady barrier",
)

baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61"
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: change escaped B ping-pong sites: " + baseline_sha256)

required = (
    "constexpr int kMmaSharedAOffset = 0;",
    "constexpr int kMmaSharedB0Offset = kMmaSharedAOffset + kMmaSharedABytes;",
    "constexpr int kMmaSharedB1Offset = kMmaSharedB0Offset + kMmaSharedBBytes;",
    "constexpr int kMmaSharedBytes = kMmaSharedB1Offset + kMmaSharedBBytes;",
    "int8_t* shared_a = shared_data + kMmaSharedAOffset;",
    "int8_t* shared_b_active = shared_data + kMmaSharedB0Offset;",
    "int8_t* shared_b_inactive = shared_data + kMmaSharedB1Offset;",
    "shared_b_active_tensor(lds_row_b[rowi], lds_col[coli])",
    "shared_b_active_tensor(store_row_b[i], store_col)",
    "Tensor shared_b_swap_tensor = shared_b_active_tensor;",
    "shared_b_active_tensor = shared_b_inactive_tensor;",
    "shared_b_inactive_tensor = shared_b_swap_tensor;",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: B ping-pong token missing: " + token)


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
wave_m = read_const("kMmaWaveM")
if (tile_m, tile_n, tile_k, threads, wave_size, wave_m) != (128, 128, 128, 256, 64, 4):
    sys.exit("SOURCE CHECK FAIL: formal-best 4x1 CTA geometry changed")

a_bytes = tile_m * tile_k
b_bytes = tile_n * tile_k
regions = ((0, a_bytes), (a_bytes, a_bytes + b_bytes),
           (a_bytes + b_bytes, a_bytes + 2 * b_bytes))
shared_bytes = a_bytes + 2 * b_bytes
coverage = [0] * shared_bytes
for begin, end in regions:
    if begin % 16 or end % 16 or begin < 0 or end > shared_bytes:
        sys.exit("SOURCE CHECK FAIL: shared region is out of range or misaligned")
    for address in range(begin, end):
        coverage[address] += 1
if regions != ((0, 16384), (16384, 32768), (32768, 49152)):
    sys.exit("SOURCE CHECK FAIL: shared A/B0/B1 layout is not exact")
if any(count != 1 for count in coverage):
    sys.exit("SOURCE CHECK FAIL: shared A/B0/B1 regions overlap or leave holes")
if "__shared__ int8_t shared_data[kMmaSharedBytes];" not in text:
    sys.exit("SOURCE CHECK FAIL: 48-KiB shared allocation is missing")

body_start = text.index("    int store_row_a[kMmaLoadsA];")
loop_start = text.index(
    "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {",
    body_start,
)
loop_end = text.index("\n    }\n\n    int output_row[8];", loop_start)
tail_start = loop_end + len("\n    }\n\n")
output_start = text.index("    MmaInt4 output[kMmaOutputVectors];", tail_start)
initial = text[body_start:loop_start]
loop = text[loop_start:loop_end]
tail = text[tail_start:output_start]


def require_count(region, token, expected, label):
    actual = region.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: %s count %s expected=%d actual=%d"
            % (label, token, expected, actual))


for region, label, counts in (
    (initial, "initial", {
        "XH_MMA_STS(shared_a_tensor(": 2,
        "XH_MMA_STS(shared_b_active_tensor(": 1,
        "XH_MMA_STS(shared_b_inactive_tensor(": 0,
        "XH_LDS_A_B128(": 1,
        "XH_LDS_B_B128(": 4,
        "__syncthreadshared();": 1,
    }),
    (loop, "steady", {
        "XH_LDG_A_STAGE_I(": 4,
        "XH_LDG_B_STAGE_I(": 4,
        "XH_MMA_STAGE_MNKX2(": 64,
        "XH_LDS_A_B128(": 4,
        "XH_LDS_B_B128(": 16,
        "XH_MMA_STS(shared_a_tensor(": 4,
        "XH_MMA_STS(shared_b_inactive_tensor(": 4,
        "XH_MMA_STS(shared_b_active_tensor(": 0,
        "__syncthreadshared();": 1,
        "Tensor shared_b_swap_tensor": 1,
    }),
    (tail, "tail", {
        "XH_LDG_": 0,
        "XH_MMA_STAGE_MNKX2(": 64,
        "XH_LDS_A_B128(": 3,
        "XH_LDS_B_B128(": 12,
        "XH_MMA_STS(shared_a_tensor(": 2,
        "XH_MMA_STS(shared_b_": 0,
        "__syncthreadshared();": 0,
        "shared_b_swap_tensor": 0,
    }),
):
    for token, expected in counts.items():
        require_count(region, token, expected, label)

if text.count("__syncthreadshared();") != 2:
    sys.exit("SOURCE CHECK FAIL: source must contain initial and one steady barrier only")

initial_barrier = initial.index("__syncthreadshared();")
initial_last_store = max(
    initial.rindex("XH_MMA_STS(shared_a_tensor("),
    initial.rindex("XH_MMA_STS(shared_b_active_tensor("),
)
initial_first_read = min(
    initial.index("XH_LDS_A_B128("), initial.index("XH_LDS_B_B128("))
if not initial_last_store < initial_barrier < initial_first_read:
    sys.exit("SOURCE CHECK FAIL: initial store/barrier/read order is unsafe")

steady_barrier = loop.index("__syncthreadshared();")
first_inactive_store = loop.index("XH_MMA_STS(shared_b_inactive_tensor(")
inactive_stores = [match.start() for match in re.finditer(
    r"XH_MMA_STS\(shared_b_inactive_tensor\(", loop)]
a_stores = [match.start() for match in re.finditer(
    r"XH_MMA_STS\(shared_a_tensor\(", loop)]
active_b_reads = [match.start() for match in re.finditer(r"XH_LDS_B_B128\(", loop)]
current_b_reads = [position for position in active_b_reads if position < first_inactive_store]
next_b_reads = [position for position in active_b_reads if position > steady_barrier]
current_a_last_read = loop.index("XH_LDS_A_B128(1, 1);")
swap_begin = loop.index("Tensor shared_b_swap_tensor")
swap_end = loop.index("shared_b_inactive_tensor = shared_b_swap_tensor;")
next_a_read = loop.index("XH_LDS_A_B128(0, 0);", swap_end)
if not (
    len(current_b_reads) == 12
    and len(next_b_reads) == 4
    and max(current_b_reads) < first_inactive_store
    and current_a_last_read < first_inactive_store
    and all(position < steady_barrier for position in inactive_stores + a_stores)
    and steady_barrier < swap_begin <= swap_end
    and swap_end < min(next_a_read, min(next_b_reads))
):
    sys.exit("SOURCE CHECK FAIL: steady read/store/barrier/swap/read order is unsafe")


def prove_mma_chains(region, label):
    chains = {(m, n): [] for m in range(2) for n in range(8)}
    for m, n, kk in re.findall(
            r"XH_MMA_STAGE_MNKX2\((\d),\s*(\d),\s*(\d)\);", region):
        m, n, kk = int(m), int(n), int(kk)
        if (m, n) not in chains:
            sys.exit("SOURCE CHECK FAIL: %s MMA chain out of range" % label)
        chains[m, n].extend((kk, kk + 1))
    for chain, depths in chains.items():
        if depths != list(range(8)):
            sys.exit(
                "SOURCE CHECK FAIL: %s chain %s depths=%s"
                % (label, chain, depths))


prove_mma_chains(loop, "steady")
prove_mma_chains(tail, "tail")

# Exhaust the unchanged swizzled A stores and LDS reads. Every byte is written
# and read exactly once, and each read stays within its physical producer wave.
a_writer_wave = [-1] * a_bytes
a_writer_source = [-1] * a_bytes
a_write_count = [0] * a_bytes
for tid in range(threads):
    wave = tid // wave_size
    lane = tid % wave_size
    load_k = (lane % 8) * 16
    store_col = (((tid // 8) + (tid % 8)) % 8) * 16
    for load in range(4):
        global_row = tid // 8 + 32 * load
        store_row = wave * 32 + lane // 8 + 8 * load
        for byte in range(16):
            address = store_row * tile_k + store_col + byte
            source = global_row * tile_k + load_k + byte
            if not (0 <= address < a_bytes and 0 <= source < a_bytes):
                sys.exit("SOURCE CHECK FAIL: A store model is out of range")
            a_write_count[address] += 1
            a_writer_wave[address] = wave
            a_writer_source[address] = source

a_read_count = [0] * a_bytes
for tid in range(threads):
    wave = tid // wave_size
    lane = tid % wave_size
    for row_fragment in range(2):
        lds_row = tid % 16 + wave * 32 + 16 * row_fragment
        for half in range(2):
            lds_col = ((tid % 16 + lane // 16 + 4 * half) % 8) * 16
            for byte in range(16):
                address = lds_row * tile_k + lds_col + byte
                if a_writer_source[address] < 0 or a_writer_wave[address] != wave:
                    sys.exit("SOURCE CHECK FAIL: A LDS is not wave-private")
                a_read_count[address] += 1
if any(count != 1 for count in a_write_count + a_read_count):
    sys.exit("SOURCE CHECK FAIL: A ownership is not an exact write/read cover")

# Exhaust the unchanged B producer/consumer mapping. Each buffer byte has one
# CTA producer and is consumed once by each of the four waves.
def write_b_tile(buffer, tile_label):
    visits = [0] * b_bytes
    for tid in range(threads):
        lane = tid % wave_size
        load_k = (lane % 8) * 16
        store_col = (((tid // 8) + (tid % 8)) % 8) * 16
        for load in range(4):
            global_row = (tid // 8) * 4 + load
            store_row = tid // 8 + 32 * load
            for byte in range(16):
                address = store_row * tile_k + store_col + byte
                source = global_row * tile_k + load_k + byte
                if not (0 <= address < b_bytes and 0 <= source < b_bytes):
                    sys.exit("SOURCE CHECK FAIL: B producer model is out of range")
                visits[address] += 1
                buffer[address] = tile_label
    if any(count != 1 for count in visits):
        sys.exit("SOURCE CHECK FAIL: B producer is not an exact cover")


def consume_b_tile(buffer, expected_tile):
    visits = [0] * b_bytes
    for tid in range(threads):
        lane = tid % wave_size
        for row_fragment in range(8):
            lds_row = tid % 16 + 16 * row_fragment
            for half in range(2):
                lds_col = ((tid % 16 + lane // 16 + 4 * half) % 8) * 16
                for byte in range(16):
                    address = lds_row * tile_k + lds_col + byte
                    if buffer[address] != expected_tile:
                        sys.exit("SOURCE CHECK FAIL: B consumer observed a stale tile")
                    visits[address] += 1
    if any(count != 4 for count in visits):
        sys.exit("SOURCE CHECK FAIL: B consumer is not a four-wave exact cover")


buffers = [[-1] * b_bytes, [-1] * b_bytes]
labels = [-1, -1]
consumed_masks = [0, 0]
active = 0
write_b_tile(buffers[active], 2)
labels[active] = 2
two_iteration_order = []
for next_tile in range(2):
    consume_b_tile(buffers[active], labels[active])
    consumed_masks[active] = (1 << (threads // wave_size)) - 1
    two_iteration_order.append(labels[active])
    inactive = active ^ 1
    if labels[inactive] >= 0 and consumed_masks[inactive] != 0xf:
        sys.exit("SOURCE CHECK FAIL: B buffer reused before the CTA barrier")
    write_b_tile(buffers[inactive], next_tile)
    labels[inactive] = next_tile
    consumed_masks[inactive] = 0
    active = inactive
consume_b_tile(buffers[active], labels[active])
two_iteration_order.append(labels[active])
if two_iteration_order != [2, 0, 1]:
    sys.exit("SOURCE CHECK FAIL: two-iteration B labels are incorrect")

# Global address and predicate formulas remain the formal-best formulas. Each
# tile has one exact A and B byte cover; all public last-K predicates are true.
a_global_visits = [0] * a_bytes
b_global_visits = [0] * b_bytes
for tid in range(threads):
    load_k = (tid % 8) * 16
    for load in range(4):
        a_row = tid // 8 + 32 * load
        b_row = (tid // 8) * 4 + load
        for byte in range(16):
            a_global_visits[a_row * tile_k + load_k + byte] += 1
            b_global_visits[b_row * tile_k + load_k + byte] += 1
if any(count != 1 for count in a_global_visits + b_global_visits):
    sys.exit("SOURCE CHECK FAIL: global A/B addresses are not exact tile covers")

required_global_tokens = (
    "a_base + load_a_row_offset[ldgi] + load_k,",
    "&(global_b(load_b_row[ldgi], load_k, tile_k)),",
    "&(global_b(load_b_row[i], load_k, num_k_tiles - 1)),",
    "load_k,\n            k_head,\n            MACA_ICMP_SLT);",
)
for token in required_global_tokens:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: formal-best global access token missing: " + token)

# The formal 4x1 output mapping remains a one-to-one cover.
output_visits = [0] * (tile_m * tile_n)
for tid in range(threads):
    wave = tid // wave_size
    lane = tid % wave_size
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

lds_per_tile = 4 + 16
sts_per_tile = 4 + 4
mma_per_tile = 2 * 8 * 8
if (lds_per_tile, sts_per_tile, mma_per_tile) != (20, 8, 128):
    sys.exit("SOURCE CHECK FAIL: per-tile LDS/STS/MMA model changed")

for tile_count in (1, 2, 16, 56):
    labels = [tile_count - 1, -1]
    active = 0
    barriers = 1
    consumed = []
    for next_tile in range(tile_count - 1):
        consumed.append(labels[active])
        inactive = active ^ 1
        labels[inactive] = next_tile
        barriers += 1
        active = inactive
    consumed.append(labels[active])
    expected = [tile_count - 1] + list(range(tile_count - 1))
    if consumed != expected or barriers != tile_count:
        sys.exit("SOURCE CHECK FAIL: tile-label/barrier simulation failed")
    print(
        "SOURCE SCHEDULE tiles=%d order=last,0..last-1 barriers=%d baseline-barriers=%d"
        % (tile_count, barriers, 2 * tile_count - 1))

for em, n, k in (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
):
    k_tiles = k // tile_k
    global_a_bytes = 4 * k_tiles * 16 * threads
    global_b_bytes = 4 * k_tiles * 16 * threads
    initial_b_predicates = sum(
        ((tid % 8) * 16) < ((k - 1) % tile_k + 1)
        for tid in range(threads) for _ in range(4))
    if global_a_bytes != tile_m * k or global_b_bytes != tile_n * k:
        sys.exit("SOURCE CHECK FAIL: public global traffic changed")
    if initial_b_predicates != 4 * threads:
        sys.exit("SOURCE CHECK FAIL: public last-K predicate changed")
    print(
        "SOURCE MODEL em=%d n=%d k=%d shared=%d A-bytes/CTA=%d B-bytes/CTA=%d "
        "LDS=%d STS=%d MMA=%d barriers=%d"
        % (em, n, k, shared_bytes, global_a_bytes, global_b_bytes,
           lds_per_tile * k_tiles, sts_per_tile * k_tiles,
           mma_per_tile * k_tiles, k_tiles))

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: formal-best reconstructed; only B ping-pong sites changed; "
    "A wave-private; B producer/reuse ordered; 48-KiB exact cover; all 16 chains "
    "depth=0..7; global traffic/predicates/output/read-only contract preserved; sha256="
    + source_sha256)
PYEOF

printf 'BUILD exp-20260823-033 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
