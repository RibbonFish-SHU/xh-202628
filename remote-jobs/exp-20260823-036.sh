#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-036"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'constexpr int kMmaBFragmentCols = kMmaCols / 2;' "$submission_file"
grep -Fq 'int32_t b_frag[kMmaBFragmentCols][kMmaDepth];' "$submission_file"
grep -Fq 'REGRESSION maca-b-fragment-strip-lifecycle' "$source_file"
grep -Fq 'REGRESSION maca-b-fragment-strip-a-ownership' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaSharedScaleBBytes' \
                'shared_scale_b' 'kMmaEpilogueScaleBytes' 'shared_row_scale' \
                'shared_col_scale' 'combined_row_scale' 'XH_A_FRAG' \
                'shared_b_next' 'kMmaSharedBStages'; do
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


def canonicalize_protected_source(source):
    source = source.replace("\r\n", "\n")
    source, count = re.subn(
        r"^constexpr int kMmaBFragmentCols = kMmaCols / 2;\n",
        "",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count not in (0, 1):
        sys.exit("SOURCE CHECK FAIL: B fragment-column constant is not unique")

    start = source.index("#define XH_MMA_STAGE_MNKX2")
    end = source.index("#define XH_LDG_A_STAGE_I", start)
    source = source[:start] + "<MMA_STAGE_MACRO>\n\n" + source[end:]

    start = source.index("#define XH_LDS_B_B128")
    end = source.index("#define XH_CVT_F32_TO_BF16", start)
    source = source[:start] + "<B_LDS_MACRO>\n\n" + source[end:]

    source, count = re.subn(
        r"^    int32_t b_frag\[[^\n]+;\n",
        "    <B_FRAGMENT_DECL>\n",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        sys.exit("SOURCE CHECK FAIL: B fragment declaration is not unique")

    kernel_start = source.index("__global__ void fused_moe_i8_tn_mma_kernel(")
    schedule_start = source.index("    XH_LDS_A_B128(0, 0);", kernel_start)
    schedule_end = source.index(
        "    MmaInt4 output[kMmaOutputVectors];", schedule_start)
    source = (
        source[:schedule_start]
        + "    <B_FRAGMENT_SCHEDULE>\n\n"
        + source[schedule_end:]
    )
    return source


protected = canonicalize_protected_source(text)
protected_sha256 = hashlib.sha256(protected.encode()).hexdigest()
expected_protected_sha256 = (
    "f3f21475518bbf9a8526d140f16f5e16c17ad12924412d2368a0a7227e620727")
print("SOURCE PROTECTED SHA256=" + protected_sha256)
if protected_sha256 != expected_protected_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: source outside the assigned B-fragment path changed: "
        + protected_sha256)

local_baseline = os.environ.get("XH_VERIFY_BASELINE_COMMIT")
if local_baseline:
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            local_baseline
            + ":operators/fused_moe_i8_tn/cuda_maca/submission.cu",
        ],
        text=True,
    )
    baseline_protected = canonicalize_protected_source(baseline)
    if baseline_protected != protected:
        sys.exit("SOURCE CHECK FAIL: local baseline protected projection differs")
    baseline_sha256 = hashlib.sha256(
        baseline.replace("\r\n", "\n").encode()).hexdigest()
    if baseline_sha256 != (
            "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61"):
        sys.exit("SOURCE CHECK FAIL: local performance baseline source hash changed")


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
    sys.exit("SOURCE CHECK FAIL: CTA/tile/4x1-wave geometry changed")
if tile_m * tile_k + tile_n * tile_k != 32 * 1024:
    sys.exit("SOURCE CHECK FAIL: shared-memory footprint changed")

required_exact = (
    "constexpr int kMmaBFragmentCols = kMmaCols / 2;",
    "MmaInt4 accum[kMmaRows][kMmaCols] = {0};",
    "int32_t a_frag[kMmaRows][kMmaDepth];",
    "int32_t b_frag[kMmaBFragmentCols][kMmaDepth];",
    "#define XH_MMA_STAGE_MNKX2(m, nn, bn, kk)",
    "b_frag[bn][kk]",
    "b_frag[bn][kk + 1]",
    "#define XH_LDS_B_B128(slot, rowi, coli)",
    "XH_MMA_LDS(b_frag[slot][coli * 4]",
    "__shared__ int8_t shared_data[kMmaSharedBytes];",
)
for token in required_exact:
    if text.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: required strip token is not exact/unique: " + token)
if "b_frag[nn][" in text:
    sys.exit("SOURCE CHECK FAIL: MMA still indexes a logical eight-column B array")

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
loop_marker = "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {"
loop_start = kernel.index(loop_marker)
loop_end = kernel.index("\n    }\n\n    int output_row[8];", loop_start)
tail_start = loop_end + len("\n    }\n\n")
output_start = kernel.index("    MmaInt4 output[kMmaOutputVectors];", tail_start)
entry_start = kernel.index("    int store_row_a[kMmaLoadsA];")
entry = kernel[entry_start:loop_start]
loop = kernel[loop_start:loop_end]
tail = kernel[tail_start:output_start]


def count(region, token, expected, label):
    actual = region.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: %s count %s expected=%d actual=%d"
            % (label, token, expected, actual))


for region, label, expected in (
    (entry, "entry", {
        "XH_MMA_STS(shared_a_tensor(": 2,
        "XH_MMA_STS(shared_b_tensor(": 1,
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
        "XH_MMA_STS(shared_b_tensor(": 4,
        "__syncthreadshared();": 2,
    }),
    (tail, "tail", {
        "XH_LDG_": 0,
        "XH_MMA_STAGE_MNKX2(": 64,
        "XH_LDS_A_B128(": 3,
        "XH_LDS_B_B128(": 12,
        "XH_MMA_STS(shared_a_tensor(": 2,
        "XH_MMA_STS(shared_b_tensor(": 0,
        "__syncthreadshared();": 0,
    }),
):
    for token, expected_count in expected.items():
        count(region, token, expected_count, label)
if kernel.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: kernel must retain exactly three barrier sites")

entry_barrier = entry.index("__syncthreadshared();")
last_entry_store = max(
    entry.rindex("XH_MMA_STS(shared_a_tensor("),
    entry.rindex("XH_MMA_STS(shared_b_tensor("),
)
first_entry_lds = min(
    entry.index("XH_LDS_A_B128("), entry.index("XH_LDS_B_B128("))
if not last_entry_store < entry_barrier < first_entry_lds:
    sys.exit("SOURCE CHECK FAIL: initial store/barrier/LDS ordering is unsafe")

first_barrier = loop.index("__syncthreadshared();")
second_barrier = loop.index("__syncthreadshared();", first_barrier + 1)
all_current_lds = [
    match.start() for match in re.finditer(r"XH_LDS_[AB]_B128\(", loop[:first_barrier])
]
between_lds = re.findall(
    r"XH_LDS_[AB]_B128\(", loop[first_barrier:second_barrier])
a_stores = [
    match.start() for match in re.finditer(
        r"XH_MMA_STS\(shared_a_tensor\(", loop)
]
b_stores = [
    match.start() for match in re.finditer(
        r"XH_MMA_STS\(shared_b_tensor\(", loop)
]
next_lds = [
    match.start() for match in re.finditer(
        r"XH_LDS_[AB]_B128\(", loop[second_barrier:])
]
if len(all_current_lds) != 15 or between_lds or len(next_lds) != 5:
    sys.exit("SOURCE CHECK FAIL: current/next LDS phase cardinality changed")
if not (
        len(a_stores) == 4 and len(b_stores) == 4
        and all(position < first_barrier for position in a_stores[:2])
        and all(first_barrier < position < second_barrier
                for position in a_stores[2:] + b_stores)):
    sys.exit("SOURCE CHECK FAIL: current/next shared-store barrier phase changed")
if "XH_LDS_A_B128(1, 1);" not in loop[:first_barrier]:
    sys.exit("SOURCE CHECK FAIL: row-1 high A fragment is not loaded before barrier 1")

mma_re = re.compile(
    r"^\s*XH_MMA_STAGE_MNKX2\((\d),\s*(\d),\s*(\d),\s*(\d)\);$")
a_lds_re = re.compile(r"^\s*XH_LDS_A_B128\((\d),\s*(\d)\);$")
b_lds_re = re.compile(
    r"^\s*XH_LDS_B_B128\((\d),\s*(\d),\s*(\d)\);$")


def parse_events(region):
    events = []
    for line_number, line in enumerate(region.splitlines()):
        mma = mma_re.match(line)
        a_lds = a_lds_re.match(line)
        b_lds = b_lds_re.match(line)
        if mma:
            events.append(("mma", line_number) + tuple(map(int, mma.groups())))
        elif a_lds:
            events.append(("a_lds", line_number) + tuple(map(int, a_lds.groups())))
        elif b_lds:
            events.append(("b_lds", line_number) + tuple(map(int, b_lds.groups())))
        elif line.strip() == "__syncthreadshared();":
            events.append(("barrier", line_number))
    return events


entry_events = parse_events(entry)
loop_events = parse_events(loop)
tail_events = parse_events(tail)
expected_entry_lds = [
    ("a_lds", 0, 0),
    ("b_lds", 0, 0, 0),
    ("b_lds", 1, 1, 0),
    ("b_lds", 2, 2, 0),
    ("b_lds", 3, 3, 0),
]
actual_entry_lds = [event[:1] + event[2:] for event in entry_events
                    if event[0] in ("a_lds", "b_lds")]
if actual_entry_lds != expected_entry_lds:
    sys.exit("SOURCE CHECK FAIL: initial fragment preload identity/order changed")


def prove_macro_chains(events, label):
    chains = {(m, n): [] for m in range(2) for n in range(8)}
    macro_count = 0
    for event in events:
        if event[0] != "mma":
            continue
        _, _, m, n, slot, depth = event
        macro_count += 1
        if slot != n % 4 or depth not in (0, 2, 4, 6):
            sys.exit("SOURCE CHECK FAIL: %s MMA slot/depth is invalid" % label)
        chains[(m, n)].extend((depth, depth + 1))
    if macro_count != 64:
        sys.exit("SOURCE CHECK FAIL: %s does not contain 64 x2 MMA macros" % label)
    for chain, depths in chains.items():
        if depths != list(range(8)):
            sys.exit(
                "SOURCE CHECK FAIL: %s chain=%r depth order=%r"
                % (label, chain, depths))


prove_macro_chains(loop_events, "steady")
prove_macro_chains(tail_events, "tail")


def prove_lifecycle(num_k_tiles):
    a_state = [[None] * 8 for _ in range(2)]
    b_state = [[None] * 8 for _ in range(4)]
    a_loads = collections.Counter()
    a_uses = collections.Counter()
    b_loads = collections.Counter()
    b_uses = collections.Counter()
    chains = {(m, n): [] for m in range(2) for n in range(8)}

    def load_a(tile, m, half):
        for depth in range(half * 4, half * 4 + 4):
            previous = a_state[m][depth]
            if previous is not None and a_uses[previous] != 8:
                sys.exit("SOURCE CHECK FAIL: A slot overwritten before final use")
            identity = (tile, m, depth)
            a_state[m][depth] = identity
            a_loads[identity] += 1

    def load_b(tile, slot, logical_n, half):
        if slot != logical_n % 4:
            sys.exit("SOURCE CHECK FAIL: B logical column maps to wrong slot")
        for depth in range(half * 4, half * 4 + 4):
            previous = b_state[slot][depth]
            if previous is not None and b_uses[previous] != 2:
                sys.exit("SOURCE CHECK FAIL: B slot overwritten before final use")
            identity = (tile, logical_n, depth)
            b_state[slot][depth] = identity
            b_loads[identity] += 1

    first_tile = num_k_tiles - 1
    load_a(first_tile, 0, 0)
    for slot in range(4):
        load_b(first_tile, slot, slot, 0)

    def execute(events, current_tile, next_tile):
        phase = 0
        for event in events:
            kind = event[0]
            if kind == "barrier":
                phase += 1
                continue
            if kind == "a_lds":
                _, _, m, half = event
                if phase == 1:
                    sys.exit("SOURCE CHECK FAIL: A LDS occurs between loop barriers")
                tile = current_tile if phase == 0 else next_tile
                if tile is None:
                    sys.exit("SOURCE CHECK FAIL: tail attempted next-tile A LDS")
                load_a(tile, m, half)
                continue
            if kind == "b_lds":
                _, _, slot, logical_n, half = event
                if phase == 1:
                    sys.exit("SOURCE CHECK FAIL: B LDS occurs between loop barriers")
                tile = current_tile if phase == 0 else next_tile
                if tile is None:
                    sys.exit("SOURCE CHECK FAIL: tail attempted next-tile B LDS")
                load_b(tile, slot, logical_n, half)
                continue
            _, _, m, n, slot, macro_depth = event
            for depth in (macro_depth, macro_depth + 1):
                a_identity = (current_tile, m, depth)
                b_identity = (current_tile, n, depth)
                if a_state[m][depth] != a_identity:
                    sys.exit("SOURCE CHECK FAIL: MMA consumed wrong A identity")
                if b_state[slot][depth] != b_identity:
                    sys.exit("SOURCE CHECK FAIL: MMA consumed wrong B strip identity")
                a_uses[a_identity] += 1
                b_uses[b_identity] += 1
                chains[(m, n)].append(current_tile * 8 + depth)

    for tile in range(num_k_tiles - 1):
        current_tile = first_tile if tile == 0 else tile - 1
        execute(loop_events, current_tile, tile)
    tail_tile = 0 if num_k_tiles == 1 else num_k_tiles - 2
    execute(tail_events, tail_tile, None)

    expected_a = {
        (tile, m, depth)
        for tile in range(num_k_tiles) for m in range(2) for depth in range(8)
    }
    expected_b = {
        (tile, n, depth)
        for tile in range(num_k_tiles) for n in range(8) for depth in range(8)
    }
    if set(a_loads) != expected_a or any(a_loads[key] != 1 for key in expected_a):
        sys.exit("SOURCE CHECK FAIL: A fragment load identity/cardinality changed")
    if set(b_loads) != expected_b or any(b_loads[key] != 1 for key in expected_b):
        sys.exit("SOURCE CHECK FAIL: B fragment load identity/cardinality changed")
    if any(a_uses[key] != 8 for key in expected_a):
        sys.exit("SOURCE CHECK FAIL: A fragment use cardinality changed")
    if any(b_uses[key] != 2 for key in expected_b):
        sys.exit("SOURCE CHECK FAIL: B fragment use cardinality changed")

    tile_order = [first_tile] + list(range(num_k_tiles - 1))
    expected_chain = [
        tile * 8 + depth for tile in tile_order for depth in range(8)
    ]
    for chain, actual in chains.items():
        if actual != expected_chain:
            sys.exit(
                "SOURCE CHECK FAIL: k-tiles=%d chain=%r tile/depth order changed"
                % (num_k_tiles, chain))
    print(
        "SOURCE LIFECYCLE k-tiles=%d chains=16 tile-order=last,0..last-1 "
        "depth-order=0..7 overwrite-after-final-use=PASS"
        % num_k_tiles)


for tiles in (1, 2, 16, 56):
    prove_lifecycle(tiles)

# Exhaustively prove that each physical slot/group/depth is the exact baseline
# shared vector and that every wave reads the full B tile exactly once.
for thread_id in range(threads):
    lane = thread_id % wave_size
    baseline = [[None] * 8 for _ in range(8)]
    for logical_n in range(8):
        row = lane % 16 + 16 * logical_n
        for half in range(2):
            chunk = ((thread_id % 16) + lane // 16 + 4 * half) % 8
            for offset in range(4):
                baseline[logical_n][4 * half + offset] = (
                    row * 128 + chunk * 16 + offset * 4)
    for group in range(2):
        for slot in range(4):
            logical_n = 4 * group + slot
            row = lane % 16 + 16 * logical_n
            for half in range(2):
                chunk = ((thread_id % 16) + lane // 16 + 4 * half) % 8
                for offset in range(4):
                    candidate = row * 128 + chunk * 16 + offset * 4
                    if candidate != baseline[logical_n][4 * half + offset]:
                        sys.exit("SOURCE CHECK FAIL: B slot is not baseline-identical")

for wave in range(4):
    visits = [0] * (128 * 8)
    for lane in range(wave_size):
        thread_id = wave * wave_size + lane
        for logical_n in range(8):
            row = lane % 16 + 16 * logical_n
            for half in range(2):
                chunk = ((thread_id % 16) + lane // 16 + 4 * half) % 8
                visits[row * 8 + chunk] += 1
    if visits != [1] * len(visits):
        sys.exit("SOURCE CHECK FAIL: per-wave B LDS mapping is not an exact cover")

# The moved row-1 high-half A LDS is before barrier 1. Its producer vectors are
# wave-private load_a[2:4] stores, so no cross-wave CTA synchronization is needed.
owner_wave = [-1] * (128 * 8)
owner_load = [-1] * (128 * 8)
for thread_id in range(threads):
    wave = thread_id // wave_size
    lane = thread_id % wave_size
    store_chunk = ((thread_id // 8) + (thread_id % 8)) % 8
    for load in range(4):
        store_row = wave * 32 + lane // 8 + load * 8
        index = store_row * 8 + store_chunk
        if owner_wave[index] != -1:
            sys.exit("SOURCE CHECK FAIL: A shared producer vectors alias")
        owner_wave[index] = wave
        owner_load[index] = load
for thread_id in range(threads):
    wave = thread_id // wave_size
    lane = thread_id % wave_size
    for half in range(2):
        row = (thread_id % 16) + wave * 32 + 16
        chunk = ((thread_id % 16) + lane // 16 + 4 * half) % 8
        index = row * 8 + chunk
        if owner_wave[index] != wave or owner_load[index] not in (2, 3):
            sys.exit("SOURCE CHECK FAIL: pre-barrier A row-1 LDS crosses waves")

output_tokens = (
    "int output_row[8];",
    "const int row_thread_base = row_base + mma_output_row_local(tid, 0, 0);",
    "output_row[j] = row_thread_base + j;",
    "output_row[4 + j] = row_base + mma_output_row_local(tid, 1, j);",
)
for token in output_tokens:
    if kernel.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: output mapping token changed: " + token)

a_fragments = 2 * 8
baseline_b_fragments = 8 * 8
strip_b_fragments = 4 * 8
if (a_fragments, baseline_b_fragments, strip_b_fragments,
        baseline_b_fragments - strip_b_fragments) != (16, 64, 32, 32):
    sys.exit("SOURCE CHECK FAIL: fragment resource model changed")

for k_tiles in (1, 2, 16, 56):
    a_bytes = 4 * k_tiles * 16 * threads
    b_bytes = 4 * k_tiles * 16 * threads
    a_sts = 4 * k_tiles
    b_sts = 4 * k_tiles
    a_lds = 4 * k_tiles
    b_lds = 16 * k_tiles
    mma = 128 * k_tiles
    barriers = 1 + 2 * (k_tiles - 1)
    if a_bytes != tile_m * tile_k * k_tiles:
        sys.exit("SOURCE CHECK FAIL: A global bytes changed")
    if b_bytes != tile_n * tile_k * k_tiles:
        sys.exit("SOURCE CHECK FAIL: B global bytes changed")
    if (a_sts, b_sts, a_lds, b_lds) != (
            4 * k_tiles, 4 * k_tiles, 4 * k_tiles, 16 * k_tiles):
        sys.exit("SOURCE CHECK FAIL: shared traffic changed")
    print(
        "SOURCE MODEL k-tiles=%d LDS-bytes=32768 A/B-LDG/thread=%d/%d "
        "A/B-STS/thread=%d/%d A/B-LDS/thread=%d/%d MMA/thread=%d barriers=%d"
        % (k_tiles, 4 * k_tiles, 4 * k_tiles, a_sts, b_sts,
           a_lds, b_lds, mma, barriers))

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: protected baseline exact outside B-fragment path; "
    "4x1 waves; slots=4 groups=2; B-frag int32 64->32; delta=-32/thread; "
    "all 256-thread identities and 16 chains exact; A same-wave ownership; "
    "A/B LDG=4/4 STS=4/4 LDS=4/16; MMA=128/tile; barrier sites=3; "
    "source_sha256=" + source_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD exp-20260823-036 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
