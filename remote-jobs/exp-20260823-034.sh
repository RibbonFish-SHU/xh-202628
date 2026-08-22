#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-034"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'MmaLoad128 a_frag[kMmaRows][2];' "$submission_file"
grep -Fq 'maca-direct-a-fragment-map' "$source_file"
grep -Fq 'maca-direct-a-lifecycle' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'load_b_ptr' 'XH_MMA_STAGE_PAIR_INTERLEAVED'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
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

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")


def remove_a_load_loops(source):
    marker = (
        "#pragma unroll\n"
        "    for (uint32_t i = 0; i < kMmaLoadsA; ++i) {")
    removed = 0
    while marker in source:
        start = source.index(marker)
        close = source.index("\n    }", start) + len("\n    }")
        source = source[:start] + source[close:]
        removed += 1
    if removed not in (0, 3):
        sys.exit("SOURCE CHECK FAIL: unexpected baseline A-loop count: %d" % removed)
    return source


def canonicalize_a_path(source):
    source = source.replace("\r\n", "\n")
    for name in ("kMmaSharedABytes", "kMmaLoadsA", "kMmaLdsA"):
        source, count = re.subn(
            r"^constexpr int %s = .*;\n" % name, "", source,
            count=1, flags=re.MULTILINE)
        if count not in (0, 1):
            sys.exit("SOURCE CHECK FAIL: non-unique A constant: " + name)
    source, count = re.subn(
        r"^constexpr int kMmaSharedBytes = .*;$",
        "constexpr int kMmaSharedBytes = <A_SHARED_FOOTPRINT>;",
        source, count=1, flags=re.MULTILINE)
    if count != 1:
        sys.exit("SOURCE CHECK FAIL: shared footprint declaration is not unique")

    source = re.sub(
        r"^#define XH_A_FRAG\(m, kk\).*\n", "", source,
        count=1, flags=re.MULTILINE)
    first_variants = (
        "XH_MMA_I8(a_frag[m][kk],",
        "XH_MMA_I8(XH_A_FRAG(m, kk),",
    )
    second_variants = (
        "XH_MMA_I8(a_frag[m][kk + 1],",
        "XH_MMA_I8(XH_A_FRAG(m, (kk) + 1),",
    )
    for variants, replacement in (
        (first_variants, "XH_MMA_I8(<A_FRAGMENT_0>,"),
        (second_variants, "XH_MMA_I8(<A_FRAGMENT_1>,"),
    ):
        matches = sum(source.count(variant) for variant in variants)
        if matches != 1:
            sys.exit("SOURCE CHECK FAIL: A MMA operand is not exact and unique")
        for variant in variants:
            source = source.replace(variant, replacement)

    a_ldg_start = source.index("#define XH_LDG_A_")
    b_ldg_start = source.index("#define XH_LDG_B_STAGE_I", a_ldg_start)
    source = source[:a_ldg_start] + source[b_ldg_start:]

    b_lds_start = source.index("#define XH_LDS_B_B128")
    a_lds_start = source.rfind("#define XH_LDS_A_B128", 0, b_lds_start)
    b_ldg_start = source.rfind("#define XH_LDG_B_STAGE_I", 0, b_lds_start)
    if a_lds_start > b_ldg_start:
        source = source[:a_lds_start] + source[b_lds_start:]

    shared_decl = "    __shared__ int8_t shared_data[kMmaSharedBytes];"
    shared_decl_end = source.index("\n", source.index(shared_decl)) + 1
    expert_start = source.index("    const int expert =", shared_decl_end)
    source = source[:shared_decl_end] + "    <A_SHARED_LAYOUT>\n\n" + source[expert_start:]

    source = remove_a_load_loops(source)
    direct_setup = (
        "    const int a_frag_row =\n"
        "        row_base + 8 * wave + (lane % 8) + 32 * ((lane % 16) / 8);\n"
        "    const uint64_t a_frag_row_offset = static_cast<uint64_t>(a_frag_row) * k;\n"
        "    const int a_frag_k = (lane / 16) * 16;\n")
    source = source.replace(direct_setup, "")

    tensor_marker = "    Tensor shared_a_tensor = make_tensor("
    if tensor_marker in source:
        start = source.index(tensor_marker)
        end = source.index("    Tensor shared_b_tensor = make_tensor(", start)
        source = source[:start] + source[end:]

    filtered = []
    a_line_prefixes = (
        "MmaLoad128 load_a[",
        "MmaLoad128 a_frag[",
        "int32_t a_frag[",
        "int load_a_row_offset[",
        "const int load_a_row_base =",
        "int store_row_a[",
        "int lds_row_a[",
        "lds_row_a[i] =",
        "XH_LDG_A_FRAG(",
        "XH_LDG_A_STAGE_I(",
        "XH_LDS_A_B128(",
        "XH_MMA_STS(shared_a_tensor(",
        "a_base +=",
        "#undef XH_LDG_A_FRAG",
        "#undef XH_LDG_A_STAGE_I",
        "#undef XH_LDS_A_B128",
        "#undef XH_A_FRAG",
    )
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(a_line_prefixes):
            continue
        filtered.append("".join(line.split()))
    return "\n".join(filtered) + "\n"


protected = canonicalize_a_path(text)
protected_sha256 = hashlib.sha256(protected.encode()).hexdigest()
expected_protected_sha256 = "3f08f99f4e8afab269364c62286e0f99e8a0dbcf2a404dd22e2f416d3006997e"
print("SOURCE PROTECTED SHA256=" + protected_sha256)
if protected_sha256 != expected_protected_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: source outside the A path changed: "
        + protected_sha256)

local_baseline = os.environ.get("XH_VERIFY_BASELINE_COMMIT")
if local_baseline:
    baseline = subprocess.check_output(
        ["git", "show", local_baseline + ":operators/fused_moe_i8_tn/cuda_maca/submission.cu"],
        text=True,
    )
    if canonicalize_a_path(baseline) != protected:
        sys.exit("SOURCE CHECK FAIL: local baseline protected projection differs")


def count(region, token, expected, label):
    actual = region.count(token)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: %s count %s expected=%d actual=%d"
            % (label, token, expected, actual))


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
    sys.exit("SOURCE CHECK FAIL: CTA or wave geometry changed")
if text.count("constexpr int kMmaSharedBytes = kMmaSharedBBytes;") != 1:
    sys.exit("SOURCE CHECK FAIL: LDS is not the exact 16 KiB B tile")
for removed in (
    "kMmaSharedABytes", "kMmaLoadsA", "kMmaLdsA", "shared_a_tensor",
    "XH_LDS_A_B128", "XH_LDG_A_STAGE_I", "load_a_row_offset", "store_row_a",
):
    if removed in text:
        sys.exit("SOURCE CHECK FAIL: removed A staging token remains: " + removed)

required_direct = (
    "#define XH_A_FRAG(m, kk) a_frag[m][(kk) / 4][(kk) % 4]",
    "MmaLoad128 a_frag[kMmaRows][2];",
    "row_base + 8 * wave + (lane % 8) + 32 * ((lane % 16) / 8);",
    "const uint64_t a_frag_row_offset = static_cast<uint64_t>(a_frag_row) * k;",
    "const int a_frag_k = (lane / 16) * 16;",
    "a_base + a_frag_row_offset + static_cast<uint64_t>(64 * (m)) * k",
    "+ a_frag_k + 64 * (half),",
)
for token in required_direct:
    if text.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: direct A token is not exact and unique: " + token)

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
loop_marker = "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {"
loop_start = kernel.index(loop_marker)
loop_end = kernel.index("\n    }\n\n    int output_row[8];", loop_start)
body_start = kernel.index("    MmaLoad128 a_frag[kMmaRows][2];")
initial = kernel[body_start:loop_start]
loop = kernel[loop_start:loop_end]
tail_start = loop_end + len("\n    }\n\n")
output_start = kernel.index("    MmaInt4 output[kMmaOutputVectors];", tail_start)
tail = kernel[tail_start:output_start]

direct_load_re = re.compile(r"^\s*XH_LDG_A_FRAG\((\d),\s*(\d)\);$", re.MULTILINE)
expected_load_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
for region, label, expected in (
    (initial, "initial", expected_load_order),
    (loop, "steady", expected_load_order),
    (tail, "tail", []),
):
    actual = [tuple(map(int, match)) for match in direct_load_re.findall(region)]
    if actual != expected:
        sys.exit("SOURCE CHECK FAIL: %s direct A load order is %r" % (label, actual))

mma_re = re.compile(r"^\s*XH_MMA_STAGE_MNKX2\((\d),\s*(\d),\s*(\d)\);$")
load_re = re.compile(r"^\s*XH_LDG_A_FRAG\((\d),\s*(\d)\);$")


def parse_events(region):
    events = []
    for line_number, line in enumerate(region.splitlines()):
        mma = mma_re.match(line)
        load = load_re.match(line)
        if mma:
            m, n, depth = map(int, mma.groups())
            events.append(("mma", line_number, m, n, depth))
        elif load:
            m, half = map(int, load.groups())
            events.append(("load", line_number, m, half))
    return events


loop_events = parse_events(loop)
for m in range(2):
    for half in range(2):
        uses = [
            event for event in loop_events
            if event[0] == "mma" and event[2] == m and event[4] // 4 == half
        ]
        loads = [
            event for event in loop_events
            if event[0] == "load" and event[2:] == (m, half)
        ]
        if len(uses) != 16 or len(loads) != 1:
            sys.exit("SOURCE CHECK FAIL: direct fragment use/load cardinality changed")
        final_use = uses[-1]
        expected_final_depth = 2 + 4 * half
        if final_use[2:] != (m, 7, expected_final_depth):
            sys.exit(
                "SOURCE CHECK FAIL: fragment (%d,%d) final use is %r"
                % (m, half, final_use[2:]))
        if loads[0][1] <= final_use[1]:
            sys.exit("SOURCE CHECK FAIL: direct A overwrite precedes its final use")
        if any(use[1] > loads[0][1] for use in uses):
            sys.exit("SOURCE CHECK FAIL: direct A fragment is used after overwrite")

pointer_updates = [
    index for index, line in enumerate(loop.splitlines())
    if line.strip() == "a_base += kMmaTileK;"
]
if len(pointer_updates) != 1:
    sys.exit("SOURCE CHECK FAIL: direct A base update count changed")
last_a_load_line = max(event[1] for event in loop_events if event[0] == "load")
if pointer_updates[0] <= last_a_load_line:
    sys.exit("SOURCE CHECK FAIL: A base advances before all four direct loads")


def prove_mma_chains(region, label):
    chains = {(m, n): [] for m in range(2) for n in range(8)}
    for event in parse_events(region):
        if event[0] != "mma":
            continue
        _, _, m, n, depth = event
        if (m, n) not in chains:
            sys.exit("SOURCE CHECK FAIL: %s MMA chain out of range" % label)
        chains[m, n].extend((depth, depth + 1))
    for chain, depths in chains.items():
        if depths != list(range(8)):
            sys.exit(
                "SOURCE CHECK FAIL: %s chain %r depth order is %r"
                % (label, chain, depths))
    if sum(len(depths) for depths in chains.values()) != 128:
        sys.exit("SOURCE CHECK FAIL: %s tile does not contain 128 MMAs" % label)


prove_mma_chains(loop, "steady")
prove_mma_chains(tail, "tail")


def prove_lifecycle(num_k_tiles):
    fragments = {(m, half): num_k_tiles - 1 for m in range(2) for half in range(2)}
    chains = {(m, n): [] for m in range(2) for n in range(8)}

    def execute(events, next_tile=None):
        for event in events:
            if event[0] == "mma":
                _, _, m, n, depth = event
                half = depth // 4
                tile = fragments[(m, half)]
                chains[(m, n)].extend((tile * 8 + depth, tile * 8 + depth + 1))
            elif event[0] == "load":
                _, _, m, half = event
                if next_tile is None:
                    sys.exit("SOURCE CHECK FAIL: tail unexpectedly loads A")
                fragments[(m, half)] = next_tile

    for tile in range(num_k_tiles - 1):
        execute(loop_events, tile)
    execute(parse_events(tail))
    tile_order = [num_k_tiles - 1] + list(range(num_k_tiles - 1))
    expected = [tile * 8 + depth for tile in tile_order for depth in range(8)]
    for chain, trace in chains.items():
        if trace != expected:
            sys.exit(
                "SOURCE CHECK FAIL: k_tiles=%d chain=%r lifecycle mismatch"
                % (num_k_tiles, chain))
    print(
        "SOURCE LIFECYCLE k-tiles=%d chains=16 tile-order=last,0..last-1 "
        "depth-order=0..7 PASS" % num_k_tiles)


prove_lifecycle(56)
prove_lifecycle(16)

b_stage_start = text.index("#define XH_LDG_B_STAGE_I")
b_stage_end = text.index("#define XH_LDS_B_B128", b_stage_start)
b_stage = text[b_stage_start:b_stage_end]
required_b_stage = (
    "load_b[ldgi] = __builtin_mxc_ldg_b128(",
    "&(global_b(load_b_row[ldgi], load_k, tile_k)),",
    "        0,",
    "        -1,",
    "        true,",
    "        false)",
)
for token in required_b_stage:
    if token not in b_stage:
        sys.exit("SOURCE CHECK FAIL: steady B address/control changed: " + token)

required_b = (
    "b_ptr + static_cast<uint64_t>(expert) * n * k;",
    "make_shape(n, k),",
    "make_stride(k, Int<1>{}));",
    "make_tile(Int<kMmaTileN>{}, Int<kMmaTileK>{}),",
    "make_coord(tile_n, _));",
    "const int k_head = (k - 1) % kMmaTileK + 1;",
    "const int load_b_row_base = tid / 8 * kMmaLoadsB;",
    "const int load_k = (lane % 8) * 16;",
    "const int candidate_col = load_b_row_base + i;",
    "load_b_row[i] = candidate_col < col_limit ? candidate_col : col_limit - 1;",
    "&(global_b(load_b_row[i], load_k, num_k_tiles - 1)),",
    "load_k,\n            k_head,\n            MACA_ICMP_SLT);",
    "store_row_b[i] = tid / 8 + kMmaRowsPerLoad * i;",
    "XH_MMA_STS(shared_b_tensor(store_row_b[i], store_col), load_b[i], MmaLoad128);",
    "lds_row_b[i] = (tid % 16) + 16 * i;",
)
for token in required_b:
    if text.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: B address/predicate/STS/LDS token changed: " + token)

trace_prefixes = (
    "load_b[i] = __builtin_mxc_ldg_b128_predicator(",
    "XH_LDG_B_STAGE_I(",
    "XH_LDS_B_B128(",
    "XH_MMA_STS(shared_b_tensor(",
    "__syncthreadshared();",
)
b_trace = []
trace_start = kernel.index("    MmaLoad128 a_frag[kMmaRows][2];")
for line in kernel[trace_start:output_start].splitlines():
    stripped = line.strip()
    if stripped.startswith(trace_prefixes):
        b_trace.append("".join(stripped.split()))
b_trace_sha256 = hashlib.sha256("\n".join(b_trace).encode()).hexdigest()
expected_b_trace_sha256 = "bbc4e10232a4f0055a52744fdfb8d51f5a70f1396bba49b5738215549fe9e17f"
print("SOURCE B TRACE SHA256=" + b_trace_sha256)
if b_trace_sha256 != expected_b_trace_sha256:
    sys.exit("SOURCE CHECK FAIL: B operation/barrier trace changed: " + b_trace_sha256)
expected_b_counts = {
    "load_b[i]=__builtin_mxc_ldg_b128_predicator(": 1,
    "XH_LDG_B_STAGE_I(": 4,
    "XH_LDS_B_B128(": 32,
    "XH_MMA_STS(shared_b_tensor(": 5,
    "__syncthreadshared();": 3,
}
for prefix, expected in expected_b_counts.items():
    actual = sum(item.startswith(prefix) for item in b_trace)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: B trace count %s expected=%d actual=%d"
            % (prefix, expected, actual))

if initial.count("__syncthreadshared();") != 1 \
        or loop.count("__syncthreadshared();") != 2 \
        or tail.count("__syncthreadshared();") != 0 \
        or kernel.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: barrier call sites changed")
initial_barrier = initial.index("__syncthreadshared();")
if not (
    initial.rindex("XH_MMA_STS(shared_b_tensor(") < initial_barrier
    < initial.index("XH_LDS_B_B128(")):
    sys.exit("SOURCE CHECK FAIL: initial B store/barrier/LDS order is unsafe")
first_barrier = loop.index("__syncthreadshared();")
second_barrier = loop.index("__syncthreadshared();", first_barrier + 1)
second_half_b_lds = [
    loop.index("XH_LDS_B_B128(%d, 1);" % fragment) for fragment in range(8)
]
b_stores = [match.start() for match in re.finditer(
    r"XH_MMA_STS\(shared_b_tensor\(", loop)]
next_first_half_b_lds = [
    loop.index("XH_LDS_B_B128(%d, 0);" % fragment, second_barrier)
    for fragment in range(4)
]
if not (
    max(second_half_b_lds) < first_barrier
    and all(first_barrier < position < second_barrier for position in b_stores)
    and min(next_first_half_b_lds) > second_barrier):
    sys.exit("SOURCE CHECK FAIL: B consume/barrier/produce ownership changed")

shared_owner = [-1] * (128 * 8)
direct_visits = [0] * (128 * 8)
b_global_visits = [0] * (128 * 8)
b_shared_visits = [0] * (128 * 8)
for thread_id in range(256):
    wave = thread_id // 64
    lane = thread_id % 64
    store_chunk = ((thread_id // 8) + (thread_id % 8)) % 8
    for load in range(4):
        global_row = thread_id // 8 + 32 * load
        shared_row = wave * 32 + lane // 8 + 8 * load
        shared_index = shared_row * 8 + store_chunk
        if shared_owner[shared_index] != -1:
            sys.exit("SOURCE CHECK FAIL: baseline A STS aliases shared vector")
        shared_owner[shared_index] = global_row * 8 + lane % 8

        b_global_row = (thread_id // 8) * 4 + load
        b_shared_row = thread_id // 8 + 32 * load
        b_global_visits[b_global_row * 8 + lane % 8] += 1
        b_shared_visits[b_shared_row * 8 + store_chunk] += 1
if any(owner < 0 for owner in shared_owner):
    sys.exit("SOURCE CHECK FAIL: baseline A STS shared cover is incomplete")
if any(count != 1 for count in b_global_visits + b_shared_visits):
    sys.exit("SOURCE CHECK FAIL: B producer cover changed")

identity_mismatches = 0
for wave in range(4):
    for lane in range(64):
        r = lane % 16
        q = lane // 16
        for mma_row in range(2):
            for half in range(2):
                shared_row = r + wave * 32 + 16 * mma_row
                shared_chunk = (r + q + 4 * half) % 8
                baseline_vector = shared_owner[shared_row * 8 + shared_chunk]
                direct_row = 64 * mma_row + 32 * (r // 8) + 8 * wave + r % 8
                direct_chunk = q + 4 * half
                direct_vector = direct_row * 8 + direct_chunk
                identity_mismatches += baseline_vector != direct_vector
                direct_visits[direct_vector] += 1
if identity_mismatches or any(count != 1 for count in direct_visits):
    sys.exit("SOURCE CHECK FAIL: direct A mapping is not exact baseline identity/cover")

for wave in range(4):
    b_lds_visits = [0] * (128 * 8)
    for lane in range(64):
        for mma_col in range(8):
            row = lane % 16 + 16 * mma_col
            for half in range(2):
                chunk = ((lane % 16) + lane // 16 + 4 * half) % 8
                b_lds_visits[row * 8 + chunk] += 1
    if any(count != 1 for count in b_lds_visits):
        sys.exit("SOURCE CHECK FAIL: per-wave B LDS cover changed")


def make_baseline_groups():
    return [
        [(lane, wave * 8 + lane // 8 + 32 * load, lane % 8) for lane in range(64)]
        for wave in range(4) for load in range(4)
    ]


def make_direct_groups():
    return [
        [
            (lane,
             64 * mma_row + 32 * ((lane % 16) // 8) + 8 * wave + lane % 8,
             lane // 16 + 4 * half)
            for lane in range(64)
        ]
        for wave in range(4) for mma_row in range(2) for half in range(2)
    ]


def segment_requests(groups, segment_bytes, base_offset, coalescer_lanes):
    requests = 0
    for group in groups:
        for first_lane in range(0, 64, coalescer_lanes):
            for row in range(128):
                segments = set()
                for lane, vector_row, chunk in group:
                    if first_lane <= lane < first_lane + coalescer_lanes and vector_row == row:
                        first_byte = base_offset + vector_row * 128 + chunk * 16
                        segments.add(first_byte // segment_bytes)
                        segments.add((first_byte + 15) // segment_bytes)
                requests += len(segments)
    return requests


def unique_segments(groups, segment_bytes, base_offset):
    segments = set()
    for group in groups:
        for _, row, chunk in group:
            first_byte = base_offset + row * 128 + chunk * 16
            segments.add(first_byte // segment_bytes)
            segments.add((first_byte + 15) // segment_bytes)
    return len(segments)


baseline_groups = make_baseline_groups()
direct_groups = make_direct_groups()
request_cases = (
    (32, 0, 512, 512, 512),
    (32, 16, 640, 768, 513),
    (64, 0, 256, 256, 256),
    (64, 16, 384, 512, 257),
    (64, 32, 384, 512, 257),
    (64, 48, 384, 512, 257),
    (128, 0, 128, 256, 128),
    (128, 16, 256, 384, 129),
    (128, 32, 256, 384, 129),
    (128, 48, 256, 384, 129),
    (128, 64, 256, 256, 129),
    (128, 80, 256, 384, 129),
    (128, 96, 256, 384, 129),
    (128, 112, 256, 384, 129),
)
for segment, offset, expected_baseline, expected_direct, expected_unique in request_cases:
    baseline_requests = segment_requests(baseline_groups, segment, offset, 64)
    direct_requests = segment_requests(direct_groups, segment, offset, 64)
    baseline_unique = unique_segments(baseline_groups, segment, offset)
    direct_unique = unique_segments(direct_groups, segment, offset)
    if (baseline_requests, direct_requests, baseline_unique, direct_unique) != (
            expected_baseline, expected_direct, expected_unique, expected_unique):
        sys.exit("SOURCE CHECK FAIL: A request/coalescing table changed")
for segment in (32, 64, 128):
    if segment_requests(baseline_groups, segment, 0, 16) != 16384 // segment \
            or segment_requests(direct_groups, segment, 0, 16) != 1024:
        sys.exit("SOURCE CHECK FAIL: 16-lane A coalescer diagnostic changed")

shared_bytes = tile_n * tile_k
a_ldg_per_tile_per_thread = len(expected_load_order)
a_sts_per_tile_per_thread = 0
a_lds_per_tile_per_thread = 0
mma_per_tile_per_thread = 2 * 8 * 8
if (shared_bytes, a_ldg_per_tile_per_thread, a_sts_per_tile_per_thread,
        a_lds_per_tile_per_thread, mma_per_tile_per_thread) != (16384, 4, 0, 0, 128):
    sys.exit("SOURCE CHECK FAIL: direct A resource model changed")
for k in (7168, 2048):
    k_tiles = k // tile_k
    barriers = 1 + 2 * (k_tiles - 1)
    a_bytes = 4 * k_tiles * 16 * threads
    b_bytes = 4 * k_tiles * 16 * threads
    if a_bytes != tile_m * k or b_bytes != tile_n * k:
        sys.exit("SOURCE CHECK FAIL: A/B byte model changed")
    print(
        "SOURCE MODEL k=%d k-tiles=%d LDS-bytes=%d A-LDG/thread=%d "
        "A-STS/thread=0 A-LDS/thread=0 B-bytes/CTA=%d MMA/thread=%d barriers=%d"
        % (k, k_tiles, shared_bytes, 4 * k_tiles, b_bytes,
           mma_per_tile_per_thread * k_tiles, barriers))

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: protected baseline exact outside A path; 1024 direct vectors "
    "match baseline STS-to-LDS identity and exact cover; overwrites follow final uses; "
    "all 16 chains preserve last,0..last-1 and depth 0..7; B addresses/predicates/"
    "STS/LDS/barriers exact; LDS=16384; A STS/LDS=0; source-sha256=" + source_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD exp-20260823-034 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
