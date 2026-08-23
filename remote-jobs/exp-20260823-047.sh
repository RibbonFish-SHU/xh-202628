#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-047"
submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_ldg_b128_bsm(' "$submission_file"
grep -Fq '__syncthreads();' "$submission_file"
grep -Fq 'REGRESSION maca-safe-six-bsm' "$source_file"

for rejected in '__dp4a' 'mma_kernel_64' 'wide_mma' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaEpilogueScaleBytes' \
                'shared_row_scale' 'shared_col_scale' 'combined_row_scale' \
                'kMmaSharedScaleBBytes' 'shared_b_alt' 'shared_b_next' \
                '2 * kMmaSharedBytes' 'kMmaSharedBytes * 2' \
                '__builtin_mxc_arrive' '__builtin_mxc_barrier_inst' \
                '__builtin_mxc_barrier_and_wait'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import re
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as source:
    text = source.read().replace("\r\n", "\n").replace("\r", "\n")


def replace_one(source, candidate, baseline, label):
    count = source.count(candidate)
    if count != 1:
        sys.exit("SOURCE CHECK FAIL: %s count=%d" % (label, count))
    return source.replace(candidate, baseline, 1)


candidate_macros = """#define XH_LDG_A_STAGE_I(ldgi)                                                                    \\
    load_a[ldgi] = __builtin_mxc_ldg_b128(                                                        \\
        a_base + load_a_row_offset[ldgi] + load_k,                                                \\
        0,                                                                                         \\
        -1,                                                                                        \\
        true,                                                                                      \\
        true,                                                                                      \\
        false,                                                                                     \\
        false)

#define XH_BSM_A_STAGE_I(bsmi, tile_index)                                                        \\
    __builtin_mxc_ldg_b128_bsm(                                                                    \\
        shared_a + store_row_a[bsmi] * kMmaTileK + store_col,                                    \\
        const_cast<void*>(reinterpret_cast<const void*>(                                          \\
            a_ptr + load_a_row_offset[bsmi] + (tile_index) * kMmaTileK + load_k)),                 \\
        0,                                                                                         \\
        -1,                                                                                        \\
        true,                                                                                      \\
        true,                                                                                      \\
        false,                                                                                     \\
        false)

#define XH_BSM_B_STAGE_I(bsmi, tile_index)                                                        \\
    __builtin_mxc_ldg_b128_bsm(                                                                    \\
        shared_b + store_row_b[bsmi] * kMmaTileK + store_col,                                    \\
        const_cast<void*>(reinterpret_cast<const void*>(                                          \\
            expert_b + static_cast<uint64_t>(                                                     \\
                tile_n * kMmaTileN + load_b_row[bsmi]) * k                                        \\
                + (tile_index) * kMmaTileK + load_k)),                                            \\
        0,                                                                                         \\
        -1,                                                                                        \\
        true,                                                                                      \\
        true,                                                                                      \\
        false,                                                                                     \\
        false)"""

baseline_macros = """#define XH_LDG_A_STAGE_I(ldgi)                                                                    \\
    load_a[ldgi] = __builtin_mxc_ldg_b128(                                                        \\
        a_base + load_a_row_offset[ldgi] + load_k,                                                \\
        0,                                                                                         \\
        -1,                                                                                        \\
        true,                                                                                      \\
        true,                                                                                      \\
        false,                                                                                     \\
        false)

#define XH_LDG_B_STAGE_I(ldgi)                                                                    \\
    load_b[ldgi] = __builtin_mxc_ldg_b128(                                                        \\
        &(global_b(load_b_row[ldgi], load_k, tile_k)),                                            \\
        0,                                                                                         \\
        -1,                                                                                        \\
        true,                                                                                      \\
        true,                                                                                      \\
        false,                                                                                     \\
        false)"""

reconstructed = replace_one(
    text, candidate_macros, baseline_macros, "safe six-path macros")


def kernel_slice(source):
    start = source.index("__global__ void fused_moe_i8_tn_mma_kernel(")
    end = source.index("\n#else", start)
    return source[start:end]


kernel = kernel_slice(reconstructed)
loop_anchor = "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {"
tail_anchor = "\n    int output_row[8];"
loop_start = kernel.index(loop_anchor)
loop_end = kernel.index(tail_anchor, loop_start)
loop = kernel[loop_start:loop_end]

loop = replace_one(
    loop,
    """    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {
        XH_MMA_STAGE_MNKX2(0, 0, 0);""",
    """    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {
        XH_LDG_B_STAGE_I(0);
        XH_LDG_B_STAGE_I(1);
        XH_MMA_STAGE_MNKX2(0, 0, 0);""",
    "steady B LDG 0/1")
loop = replace_one(
    loop,
    """        XH_LDS_B_B128(6, 0);
        XH_MMA_STAGE_MNKX2(0, 1, 2);""",
    """        XH_LDS_B_B128(6, 0);
        XH_LDG_B_STAGE_I(2);
        XH_MMA_STAGE_MNKX2(0, 1, 2);""",
    "steady B LDG 2")
loop = replace_one(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 2, 0);
        XH_MMA_STAGE_MNKX2(0, 2, 2);""",
    """        XH_MMA_STAGE_MNKX2(0, 2, 0);
        XH_LDG_B_STAGE_I(3);
        XH_MMA_STAGE_MNKX2(0, 2, 2);""",
    "steady B LDG 3")
loop = replace_one(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 3, 0);
        XH_MMA_STAGE_MNKX2(0, 3, 2);""",
    """        XH_MMA_STAGE_MNKX2(0, 3, 0);
        XH_LDG_A_STAGE_I(0);
        XH_MMA_STAGE_MNKX2(0, 3, 2);
        XH_LDG_A_STAGE_I(1);""",
    "steady A LDG 0/1")

for operand, index, baseline in (
        ("B", 0, "XH_MMA_STS(shared_b_tensor(store_row_b[0], store_col), load_b[0], MmaLoad128);"),
        ("B", 1, "XH_MMA_STS(shared_b_tensor(store_row_b[1], store_col), load_b[1], MmaLoad128);"),
        ("B", 2, "XH_MMA_STS(shared_b_tensor(store_row_b[2], store_col), load_b[2], MmaLoad128);"),
        ("B", 3, "XH_MMA_STS(shared_b_tensor(store_row_b[3], store_col), load_b[3], MmaLoad128);"),
        ("A", 0, "XH_MMA_STS(shared_a_tensor(store_row_a[0], store_col), load_a[0], MmaLoad128);"),
        ("A", 1, "XH_MMA_STS(shared_a_tensor(store_row_a[1], store_col), load_a[1], MmaLoad128);")):
    loop = replace_one(
        loop,
        "XH_BSM_%s_STAGE_I(%d, tile_k);" % (operand, index),
        baseline,
        "steady %s%d BSM site" % (operand, index))

loop = replace_one(
    loop,
    """        XH_MMA_STAGE_MNKX2(1, 5, 4);
        __syncthreads();
        XH_MMA_STAGE_MNKX2(1, 5, 6);""",
    """        XH_MMA_STAGE_MNKX2(1, 5, 4);
        __syncthreadshared();
        XH_MMA_STAGE_MNKX2(1, 5, 6);""",
    "compiler-managed completion barrier")

reconstructed_kernel = kernel[:loop_start] + loop + kernel[loop_end:]
reconstructed = reconstructed.replace(kernel, reconstructed_kernel, 1)
reconstructed = replace_one(
    reconstructed,
    """#undef XH_BSM_B_STAGE_I
#undef XH_BSM_A_STAGE_I
#undef XH_LDG_A_STAGE_I""",
    """#undef XH_LDG_B_STAGE_I
#undef XH_LDG_A_STAGE_I""",
    "stage helper undefs")

expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
if baseline_sha256 != expected_baseline_sha256:
    sys.exit("SOURCE CHECK FAIL: reverse hash mismatch: " + baseline_sha256)

candidate_kernel = kernel_slice(text)
baseline_kernel = kernel_slice(reconstructed)
candidate_loop_start = candidate_kernel.index(loop_anchor)
candidate_loop_end = candidate_kernel.index(tail_anchor, candidate_loop_start)
candidate_loop = candidate_kernel[candidate_loop_start:candidate_loop_end]
tail_end_anchor = "\n#undef XH_CVT_F32_TO_BF16"
candidate_tail_end = candidate_kernel.index(tail_end_anchor, candidate_loop_end)
baseline_tail_start = baseline_kernel.index(tail_anchor)
baseline_tail_end = baseline_kernel.index(tail_end_anchor, baseline_tail_start)
candidate_tail = candidate_kernel[candidate_loop_end:candidate_tail_end]
baseline_tail = baseline_kernel[baseline_tail_start:baseline_tail_end]

if text.count(candidate_macros) != 1 or text.count("__builtin_mxc_ldg_b128_bsm(") != 2:
    sys.exit("SOURCE CHECK FAIL: exact flat-pointer is_async=false macros changed")
if any(token in text for token in (
        "__builtin_mxc_arrive", "__builtin_mxc_barrier_inst",
        "__builtin_mxc_barrier_and_wait")):
    sys.exit("SOURCE CHECK FAIL: manual BSM completion mechanism found")
if candidate_kernel.count("__syncthreadshared();") != 2 \
        or candidate_kernel.count("__syncthreads();") != 1:
    sys.exit("SOURCE CHECK FAIL: three source barrier sites changed")
if candidate_loop.count("__syncthreadshared();") != 1 \
        or candidate_loop.count("__syncthreads();") != 1:
    sys.exit("SOURCE CHECK FAIL: steady 2T boundary sites changed")

candidate_prologue_start = candidate_kernel.index("    const int tid = threadIdx.x;")
candidate_prologue_end = candidate_kernel.index("    __syncthreadshared();")
baseline_prologue_start = baseline_kernel.index("    const int tid = threadIdx.x;")
baseline_prologue_end = baseline_kernel.index("    __syncthreadshared();")
if (candidate_kernel[candidate_prologue_start:candidate_prologue_end]
        != baseline_kernel[baseline_prologue_start:baseline_prologue_end]):
    sys.exit("SOURCE CHECK FAIL: formal-best initial prologue changed")
if candidate_tail != baseline_tail:
    sys.exit("SOURCE CHECK FAIL: formal-best peeled A2/A3 tail changed")

calls = re.findall(r"^\s*XH_BSM_([AB])_STAGE_I\((\d), ([^)]+)\);$", candidate_loop, re.M)
expected_calls = [
    ("B", "0", "tile_k"), ("B", "1", "tile_k"),
    ("B", "2", "tile_k"), ("B", "3", "tile_k"),
    ("A", "0", "tile_k"), ("A", "1", "tile_k")]
if calls != expected_calls:
    sys.exit("SOURCE CHECK FAIL: six BSM identities/order changed: %r" % (calls,))
if re.findall(r"^\s*XH_LDG_A_STAGE_I\((\d)\);$", candidate_loop, re.M) != ["2", "3"]:
    sys.exit("SOURCE CHECK FAIL: A2/A3-only steady register production changed")
if "XH_LDG_B_STAGE_I" in candidate_kernel:
    sys.exit("SOURCE CHECK FAIL: steady B register LDG helper remains")

placement_snippets = (
    """        XH_MMA_STAGE_MNKX2(1, 4, 0);
        XH_BSM_B_STAGE_I(0, tile_k);
        XH_MMA_STAGE_MNKX2(1, 4, 2);""",
    """        XH_MMA_STAGE_MNKX2(1, 5, 2);
        XH_BSM_B_STAGE_I(1, tile_k);
        XH_MMA_STAGE_MNKX2(1, 6, 0);""",
    """        XH_MMA_STAGE_MNKX2(1, 7, 0);
        XH_BSM_B_STAGE_I(2, tile_k);
        XH_MMA_STAGE_MNKX2(1, 7, 2);""",
    """        XH_MMA_STAGE_MNKX2(1, 0, 6);
        XH_BSM_B_STAGE_I(3, tile_k);
        XH_MMA_STAGE_MNKX2(1, 1, 4);""",
    """        XH_MMA_STAGE_MNKX2(1, 2, 4);
        XH_BSM_A_STAGE_I(0, tile_k);
        XH_MMA_STAGE_MNKX2(1, 2, 6);""",
    """        XH_MMA_STAGE_MNKX2(1, 3, 6);
        XH_BSM_A_STAGE_I(1, tile_k);

        XH_MMA_STAGE_MNKX2(1, 4, 4);""")
if any(candidate_loop.count(snippet) != 1 for snippet in placement_snippets):
    sys.exit("SOURCE CHECK FAIL: BSM moved from formal-best STS position")

a2_sts = "XH_MMA_STS(shared_a_tensor(store_row_a[2], store_col), load_a[2], MmaLoad128);"
a3_sts = "XH_MMA_STS(shared_a_tensor(store_row_a[3], store_col), load_a[3], MmaLoad128);"
if candidate_kernel.count(a2_sts) != 2 or candidate_kernel.count(a3_sts) != 2:
    sys.exit("SOURCE CHECK FAIL: loop/tail ordinary A2/A3 STS cardinality changed")
loop_order = (
    candidate_loop.index(a2_sts), candidate_loop.index(a3_sts),
    candidate_loop.index("XH_LDG_A_STAGE_I(2);"),
    candidate_loop.index("XH_LDG_A_STAGE_I(3);"),
    candidate_loop.index("XH_LDS_A_B128(1, 0);"),
    candidate_loop.index("__syncthreadshared();"),
    candidate_loop.index("XH_LDS_A_B128(1, 1);"))
if list(loop_order) != sorted(loop_order):
    sys.exit("SOURCE CHECK FAIL: A2/A3 STS-production-LDS rotation order changed")
tail_order = (
    candidate_tail.index(a2_sts), candidate_tail.index(a3_sts),
    candidate_tail.index("XH_LDS_A_B128(1, 0);"),
    candidate_tail.index("XH_LDS_A_B128(1, 1);"))
if list(tail_order) != sorted(tail_order):
    sys.exit("SOURCE CHECK FAIL: peeled-tail A2/A3 STS-before-LDS order changed")

first_barrier = candidate_loop.index("__syncthreadshared();")
completion = candidate_loop.index("__syncthreads();")
last_bsm = candidate_loop.index("XH_BSM_A_STAGE_I(1, tile_k);")
first_next_lds = candidate_loop.index("XH_LDS_A_B128(0, 0);", completion)
if not first_barrier < last_bsm < completion < first_next_lds:
    sys.exit("SOURCE CHECK FAIL: six BSM completion interval changed")
if "XH_MMA_STS(shared_b_tensor" in candidate_loop \
        or "load_a[0] =" in candidate_loop or "load_a[1] =" in candidate_loop:
    sys.exit("SOURCE CHECK FAIL: converted register path remains")

mma_sequence = re.findall(r"XH_MMA_STAGE_MNKX2\([^\n]+", candidate_kernel)
baseline_mma_sequence = re.findall(r"XH_MMA_STAGE_MNKX2\([^\n]+", baseline_kernel)
lds_sequence = re.findall(r"XH_LDS_[AB]_B128\([^\n]+", candidate_kernel)
baseline_lds_sequence = re.findall(r"XH_LDS_[AB]_B128\([^\n]+", baseline_kernel)
if mma_sequence != baseline_mma_sequence or len(mma_sequence) != 129:
    sys.exit("SOURCE CHECK FAIL: formal-best MMA order changed")
if lds_sequence != baseline_lds_sequence or len(lds_sequence) != 42:
    sys.exit("SOURCE CHECK FAIL: formal-best LDS order changed")


def literal(name):
    match = re.search(r"constexpr int %s = (\d+);" % name, text)
    if not match:
        sys.exit("SOURCE CHECK FAIL: missing constant " + name)
    return int(match.group(1))


tile_m, tile_n, tile_k = map(literal, ("kMmaTileM", "kMmaTileN", "kMmaTileK"))
threads, wave_size = map(literal, ("kMmaThreads", "kMmaWaveSize"))
if (tile_m, tile_n, tile_k, threads, wave_size) != (128, 128, 128, 256, 64):
    sys.exit("SOURCE CHECK FAIL: geometry changed")
if "__shared__ int8_t shared_data[kMmaSharedBytes];" not in candidate_kernel:
    sys.exit("SOURCE CHECK FAIL: 32-KiB single image changed")


def operand_map(operand):
    vectors = 1024
    src_visits = [0] * vectors
    dst_visits = [0] * vectors
    image = [-1] * vectors
    destinations = [[] for _ in range(4)]
    producers = {}
    for tid in range(threads):
        wave, lane = divmod(tid, wave_size)
        src_chunk = lane % 8
        dst_chunk = ((tid // 8) + (tid % 8)) % 8
        for load in range(4):
            if operand == "A":
                src_row = tid // 8 + 32 * load
                dst_row = wave * 32 + lane // 8 + 8 * load
            else:
                src_row = (tid // 8) * 4 + load
                dst_row = tid // 8 + 32 * load
            src = src_row * 8 + src_chunk
            dst = dst_row * 8 + dst_chunk
            src_visits[src] += 1
            dst_visits[dst] += 1
            image[dst] = src
            destinations[load].append(dst)
            producers[dst] = tid
    if src_visits != [1] * vectors or dst_visits != [1] * vectors or -1 in image:
        sys.exit("SOURCE CHECK FAIL: %s 1024-vector bijection changed" % operand)
    return destinations, producers


a_destinations, a_producers = operand_map("A")
operand_map("B")
a23_destinations = set(a_destinations[2] + a_destinations[3])
a23_reads = []
cross_lane = 0
for tid in range(threads):
    wave, lane = divmod(tid, wave_size)
    for col in range(2):
        row = tid % 16 + wave * 32 + 16
        chunk = ((tid % 16) + (lane // 16) + 4 * col) % 8
        vector = row * 8 + chunk
        producer = a_producers[vector]
        if vector not in a23_destinations or producer // wave_size != wave:
            sys.exit("SOURCE CHECK FAIL: A2/A3 LDS is not same-wave owned")
        cross_lane += producer != tid
        a23_reads.append(vector)
if set(a23_reads) != a23_destinations or cross_lane == 0:
    sys.exit("SOURCE CHECK FAIL: A2/A3 same-wave cross-lane cover changed")

for tile_count in (1, 2, 16, 56):
    shared_a = [-1] * 1024
    shared_b = [tile_count - 1] * 1024
    staged_a23 = tile_count - 1
    consumed = []
    boundaries = 1
    bsm_calls = 0
    ordinary_stores = 0
    pending_checks = 0

    def fill_a(load, tile):
        for dst in a_destinations[load]:
            shared_a[dst] = tile

    def check_a(load, tile):
        if any(shared_a[dst] != tile for dst in a_destinations[load]):
            sys.exit("SOURCE CHECK FAIL: stale A lifecycle partition")

    fill_a(0, tile_count - 1)
    fill_a(1, tile_count - 1)
    for next_tile in range(tile_count - 1):
        current = tile_count - 1 if next_tile == 0 else next_tile - 1
        check_a(0, current)
        check_a(1, current)
        if shared_b != [current] * 1024:
            sys.exit("SOURCE CHECK FAIL: stale B lifecycle image")
        fill_a(2, staged_a23)
        fill_a(3, staged_a23)
        ordinary_stores += 2
        check_a(2, current)
        check_a(3, current)
        consumed.append(current)
        staged_a23 = next_tile
        boundaries += 1

        pending_b = [next_tile] * 1024
        pending_a01 = next_tile
        bsm_calls += 6
        check_a(0, current)
        check_a(1, current)
        if shared_b != [current] * 1024:
            sys.exit("SOURCE CHECK FAIL: BSM modeled as instantaneously visible")
        pending_checks += 1
        shared_b = pending_b
        fill_a(0, pending_a01)
        fill_a(1, pending_a01)
        boundaries += 1

    tail = tile_count - 1 if tile_count == 1 else tile_count - 2
    check_a(0, tail)
    check_a(1, tail)
    if shared_b != [tail] * 1024:
        sys.exit("SOURCE CHECK FAIL: stale peeled-tail B")
    fill_a(2, staged_a23)
    fill_a(3, staged_a23)
    ordinary_stores += 2
    check_a(2, tail)
    check_a(3, tail)
    consumed.append(tail)
    expected = [tile_count - 1] + list(range(tile_count - 1))
    if (consumed != expected or boundaries != 2 * tile_count - 1
            or bsm_calls != 6 * (tile_count - 1)
            or ordinary_stores != 2 * tile_count
            or pending_checks != tile_count - 1
            or 6 + bsm_calls + ordinary_stores != 8 * tile_count):
        sys.exit("SOURCE CHECK FAIL: lifecycle counts/labels/boundaries changed")
    print(
        "SOURCE SCHEDULE T=%d labels=%s boundaries=%d BSM=%d A23-ordinary=%d pending-checks=%d"
        % (tile_count, consumed, boundaries, bsm_calls, ordinary_stores, pending_checks))

for em, n, k in (
        (4096, 4096, 7168), (32768, 4096, 7168),
        (4096, 7168, 2048), (32768, 7168, 2048)):
    tiles = k // tile_k
    if em % tile_m or n % tile_n or k % tile_k:
        sys.exit("SOURCE CHECK FAIL: public shape alignment changed")
    print(
        "SOURCE MODEL em=%d n=%d k=%d T=%d A/B-vectors=1024/1024 "
        "A/B-bytes=16384/16384 MMA/thread/tile=128 boundaries=%d"
        % (em, n, k, tiles, 2 * tiles - 1))

readonly = (
    "const int8_t* __restrict__ a_ptr", "const int8_t* __restrict__ b_ptr",
    "const float* __restrict__ scale_a_ptr", "const float* __restrict__ scale_b_ptr",
    "const float* __restrict__ moe_weights_ptr",
    "const int32_t* __restrict__ expert_ids_ptr")
if any(argument not in candidate_kernel for argument in readonly):
    sys.exit("SOURCE CHECK FAIL: read-only argument changed")
if candidate_kernel.count("__builtin_mxc_stg_b64_predicator(") != 2:
    sys.exit("SOURCE CHECK FAIL: output stores changed")

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: exact formal-best reconstructed; six flat is_async=false BSM "
    "calls at old STS sites; pending until exact __syncthreads; A2/A3 ordinary "
    "register LDG/STS and peeled tail exact; same-wave cross-lane A23=%d; "
    "T=1/2/16/56, mappings, traffic and read-only PASS; source_sha256=%s"
    % (cross_lane, source_sha256))
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  printf 'STATIC-ONLY exp-20260823-047 PASS\n'
  exit 0
fi

mkdir -p -- "$build_dir"
cuda_home=${CUDA_HOME:-/usr/local/cuda}
printf 'proxy/NVIDIA BUILD exp-20260823-047 compiler=%s\n' "$cuda_home/bin/nvcc"
"$cuda_home/bin/nvcc" \
  -O3 \
  -std=c++14 \
  -arch=sm_86 \
  -lineinfo \
  -Xcompiler=-Wall \
  "$source_file" \
  -lcuda \
  -o "$binary"
printf 'proxy/NVIDIA BUILD PASS (fallback only; MACA safe-six BSM branch not executed)\n'

for mode in --correctness --benchmark --regression; do
  printf 'proxy/NVIDIA RUN %s\n' "$mode"
  "$binary" "$mode" | sed 's/^/proxy\/NVIDIA /'
done
