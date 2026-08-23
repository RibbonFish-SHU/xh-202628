#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-045"
submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq '__builtin_mxc_ldg_b128_bsm(' "$submission_file"
grep -Fq '__builtin_mxc_arrive(64);' "$submission_file"
grep -Fq '__builtin_mxc_barrier_inst();' "$submission_file"
grep -Fq 'REGRESSION maca-full-bsm-zero' "$source_file"

for rejected in '__dp4a' 'mma_kernel_64' 'wide_mma' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaEpilogueScaleBytes' \
                'shared_row_scale' 'shared_col_scale' 'combined_row_scale' \
                'kMmaSharedScaleBBytes' 'shared_b_alt' 'shared_b_next' \
                '2 * kMmaSharedBytes' 'kMmaSharedBytes * 2' \
                '__builtin_mxc_arrive(64 +' 'XH_BSM_ISSUE_TILE'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected historical mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import difflib
import os
import re
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as source:
    text = source.read().replace("\r\n", "\n").replace("\r", "\n")


def replace_unique(source, candidate, baseline, label):
    count = source.count(candidate)
    if count != 1:
        sys.exit(
            "SOURCE CHECK FAIL: %s block count expected=1 actual=%d"
            % (label, count))
    return source.replace(candidate, baseline, 1)


candidate_macros = """#define XH_BSM_A_STAGE_I(bsmi, tile_index)                                                        \\
    __builtin_mxc_ldg_b128_bsm(                                                                    \\
        &(shared_a_tensor(store_row_a[bsmi], store_col)),                                         \\
        const_cast<void*>(reinterpret_cast<const void*>(                                          \\
            a_ptr + load_a_row_offset[bsmi] + (tile_index) * kMmaTileK + load_k)),                 \\
        0,                                                                                         \\
        -1,                                                                                        \\
        true,                                                                                      \\
        true,                                                                                      \\
        false,                                                                                     \\
        true)

#define XH_BSM_B_STAGE_I(bsmi, tile_index)                                                        \\
    __builtin_mxc_ldg_b128_bsm(                                                                    \\
        &(shared_b_tensor(store_row_b[bsmi], store_col)),                                         \\
        const_cast<void*>(reinterpret_cast<const void*>(                                          \\
            &(global_b(load_b_row[bsmi], load_k, tile_index)))),                                  \\
        0,                                                                                         \\
        -1,                                                                                        \\
        true,                                                                                      \\
        true,                                                                                      \\
        false,                                                                                     \\
        true)"""

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

reconstructed = replace_unique(
    text, candidate_macros, baseline_macros, "official full BSM macros")

reconstructed = replace_unique(
    reconstructed,
    """    const int k_head = (k - 1) % kMmaTileK + 1;""",
    """    MmaLoad128 load_a[kMmaLoadsA];
    MmaLoad128 load_b[kMmaLoadsB];
    const int k_head = (k - 1) % kMmaTileK + 1;""",
    "initial register declarations")
reconstructed = replace_unique(
    reconstructed,
    """    const int num_k_tiles = (k + kMmaTileK - 1) / kMmaTileK;

#pragma unroll""",
    """    const int num_k_tiles = (k + kMmaTileK - 1) / kMmaTileK;

    int8_t* a_base = const_cast<int8_t*>(a_ptr) + (num_k_tiles - 1) * kMmaTileK;

#pragma unroll""",
    "baseline A pointer")
reconstructed = replace_unique(
    reconstructed,
    """#pragma unroll
    for (uint32_t i = 0; i < kMmaLoadsB; ++i) {
        const int candidate_col = load_b_row_base + i;
        load_b_row[i] = candidate_col < col_limit ? candidate_col : col_limit - 1;
    }

    Tensor shared_a_tensor""",
    """#pragma unroll
    for (uint32_t i = 0; i < kMmaLoadsB; ++i) {
        const int candidate_col = load_b_row_base + i;
        load_b_row[i] = candidate_col < col_limit ? candidate_col : col_limit - 1;
        load_b[i] = __builtin_mxc_ldg_b128_predicator(
            &(global_b(load_b_row[i], load_k, num_k_tiles - 1)),
            0,
            true,
            true,
            false,
            false,
            load_k,
            k_head,
            MACA_ICMP_SLT);
    }
#pragma unroll
    for (uint32_t i = 0; i < kMmaLoadsA; ++i) {
        load_a[i] = __builtin_mxc_ldg_b128(
            a_base + load_a_row_offset[i] + load_k,
            0,
            -1,
            true,
            true,
            false,
            false);
    }

    Tensor shared_a_tensor""",
    "initial last-tile LDG")
reconstructed = replace_unique(
    reconstructed,
    """    for (uint32_t i = 0; i < kMmaLoadsB; ++i) {
        store_row_b[i] = tid / 8 + kMmaRowsPerLoad * i;
    }""",
    """    for (uint32_t i = 0; i < kMmaLoadsB; ++i) {
        store_row_b[i] = tid / 8 + kMmaRowsPerLoad * i;
        XH_MMA_STS(shared_b_tensor(store_row_b[i], store_col), load_b[i], MmaLoad128);
    }""",
    "initial B STS")

prologue_scope_start = reconstructed.index(
    "    {\n        MmaLoad128 load_a[kMmaLoadsA];") - 1
prologue_scope_end_marker = "    }\n\n    MmaInt4 accum"
prologue_scope_end = reconstructed.index(
    prologue_scope_end_marker, prologue_scope_start)
prologue_scope = reconstructed[prologue_scope_start:prologue_scope_end + len("    }\n")]
if (prologue_scope.count("__builtin_mxc_ldg_b128_predicator(") != 1
        or prologue_scope.count("__builtin_mxc_ldg_b128(") != 1
        or prologue_scope.count("XH_MMA_STS(shared_b_tensor(") != 1
        or prologue_scope.count("XH_MMA_STS(shared_a_tensor(") != 4):
    sys.exit("SOURCE CHECK FAIL: full initial prologue scope changed")
reconstructed = (
    reconstructed[:prologue_scope_start]
    + """    XH_MMA_STS(shared_a_tensor(store_row_a[0], store_col), load_a[0], MmaLoad128);
    XH_MMA_STS(shared_a_tensor(store_row_a[1], store_col), load_a[1], MmaLoad128);
"""
    + reconstructed[prologue_scope_end + len("    }\n"):])


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

loop = replace_unique(
    loop,
    """    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {
        XH_MMA_STAGE_MNKX2(0, 0, 0);""",
    """    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {
        XH_LDG_B_STAGE_I(0);
        XH_LDG_B_STAGE_I(1);
        XH_MMA_STAGE_MNKX2(0, 0, 0);""",
    "steady B LDG 0/1")
loop = replace_unique(
    loop,
    """        XH_LDS_B_B128(6, 0);
        XH_MMA_STAGE_MNKX2(0, 1, 2);""",
    """        XH_LDS_B_B128(6, 0);
        XH_LDG_B_STAGE_I(2);
        XH_MMA_STAGE_MNKX2(0, 1, 2);""",
    "steady B LDG 2")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 2, 0);
        XH_MMA_STAGE_MNKX2(0, 2, 2);""",
    """        XH_MMA_STAGE_MNKX2(0, 2, 0);
        XH_LDG_B_STAGE_I(3);
        XH_MMA_STAGE_MNKX2(0, 2, 2);""",
    "steady B LDG 3")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 3, 0);
        XH_MMA_STAGE_MNKX2(0, 3, 2);""",
    """        XH_MMA_STAGE_MNKX2(0, 3, 0);
        XH_LDG_A_STAGE_I(0);
        XH_MMA_STAGE_MNKX2(0, 3, 2);
        XH_LDG_A_STAGE_I(1);""",
    "steady A LDG 0/1")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 2, 6);
        XH_MMA_STAGE_MNKX2(0, 3, 4);""",
    """        XH_MMA_STAGE_MNKX2(0, 2, 6);
        XH_MMA_STS(shared_a_tensor(store_row_a[2], store_col), load_a[2], MmaLoad128);
        XH_MMA_STAGE_MNKX2(0, 3, 4);""",
    "current A STS 2")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 3, 6);

        XH_MMA_STAGE_MNKX2(0, 4, 4);""",
    """        XH_MMA_STAGE_MNKX2(0, 3, 6);
        XH_MMA_STS(shared_a_tensor(store_row_a[3], store_col), load_a[3], MmaLoad128);

        XH_MMA_STAGE_MNKX2(0, 4, 4);""",
    "current A STS 3")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 4, 4);
        XH_MMA_STAGE_MNKX2(0, 4, 6);""",
    """        XH_MMA_STAGE_MNKX2(0, 4, 4);
        XH_LDG_A_STAGE_I(2);
        XH_MMA_STAGE_MNKX2(0, 4, 6);
        XH_LDG_A_STAGE_I(3);""",
    "steady A LDG 2/3")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(0, 7, 4);
        XH_MMA_STAGE_MNKX2(0, 7, 6);""",
    """        XH_MMA_STAGE_MNKX2(0, 7, 4);
        a_base += kMmaTileK;
        XH_MMA_STAGE_MNKX2(0, 7, 6);""",
    "steady A pointer increment")

bsm_batch = """        XH_BSM_A_STAGE_I(0, tile_k);
        XH_BSM_A_STAGE_I(1, tile_k);
        XH_BSM_A_STAGE_I(2, tile_k);
        XH_BSM_A_STAGE_I(3, tile_k);
        XH_BSM_B_STAGE_I(0, tile_k);
        XH_BSM_B_STAGE_I(1, tile_k);
        XH_BSM_B_STAGE_I(2, tile_k);
        XH_BSM_B_STAGE_I(3, tile_k);
"""
loop = replace_unique(loop, bsm_batch, "", "eight-call full BSM batch")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(1, 4, 0);
        XH_MMA_STAGE_MNKX2(1, 4, 2);""",
    """        XH_MMA_STAGE_MNKX2(1, 4, 0);
        XH_MMA_STS(shared_b_tensor(store_row_b[0], store_col), load_b[0], MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 4, 2);""",
    "steady B STS 0")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(1, 5, 2);
        XH_MMA_STAGE_MNKX2(1, 6, 0);""",
    """        XH_MMA_STAGE_MNKX2(1, 5, 2);
        XH_MMA_STS(shared_b_tensor(store_row_b[1], store_col), load_b[1], MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 6, 0);""",
    "steady B STS 1")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(1, 7, 0);
        XH_MMA_STAGE_MNKX2(1, 7, 2);""",
    """        XH_MMA_STAGE_MNKX2(1, 7, 0);
        XH_MMA_STS(shared_b_tensor(store_row_b[2], store_col), load_b[2], MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 7, 2);""",
    "steady B STS 2")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(1, 0, 6);
        XH_MMA_STAGE_MNKX2(1, 1, 4);""",
    """        XH_MMA_STAGE_MNKX2(1, 0, 6);
        XH_MMA_STS(shared_b_tensor(store_row_b[3], store_col), load_b[3], MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 1, 4);""",
    "steady B STS 3")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(1, 2, 4);
        XH_MMA_STAGE_MNKX2(1, 2, 6);""",
    """        XH_MMA_STAGE_MNKX2(1, 2, 4);
        XH_MMA_STS(shared_a_tensor(store_row_a[0], store_col), load_a[0], MmaLoad128);
        XH_MMA_STAGE_MNKX2(1, 2, 6);""",
    "steady A STS 0")
loop = replace_unique(
    loop,
    """        XH_MMA_STAGE_MNKX2(1, 3, 6);

        XH_MMA_STAGE_MNKX2(1, 4, 4);""",
    """        XH_MMA_STAGE_MNKX2(1, 3, 6);
        XH_MMA_STS(shared_a_tensor(store_row_a[1], store_col), load_a[1], MmaLoad128);

        XH_MMA_STAGE_MNKX2(1, 4, 4);""",
    "steady A STS 1")
loop = replace_unique(
    loop,
    """        __builtin_mxc_arrive(64);
        __builtin_mxc_barrier_inst();""",
    """        __syncthreadshared();""",
    "count-zero wait")

reconstructed_kernel = kernel[:loop_start] + loop + kernel[loop_end:]
reconstructed = reconstructed.replace(kernel, reconstructed_kernel, 1)
reconstructed = replace_unique(
    reconstructed,
    """    XH_MMA_STAGE_MNKX2(0, 2, 4);
    XH_MMA_STAGE_MNKX2(0, 2, 6);""",
    """    XH_MMA_STAGE_MNKX2(0, 2, 4);
    XH_MMA_STS(shared_a_tensor(store_row_a[2], store_col), load_a[2], MmaLoad128);
    XH_MMA_STAGE_MNKX2(0, 2, 6);""",
    "peeled-tail A STS 2")
reconstructed = replace_unique(
    reconstructed,
    """    XH_MMA_STAGE_MNKX2(0, 3, 6);

    XH_MMA_STAGE_MNKX2(0, 4, 4);""",
    """    XH_MMA_STAGE_MNKX2(0, 3, 6);
    XH_MMA_STS(shared_a_tensor(store_row_a[3], store_col), load_a[3], MmaLoad128);

    XH_MMA_STAGE_MNKX2(0, 4, 4);""",
    "peeled-tail A STS 3")
reconstructed = replace_unique(
    reconstructed,
    """    const int loop_k_tiles = num_k_tiles - 1;
    for (uint32_t tile_k""",
    """    const int loop_k_tiles = num_k_tiles - 1;
    a_base = const_cast<int8_t*>(a_ptr);
    for (uint32_t tile_k""",
    "steady A pointer reset")
reconstructed = replace_unique(
    reconstructed,
    """#undef XH_BSM_B_STAGE_I
#undef XH_BSM_A_STAGE_I""",
    """#undef XH_LDG_B_STAGE_I
#undef XH_LDG_A_STAGE_I""",
    "stage helper undefs")

expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
if baseline_sha256 != expected_baseline_sha256:
    debug_baseline = os.environ.get("XH_DEBUG_BASELINE_FILE")
    if debug_baseline:
        with open(debug_baseline, encoding="utf-8", newline="") as source:
            expected_text = source.read().replace("\r\n", "\n").replace("\r", "\n")
        sys.stderr.writelines(difflib.unified_diff(
            expected_text.splitlines(True), reconstructed.splitlines(True),
            fromfile="formal-best", tofile="reconstructed"))
    sys.exit(
        "SOURCE CHECK FAIL: exact formal-best reconstruction mismatch: "
        + baseline_sha256)

candidate_kernel = kernel_slice(text)
baseline_kernel = kernel_slice(reconstructed)
candidate_loop_start = candidate_kernel.index(loop_anchor)
candidate_loop_end = candidate_kernel.index(tail_anchor, candidate_loop_start)
candidate_loop = candidate_kernel[candidate_loop_start:candidate_loop_end]
candidate_tail = candidate_kernel[candidate_loop_end:]

if text.count(candidate_macros) != 1 or text.count("__builtin_mxc_ldg_b128_bsm(") != 2:
    sys.exit("SOURCE CHECK FAIL: exact official A/B BSM macro forms changed")
if re.findall(r"__builtin_mxc_arrive\s*\(([^)]*)\)", text) != ["64"]:
    sys.exit("SOURCE CHECK FAIL: nonzero or extra arrive form found")
if text.count("__builtin_mxc_barrier_inst();") != 1:
    sys.exit("SOURCE CHECK FAIL: expected one BSM completion barrier")
if "a_base" in candidate_kernel:
    sys.exit("SOURCE CHECK FAIL: cross-iteration A pointer lifetime remains")
if "a_ptr + load_a_row_offset[bsmi] + (tile_index) * kMmaTileK + load_k" not in text:
    sys.exit("SOURCE CHECK FAIL: A BSM does not explicitly address tile_index")

calls = re.findall(r"^\s*XH_BSM_([AB])_STAGE_I\((\d), tile_k\);$", candidate_loop, re.M)
if calls != [("A", "0"), ("A", "1"), ("A", "2"), ("A", "3"),
             ("B", "0"), ("B", "1"), ("B", "2"), ("B", "3")]:
    sys.exit("SOURCE CHECK FAIL: eight-call A0..A3/B0..B3 batch changed: %r" % (calls,))
batch_start = candidate_loop.index("        XH_BSM_A_STAGE_I(0, tile_k);")
batch_end = candidate_loop.index(
    "        XH_MMA_STAGE_MNKX2(1, 0, 2);", batch_start)
if candidate_loop[batch_start:batch_end] != bsm_batch:
    sys.exit("SOURCE CHECK FAIL: BSM batch is not contiguous")
before_batch = candidate_loop[:batch_start].rstrip().splitlines()
if before_batch[-1].strip() != "XH_LDS_A_B128(1, 1);":
    sys.exit("SOURCE CHECK FAIL: BSM batch does not immediately follow final current A LDS")

first_cta = candidate_loop.index("        __syncthreadshared();")
final_current_a_lds = candidate_loop.index("        XH_LDS_A_B128(1, 1);")
wait = candidate_loop.index("        __builtin_mxc_arrive(64);")
barrier = candidate_loop.index("        __builtin_mxc_barrier_inst();")
first_next_lds = candidate_loop.index("        XH_LDS_A_B128(0, 0);", barrier)
if not first_cta < final_current_a_lds < batch_start < wait < barrier < first_next_lds:
    sys.exit("SOURCE CHECK FAIL: consume/BSM/wait/next-LDS order changed")
if "XH_LDS_" in candidate_loop[batch_start:barrier]:
    sys.exit("SOURCE CHECK FAIL: LDS overlaps the contiguous BSM batch or pending interval")
if "XH_MMA_STS(" in candidate_loop or "__builtin_mxc_ldg_b128(" in candidate_loop:
    sys.exit("SOURCE CHECK FAIL: steady register LDG or ordinary STS remains")
if "XH_MMA_STS(" in candidate_tail:
    sys.exit("SOURCE CHECK FAIL: peeled-tail stale STS remains")
if candidate_kernel.count("__syncthreadshared();") != 2 \
        or candidate_loop.count("__syncthreadshared();") != 1:
    sys.exit("SOURCE CHECK FAIL: initial/steady CTA barrier sites changed")

prologue_end = candidate_kernel.index("    __syncthreadshared();")
prologue = candidate_kernel[:prologue_end]
scope_start = prologue.index("    {\n        MmaLoad128 load_a[kMmaLoadsA];")
scope = prologue[scope_start:]
if (scope.count("XH_MMA_STS(shared_a_tensor(") != 4
        or scope.count("XH_MMA_STS(shared_b_tensor(") != 1
        or scope.count("load_a[") != 6 or scope.count("load_b[") != 3):
    sys.exit("SOURCE CHECK FAIL: prologue does not fully initialize A/B")
if "load_a[" in candidate_kernel[prologue_end:] or "load_b[" in candidate_kernel[prologue_end:]:
    sys.exit("SOURCE CHECK FAIL: initial register stage lifetime crosses the first barrier")

mma_sequence = re.findall(r"XH_MMA_STAGE_MNKX2\([^\n]+", candidate_kernel)
baseline_mma_sequence = re.findall(r"XH_MMA_STAGE_MNKX2\([^\n]+", baseline_kernel)
lds_sequence = re.findall(r"XH_LDS_[AB]_B128\([^\n]+", candidate_kernel)
baseline_lds_sequence = re.findall(r"XH_LDS_[AB]_B128\([^\n]+", baseline_kernel)
if mma_sequence != baseline_mma_sequence or len(mma_sequence) != 129:
    sys.exit("SOURCE CHECK FAIL: formal-best MMA order changed")
if lds_sequence != baseline_lds_sequence or len(lds_sequence) != 42:
    sys.exit("SOURCE CHECK FAIL: formal-best LDS order changed")


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
if (tile_m, tile_n, tile_k, threads, wave_size) != (128, 128, 128, 256, 64):
    sys.exit("SOURCE CHECK FAIL: CTA/tile/thread geometry changed")
if "__shared__ int8_t shared_data[kMmaSharedBytes];" not in candidate_kernel:
    sys.exit("SOURCE CHECK FAIL: 32-KiB single-buffer declaration changed")


def prove_operand_mapping(operand):
    vectors = 128 * 8
    source_visits = [0] * vectors
    destination_visits = [0] * vectors
    baseline_image = [-1] * (vectors * 16)
    bsm_image = [-1] * (vectors * 16)
    for tid in range(threads):
        wave, lane = divmod(tid, wave_size)
        source_chunk = lane % 8
        destination_chunk = ((tid // 8) + (tid % 8)) % 8
        for load in range(4):
            if operand == "A":
                source_row = tid // 8 + 32 * load
                destination_row = wave * 32 + lane // 8 + 8 * load
            else:
                source_row = (tid // 8) * 4 + load
                destination_row = tid // 8 + 32 * load
            source_vector = source_row * 8 + source_chunk
            destination_vector = destination_row * 8 + destination_chunk
            source_visits[source_vector] += 1
            destination_visits[destination_vector] += 1
            for byte in range(16):
                baseline_image[destination_vector * 16 + byte] = source_vector * 16 + byte
                bsm_image[destination_vector * 16 + byte] = source_vector * 16 + byte
    if source_visits != [1] * vectors or destination_visits != [1] * vectors:
        sys.exit("SOURCE CHECK FAIL: %s 1024-vector source/destination cover changed" % operand)
    if baseline_image != bsm_image or -1 in bsm_image:
        sys.exit("SOURCE CHECK FAIL: %s BSM shared byte image is incomplete or stale" % operand)
    return len(bsm_image)


a_bytes = prove_operand_mapping("A")
b_bytes = prove_operand_mapping("B")
if (a_bytes, b_bytes, a_bytes + b_bytes) != (16384, 16384, 32768):
    sys.exit("SOURCE CHECK FAIL: A/B/total LDS byte images changed")

for tile_count in (1, 2, 16, 56):
    shared_a = [tile_count - 1] * 1024
    shared_b = [tile_count - 1] * 1024
    consumed = []
    pending = False
    boundaries = 1
    for next_tile in range(tile_count - 1):
        current = tile_count - 1 if next_tile == 0 else next_tile - 1
        if shared_a != [current] * 1024 or shared_b != [current] * 1024:
            sys.exit("SOURCE CHECK FAIL: stale/uninitialized steady shared image")
        consumed.append(current)
        boundaries += 1
        if pending:
            sys.exit("SOURCE CHECK FAIL: pending BSM crossed first CTA barrier")
        pending = True
        pending_a = [next_tile] * 1024
        pending_b = [next_tile] * 1024
        boundaries += 1
        shared_a, shared_b = pending_a, pending_b
        pending = False
    tail = tile_count - 1 if tile_count == 1 else tile_count - 2
    if pending or shared_a != [tail] * 1024 or shared_b != [tail] * 1024:
        sys.exit("SOURCE CHECK FAIL: T=1/T=2 peeled tail is stale or pending")
    consumed.append(tail)
    expected = [tile_count - 1] + list(range(tile_count - 1))
    if consumed != expected or boundaries != 2 * tile_count - 1:
        sys.exit("SOURCE CHECK FAIL: tile labels or 2T-1 boundary count changed")
    print(
        "SOURCE SCHEDULE T=%d labels=%s boundaries=%d pending-at-cta=0 tail-stale=0"
        % (tile_count, consumed, boundaries))

for em, n, k in (
        (4096, 4096, 7168),
        (32768, 4096, 7168),
        (4096, 7168, 2048),
        (32768, 7168, 2048)):
    tiles = k // tile_k
    if em % tile_m or n % tile_n or k % tile_k:
        sys.exit("SOURCE CHECK FAIL: public shape is not exact-tile aligned")
    print(
        "SOURCE MODEL em=%d n=%d k=%d T=%d A/B-vectors=1024/1024 "
        "A/B-bytes=16384/16384 MMA/thread/tile=128 boundaries=%d"
        % (em, n, k, tiles, 2 * tiles - 1))

readonly_arguments = (
    "const int8_t* __restrict__ a_ptr",
    "const int8_t* __restrict__ b_ptr",
    "const float* __restrict__ scale_a_ptr",
    "const float* __restrict__ scale_b_ptr",
    "const float* __restrict__ moe_weights_ptr",
    "const int32_t* __restrict__ expert_ids_ptr",
)
if any(argument not in candidate_kernel for argument in readonly_arguments):
    sys.exit("SOURCE CHECK FAIL: a read-only kernel argument changed")
if candidate_kernel.count("__builtin_mxc_stg_b64_predicator(") != 2:
    sys.exit("SOURCE CHECK FAIL: output global-store sites changed")

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: exact formal-best reconstructed; official A+B BSM args; "
    "initial full A/B image; eight contiguous A0..A3/B0..B3 calls after final LDS; "
    "arrive(64)+barrier before next LDS; no steady LDG/STS or tail stale overwrite; "
    "A/B 1024-vector and 16384-byte images exact; T=1/2/16/56 and read-only PASS; "
    "source_sha256=" + source_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  printf 'STATIC-ONLY exp-20260823-045 PASS\n'
  exit 0
fi

mkdir -p -- "$build_dir"
cuda_home=${CUDA_HOME:-/usr/local/cuda}
printf 'proxy/NVIDIA BUILD exp-20260823-045 compiler=%s\n' "$cuda_home/bin/nvcc"
"$cuda_home/bin/nvcc" \
  -O3 \
  -std=c++14 \
  -arch=sm_86 \
  -lineinfo \
  -Xcompiler=-Wall \
  "$source_file" \
  -lcuda \
  -o "$binary"
printf 'proxy/NVIDIA BUILD PASS (fallback only; MACA full BSM branch not executed)\n'

for mode in --correctness --benchmark --regression; do
  printf 'proxy/NVIDIA RUN %s\n' "$mode"
  "$binary" "$mode" | sed 's/^/proxy\/NVIDIA /'
done
