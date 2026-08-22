#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-041"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'shared_b_tensor((tid % 16) + 16 * rowi, lds_col[coli])' "$submission_file"
grep -Fq 'REGRESSION maca-b-lds-row-inline' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaBFragmentCols' \
                'kMmaSharedScaleBBytes' 'shared_scale_b' 'kMmaEpilogueScaleBytes' \
                'shared_row_scale' 'shared_col_scale' 'combined_row_scale' \
                '__shfl_sync' 'owned_row_factor' 'XH_A_FRAG' 'shared_b_next' \
                'kMmaSharedBStages'; do
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
    "e22f4d5b2eabbfe3027a32338285621a5768d9e221d812cb9b5fa6f43bbba7b4")
if candidate_sha256 != expected_candidate_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate source hash changed: " + candidate_sha256)


def normalized_macro_line(line):
    value = line.strip()
    if value.endswith("\\"):
        return value[:-1].rstrip() + " <CONT>"
    return value


macro_start = text.index("#define XH_LDS_B_B128(rowi, coli)")
macro_end = text.index("\n\n#define XH_CVT_F32_TO_BF16", macro_start)
macro_lines = text[macro_start:macro_end].splitlines()
expected_macro_lines = (
    "#define XH_LDS_B_B128(rowi, coli) <CONT>",
    "XH_MMA_LDS(b_frag[rowi][coli * 4], "
    "shared_b_tensor((tid % 16) + 16 * rowi, lds_col[coli]), MmaLoad128)",
)
if tuple(normalized_macro_line(line) for line in macro_lines) != expected_macro_lines:
    sys.exit("SOURCE CHECK FAIL: B LDS inline macro is not the assigned exact form")

baseline_macro = (
    macro_lines[0]
    + "\n"
    + "    XH_MMA_LDS(b_frag[rowi][coli * 4], "
      "shared_b_tensor(lds_row_b[rowi], lds_col[coli]), MmaLoad128)"
)
reconstructed = text[:macro_start] + baseline_macro + text[macro_end:]

candidate_declaration = "    int lds_row_a[2];\n    int lds_col[2];"
baseline_declaration = (
    "    int lds_row_a[2];\n"
    "    int lds_row_b[8];\n"
    "    int lds_col[2];"
)
if reconstructed.count(candidate_declaration) != 1:
    sys.exit("SOURCE CHECK FAIL: B row declaration anchor changed")
reconstructed = reconstructed.replace(
    candidate_declaration, baseline_declaration, 1)

candidate_initialization_anchor = (
    "        lds_row_a[i] = (tid % 16) + wave * 32 + 16 * i;\n"
    "    }\n\n"
    "    __syncthreadshared();"
)
baseline_initialization_anchor = (
    "        lds_row_a[i] = (tid % 16) + wave * 32 + 16 * i;\n"
    "    }\n"
    "#pragma unroll\n"
    "    for (int i = 0; i < 8; ++i) {\n"
    "        lds_row_b[i] = (tid % 16) + 16 * i;\n"
    "    }\n\n"
    "    __syncthreadshared();"
)
if reconstructed.count(candidate_initialization_anchor) != 1:
    sys.exit("SOURCE CHECK FAIL: B row initialization anchor changed")
reconstructed = reconstructed.replace(
    candidate_initialization_anchor, baseline_initialization_anchor, 1)

baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped assigned B LDS row blocks: "
        + baseline_sha256)

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
    ).replace("\r\n", "\n")
    if baseline != reconstructed:
        sys.exit("SOURCE CHECK FAIL: local performance baseline differs")
    if hashlib.sha256(baseline.encode()).hexdigest() != expected_baseline_sha256:
        sys.exit("SOURCE CHECK FAIL: local performance baseline hash changed")


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
if "constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;" not in text:
    sys.exit("SOURCE CHECK FAIL: baseline shared allocation expression changed")

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
baseline_kernel = reconstructed[kernel_start:kernel_end]
if "lds_row_b" in kernel:
    sys.exit("SOURCE CHECK FAIL: long-lived B row array or initialization remains")
for token in (
        "int lds_row_a[2];",
        "int lds_col[2];",
        "lds_col[i] = (((tid % 16) + (lane / 16) + 4 * i) % 8) * 16;",
        "lds_row_a[i] = (tid % 16) + wave * 32 + 16 * i;"):
    if kernel.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: protected A/column row token changed: " + token)

unchanged_counts = {
    "__builtin_mxc_ldg_b32_predicator(": 2,
    "__builtin_mxc_ldg_b128_predicator(": 2,
    "__builtin_mxc_stg_b64_predicator(": 2,
    "XH_MMA_STAGE_MNKX2(": 129,
    "XH_LDG_A_STAGE_I(": 5,
    "XH_LDG_B_STAGE_I(": 5,
    "XH_LDS_A_B128(": 9,
    "XH_LDS_B_B128(": 33,
    "XH_MMA_LDS(": 2,
    "XH_MMA_STS(": 13,
    "__syncthreadshared();": 3,
}
for token, expected in unchanged_counts.items():
    candidate_count = kernel.count(token)
    baseline_count = baseline_kernel.count(token)
    if candidate_count != expected or baseline_count != expected:
        sys.exit(
            "SOURCE CHECK FAIL: unchanged count %s expected=%d candidate=%d baseline=%d"
            % (token, expected, candidate_count, baseline_count))

loop_marker = "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {"
loop_start = kernel.index(loop_marker)
loop_end = kernel.index("\n    }\n\n    int output_row[8];", loop_start)
tail_start = loop_end + len("\n    }\n\n")
output_start = kernel.index("    MmaInt4 output[kMmaOutputVectors];", tail_start)
entry_start = kernel.index("    int store_row_a[kMmaLoadsA];")
entry = kernel[entry_start:loop_start]
loop = kernel[loop_start:loop_end]
tail = kernel[tail_start:output_start]


def require_count(region, token, expected, label):
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
        require_count(region, token, expected_count, label)

call_pattern = re.compile(
    r"^\s*XH_LDS_B_B128\((\d),\s*(\d)\);$", re.MULTILINE)
raw_call_pattern = re.compile(
    r"^\s*XH_LDS_B_B128\(([^)]*)\);$", re.MULTILINE)
calls = [tuple(map(int, match.groups())) for match in call_pattern.finditer(kernel)]
raw_calls = raw_call_pattern.findall(kernel)
baseline_calls = [
    tuple(map(int, match.groups())) for match in call_pattern.finditer(baseline_kernel)]
expected_calls = [
    (0, 0), (1, 0), (2, 0), (3, 0),
    (4, 0), (5, 0), (6, 0), (7, 0),
    (0, 1), (1, 1), (2, 1), (3, 1),
    (4, 1), (5, 1), (6, 1), (7, 1),
    (0, 0), (1, 0), (2, 0), (3, 0),
    (4, 0), (5, 0), (6, 0), (7, 0),
    (0, 1), (1, 1), (2, 1), (3, 1),
    (4, 1), (5, 1), (6, 1), (7, 1),
]
if len(raw_calls) != 32 or calls != expected_calls or baseline_calls != expected_calls:
    sys.exit("SOURCE CHECK FAIL: B LDS literal call multiset or source order changed")
if any(not 0 <= rowi < 8 or not 0 <= coli < 2 for rowi, coli in calls):
    sys.exit("SOURCE CHECK FAIL: B LDS row/column literal is out of range")

vector_visits = {
    (wave, phase): collections.Counter()
    for wave in range(4) for phase in range(2)
}
checked_executions = 0
checked_fragments = 0
for thread_id in range(threads):
    lane = thread_id % wave_size
    baseline_rows = [(thread_id % 16) + 16 * rowi for rowi in range(8)]
    candidate_trace = []
    baseline_trace = []
    for call_index, (rowi, coli) in enumerate(calls):
        lds_col = (((thread_id % 16) + (lane // 16) + 4 * coli) % 8) * 16
        baseline_row = baseline_rows[rowi]
        candidate_row = (thread_id % 16) + 16 * rowi
        for word in range(4):
            baseline_address = baseline_row * tile_k + lds_col + word * 4
            candidate_address = candidate_row * tile_k + lds_col + word * 4
            baseline_fragment = rowi * 8 + coli * 4 + word
            candidate_fragment = rowi * 8 + coli * 4 + word
            baseline_trace.append((baseline_address, baseline_fragment))
            candidate_trace.append((candidate_address, candidate_fragment))
            checked_fragments += 1
        phase = call_index // 16
        vector = candidate_row * (tile_k // 16) + lds_col // 16
        vector_visits[(thread_id // wave_size, phase)][vector] += 1
        checked_executions += 1
    if candidate_trace != baseline_trace:
        sys.exit("SOURCE CHECK FAIL: B LDS address/fragment trace changed")

expected_vectors = set(range(tile_m * (tile_k // 16)))
for key, visits in vector_visits.items():
    if set(visits) != expected_vectors or any(count != 1 for count in visits.values()):
        sys.exit("SOURCE CHECK FAIL: B LDS phase is not an exact per-wave cover: %r" % (key,))
if (checked_executions, checked_fragments) != (8192, 32768):
    sys.exit("SOURCE CHECK FAIL: exhaustive B LDS cardinality changed")

for k_tiles in (1, 2, 16, 56):
    b_lds = 4 + 16 * (k_tiles - 1) + 12
    barriers = 2 * k_tiles - 1
    if b_lds != 16 * k_tiles:
        sys.exit("SOURCE CHECK FAIL: dynamic B LDS count changed")
    print(
        "SOURCE MODEL k-tiles=%d B-LDS/thread=%d MMA/thread=%d barriers=%d "
        "B-row-i32=8->0 B-row-init=8->0"
        % (k_tiles, b_lds, 128 * k_tiles, barriers))

print(
    "SOURCE CHECK PASS: exact performance baseline reconstructed; only B LDS row "
    "expression inlined; all 32 call sites retain literal/order identity; all "
    "256 threads and 32768 fragment words retain exact address/destination order; "
    "two per-wave B-tile phases are exact covers; B-row i32 8->0; init ops 8->0; "
    "A/B traffic, MMA, barriers, 4x1 waves and 32-KiB LDS unchanged; "
    "source_sha256=" + candidate_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD exp-20260823-041 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
