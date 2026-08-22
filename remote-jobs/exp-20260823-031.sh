#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-031"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'XH_MMA_STAGE_PAIR_INTERLEAVED' "$submission_file"
grep -Fq 'maca-mma-chain-interleave' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'load_b_ptr'; do
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

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")

pair_definition = "#define XH_MMA_STAGE_PAIR_INTERLEAVED"
pair_definition_start = text.index(pair_definition)
pair_definition_end = text.index("#define XH_LDG_A_STAGE_I", pair_definition_start)
reconstructed = text[:pair_definition_start] + text[pair_definition_end:]

pair_call_re = re.compile(
    r"^([ \t]*)XH_MMA_STAGE_PAIR_INTERLEAVED\("
    r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\);$",
    re.MULTILINE,
)


def restore_pair(match):
    indent, m, n0, k0, n1, k1 = match.groups()
    return (
        f"{indent}XH_MMA_STAGE_MNKX2({m}, {n0}, {k0});\n"
        f"{indent}XH_MMA_STAGE_MNKX2({m}, {n1}, {k1});"
    )


reconstructed, restored_pairs = pair_call_re.subn(restore_pair, reconstructed)
reconstructed = reconstructed.replace(
    "#undef XH_MMA_STAGE_PAIR_INTERLEAVED\n", "")
if restored_pairs != 13:
    sys.exit(
        "SOURCE CHECK FAIL: expected 13 interleaved pairs, found %d"
        % restored_pairs)

expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61"
)
reconstructed_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
if reconstructed_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: reversing only MMA interleaving did not reconstruct "
        "the exact assigned baseline: " + reconstructed_sha256)

loop_marker = "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {"
loop_start = text.index(loop_marker)
loop_end = text.index("\n    }\n\n    int output_row[8];", loop_start)
candidate_loop = text[loop_start:loop_end]
baseline_loop_start = reconstructed.index(loop_marker)
baseline_loop_end = reconstructed.index(
    "\n    }\n\n    int output_row[8];", baseline_loop_start)
baseline_loop = reconstructed[baseline_loop_start:baseline_loop_end]

original_call_re = re.compile(
    r"^\s*XH_MMA_STAGE_MNKX2\((\d+),\s*(\d+),\s*(\d+)\);$")
pair_line_re = re.compile(
    r"^\s*XH_MMA_STAGE_PAIR_INTERLEAVED\("
    r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\);$")


def original_macro(line):
    match = original_call_re.match(line)
    return tuple(map(int, match.groups())) if match else None


def pair_macro(line):
    match = pair_line_re.match(line)
    return tuple(map(int, match.groups())) if match else None


def atomic_mmas(source):
    result = []
    for line in source.splitlines():
        original = original_macro(line)
        if original is not None:
            m, n, k = original
            result.extend(((m, n, k), (m, n, k + 1)))
            continue
        pair = pair_macro(line)
        if pair is not None:
            m, n0, k0, n1, k1 = pair
            result.extend(
                ((m, n0, k0), (m, n1, k1),
                 (m, n0, k0 + 1), (m, n1, k1 + 1)))
    return result


baseline_lines = baseline_loop.splitlines()
expected_pairs = []
line_index = 0
while line_index < len(baseline_lines):
    if original_macro(baseline_lines[line_index]) is None:
        line_index += 1
        continue
    run = []
    while line_index < len(baseline_lines):
        macro = original_macro(baseline_lines[line_index])
        if macro is None:
            break
        run.append(macro)
        line_index += 1
    run_index = 0
    while run_index < len(run):
        if (run_index + 1 < len(run)
                and run[run_index][:2] != run[run_index + 1][:2]):
            first = run[run_index]
            second = run[run_index + 1]
            if first[0] != second[0]:
                sys.exit("SOURCE CHECK FAIL: pairing crosses an M accumulator group")
            expected_pairs.append(
                (first[0], first[1], first[2], second[1], second[2]))
            run_index += 2
        else:
            run_index += 1

actual_pairs = [
    pair_macro(line) for line in candidate_loop.splitlines()
    if pair_macro(line) is not None
]
if actual_pairs != expected_pairs or len(actual_pairs) != 13:
    sys.exit(
        "SOURCE CHECK FAIL: candidate is not the deterministic left-to-right "
        "pairing of adjacent independent baseline chains")
if any(n0 == n1 for _, n0, _, n1, _ in actual_pairs):
    sys.exit("SOURCE CHECK FAIL: an interleaved pair aliases one accumulator")

baseline_mmas = atomic_mmas(baseline_loop)
candidate_mmas = atomic_mmas(candidate_loop)
if len(baseline_mmas) != 128 or len(candidate_mmas) != 128:
    sys.exit(
        "SOURCE CHECK FAIL: steady tile must contain exactly 128 MMAs: "
        "baseline=%d candidate=%d"
        % (len(baseline_mmas), len(candidate_mmas)))
if collections.Counter(candidate_mmas) != collections.Counter(baseline_mmas):
    sys.exit("SOURCE CHECK FAIL: steady MMA instruction multiset changed")
if candidate_mmas == baseline_mmas:
    sys.exit("SOURCE CHECK FAIL: MMA issue order did not change")

for m in range(2):
    for n in range(8):
        depths = [k for mm, nn, k in candidate_mmas if (mm, nn) == (m, n)]
        if depths != list(range(8)):
            sys.exit(
                "SOURCE CHECK FAIL: accumulator (%d,%d) depth order is %r"
                % (m, n, depths))

anchor_prefixes = (
    "XH_LDG_A_STAGE_I(",
    "XH_LDG_B_STAGE_I(",
    "XH_LDS_A_B128(",
    "XH_LDS_B_B128(",
    "XH_MMA_STS(shared_a_tensor(",
    "XH_MMA_STS(shared_b_tensor(",
    "__syncthreadshared();",
)


def anchored_non_mma_trace(source):
    trace = []
    mma_index = 0
    for line in source.splitlines():
        stripped = line.strip()
        original = original_macro(line)
        pair = pair_macro(line)
        if original is not None:
            mma_index += 2
        elif pair is not None:
            mma_index += 4
        elif stripped.startswith(anchor_prefixes):
            trace.append((mma_index, "".join(stripped.split())))
    return trace


candidate_anchors = anchored_non_mma_trace(candidate_loop)
baseline_anchors = anchored_non_mma_trace(baseline_loop)
if candidate_anchors != baseline_anchors:
    sys.exit(
        "SOURCE CHECK FAIL: LDG/LDS/STS/barrier sequence or MMA-relative index changed")

anchor_counts = collections.Counter(
    next(prefix for prefix in anchor_prefixes if event.startswith(prefix))
    for _, event in candidate_anchors)
expected_anchor_counts = {
    "XH_LDG_A_STAGE_I(": 4,
    "XH_LDG_B_STAGE_I(": 4,
    "XH_LDS_A_B128(": 4,
    "XH_LDS_B_B128(": 16,
    "XH_MMA_STS(shared_a_tensor(": 4,
    "XH_MMA_STS(shared_b_tensor(": 4,
    "__syncthreadshared();": 2,
}
if anchor_counts != collections.Counter(expected_anchor_counts):
    sys.exit("SOURCE CHECK FAIL: unexpected steady non-MMA operation counts")

fragment_start = text.index("    XH_LDS_A_B128(0, 0);")
fragment_source = text[fragment_start:loop_end]
a_available = [[False] * 8 for _ in range(2)]
b_available = [[False] * 8 for _ in range(8)]
lds_a_re = re.compile(r"^\s*XH_LDS_A_B128\((\d+),\s*(\d+)\);$")
lds_b_re = re.compile(r"^\s*XH_LDS_B_B128\((\d+),\s*(\d+)\);$")
for line in fragment_source.splitlines():
    lds_a = lds_a_re.match(line)
    lds_b = lds_b_re.match(line)
    if lds_a:
        row, col = map(int, lds_a.groups())
        for depth in range(4 * col, 4 * col + 4):
            a_available[row][depth] = True
        continue
    if lds_b:
        row, col = map(int, lds_b.groups())
        for depth in range(4 * col, 4 * col + 4):
            b_available[row][depth] = True
        continue
    original = original_macro(line)
    pair = pair_macro(line)
    events = []
    if original is not None:
        m, n, k = original
        events = ((m, n, k), (m, n, k + 1))
    elif pair is not None:
        m, n0, k0, n1, k1 = pair
        events = (
            (m, n0, k0), (m, n1, k1),
            (m, n0, k0 + 1), (m, n1, k1 + 1))
    for m, n, k in events:
        if not a_available[m][k] or not b_available[n][k]:
            sys.exit(
                "SOURCE CHECK FAIL: MMA uses fragment before LDS: (%d,%d,%d)"
                % (m, n, k))

tail_marker = "    int output_row[8];"
tail_end_marker = "    MmaInt4 output[kMmaOutputVectors];"
candidate_tail_start = text.index(tail_marker, loop_end)
candidate_tail_end = text.index(tail_end_marker, candidate_tail_start)
candidate_tail = text[candidate_tail_start:candidate_tail_end]
baseline_tail_start = reconstructed.index(tail_marker, baseline_loop_end)
baseline_tail_end = reconstructed.index(tail_end_marker, baseline_tail_start)
baseline_tail = reconstructed[baseline_tail_start:baseline_tail_end]
if candidate_tail != baseline_tail or pair_definition in candidate_tail:
    sys.exit("SOURCE CHECK FAIL: peeled final-tile schedule changed")
if len(atomic_mmas(candidate_tail)) != 128:
    sys.exit("SOURCE CHECK FAIL: peeled final tile must retain 128 baseline MMAs")

required_resources = (
    "constexpr int kMmaTileM = 128;",
    "constexpr int kMmaTileN = 128;",
    "constexpr int kMmaTileK = 128;",
    "constexpr int kMmaThreads = 256;",
    "MmaInt4 accum[kMmaRows][kMmaCols] = {0};",
    "int32_t a_frag[kMmaRows][kMmaDepth];",
    "int32_t b_frag[kMmaCols][kMmaDepth];",
)
for token in required_resources:
    if text.count(token) != 1:
        sys.exit("SOURCE CHECK FAIL: resource declaration changed: " + token)
if text.count("__syncthreadshared();") != 3:
    sys.exit("SOURCE CHECK FAIL: total barrier count changed")

candidate_sha256 = hashlib.sha256(text.encode()).hexdigest()
tail_sha256 = hashlib.sha256(candidate_tail.encode()).hexdigest()
print(
    "SOURCE CHECK PASS: exact assigned baseline reconstructed; "
    "steady MMAs=128; interleaved-pairs=13; interleaved-MMAs=52; "
    "all 16 accumulator depth chains=0..7; memory anchors/barriers unchanged; "
    "fragments loaded before use; peeled-tail exact; resources exact")
print("SOURCE SHA256 candidate=%s peeled-tail=%s" % (candidate_sha256, tail_sha256))
PYEOF

printf 'BUILD exp-20260823-031 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
