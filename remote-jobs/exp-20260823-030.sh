#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-030"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'REGRESSION maca-bf16-ties-away' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import math
import re
import struct
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")

candidate_macro = """#define XH_CVT_F32_TO_BF16(dst, src0, src1)                                                       \\
    src0 += 0x8000;                                                                               \\
    src1 += 0x8000;                                                                               \\
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)"""
baseline_macro = """#define XH_CVT_F32_TO_BF16(dst, src0, src1)                                                       \\
    src0 = ((src0 >> 16) & 1) + src0 + 0x7fff;                                                   \\
    src1 = ((src1 >> 16) & 1) + src1 + 0x7fff;                                                   \\
    dst = __builtin_mxc_byte_perm(src0, src1, 0x03020706)"""

if text.count(candidate_macro) != 1:
    sys.exit("SOURCE CHECK FAIL: ties-away pack macro is not exact and unique")
reconstructed = text.replace(candidate_macro, baseline_macro, 1)
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61"
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: source outside the two tie-rule statements changed: "
        + baseline_sha256)

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
pack_call_sites = re.findall(
    r"^\s+XH_CVT_F32_TO_BF16\($", kernel, re.MULTILINE)
if len(pack_call_sites) != 4:
    sys.exit("SOURCE CHECK FAIL: output pack call-site count changed")
if text.count("__builtin_mxc_stg_b64_predicator(") != 2:
    sys.exit("SOURCE CHECK FAIL: b64 output store count changed")
if "__builtin_mxc_stg_b128" in text:
    sys.exit("SOURCE CHECK FAIL: output store mechanism widened")
if candidate_macro.count("__builtin_mxc_byte_perm") != 1:
    sys.exit("SOURCE CHECK FAIL: byte permutation changed")

macro_calls_per_thread = 4 * 2 * 4
removed_ops_per_macro = 6
removed_ops_per_thread = macro_calls_per_thread * removed_ops_per_macro
if (macro_calls_per_thread, removed_ops_per_thread) != (32, 192):
    sys.exit("SOURCE CHECK FAIL: dynamic output-pack instruction model changed")

store_pattern = re.compile(
    r"\s*__builtin_mxc_stg_b64_predicator\(.*?\n\s*MACA_ICMP_EQ\);",
    re.DOTALL,
)
candidate_stores = store_pattern.findall(text)
baseline_stores = store_pattern.findall(reconstructed)
if candidate_stores != baseline_stores or len(candidate_stores) != 2:
    sys.exit("SOURCE CHECK FAIL: output store arguments/order changed")

print(
    "SOURCE CHECK PASS: baseline reconstructed exactly; only BF16 ties-even "
    "extraction changed; byte_perm/stores exact; macro-calls/thread=32; "
    "estimated scalar ALU removed/thread=192")
for k in (7168, 2048):
    mma_per_thread = 128 * (k // 128)
    print(
        "SOURCE MODEL k=%d MMA/thread=%d removed-pack-ALU/thread=%d "
        "removed-vs-MMA=%.3f%%"
        % (k, mma_per_thread, removed_ops_per_thread,
           100.0 * removed_ops_per_thread / mma_per_thread))


def bf16_to_float(value):
    return struct.unpack(">f", struct.pack(">I", value << 16))[0]


low_words = (0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff)
checked = 0
finite_checked = 0
nonfinite_checked = 0
changed_ties = 0
max_relative_error = 0.0
for high in range(0x10000):
    for low in low_words:
        bits = (high << 16) | low
        rne = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) >> 16
        ties_away = ((bits + 0x8000) & 0xffffffff) >> 16
        reference = bf16_to_float(rne)
        candidate = bf16_to_float(ties_away)
        checked += 1

        if rne != ties_away:
            if not (low == 0x8000 and (high & 1) == 0
                    and ties_away == ((rne + 1) & 0xffff)):
                sys.exit("NUMERIC CHECK FAIL: non-tie result differs")
            changed_ties += 1

        if math.isfinite(reference):
            if not math.isfinite(candidate):
                sys.exit("NUMERIC CHECK FAIL: finite RNE became nonfinite")
            error = abs(candidate - reference)
            tolerance = 0.005 + 0.02 * abs(reference)
            if error > tolerance:
                sys.exit(
                    "NUMERIC CHECK FAIL: bits=%08x ref=%r candidate=%r error=%r tol=%r"
                    % (bits, reference, candidate, error, tolerance))
            if abs(reference) >= 0.005:
                max_relative_error = max(
                    max_relative_error, error / abs(reference))
            finite_checked += 1
        else:
            input_is_nan = (bits & 0x7fffffff) > 0x7f800000
            if not input_is_nan and math.isinf(reference) and not (
                    math.isinf(candidate)
                    and math.copysign(1.0, candidate) == math.copysign(1.0, reference)):
                sys.exit("NUMERIC CHECK FAIL: infinity/overflow behavior differs")
            if (ties_away & 0x7f80) != 0x7f80:
                sys.exit("NUMERIC CHECK FAIL: nonfinite boundary became finite")
            nonfinite_checked += 1

if checked != 393216 or changed_ties != 32768:
    sys.exit("NUMERIC CHECK FAIL: boundary coverage changed")

named = {
    "+zero": 0x00000000,
    "-zero": 0x80000000,
    "+min-subnormal": 0x00000001,
    "-min-subnormal": 0x80000001,
    "+max-finite-below-midpoint": 0x7f7f7fff,
    "+max-finite-midpoint": 0x7f7f8000,
    "-max-finite-midpoint": 0xff7f8000,
    "+infinity": 0x7f800000,
    "-infinity": 0xff800000,
    "+quiet-nan": 0x7fc00000,
    "-quiet-nan": 0xffc00000,
}
for name, bits in named.items():
    rne = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffffffff) >> 16
    ties_away = ((bits + 0x8000) & 0xffffffff) >> 16
    print("NUMERIC SPECIAL name=%s input=%08x rne=%04x ties-away=%04x"
          % (name, bits, rne, ties_away))

print(
    "NUMERIC CHECK PASS: boundary-pairs=%d finite=%d nonfinite=%d "
    "changed-even-ties=%d max-relative-error=%.9f"
    % (checked, finite_checked, nonfinite_checked,
       changed_ties, max_relative_error))
PYEOF

printf 'BUILD exp-20260823-030 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
