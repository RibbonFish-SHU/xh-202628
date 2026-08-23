#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-055"
baseline_dir="$build_dir/baseline"
mkdir -p -- "$baseline_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
candidate_binary="$build_dir/test_fused_moe_i8_tn-candidate"
baseline_binary="$build_dir/test_fused_moe_i8_tn-baseline"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'REGRESSION maca-row-metadata-unpredicated case=' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' '__shfl_sync' \
                'mma_kernel_64' 'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' \
                'kCase2L2NGroup' 'kSerpentineN' 'kCooperativeEpilogue' \
                'mma_adjacent_m' 'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaBFragmentCols' \
                'kMmaSharedScaleBBytes' 'shared_scale_b' 'kMmaEpilogueScaleBytes' \
                'shared_row_scale' 'shared_col_scale' 'combined_row_scale' \
                'owned_row_factor' 'row_factor_lane' 'row_factor' 'XH_A_FRAG' \
                'shared_b_next' '+ 0x8000' 'template <int kFixedEm'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected historical mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" "$source_file" "$baseline_dir" <<'PYEOF'
import hashlib
import pathlib
import sys

submission_path = pathlib.Path(sys.argv[1])
test_path = pathlib.Path(sys.argv[2])
baseline_dir = pathlib.Path(sys.argv[3])
text = submission_path.read_text(encoding="utf-8").replace("\r\n", "\n")
test_text = test_path.read_text(encoding="utf-8").replace("\r\n", "\n")

candidate_sha256 = hashlib.sha256(text.encode()).hexdigest()
expected_candidate_sha256 = (
    "419634b6c6ebbfdd419c31cd86bffcbfc1bc3110d6061fe3d85bbe860d2a3408")
if candidate_sha256 != expected_candidate_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate source hash changed: " + candidate_sha256)

candidate_test_sha256 = hashlib.sha256(test_text.encode()).hexdigest()
expected_candidate_test_sha256 = (
    "ab06f9e5c042ca374b4d47f3974b37625de0dbe4fc2241be6a73e0e3d4afd52d")
if candidate_test_sha256 != expected_candidate_test_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate test hash changed: " + candidate_test_sha256)

candidate_start = text.index("    float weights[2][4];")
candidate_end = text.index(
    "\n#pragma unroll\n"
    "    for (uint32_t i = 0; i < 2; ++i) {\n"
    "        const float* ptr =",
    candidate_start,
)
candidate = text[candidate_start:candidate_end]

required = (
    "float weights[2][4];",
    "float row_scale[2][4];",
    "const int row = output_row[i * 4 + j];",
    "const_cast<float*>(moe_weights_ptr + row)",
    "const_cast<float*>(scale_a_ptr + row)",
    "0,\n                    -1,\n                    true,\n"
    "                    true,\n                    false,\n                    false);",
)
for token in required:
    if token not in candidate:
        sys.exit("SOURCE CHECK FAIL: row-metadata token missing: " + token)
if candidate.count("for (uint32_t i = 0; i < 2; ++i)") != 1:
    sys.exit("SOURCE CHECK FAIL: row-group loop changed")
if candidate.count("for (uint32_t j = 0; j < 4; ++j)") != 1:
    sys.exit("SOURCE CHECK FAIL: row-in-group loop changed")
if candidate.count("__builtin_mxc_ldg_b32(") != 2:
    sys.exit("SOURCE CHECK FAIL: expected exactly two unpredicated b32 sites")
if "__builtin_mxc_ldg_b32_predicator(" in candidate:
    sys.exit("SOURCE CHECK FAIL: predicated row-metadata load remains")
if "__builtin_mxc_ldg_b128" in candidate:
    sys.exit("SOURCE CHECK FAIL: exp-027 b128 metadata mechanism is present")

baseline = '''    float weights[2][4];
    float row_scale[2][4];
    MmaFloat4 col_scale[2];

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const int row = output_row[i * 4 + j];
            *(reinterpret_cast<MmaInt1*>(&weights[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(moe_weights_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
            *(reinterpret_cast<MmaInt1*>(&row_scale[i]) + j) =
                __builtin_mxc_ldg_b32_predicator(
                    const_cast<float*>(scale_a_ptr + row),
                    0,
                    true,
                    true,
                    false,
                    false,
                    row,
                    em,
                    MACA_ICMP_SLT);
        }
    }
'''
reconstructed = text[:candidate_start] + baseline + text[candidate_end:]
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: change escaped assigned row-metadata block: "
        + baseline_sha256)

test_start = test_text.index("void verify_mma_row_metadata_unpredicated() {")
test_end = test_text.index("void run_regression()", test_start)
baseline_test = test_text[:test_start] + test_text[test_end:]
invocation = "    verify_mma_row_metadata_unpredicated();\n"
if baseline_test.count(invocation) != 1:
    sys.exit("SOURCE CHECK FAIL: metadata regression invocation changed")
baseline_test = baseline_test.replace(invocation, "", 1)
baseline_test_sha256 = hashlib.sha256(baseline_test.encode()).hexdigest()
expected_baseline_test_sha256 = (
    "a46531c3ea79be9282f52002364e6ae3a521dc09cc480f05e554261eddd11fcc")
if baseline_test_sha256 != expected_baseline_test_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: test does not reverse to formal-best harness: "
        + baseline_test_sha256)

baseline_dir.mkdir(parents=True, exist_ok=True)
with (baseline_dir / "submission.cu").open(
        "w", encoding="utf-8", newline="\n") as output:
    output.write(reconstructed)
with (baseline_dir / "test_fused_moe_i8_tn.cu").open(
        "w", encoding="utf-8", newline="\n") as output:
    output.write(baseline_test)

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
baseline_kernel_start = reconstructed.index(
    "__global__ void fused_moe_i8_tn_mma_kernel(")
baseline_kernel_end = reconstructed.index("\n#else", baseline_kernel_start)
baseline_kernel = reconstructed[baseline_kernel_start:baseline_kernel_end]
protected_counts = {
    "__builtin_mxc_ldg_b32(": (2, 0),
    "__builtin_mxc_ldg_b32_predicator(": (0, 2),
    "__builtin_mxc_ldg_b128_predicator(": (2, 2),
    "__builtin_mxc_stg_b64_predicator(": (2, 2),
    "XH_MMA_STAGE_MNKX2(": (129, 129),
    "XH_LDG_A_STAGE_I(": (5, 5),
    "XH_LDG_B_STAGE_I(": (5, 5),
    "XH_MMA_LDS(": (2, 2),
    "XH_MMA_STS(": (13, 13),
    "__syncthreadshared();": (3, 3),
}
for token, (candidate_expected, baseline_expected) in protected_counts.items():
    candidate_count = kernel.count(token)
    baseline_count = baseline_kernel.count(token)
    if candidate_count != candidate_expected or baseline_count != baseline_expected:
        sys.exit(
            "SOURCE CHECK FAIL: protected count %s candidate=%d/%d baseline=%d/%d"
            % (token, candidate_count, candidate_expected,
               baseline_count, baseline_expected))

for token in (
        "load_b[i] = __builtin_mxc_ldg_b128_predicator(",
        "load_k,\n            k_head,\n            MACA_ICMP_SLT",
        "col_scale[i] = __builtin_mxc_ldg_b128_predicator(",
        "output_col_mask[i],\n            1,\n            MACA_ICMP_EQ",
        "(output_row[i * 4 + j] < em) && output_col_mask[0]",
        "(output_row[i * 4 + j] < em) && output_col_mask[1]",
):
    if token not in kernel:
        sys.exit(
            "SOURCE CHECK FAIL: exp-012 B/scale-B/store predicate scope leaked: "
            + token)

tile_rows = 128
threads = 256
row_groups = 2
rows_per_group = 4
metadata_arrays = 2
loads_per_thread = row_groups * rows_per_group * metadata_arrays
loads_per_cta = threads * loads_per_thread
dynamic_bytes_per_cta = loads_per_cta * 4
unique_bytes_per_cta = tile_rows * metadata_arrays * 4
if (loads_per_thread, loads_per_cta, dynamic_bytes_per_cta,
        unique_bytes_per_cta) != (16, 4096, 16384, 1024):
    sys.exit("SOURCE CHECK FAIL: row-metadata traffic model changed")

for em, n, k in (
        (4096, 4096, 7168),
        (32768, 4096, 7168),
        (4096, 7168, 2048),
        (32768, 7168, 2048)):
    if em % tile_rows:
        sys.exit("SOURCE CHECK FAIL: public EM has a partial row tile")
    print(
        "SOURCE MODEL em=%d n=%d k=%d exact-row-tiles=%d loads/thread=%d "
        "loads/CTA=%d dynamic-bytes/CTA=%d unique-bytes/CTA=%d "
        "predicate-compares-removed/CTA=%d"
        % (em, n, k, em // tile_rows, loads_per_thread, loads_per_cta,
           dynamic_bytes_per_cta, unique_bytes_per_cta, loads_per_cta))

print(
    "SOURCE CHECK PASS: exact formal best reconstructed; only eight weights and "
    "eight scale-A b32 expressions per thread changed from predicated to ordinary; "
    "B-final, scale-B and store predicates retained; source_sha256="
    + candidate_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD candidate exp-20260823-055 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" \
  -O3 \
  -std=c++14 \
  -arch=sm_86 \
  -lineinfo \
  -Xcompiler=-Wall \
  "$source_file" \
  -lcuda \
  -o "$candidate_binary"
printf 'BUILD candidate PASS\n'

printf 'BUILD exact-baseline exp-20260823-055 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" \
  -O3 \
  -std=c++14 \
  -arch=sm_86 \
  -lineinfo \
  -Xcompiler=-Wall \
  "$baseline_dir/test_fused_moe_i8_tn.cu" \
  -lcuda \
  -o "$baseline_binary"
printf 'BUILD exact-baseline PASS\n'

"$candidate_binary" --correctness
"$candidate_binary" --regression

"$baseline_binary" --benchmark | tee "$build_dir/baseline-benchmark-a.log"
"$candidate_binary" --benchmark | tee "$build_dir/candidate-benchmark-a.log"
"$candidate_binary" --benchmark | tee "$build_dir/candidate-benchmark-b.log"
"$baseline_binary" --benchmark | tee "$build_dir/baseline-benchmark-b.log"

python3 - "$build_dir" <<'PYEOF'
import pathlib
import re
import statistics
import sys

build_dir = pathlib.Path(sys.argv[1])
conditions = {
    "baseline": (
        build_dir / "baseline-benchmark-a.log",
        build_dir / "baseline-benchmark-b.log",
    ),
    "candidate": (
        build_dir / "candidate-benchmark-a.log",
        build_dir / "candidate-benchmark-b.log",
    ),
}
pattern = re.compile(
    r"BENCHMARK proxy=NVIDIA case=(\S+) samples_ms=([0-9.,]+) median_ms=")
samples = {condition: {} for condition in conditions}
for condition, paths in conditions.items():
    for path in paths:
        matches = pattern.findall(path.read_text(encoding="utf-8"))
        if len(matches) != 4:
            sys.exit(
                "PAIRED BENCHMARK FAIL: %s expected four cases in %s"
                % (condition, path))
        for case, values in matches:
            samples[condition].setdefault(case, []).extend(
                float(value) for value in values.split(","))

if set(samples["baseline"]) != set(samples["candidate"]):
    sys.exit("PAIRED BENCHMARK FAIL: baseline/candidate cases differ")
for case in samples["baseline"]:
    baseline_values = samples["baseline"][case]
    candidate_values = samples["candidate"][case]
    if len(baseline_values) != 10 or len(candidate_values) != 10:
        sys.exit("PAIRED BENCHMARK FAIL: each condition requires ten samples")
    baseline_median = statistics.median(baseline_values)
    candidate_median = statistics.median(candidate_values)
    delta_pct = 100.0 * (candidate_median / baseline_median - 1.0)
    print(
        "PAIRED proxy=NVIDIA case=%s baseline_median_ms=%.4f "
        "candidate_median_ms=%.4f candidate_delta_pct=%+.3f samples=10+10"
        % (case, baseline_median, candidate_median, delta_pct))
print("PAIRED BENCHMARK PASS: NVIDIA executes the unchanged fallback")
PYEOF
