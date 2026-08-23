#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-049"
baseline_dir="$build_dir/baseline"
mkdir -p -- "$baseline_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
candidate_binary="$build_dir/test_fused_moe_i8_tn-candidate"
baseline_binary="$build_dir/test_fused_moe_i8_tn-baseline"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'template <int kFixedEm, int kFixedN, int kFixedK>' "$submission_file"
grep -Fq 'REGRESSION maca-shape-specialization runtime-fallback=' "$source_file"

for required in \
  'fused_moe_i8_tn_mma_kernel<4096, 4096, 7168><<<grid, block>>>' \
  'fused_moe_i8_tn_mma_kernel<32768, 4096, 7168><<<grid, block>>>' \
  'fused_moe_i8_tn_mma_kernel<4096, 7168, 2048><<<grid, block>>>' \
  'fused_moe_i8_tn_mma_kernel<32768, 7168, 2048><<<grid, block>>>' \
  'fused_moe_i8_tn_mma_kernel<0, 0, 0><<<grid, block>>>'; do
  grep -Fq "$required" "$submission_file"
done

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' '__shfl_sync' \
                'mma_kernel_64' 'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' \
                'kCase2L2NGroup' 'kSerpentineN' 'kCooperativeEpilogue' \
                'mma_adjacent_m' 'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaBFragmentCols' \
                'kMmaSharedScaleBBytes' 'shared_scale_b' 'kMmaEpilogueScaleBytes' \
                'shared_row_scale' 'shared_col_scale' 'combined_row_scale' \
                'owned_row_factor' 'row_factor_lane' 'row_factor' 'XH_A_FRAG' \
                'shared_b_next' '+ 0x8000'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected historical mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" "$source_file" "$baseline_dir" <<'PYEOF'
import hashlib
import pathlib
import re
import sys

submission_path = pathlib.Path(sys.argv[1])
test_path = pathlib.Path(sys.argv[2])
baseline_dir = pathlib.Path(sys.argv[3])
text = submission_path.read_text(encoding="utf-8").replace("\r\n", "\n")
test_text = test_path.read_text(encoding="utf-8").replace("\r\n", "\n")

candidate_sha256 = hashlib.sha256(text.encode()).hexdigest()
expected_candidate_sha256 = (
    "6fb18187921f0a568e363f76d8fbeffd533f871ef26dd283016384cc252dd1f6")
if candidate_sha256 != expected_candidate_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate source hash changed: " + candidate_sha256)


def replace_unique(source, candidate, baseline, label):
    count = source.count(candidate)
    if count != 1:
        sys.exit(
            "SOURCE CHECK FAIL: %s block count expected=1 actual=%d" % (label, count))
    return source.replace(candidate, baseline, 1)


helper = """enum MmaShapeSpecialization {
    kMmaRuntimeShape = 0,
    kMmaShape4096x4096x7168,
    kMmaShape32768x4096x7168,
    kMmaShape4096x7168x2048,
    kMmaShape32768x7168x2048,
};

static inline MmaShapeSpecialization mma_shape_specialization(
    const KernelConfig& config
) {
    if (same_config(config, {4096, 4096, 7168})) {
        return kMmaShape4096x4096x7168;
    }
    if (same_config(config, {32768, 4096, 7168})) {
        return kMmaShape32768x4096x7168;
    }
    if (same_config(config, {4096, 7168, 2048})) {
        return kMmaShape4096x7168x2048;
    }
    if (same_config(config, {32768, 7168, 2048})) {
        return kMmaShape32768x7168x2048;
    }
    return kMmaRuntimeShape;
}

"""

candidate_signature = """template <int kFixedEm, int kFixedN, int kFixedK>
__global__ void fused_moe_i8_tn_mma_kernel(
    const int8_t* __restrict__ a_ptr,
    const int8_t* __restrict__ b_ptr,
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const float* __restrict__ moe_weights_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    int runtime_em,
    int runtime_n,
    int runtime_k
) {
    using namespace cute;

    const int em = kFixedEm == 0 ? runtime_em : kFixedEm;
    const int n = kFixedN == 0 ? runtime_n : kFixedN;
    const int k = kFixedK == 0 ? runtime_k : kFixedK;
"""

baseline_signature = """__global__ void fused_moe_i8_tn_mma_kernel(
    const int8_t* __restrict__ a_ptr,
    const int8_t* __restrict__ b_ptr,
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const float* __restrict__ moe_weights_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    int em,
    int n,
    int k
) {
    using namespace cute;
"""

candidate_launch = """    switch (mma_shape_specialization(config)) {
        case kMmaShape4096x4096x7168:
            fused_moe_i8_tn_mma_kernel<4096, 4096, 7168><<<grid, block>>>(
                a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out,
                config.em, config.n, config.k);
            break;
        case kMmaShape32768x4096x7168:
            fused_moe_i8_tn_mma_kernel<32768, 4096, 7168><<<grid, block>>>(
                a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out,
                config.em, config.n, config.k);
            break;
        case kMmaShape4096x7168x2048:
            fused_moe_i8_tn_mma_kernel<4096, 7168, 2048><<<grid, block>>>(
                a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out,
                config.em, config.n, config.k);
            break;
        case kMmaShape32768x7168x2048:
            fused_moe_i8_tn_mma_kernel<32768, 7168, 2048><<<grid, block>>>(
                a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out,
                config.em, config.n, config.k);
            break;
        default:
            fused_moe_i8_tn_mma_kernel<0, 0, 0><<<grid, block>>>(
                a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out,
                config.em, config.n, config.k);
            break;
    }
"""

baseline_launch = """    fused_moe_i8_tn_mma_kernel<<<grid, block>>>(
        a,
        b_col_major,
        scale_a,
        scale_b,
        moe_weights,
        expert_ids,
        out,
        config.em,
        config.n,
        config.k
    );
"""

reconstructed = replace_unique(text, helper, "", "dispatch helper")
reconstructed = replace_unique(
    reconstructed, candidate_signature, baseline_signature, "kernel signature")
reconstructed = replace_unique(
    reconstructed, candidate_launch, baseline_launch, "launch dispatch")
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped assigned specialization blocks: "
        + baseline_sha256)

specialization_start = test_text.index("void verify_mma_shape_specialization() {")
specialization_end = test_text.index(
    "void verify_mma_a_load_bounds()", specialization_start)
baseline_test = test_text[:specialization_start] + test_text[specialization_end:]
baseline_test = replace_unique(
    baseline_test,
    "    verify_mma_shape_specialization();\n",
    "",
    "regression invocation")
baseline_dir.mkdir(parents=True, exist_ok=True)
with (baseline_dir / "submission.cu").open(
        "w", encoding="utf-8", newline="\n") as output:
    output.write(reconstructed)
with (baseline_dir / "test_fused_moe_i8_tn.cu").open(
        "w", encoding="utf-8", newline="\n") as output:
    output.write(baseline_test)

public_shapes = (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
)
unsupported_shapes = (
    (128, 32, 256),
    (4096, 4096, 2048),
    (4096, 7168, 7168),
    (4097, 4096, 7168),
)
fixed_launches = re.findall(
    r"fused_moe_i8_tn_mma_kernel<(\d+), (\d+), (\d+)><<<grid, block>>>",
    candidate_launch)
fixed_launches = tuple(tuple(map(int, values)) for values in fixed_launches)
if fixed_launches != public_shapes + ((0, 0, 0),):
    sys.exit("SOURCE CHECK FAIL: fixed/runtime launch set or order changed")

helper_shape_matches = tuple(
    tuple(map(int, values))
    for values in re.findall(
        r"same_config\(config, \{(\d+), (\d+), (\d+)\}\)", helper))
if helper_shape_matches != public_shapes:
    sys.exit("SOURCE CHECK FAIL: exact public helper mappings changed")
if any(shape in public_shapes for shape in unsupported_shapes):
    sys.exit("SOURCE CHECK FAIL: unsupported fallback fixture aliases a public shape")


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
shared_bytes = tile_m * tile_k + tile_n * tile_k
if (tile_m, tile_n, tile_k, threads, wave_size, shared_bytes) != (
        128, 128, 128, 256, 64, 32768):
    sys.exit("SOURCE CHECK FAIL: CTA/tile/wave/shared geometry changed")

kernel_start = text.index("template <int kFixedEm, int kFixedN, int kFixedK>")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
baseline_kernel_start = reconstructed.index(
    "__global__ void fused_moe_i8_tn_mma_kernel(")
baseline_kernel_end = reconstructed.index("\n#else", baseline_kernel_start)
baseline_kernel = reconstructed[baseline_kernel_start:baseline_kernel_end]
unchanged_counts = {
    "__builtin_mxc_ldg_b32_predicator(": 2,
    "__builtin_mxc_ldg_b128_predicator(": 2,
    "__builtin_mxc_stg_b64_predicator(": 2,
    "XH_MMA_STAGE_MNKX2(": 129,
    "XH_LDG_A_STAGE_I(": 5,
    "XH_LDG_B_STAGE_I(": 5,
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

for em, n, k in public_shapes:
    if em % tile_m or n % tile_n or k % tile_k:
        sys.exit("SOURCE CHECK FAIL: public shape does not tile exactly")
    print(
        "SOURCE MODEL em=%d n=%d k=%d fixed=yes K-tiles=%d MMA/thread=%d "
        "barriers=%d LDS=%d threads=%d"
        % (em, n, k, k // tile_k, 128 * (k // tile_k),
           2 * (k // tile_k) - 1, shared_bytes, threads))

print(
    "SOURCE CHECK PASS: exact performance baseline reconstructed; all four public "
    "EM/N/K tuples select distinct fixed template instances; unsupported shapes "
    "retain <0,0,0>; matrix/shared/epilogue/store body protected; source_sha256="
    + candidate_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  exit 0
fi

printf 'BUILD candidate exp-20260823-049 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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

printf 'BUILD exact-baseline exp-20260823-049 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
