#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-042"
submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -Fq '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq '__global__ __launch_bounds__(256, 2) void fused_moe_i8_tn_mma_kernel(' "$submission_file"
grep -Fq 'REGRESSION maca-launch-bounds' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' '__shfl_sync' \
                'mma_kernel_64' 'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' \
                'kCase2L2NGroup' 'kSerpentineN' 'kCooperativeEpilogue' \
                'mma_adjacent_m' 'fused_moe_i8_tn_mma_kernel_n64' 'load_b_ptr' \
                'XH_MMA_STAGE_PAIR_INTERLEAVED' 'kMmaEpilogueScaleBytes' \
                'kMmaSharedScaleBBytes' 'shared_scale_b' 'shared_row_scale' \
                'shared_col_scale' 'combined_row_scale' 'kMmaBFragmentCols' \
                'shared_b_next' 'kMmaSharedBStages' 'XH_A_FRAG'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected historical mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import re
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as source:
    text = source.read().replace("\r\n", "\n").replace("\r", "\n")

candidate_decl = (
    "__global__ __launch_bounds__(256, 2) void "
    "fused_moe_i8_tn_mma_kernel(")
baseline_decl = "__global__ void fused_moe_i8_tn_mma_kernel("
if text.count(candidate_decl) != 1:
    sys.exit("SOURCE CHECK FAIL: target launch-bounds annotation is not exact")
if text.count("__launch_bounds__") != 1:
    sys.exit("SOURCE CHECK FAIL: launch-bounds annotation escaped target kernel")

reconstructed = text.replace(candidate_decl, baseline_decl, 1)
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = (
    "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61")
if baseline_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: changes escaped the target kernel annotation: "
        + baseline_sha256)

source_sha256 = hashlib.sha256(text.encode()).hexdigest()
expected_source_sha256 = "6d63828e2a2e49573db8a5505cdf93804bc960ced7a077ba998b7e02ff767ffd"
if source_sha256 != expected_source_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate source hash changed: " + source_sha256)

def read_literal_const(name):
    match = re.search(r"constexpr int %s = (\d+);" % name, text)
    if not match:
        sys.exit("SOURCE CHECK FAIL: literal constant not found: " + name)
    return int(match.group(1))

tile_m = read_literal_const("kMmaTileM")
tile_n = read_literal_const("kMmaTileN")
tile_k = read_literal_const("kMmaTileK")
threads = read_literal_const("kMmaThreads")
shared_bytes = tile_m * tile_k + tile_n * tile_k
if (tile_m, tile_n, tile_k, threads, shared_bytes) != (128, 128, 128, 256, 32768):
    sys.exit("SOURCE CHECK FAIL: CTA/tile/shared geometry changed")

kernel_start = text.index(candidate_decl)
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
baseline_kernel_start = reconstructed.index(baseline_decl)
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
    actual = kernel.count(token)
    baseline_actual = baseline_kernel.count(token)
    if actual != expected or baseline_actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: protected count %s expected=%d candidate=%d baseline=%d"
            % (token, expected, actual, baseline_actual))

fallback_start = text.index("__global__ void fused_moe_i8_tn_kernel(")
fallback = text[fallback_start:]
if "__launch_bounds__" in fallback:
    sys.exit("SOURCE CHECK FAIL: NVIDIA fallback received launch bounds")

for em, n, k in (
        (4096, 4096, 7168),
        (32768, 4096, 7168),
        (4096, 7168, 2048),
        (32768, 7168, 2048)):
    if em % tile_m or n % tile_n or k % tile_k:
        sys.exit("SOURCE CHECK FAIL: public shape does not tile exactly")
    tiles = k // tile_k
    barriers = 2 * tiles - 1
    print(
        "SOURCE MODEL em=%d n=%d k=%d max-threads=256 min-blocks=2 "
        "two-block-threads=512 two-block-LDS=65536 MMA/thread=%d barriers=%d"
        % (em, n, k, 128 * tiles, barriers))

print(
    "SOURCE CHECK PASS: exact baseline reconstructed; one target-only "
    "__launch_bounds__(256, 2); fallback/instructions/traffic/barriers/geometry "
    "protected; source_sha256=" + source_sha256)
PYEOF

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  printf 'STATIC-ONLY exp-20260823-042 PASS\n'
  exit 0
fi

mkdir -p -- "$build_dir"
cuda_home=${CUDA_HOME:-/usr/local/cuda}
printf 'proxy/NVIDIA BUILD exp-20260823-042 compiler=%s\n' "$cuda_home/bin/nvcc"
"$cuda_home/bin/nvcc" \
  -O3 \
  -std=c++14 \
  -arch=sm_86 \
  -lineinfo \
  -Xcompiler=-Wall \
  "$source_file" \
  -lcuda \
  -o "$binary"
printf 'proxy/NVIDIA BUILD PASS (fallback only; target launch bounds not compiled)\n'

for mode in --correctness --benchmark --regression; do
  printf 'proxy/NVIDIA RUN %s\n' "$mode"
  "$binary" "$mode" | sed 's/^/proxy\/NVIDIA /'
done
