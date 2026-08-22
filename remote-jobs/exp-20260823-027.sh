#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-027"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'REGRESSION maca-row-metadata case=' "$source_file"
for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m' \
                'fused_moe_i8_tn_mma_kernel_n64'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read().replace("\r\n", "\n")

candidate_start = text.index("    MmaFloat4 weights[2];")
candidate_end = text.index(
    "\n#pragma unroll\n"
    "    for (uint32_t i = 0; i < 2; ++i) {\n"
    "        const float* ptr =",
    candidate_start,
)
candidate = text[candidate_start:candidate_end]

required = (
    "MmaFloat4 weights[2];",
    "MmaFloat4 row_scale[2];",
    "const int row = output_row[i * 4];",
    "const_cast<float*>(moe_weights_ptr + row)",
    "const_cast<float*>(scale_a_ptr + row)",
    "row + 3,\n            em,\n            MACA_ICMP_SLT",
)
for token in required:
    if token not in candidate:
        sys.exit("SOURCE CHECK FAIL: row-metadata token missing: " + token)

if candidate.count("for (uint32_t i = 0; i < 2; ++i)") != 1:
    sys.exit("SOURCE CHECK FAIL: row-metadata group loop changed")
if "for (uint32_t j" in candidate:
    sys.exit("SOURCE CHECK FAIL: scalar row-metadata loop remains")
if candidate.count("__builtin_mxc_ldg_b32_predicator") != 0:
    sys.exit("SOURCE CHECK FAIL: scalar row-metadata load remains")
if candidate.count("__builtin_mxc_ldg_b128_predicator") != 2:
    sys.exit("SOURCE CHECK FAIL: expected two b128 sites in the two-way unrolled loop")

baseline_scalar_sites = 2
baseline_rows_per_group = 4
row_groups = 2
candidate_vector_sites = 2
baseline_loads = baseline_scalar_sites * baseline_rows_per_group * row_groups
candidate_loads = candidate_vector_sites * row_groups
if (baseline_loads, candidate_loads) != (16, 4):
    sys.exit("SOURCE CHECK FAIL: row-metadata load-count proof changed")

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
reconstructed_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61"
if reconstructed_sha256 != expected_baseline_sha256:
    sys.exit(
        "SOURCE CHECK FAIL: change escaped assigned row-metadata block: "
        + reconstructed_sha256
    )

print(
    "SOURCE CHECK PASS: exact assigned baseline reconstructed; output mapping unchanged; "
    "row metadata LDG 16xb32->4xb128, bytes=64, predicate guards final row"
)
PYEOF

printf 'BUILD exp-20260823-027 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
