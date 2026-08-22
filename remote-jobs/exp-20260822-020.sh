#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260822-020"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel<<<grid, block>>>' "$submission_file"
grep -Fq 'fused_moe_i8_tn_mma_kernel_pair_n<<<pair_grid, pair_block>>>' "$submission_file"
grep -Fq 'const int subgroup = threadIdx.x / kMmaThreads;' "$submission_file"
grep -Fq 'const int tid = threadIdx.x % kMmaThreads;' "$submission_file"
grep -Fq 'shared_data + 2 * kMmaSharedABytes + subgroup * kMmaSharedBBytes' "$submission_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' 'wide_mma'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")
marker = "// exp-20260822-020: two baseline 128x128 pipelines share A in one 512-thread CTA."

baseline_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
baseline_end = text.index(marker)
baseline_kernel = text[baseline_start:baseline_end].rstrip()
baseline_sha256 = hashlib.sha256(baseline_kernel.encode()).hexdigest()
if baseline_sha256 != "eab0c946da8403f461122f38cd072b3f95db73e22652545277ee0fb373a58d5f":
    sys.exit(
        "SOURCE CHECK FAIL: non-case-2 kernel diverged from OJ #117114 baseline: "
        + baseline_sha256
    )

pair_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel_pair_n(")
pair_end = text.index("\n#else", pair_start)
pair_kernel = text[pair_start:pair_end]

required = (
    "constexpr int kPairMmaSubgroups = 2;",
    "constexpr int kPairMmaThreads = kMmaThreads * kPairMmaSubgroups;",
    "constexpr int kPairMmaTileN = kMmaTileN * kPairMmaSubgroups;",
    "2 * kMmaSharedABytes + kPairMmaSubgroups * kMmaSharedBBytes;",
    "__shared__ int8_t shared_data[kPairMmaSharedBytes];",
    "int8_t* shared_a_active = shared_data;",
    "int8_t* shared_a_stage = shared_a_active + kMmaSharedABytes;",
    "shared_a_active = shared_a_stage;",
    "shared_a_stage = shared_a_swap;",
    "const int tile_n = blockIdx.y * kPairMmaSubgroups + subgroup;",
    "if (subgroup == 0)",
    "const dim3 pair_block(kPairMmaThreads);",
    "config.n / kPairMmaTileN",
    "config.em / kMmaTileM",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: required paired-N token missing: " + token)

comparisons = (
    ("XH_MMA_STAGE_MNKX2(", "XH_PAIR_MMA_STAGE_MNKX2("),
    ("XH_LDS_A_B128(", "XH_PAIR_LDS_A_B128("),
    ("XH_LDS_B_B128(", "XH_PAIR_LDS_B_B128("),
    ("XH_LDG_A_STAGE_I(", "XH_PAIR_LDG_A_STAGE_I("),
    ("XH_LDG_B_STAGE_I(", "XH_PAIR_LDG_B_STAGE_I("),
)
for baseline_token, pair_token in comparisons:
    baseline_count = baseline_kernel.count(baseline_token) - 1
    pair_count = pair_kernel.count(pair_token) - 1
    if baseline_count != pair_count:
        sys.exit(
            f"SOURCE CHECK FAIL: schedule count differs for {baseline_token}: "
            f"baseline={baseline_count} pair={pair_count}"
        )

baseline_a_sts = baseline_kernel.count("XH_MMA_STS(shared_a_tensor(")
pair_a_sts = (
    pair_kernel.count("XH_PAIR_STS_A_ACTIVE_B128(") - 1
    + pair_kernel.count("XH_PAIR_STS_A_STAGE_B128(") - 1
)
if baseline_a_sts != pair_a_sts:
    sys.exit(
        f"SOURCE CHECK FAIL: A STS schedule differs: baseline={baseline_a_sts} pair={pair_a_sts}"
    )
if baseline_kernel.count("__syncthreadshared()") != pair_kernel.count("__syncthreadshared()"):
    sys.exit("SOURCE CHECK FAIL: paired-N barrier schedule differs from the baseline")

if pair_kernel.count("XH_PAIR_MMA_STAGE_MNKX2(") - 1 != 128:
    sys.exit("SOURCE CHECK FAIL: paired-N kernel must retain 128 baseline MMA stage groups")
if pair_kernel.count("__syncthreadshared()") != 3:
    sys.exit("SOURCE CHECK FAIL: paired-N kernel must retain three barrier sites")

first_barrier = pair_kernel.index("__syncthreadshared()")
for row in range(4):
    token = f"XH_PAIR_STS_A_ACTIVE_B128({row});"
    if pair_kernel.index(token) > first_barrier:
        sys.exit("SOURCE CHECK FAIL: initial A buffer must be complete before first barrier")

loop_start = pair_kernel.index("for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k)")
loop_end = pair_kernel.index("\n    int output_row[8];", loop_start)
loop = pair_kernel[loop_start:loop_end]
second_barrier = loop.rindex("__syncthreadshared()")
swap = loop.index("shared_a_active = shared_a_stage;")
if swap < second_barrier:
    sys.exit("SOURCE CHECK FAIL: A buffers must swap only after the stage-complete barrier")
for row in range(4):
    token = f"XH_PAIR_STS_A_STAGE_B128({row});"
    if loop.index(token) > second_barrier:
        sys.exit("SOURCE CHECK FAIL: staged A tile is incomplete at the swap barrier")
for row in (2, 3):
    load = loop.index(f"XH_PAIR_LDG_A_STAGE_I({row});")
    store = loop.index(f"XH_PAIR_STS_A_STAGE_B128({row});")
    if load > store:
        sys.exit("SOURCE CHECK FAIL: A2/A3 must load before writing the inactive buffer")

print(
    "SOURCE CHECK PASS: baseline hash exact; paired-N=512 threads, 64 KiB, "
    "double-buffered shared A and disjoint B, 128 MMA stages"
)
PYEOF

printf 'BUILD exp-20260822-020 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
