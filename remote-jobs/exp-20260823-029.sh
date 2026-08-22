#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-029"
mkdir -p -- "$build_dir"

submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
source_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
binary="$build_dir/test_fused_moe_i8_tn"

grep -q '__builtin_mxc_mma_16x16x16i8' "$submission_file"
grep -Fq 'load_b_ptr[ldgi]' "$submission_file"
grep -Fq 'load_b_ptr[i] += kMmaTileK;' "$submission_file"
grep -Fq 'maca-b-pointer-induction' "$source_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' 'mma_kernel_64' \
                'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' 'kCase2L2NGroup' \
                'kSerpentineN' 'kCooperativeEpilogue' 'mma_adjacent_m'; do
  if grep -q "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected token remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" <<'PYEOF'
import hashlib
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read().replace("\r\n", "\n")


def canonicalize_non_address_source(source):
    start = source.index("#define XH_LDG_B_STAGE_I(ldgi)")
    end = source.index("#define XH_LDS_A_B128", start)
    source = source[:start] + "<B_STAGE_ADDRESS>\n" + source[end:]

    start = source.index("    const int8_t* expert_b =")
    end = source.index("    int8_t* a_base =", start)
    source = source[:start] + "<B_POINTER_SETUP>\n" + source[end:]

    search_from = source.index("    int8_t* a_base =")
    start = source.index(
        "#pragma unroll\n"
        "    for (uint32_t i = 0; i < kMmaLoadsB; ++i) {",
        search_from,
    )
    end = source.index(
        "#pragma unroll\n"
        "    for (uint32_t i = 0; i < kMmaLoadsA; ++i) {",
        start,
    )
    source = source[:start] + "<B_LAST_TILE_PRELOAD>\n" + source[end:]

    loop = source.index(
        "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {")
    last_pipeline_op = source.index(
        "        XH_MMA_STAGE_MNKX2(1, 7, 6);", loop)
    last_pipeline_end = source.index("\n", last_pipeline_op) + 1
    loop_close = source.index("    }\n\n    int output_row[8];", last_pipeline_end)
    return source[:last_pipeline_end] + source[loop_close:]


protected_sha256 = hashlib.sha256(
    canonicalize_non_address_source(text).encode()).hexdigest()
if protected_sha256 != "5e57a6881e5ac8d96ccbbdbdd7baeda13f54caab80bcc27c71fe8e45b20c8eb3":
    sys.exit(
        "SOURCE CHECK FAIL: source outside B address formation changed: "
        + protected_sha256)

required = (
    "load_b_ptr[ldgi]",
    "int8_t* load_b_ptr[kMmaLoadsB];",
    "static_cast<uint64_t>(tile_n) * kMmaTileN * k + load_k;",
    "static_cast<uint64_t>(num_k_tiles - 1) * kMmaTileK;",
    "load_b_tile_base + static_cast<uint64_t>(load_b_row) * k;",
    "load_b_ptr[i] + last_b_tile_offset,",
    "load_b_ptr[i] += kMmaTileK;",
    "load_k,\n            k_head,\n            MACA_ICMP_SLT);",
)
for token in required:
    if token not in text:
        sys.exit("SOURCE CHECK FAIL: B pointer-induction token missing: " + token)
if "Tensor matrix_b" in text or "Tensor global_b" in text or "global_b(" in text:
    sys.exit("SOURCE CHECK FAIL: CUTE global-B coordinate formation remains")
if text.count("load_b_ptr[i] += kMmaTileK;") != 1:
    sys.exit("SOURCE CHECK FAIL: B pointer update must appear exactly once in source")

loop_start = text.index(
    "    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {")
loop_end = text.index("\n    }\n\n    int output_row[8];", loop_start)
loop = text[loop_start:loop_end]
loop_body = loop[loop.index("\n") + 1:]
if "tile_k" in loop_body:
    sys.exit("SOURCE CHECK FAIL: K-loop B addresses still depend on tile_k coordinates")
last_pipeline_op = loop.rindex("XH_MMA_STAGE_MNKX2(1, 7, 6);")
pointer_update = loop.index("load_b_ptr[i] += kMmaTileK;")
if pointer_update < last_pipeline_op:
    sys.exit("SOURCE CHECK FAIL: B pointers advance before the pipeline finishes")

trace_start = text.index("    MmaLoad128 load_a[kMmaLoadsA];")
trace_end = text.index("    MmaInt4 output[kMmaOutputVectors];", trace_start)
trace_prefixes = (
    "load_b[i] = __builtin_mxc_ldg_b128_predicator(",
    "load_a[i] = __builtin_mxc_ldg_b128(",
    "XH_LDG_A_STAGE_I(",
    "XH_LDG_B_STAGE_I(",
    "XH_LDS_A_B128(",
    "XH_LDS_B_B128(",
    "XH_MMA_STAGE_MNKX2(",
    "XH_MMA_STS(shared_a_tensor(",
    "XH_MMA_STS(shared_b_tensor(",
    "__syncthreadshared();",
)
trace = []
for line in text[trace_start:trace_end].splitlines():
    stripped = line.strip()
    if stripped.startswith(trace_prefixes):
        trace.append("".join(stripped.split()))
trace_sha256 = hashlib.sha256("\n".join(trace).encode()).hexdigest()
if trace_sha256 != "34ed816cbbc06ffa9b38e967f29e91225042162c0ca80eb2cd9433f0abb66418":
    sys.exit("SOURCE CHECK FAIL: LDG/MMA/LDS/STS/barrier order changed: " + trace_sha256)

expected_counts = {
    "load_b[i]=__builtin_mxc_ldg_b128_predicator(": 1,
    "load_a[i]=__builtin_mxc_ldg_b128(": 1,
    "XH_LDG_A_STAGE_I(": 4,
    "XH_LDG_B_STAGE_I(": 4,
    "XH_LDS_A_B128(": 8,
    "XH_LDS_B_B128(": 32,
    "XH_MMA_STAGE_MNKX2(": 128,
    "XH_MMA_STS(shared_a_tensor(": 8,
    "XH_MMA_STS(shared_b_tensor(": 5,
    "__syncthreadshared();": 3,
}
for prefix, expected in expected_counts.items():
    actual = sum(item.startswith(prefix) for item in trace)
    if actual != expected:
        sys.exit(
            "SOURCE CHECK FAIL: operation count %s expected=%d actual=%d"
            % (prefix, expected, actual))


def read_const(name):
    match = re.search(r"constexpr int %s = (\d+);" % name, text)
    if not match:
        sys.exit("SOURCE CHECK FAIL: constant not found: " + name)
    return int(match.group(1))


tile_m = read_const("kMmaTileM")
tile_n = read_const("kMmaTileN")
tile_k = read_const("kMmaTileK")
threads = read_const("kMmaThreads")
if (tile_m, tile_n, tile_k, threads) != (128, 128, 128, 256):
    sys.exit("SOURCE CHECK FAIL: CTA geometry changed")
if tile_m * tile_k + tile_n * tile_k != 32 * 1024:
    sys.exit("SOURCE CHECK FAIL: shared-memory footprint changed")

for em, n, k in (
    (4096, 4096, 7168),
    (32768, 4096, 7168),
    (4096, 7168, 2048),
    (32768, 7168, 2048),
):
    k_tiles = k // tile_k
    a_loads = 4 * k_tiles
    b_loads = 4 * k_tiles
    a_bytes = a_loads * 16 * threads
    b_bytes = b_loads * 16 * threads
    mma_instructions = 128 * k_tiles
    barriers = 1 + 2 * (k_tiles - 1)
    if a_bytes != tile_m * k or b_bytes != tile_n * k:
        sys.exit("SOURCE CHECK FAIL: A/B byte model changed")
    print(
        "SOURCE MODEL em=%d n=%d k=%d A-loads/thread=%d B-loads/thread=%d "
        "A-bytes/CTA=%d B-bytes/CTA=%d MMA=%d barriers=%d"
        % (em, n, k, a_loads, b_loads, a_bytes, b_bytes,
           mma_instructions, barriers))

print(
    "SOURCE CHECK PASS: protected baseline exact; four B row pointers advance 128 bytes; "
    "last predicate and LDG/MMA/LDS/STS/barrier trace preserved")
PYEOF

printf 'BUILD exp-20260823-029 compiler=%s\n' "$CUDA_HOME/bin/nvcc"
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
