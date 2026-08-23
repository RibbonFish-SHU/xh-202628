#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_root/build/exp-20260823-054"
baseline_dir="$build_dir/baseline"
submission_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/submission.cu"
test_file="$repo_root/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
candidate_binary="$build_dir/test-fused-moe-candidate"
baseline_binary="$build_dir/test-fused-moe-baseline"
mkdir -p -- "$baseline_dir"

grep -Fq '*reinterpret_cast<uint64_t*>(' "$submission_file"
grep -Fq 'REGRESSION maca-plain-b64-store stores=' "$test_file"
grep -Fq 'plain_b64_store_proxy_kernel' "$test_file"

for rejected in '__dp4a' 'ldg_b128_bsm' '__builtin_mxc_arrive' '__shfl_sync' \
                'mma_kernel_64' 'fused_moe_i8_tn_mma_kernel_pair_n' 'wide_mma' \
                'kCase2L2NGroup' 'kSerpentineN' 'kCooperativeEpilogue' \
                'mma_adjacent_m' 'fused_moe_i8_tn_mma_kernel_n64' \
                'MmaShapeSpecialization' 'mma_shape_specialization' \
                'template <int kFixedEm' '__builtin_mxc_stg_b128'; do
  if grep -Fq "$rejected" "$submission_file"; then
    printf 'SOURCE CHECK FAIL: rejected mechanism remains: %s\n' "$rejected" >&2
    exit 1
  fi
done

python3 - "$submission_file" "$baseline_dir/submission.cu" <<'PYEOF'
import hashlib
import pathlib
import re
import sys

submission_path = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
text = submission_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")

candidate_store0 = """            *reinterpret_cast<uint64_t*>(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n
                + output_col[0]) = *reinterpret_cast<uint64_t*>(&packed_out);"""
candidate_store1 = """            *reinterpret_cast<uint64_t*>(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n
                + output_col[1]) = *reinterpret_cast<uint64_t*>(&packed_out);"""
baseline_store0 = """            __builtin_mxc_stg_b64_predicator(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n + output_col[0],
                0,
                *reinterpret_cast<uint64_t*>(&packed_out),
                true,
                false,
                false,
                (output_row[i * 4 + j] < em) && output_col_mask[0],
                1,
                MACA_ICMP_EQ);"""
baseline_store1 = """            __builtin_mxc_stg_b64_predicator(
                out_base + static_cast<uint64_t>(output_row[i * 4 + j]) * n + output_col[1],
                0,
                *reinterpret_cast<uint64_t*>(&packed_out),
                true,
                false,
                false,
                (output_row[i * 4 + j] < em) && output_col_mask[1],
                1,
                MACA_ICMP_EQ);"""

def replace_one(source, candidate, baseline, label):
    count = source.count(candidate)
    if count != 1:
        sys.exit("SOURCE CHECK FAIL: %s count=%d" % (label, count))
    return source.replace(candidate, baseline, 1)

reconstructed = replace_one(text, candidate_store0, baseline_store0, "store 0")
reconstructed = replace_one(reconstructed, candidate_store1, baseline_store1, "store 1")
baseline_sha256 = hashlib.sha256(reconstructed.encode()).hexdigest()
expected_baseline_sha256 = "bed1887e257f7a513d4ba4db10d5e5ac88ccecf6377d24d4ca3f521fbd795b61"
if baseline_sha256 != expected_baseline_sha256:
    sys.exit("SOURCE CHECK FAIL: exact-best reconstruction mismatch: " + baseline_sha256)

candidate_sha256 = hashlib.sha256(text.encode()).hexdigest()
expected_candidate_sha256 = "f73c6ca21426e3a849df28527ac5452d1787e95e9419efd0b658dfedee745405"
if candidate_sha256 != expected_candidate_sha256:
    sys.exit("SOURCE CHECK FAIL: candidate source mismatch: " + candidate_sha256)

with baseline_path.open("w", encoding="utf-8", newline="\n") as output:
    output.write(reconstructed)

def read_const(name):
    match = re.search(r"constexpr int %s = (\d+);" % name, text)
    if not match:
        sys.exit("SOURCE CHECK FAIL: missing constant " + name)
    return int(match.group(1))

tile_m = read_const("kMmaTileM")
tile_n = read_const("kMmaTileN")
tile_k = read_const("kMmaTileK")
threads = read_const("kMmaThreads")
shared_bytes = tile_m * tile_k + tile_n * tile_k
if (tile_m, tile_n, tile_k, threads, shared_bytes) != (128, 128, 128, 256, 32768):
    sys.exit("SOURCE CHECK FAIL: kernel geometry changed")

kernel_start = text.index("__global__ void fused_moe_i8_tn_mma_kernel(")
kernel_end = text.index("\n#else", kernel_start)
kernel = text[kernel_start:kernel_end]
baseline_kernel = reconstructed[
    reconstructed.index("__global__ void fused_moe_i8_tn_mma_kernel("):
    reconstructed.index("\n#else", reconstructed.index("__global__ void fused_moe_i8_tn_mma_kernel("))]
protected = {
    "XH_MMA_STAGE_MNKX2(": 129,
    "XH_MMA_LDS(": 2,
    "XH_MMA_STS(": 13,
    "__syncthreadshared();": 3,
    "__builtin_mxc_ldg_b32_predicator(": 2,
    "__builtin_mxc_ldg_b128_predicator(": 2,
}
for token, expected in protected.items():
    if kernel.count(token) != expected or baseline_kernel.count(token) != expected:
        sys.exit("SOURCE CHECK FAIL: protected token changed: " + token)
if kernel.count("__builtin_mxc_stg_b64_predicator(") != 0:
    sys.exit("SOURCE CHECK FAIL: predicated b64 store remains")
if kernel.count("            *reinterpret_cast<uint64_t*>(\n") != 2:
    sys.exit("SOURCE CHECK FAIL: ordinary b64 store count changed")
if baseline_kernel.count("__builtin_mxc_stg_b64_predicator(") != 2:
    sys.exit("SOURCE CHECK FAIL: baseline store count changed")

def row_local(thread_id, row_group, row_in_group):
    wave = thread_id // 64
    lane = thread_id % 64
    return ((lane // 16) % 2) * 4 + wave * 8 + (lane // 32) * 32 + row_group * 64 + row_in_group

def col_local(thread_id, col_group):
    return (thread_id % 16) * 4 + col_group * 64

visits = [0] * (tile_m * tile_n)
ordered_stores = []
for thread_id in range(threads):
    for row_group in range(2):
        for row_in_group in range(4):
            row = row_local(thread_id, row_group, row_in_group)
            for col_group in range(2):
                col = col_local(thread_id, col_group)
                if not (0 <= row < tile_m and 0 <= col and col + 3 < tile_n):
                    sys.exit("SOURCE CHECK FAIL: store out of local bounds")
                byte_offset = 2 * (row * tile_n + col)
                if byte_offset % 8:
                    sys.exit("SOURCE CHECK FAIL: store is not b64 aligned")
                ordered_stores.append((thread_id, row_group, row_in_group, col_group, row, col))
                for element in range(4):
                    visits[row * tile_n + col + element] += 1
if len(ordered_stores) != 4096 or visits != [1] * (tile_m * tile_n):
    sys.exit("SOURCE CHECK FAIL: output ownership/coverage changed")

for em, n, k in (
        (4096, 4096, 7168), (32768, 4096, 7168),
        (4096, 7168, 2048), (32768, 7168, 2048)):
    if em % tile_m or n % tile_n or k % tile_k or (n * 2) % 8:
        sys.exit("SOURCE CHECK FAIL: public shape alignment changed")
    last_m = em // tile_m - 1
    last_n = n // tile_n - 1
    for _, _, _, _, local_row, local_col in ordered_stores:
        row = last_m * tile_m + local_row
        col = last_n * tile_n + local_col
        if row >= em or col + 3 >= n or (2 * (row * n + col)) % 8:
            sys.exit("SOURCE CHECK FAIL: last public tile store is unsafe")
    print("SOURCE MODEL em=%d n=%d k=%d stores/CTA=4096 bytes/CTA=32768 aligned=8" % (em, n, k))

print(
    "SOURCE CHECK PASS: exact formal best reconstructed; exactly two predicated "
    "store sites replaced by ordinary aligned uint64_t assignments; 4096 stores, "
    "32768 bytes and 16384 BF16 values/CTA; ownership/order/bounds/alignment exact; "
    "candidate_sha256=" + candidate_sha256)
PYEOF

cp -- "$test_file" "$baseline_dir/test_fused_moe_i8_tn.cu"

if [[ "${XH_STATIC_ONLY:-0}" == "1" ]]; then
  printf 'STATIC-ONLY exp-20260823-054 PASS\n'
  exit 0
fi

cuda_home=${CUDA_HOME:-/usr/local/cuda}
printf 'proxy/NVIDIA BUILD candidate compiler=%s\n' "$cuda_home/bin/nvcc"
"$cuda_home/bin/nvcc" -O3 -std=c++14 -arch=sm_86 -lineinfo -Xcompiler=-Wall \
  "$test_file" -lcuda -o "$candidate_binary"
printf 'proxy/NVIDIA BUILD exact-baseline compiler=%s\n' "$cuda_home/bin/nvcc"
"$cuda_home/bin/nvcc" -O3 -std=c++14 -arch=sm_86 -lineinfo -Xcompiler=-Wall \
  "$baseline_dir/test_fused_moe_i8_tn.cu" -lcuda -o "$baseline_binary"
printf 'proxy/NVIDIA BUILD PASS (MACA store branch not compiled)\n'

"$candidate_binary" --correctness
"$candidate_binary" --regression

"$baseline_binary" --benchmark | tee "$build_dir/baseline-a.log"
"$candidate_binary" --benchmark | tee "$build_dir/candidate-a.log"
"$candidate_binary" --benchmark | tee "$build_dir/candidate-b.log"
"$baseline_binary" --benchmark | tee "$build_dir/baseline-b.log"

python3 - "$build_dir" <<'PYEOF'
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
conditions = {
    "baseline": (root / "baseline-a.log", root / "baseline-b.log"),
    "candidate": (root / "candidate-a.log", root / "candidate-b.log"),
}
pattern = re.compile(r"BENCHMARK proxy=NVIDIA case=(\S+) samples_ms=([0-9.,]+) median_ms=")
samples = {name: {} for name in conditions}
for name, paths in conditions.items():
    for path in paths:
        matches = pattern.findall(path.read_text(encoding="utf-8"))
        if len(matches) != 4:
            sys.exit("PAIRED BENCHMARK FAIL: expected four cases")
        for case, values in matches:
            samples[name].setdefault(case, []).extend(float(value) for value in values.split(","))
for case in samples["baseline"]:
    baseline = samples["baseline"][case]
    candidate = samples["candidate"][case]
    if len(baseline) != 10 or len(candidate) != 10:
        sys.exit("PAIRED BENCHMARK FAIL: expected 10+10 samples")
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    delta = 100.0 * (candidate_median / baseline_median - 1.0)
    print(
        "PAIRED proxy=NVIDIA case=%s baseline_median_ms=%.4f candidate_median_ms=%.4f "
        "candidate_delta_pct=%+.3f samples=10+10"
        % (case, baseline_median, candidate_median, delta))
print("PAIRED BENCHMARK PASS: NVIDIA executes the exact fallback in both binaries")
PYEOF
