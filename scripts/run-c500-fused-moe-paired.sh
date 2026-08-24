#!/usr/bin/env bash
set -euo pipefail

for variable in \
  C500_CANDIDATE_SOURCE C500_BASELINE_SOURCE C500_RESULTS_DIR \
  C500_CANDIDATE_COMMIT C500_BASELINE_COMMIT C500_SUBMISSION_SOURCE \
  C500_CANDIDATE_SOURCE_SHA256 C500_BASELINE_SOURCE_SHA256 CUCC; do
  [[ -n "${!variable:-}" ]] || { printf 'missing required environment variable: %s\n' "$variable" >&2; exit 64; }
done

candidate_test="$C500_CANDIDATE_SOURCE/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
baseline_test="$C500_BASELINE_SOURCE/operators/fused_moe_i8_tn/cuda_maca/test_fused_moe_i8_tn.cu"
build_dir="$C500_CANDIDATE_SOURCE/build/c500-paired"
candidate_binary="$build_dir/test-fused-moe-candidate"
baseline_binary="$build_dir/test-fused-moe-baseline"
mkdir -p -- "$build_dir" "$C500_RESULTS_DIR"

common_flags=(-O3 -std=c++17 -arch=sm_80 -lineinfo -resource-usage)
{
  printf 'candidate_commit=%s\nbaseline_commit=%s\n' "$C500_CANDIDATE_COMMIT" "$C500_BASELINE_COMMIT"
  printf 'candidate_source_sha256=%s\nbaseline_source_sha256=%s\n' \
    "$C500_CANDIDATE_SOURCE_SHA256" "$C500_BASELINE_SOURCE_SHA256"
  printf 'cucc='; printf '%q ' "$CUCC" "${common_flags[@]}" "$candidate_test" -lcuda -o "$candidate_binary"; printf '\n'
  printf 'cucc='; printf '%q ' "$CUCC" "${common_flags[@]}" "$baseline_test" -lcuda -o "$baseline_binary"; printf '\n'
  printf 'sequence=warmup-baseline,warmup-candidate,baseline-a,candidate-a,candidate-b,baseline-b\n'
} > "$C500_RESULTS_DIR/commands.txt"

"$CUCC" "${common_flags[@]}" "$candidate_test" -lcuda -o "$candidate_binary" \
  > "$C500_RESULTS_DIR/build-candidate.log" 2>&1
"$CUCC" "${common_flags[@]}" "$baseline_test" -lcuda -o "$baseline_binary" \
  > "$C500_RESULTS_DIR/build-baseline.log" 2>&1

"$candidate_binary" --correctness > "$C500_RESULTS_DIR/correctness.log" 2>&1
"$candidate_binary" --regression > "$C500_RESULTS_DIR/regression.log" 2>&1

"$baseline_binary" --benchmark > "$C500_RESULTS_DIR/warmup-baseline.log" 2>&1
"$candidate_binary" --benchmark > "$C500_RESULTS_DIR/warmup-candidate.log" 2>&1
mx-smi --show-clocks xcore > "$C500_RESULTS_DIR/mx-smi-clock-after-warmup.txt" 2>&1 || true

"$baseline_binary" --benchmark > "$C500_RESULTS_DIR/benchmark-baseline-a.log" 2>&1
"$candidate_binary" --benchmark > "$C500_RESULTS_DIR/benchmark-candidate-a.log" 2>&1
"$candidate_binary" --benchmark > "$C500_RESULTS_DIR/benchmark-candidate-b.log" 2>&1
"$baseline_binary" --benchmark > "$C500_RESULTS_DIR/benchmark-baseline-b.log" 2>&1

python3 "$C500_CANDIDATE_SOURCE/scripts/summarize-c500-abba.py" \
  --baseline-a "$C500_RESULTS_DIR/benchmark-baseline-a.log" \
  --candidate-a "$C500_RESULTS_DIR/benchmark-candidate-a.log" \
  --candidate-b "$C500_RESULTS_DIR/benchmark-candidate-b.log" \
  --baseline-b "$C500_RESULTS_DIR/benchmark-baseline-b.log" \
  --output "$C500_RESULTS_DIR/paired-benchmark.json"

printf 'C500 build, correctness, regression, warmup and ABBA benchmark passed\n'
