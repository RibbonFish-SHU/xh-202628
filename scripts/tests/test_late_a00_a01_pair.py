#!/usr/bin/env python3
"""No-device proof for the late A00/A01 pair interaction."""

from __future__ import annotations

import hashlib
import re
import unittest
from collections import Counter
from pathlib import Path

from scripts.tests.test_case2_fixed_nk_u32_brow import reverse_to_formal_best


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260826-150.sh"

BASELINE_SHA256 = (
    "762b7581919a9cb75cde0e4732c8ac580ad236f86620f2d3f34c565d7c5a3204"
)

BASE_TEMPLATE = "template <bool kUseOutputScratchExpertSort>"
FIXED_TEMPLATE = (
    "template <bool kUseOutputScratchExpertSort, int kFixedEm, "
    "int kFixedN, int kFixedK>"
)
BASE_PARAMETERS = """    int em,
    int n,
    int k
"""
FIXED_PARAMETERS = """    int runtime_em,
    int runtime_n,
    int runtime_k
"""
FIXED_BINDINGS = """    const int em = kFixedEm == 0 ? runtime_em : kFixedEm;
    const int n = kFixedN == 0 ? runtime_n : kFixedN;
    const int k = kFixedK == 0 ? runtime_k : kFixedK;

"""
BASE_SORTED_LAUNCH = """        fused_moe_i8_tn_mma_kernel<true><<<grid, block>>>(
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
FIXED_CASE4_SORTED_LAUNCH = """        if (use_prefill_down_module_full_expert_sort(config)) {
            fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>
                <<<grid, block>>>(
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
        } else {
            fused_moe_i8_tn_mma_kernel<true, 0, 0, 0><<<grid, block>>>(
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
        }
"""

PREHEADER_CANDIDATE = """    __syncthreadshared();

    XH_LDS_B_B128(0, 0);
"""
PREHEADER_BASELINE = """    __syncthreadshared();

    XH_LDS_A_B128(0, 0);
    XH_LDS_B_B128(0, 0);
"""
LOOP_A00_CANDIDATE = """    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {
        XH_LDG_B_STAGE_I(0);
        XH_LDG_B_STAGE_I(1);
        XH_LDS_A_B128(0, 0);
        XH_MMA_STAGE_MNKX2(0, 0, 0);
"""
LOOP_A00_BASELINE = """    for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k) {
        XH_LDG_B_STAGE_I(0);
        XH_LDG_B_STAGE_I(1);
        XH_MMA_STAGE_MNKX2(0, 0, 0);
"""
BACKEDGE_CANDIDATE = """        __syncthreadshared();
        XH_MMA_STAGE_MNKX2(1, 5, 6);
        XH_LDS_B_B128(0, 0);
"""
BACKEDGE_BASELINE = """        __syncthreadshared();
        XH_MMA_STAGE_MNKX2(1, 5, 6);
        XH_LDS_A_B128(0, 0);
        XH_LDS_B_B128(0, 0);
"""
TAIL_A00_CANDIDATE = """    int output_row[8];
    XH_LDS_A_B128(0, 0);
    XH_MMA_STAGE_MNKX2(0, 0, 0);
"""
TAIL_A00_BASELINE = """    int output_row[8];
    XH_MMA_STAGE_MNKX2(0, 0, 0);
"""

STEADY_A01_EARLY_CANDIDATE = """        XH_MMA_STAGE_MNKX2(0, 4, 0);
        XH_MMA_STAGE_MNKX2(0, 4, 2);
"""
STEADY_A01_EARLY_BASELINE = """        XH_MMA_STAGE_MNKX2(0, 4, 0);
        XH_LDS_A_B128(0, 1);
        XH_MMA_STAGE_MNKX2(0, 4, 2);
"""
STEADY_A01_CONSUMER_CANDIDATE = """        XH_LDS_B_B128(4, 1);
        XH_LDS_A_B128(0, 1);
        XH_MMA_STAGE_MNKX2(0, 0, 4);
"""
STEADY_A01_CONSUMER_BASELINE = """        XH_LDS_B_B128(4, 1);
        XH_MMA_STAGE_MNKX2(0, 0, 4);
"""
TAIL_A01_EARLY_CANDIDATE = """    XH_MMA_STAGE_MNKX2(0, 4, 0);
    XH_MMA_STAGE_MNKX2(0, 4, 2);
"""
TAIL_A01_EARLY_BASELINE = """    XH_MMA_STAGE_MNKX2(0, 4, 0);
    XH_LDS_A_B128(0, 1);
    XH_MMA_STAGE_MNKX2(0, 4, 2);
"""
TAIL_A01_CONSUMER_CANDIDATE = """    XH_LDS_B_B128(4, 1);
    XH_LDS_A_B128(0, 1);
    XH_MMA_STAGE_MNKX2(0, 0, 4);
"""
TAIL_A01_CONSUMER_BASELINE = """    XH_LDS_B_B128(4, 1);
    XH_MMA_STAGE_MNKX2(0, 0, 4);
"""

ASSIGNED_LDS_RE = re.compile(
    r"^[ \t]*XH_LDS_A_B128\(0, [01]\);\n", re.MULTILINE
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"expected one occurrence, got {count}: {old!r}")
    return source.replace(old, new, 1)


def reverse_to_baseline(source: str) -> str:
    reverted = replace_once(source, LOOP_A00_CANDIDATE, LOOP_A00_BASELINE)
    reverted = replace_once(reverted, PREHEADER_CANDIDATE, PREHEADER_BASELINE)
    reverted = replace_once(reverted, BACKEDGE_CANDIDATE, BACKEDGE_BASELINE)
    reverted = replace_once(reverted, TAIL_A00_CANDIDATE, TAIL_A00_BASELINE)
    reverted = replace_once(
        reverted, STEADY_A01_EARLY_CANDIDATE, STEADY_A01_EARLY_BASELINE
    )
    reverted = replace_once(
        reverted, STEADY_A01_CONSUMER_CANDIDATE, STEADY_A01_CONSUMER_BASELINE
    )
    reverted = replace_once(
        reverted, TAIL_A01_EARLY_CANDIDATE, TAIL_A01_EARLY_BASELINE
    )
    return replace_once(
        reverted, TAIL_A01_CONSUMER_CANDIDATE, TAIL_A01_CONSUMER_BASELINE
    )


def project_optional_fixed_case4(source: str) -> str:
    if FIXED_TEMPLATE not in source:
        return source
    projected = replace_once(source, FIXED_TEMPLATE, BASE_TEMPLATE)
    projected = replace_once(projected, FIXED_PARAMETERS, BASE_PARAMETERS)
    projected = replace_once(projected, FIXED_BINDINGS, "")
    projected = replace_once(
        projected, FIXED_CASE4_SORTED_LAUNCH, BASE_SORTED_LAUNCH
    )
    return replace_once(
        projected,
        "fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>",
        "fused_moe_i8_tn_mma_kernel<false>",
    )


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


def function_signature(source: str, signature: str) -> str:
    start = source.index(signature)
    return source[start : source.index("{", start)]


def steady_body(source: str) -> str:
    start = source.index(LOOP_A00_CANDIDATE) + len(LOOP_A00_CANDIDATE)
    end = source.index("\n    }\n\n    int output_row[8];", start)
    return LOOP_A00_CANDIDATE + source[start:end]


def tail_body(source: str) -> str:
    start = source.index("    int output_row[8];")
    end = source.index("    MmaInt4 output[kMmaOutputVectors];", start)
    return source[start:end]


def invocation_lines(source: str, name: str) -> tuple[str, ...]:
    pattern = re.compile(rf"^[ \t]*({name}\([^;]+\);)$", re.MULTILINE)
    return tuple(pattern.findall(source))


def producer_vectors(
    wave: int, load_indices: range = range(4)
) -> Counter[tuple[int, int]]:
    vectors: Counter[tuple[int, int]] = Counter()
    for lane in range(64):
        tid = wave * 64 + lane
        col = (((tid // 8) + (tid % 8)) % 8) * 16
        for load_index in load_indices:
            row = wave * 32 + lane // 8 + load_index * 8
            vectors[(row, col)] += 1
    return vectors


def consumer_vectors(
    wave: int, row_indices: range = range(2), col_indices: range = range(2)
) -> Counter[tuple[int, int]]:
    vectors: Counter[tuple[int, int]] = Counter()
    for lane in range(64):
        tid = wave * 64 + lane
        for row_index in row_indices:
            row = (tid % 16) + wave * 32 + 16 * row_index
            for col_index in col_indices:
                col = (
                    ((tid % 16) + (lane // 16) + 4 * col_index) % 8
                ) * 16
                vectors[(row, col)] += 1
    return vectors


def lifecycle(num_k_tiles: int) -> tuple[list[int], Counter[str], int]:
    if num_k_tiles < 1:
        raise ValueError("the kernel requires at least one K tile")
    published_tile = num_k_tiles - 1
    consumed_tiles: list[int] = []
    barriers = 1
    for next_tile in range(num_k_tiles - 1):
        consumed_tiles.append(published_tile)
        barriers += 2
        published_tile = next_tile
    consumed_tiles.append(published_tile)
    counts = Counter(
        name
        for _tile in consumed_tiles
        for name in ("A00", "A01", "A10", "A11")
    )
    return consumed_tiles, counts, barriers


def labeled_vectors(
    tiles: list[int] | range,
    wave: int,
    vectors: Counter[tuple[int, int]],
) -> Counter[tuple[int, int, int, int]]:
    labeled: Counter[tuple[int, int, int, int]] = Counter()
    for tile in tiles:
        for (row, col), count in vectors.items():
            labeled[(tile, wave, row, col)] += count
    return labeled


class LateA00A01PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = project_optional_fixed_case4(
            reverse_to_formal_best(SOURCE.read_text(encoding="utf-8"))
        )
        cls.baseline = reverse_to_baseline(cls.source)
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")
        cls.baseline_kernel = function_body(
            cls.baseline, "fused_moe_i8_tn_mma_kernel("
        )

    def test_exact_inverse_recovers_formal_best(self) -> None:
        self.assertEqual(digest(self.baseline), BASELINE_SHA256)
        self.assertEqual(
            ASSIGNED_LDS_RE.sub("", self.source),
            ASSIGNED_LDS_RE.sub("", self.baseline),
        )

    def test_only_a00_a01_placements_change(self) -> None:
        self.assertEqual(len(ASSIGNED_LDS_RE.findall(self.source)), 4)
        self.assertEqual(len(ASSIGNED_LDS_RE.findall(self.baseline)), 4)
        for candidate_anchor in (
            PREHEADER_CANDIDATE,
            LOOP_A00_CANDIDATE,
            BACKEDGE_CANDIDATE,
            TAIL_A00_CANDIDATE,
            STEADY_A01_EARLY_CANDIDATE,
            STEADY_A01_CONSUMER_CANDIDATE,
            TAIL_A01_EARLY_CANDIDATE,
            TAIL_A01_CONSUMER_CANDIDATE,
        ):
            self.assertEqual(self.source.count(candidate_anchor), 1)
        self.assertEqual(
            invocation_lines(self.kernel, "XH_LDS_A_B128"),
            (
                "XH_LDS_A_B128(0, 0);",
                "XH_LDS_A_B128(0, 1);",
                "XH_LDS_A_B128(1, 0);",
                "XH_LDS_A_B128(1, 1);",
                "XH_LDS_A_B128(0, 0);",
                "XH_LDS_A_B128(0, 1);",
                "XH_LDS_A_B128(1, 0);",
                "XH_LDS_A_B128(1, 1);",
            ),
        )

    def test_first_legal_steady_and_tail_consumer_anchors(self) -> None:
        steady = steady_body(self.source)
        tail = tail_body(self.source)
        self.assertIn(
            "XH_LDG_B_STAGE_I(1);\n"
            "        XH_LDS_A_B128(0, 0);\n"
            "        XH_MMA_STAGE_MNKX2(0, 0, 0);",
            steady,
        )
        self.assertIn(
            "XH_LDS_B_B128(4, 1);\n"
            "        XH_LDS_A_B128(0, 1);\n"
            "        XH_MMA_STAGE_MNKX2(0, 0, 4);",
            steady,
        )
        self.assertTrue(tail.startswith(TAIL_A00_CANDIDATE))
        self.assertIn(TAIL_A01_CONSUMER_CANDIDATE, tail)
        self.assertNotIn("__syncthreadshared();", tail)

        first_a0_overwrite = steady.index(
            "XH_MMA_STS(shared_a_tensor(store_row_a[0], store_col)"
        )
        self.assertLess(steady.index("XH_LDS_A_B128(0, 0);"), first_a0_overwrite)
        self.assertLess(steady.index("XH_LDS_A_B128(0, 1);"), first_a0_overwrite)
        final_publication = steady.rfind("__syncthreadshared();")
        self.assertLess(first_a0_overwrite, final_publication)
        self.assertNotIn("XH_LDS_A_B128(0, 0);", steady[final_publication:])

    def test_four_wave_producer_consumer_vector_bijection(self) -> None:
        for wave in range(4):
            producers = producer_vectors(wave)
            consumers = consumer_vectors(wave)
            group0_producers = producer_vectors(wave, range(2))
            group0_consumers = consumer_vectors(wave, range(1), range(2))
            self.assertEqual(producers, consumers)
            self.assertEqual(group0_producers, group0_consumers)
            self.assertEqual((len(producers), sum(producers.values())), (256, 256))
            self.assertEqual(
                (len(group0_producers), sum(group0_producers.values())),
                (128, 128),
            )

    def test_real_tile_order_and_exact_per_tile_lds_cardinality(self) -> None:
        for num_k_tiles in (1, 2, 16, 56):
            consumed, counts, barriers = lifecycle(num_k_tiles)
            self.assertEqual(
                consumed, [num_k_tiles - 1, *range(num_k_tiles - 1)]
            )
            self.assertEqual(
                counts,
                Counter(
                    {
                        "A00": num_k_tiles,
                        "A01": num_k_tiles,
                        "A10": num_k_tiles,
                        "A11": num_k_tiles,
                    }
                ),
            )
            self.assertEqual(sum(counts.values()), 4 * num_k_tiles)
            self.assertEqual(barriers, 2 * num_k_tiles - 1)
            for wave in range(4):
                expected = labeled_vectors(
                    range(num_k_tiles), wave, producer_vectors(wave)
                )
                observed = labeled_vectors(
                    consumed, wave, consumer_vectors(wave)
                )
                self.assertEqual(observed, expected)

    def test_protected_projection_shared_geometry_and_abi_are_exact(self) -> None:
        self.assertEqual(
            ASSIGNED_LDS_RE.sub("", self.kernel),
            ASSIGNED_LDS_RE.sub("", self.baseline_kernel),
        )
        for name in (
            "XH_LDG_A_STAGE_I",
            "XH_LDG_B_STAGE_I",
            "XH_MMA_STS",
            "XH_LDS_B_B128",
            "XH_MMA_STAGE_MNKX2",
        ):
            self.assertEqual(
                invocation_lines(self.kernel, name),
                invocation_lines(self.baseline_kernel, name),
            )
        self.assertEqual(
            len(invocation_lines(self.kernel, "XH_MMA_STAGE_MNKX2")), 128
        )
        self.assertEqual(self.kernel.count("__syncthreadshared();"), 3)
        self.assertIn("constexpr int kMmaThreads = 256;", self.source)
        self.assertIn(
            "constexpr int kMmaSharedBytes = kMmaSharedABytes + kMmaSharedBBytes;",
            self.source,
        )
        self.assertEqual(128 * 128 + 128 * 128, 32768)

        kernel_signature = function_signature(
            self.source, "fused_moe_i8_tn_mma_kernel("
        )
        baseline_kernel_signature = function_signature(
            self.baseline, "fused_moe_i8_tn_mma_kernel("
        )
        self.assertEqual(kernel_signature, baseline_kernel_signature)
        public_signature = function_signature(self.source, 'extern "C" void run_kernel(')
        baseline_public_signature = function_signature(
            self.baseline, 'extern "C" void run_kernel('
        )
        self.assertEqual(public_signature, baseline_public_signature)
        for input_name in (
            "a",
            "b_col_major",
            "scale_a",
            "scale_b",
            "moe_weights",
            "token_ids",
            "expert_ids",
        ):
            self.assertRegex(public_signature, rf"const [^,]+\b{input_name}\b")
        self.assertIn("int64_t topk", public_signature)
        self.assertIn("__nv_bfloat16* out", public_signature)

    def test_remote_job_is_byte_identical_to_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
