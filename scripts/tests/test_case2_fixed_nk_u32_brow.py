#!/usr/bin/env python3
"""No-device proof for the fixed-N/K case-2 u32 B-row composition."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260827-180.sh"

CASE1 = (4096, 4096, 7168)
CASE2 = (32768, 4096, 7168)
CASE3 = (4096, 7168, 2048)
CASE4 = (32768, 7168, 2048)
FALLBACK = (384, 2048, 1024)
TILE_N = 128
TILE_K = 128
THREADS = 256
LOADS_B = 4
VECTOR_BYTES = 16
NUM_EXPERTS = 256
UINT32_LIMIT = 1 << 32
INT32_LIMIT = 1 << 31

BASELINE_SOURCE_SHA256 = (
    "0edbf3ba0efea172fa1958e25c6904ec4314ff86b9249deaaac56a98092341b3"
)
CANDIDATE_SOURCE_SHA256 = (
    "dfa9bdea39d86896fad6827bd7c2f57957d438f12a9013fdbcbec5406a828e07"
)

POLICY_BLOCK = """    constexpr bool kUseCase2FixedNkU32BLocalOffsets =
        kUseOutputScratchExpertSort && kFixedEm == 0
        && kFixedN == 4096 && kFixedK == 7168;
"""
BASE_CASE2_LAUNCH = (
    "fused_moe_i8_tn_mma_kernel<true, 0, 0, 0><<<grid, block>>>"
)
CANDIDATE_CASE2_LAUNCH = (
    "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168><<<grid, block>>>"
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"expected one occurrence, got {count}: {old!r}")
    return source.replace(old, new, 1)


def remove_retask_block(source: str, load_index: int) -> str:
    assignment = (
        f"        load_b_row[{load_index}] = candidate_col < col_limit "
        "? candidate_col : col_limit - 1;\n"
    )
    assignment_start = source.index(assignment)
    start = source.index(
        "        if constexpr (kUseCase2FixedNkU32BLocalOffsets) {\n",
        assignment_start + len(assignment),
    )
    end_marker = "        }\n"
    end = source.index(end_marker, start) + len(end_marker)
    return source[:start] + source[end:]


def replace_tail_address(source: str, load_index: int) -> str:
    load_anchor = f"        load_b_{load_index} = __builtin_mxc_ldg_b128_predicator(\n"
    load_start = source.index(load_anchor)
    start = source.index("            kUseCase2FixedNkU32BLocalOffsets\n", load_start)
    end = source.index("            0,\n", start)
    baseline = (
        f"            &(global_b(load_b_row[{load_index}], load_k, "
        "num_k_tiles - 1)),\n"
    )
    return source[:start] + baseline + source[end:]


def reverse_to_formal_best(source: str) -> str:
    reverted = replace_once(source, POLICY_BLOCK, "")
    macro_start = reverted.index(
        "        kUseCase2FixedNkU32BLocalOffsets",
        reverted.index("#define XH_LDG_B_STAGE_I"),
    )
    macro_end = reverted.index("        0,", macro_start)
    baseline_macro_address = (
        "        &(global_b(load_b_row[ldgi], load_k, tile_k)),"
        "                                            \\\n"
    )
    reverted = reverted[:macro_start] + baseline_macro_address + reverted[macro_end:]
    for load_index in range(LOADS_B):
        reverted = remove_retask_block(reverted, load_index)
        reverted = replace_tail_address(reverted, load_index)
    return replace_once(reverted, CANDIDATE_CASE2_LAUNCH, BASE_CASE2_LAUNCH)


def template_for(shape: tuple[int, int, int]) -> tuple[bool, int, int, int]:
    if shape == CASE4:
        return (True, 32768, 7168, 2048)
    if shape == CASE2:
        return (True, 0, 4096, 7168)
    return (False, 0, 0, 0)


def u32_policy(template: tuple[bool, int, int, int]) -> bool:
    sorted_map, fixed_em, fixed_n, fixed_k = template
    return sorted_map and fixed_em == 0 and fixed_n == 4096 and fixed_k == 7168


def row_and_k_offset(tid: int, load_index: int) -> tuple[int, int]:
    lane = tid % 64
    return (tid // 8) * LOADS_B + load_index, (lane % 8) * VECTOR_BYTES


class Case2FixedNkU32BRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.formal = reverse_to_formal_best(cls.source)
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")
        cls.launch = function_body(cls.source, "static inline void launch(")

    def test_exact_dispatch_and_compile_time_policy(self) -> None:
        expected = {
            CASE1: (False, 0, 0, 0),
            CASE2: (True, 0, 4096, 7168),
            CASE3: (False, 0, 0, 0),
            CASE4: (True, 32768, 7168, 2048),
            FALLBACK: (False, 0, 0, 0),
        }
        self.assertEqual({shape: template_for(shape) for shape in expected}, expected)
        self.assertEqual(
            {shape: u32_policy(args) for shape, args in expected.items()},
            {CASE1: False, CASE2: True, CASE3: False, CASE4: False, FALLBACK: False},
        )
        self.assertEqual(self.source.count(POLICY_BLOCK), 1)
        self.assertEqual(self.launch.count(CANDIDATE_CASE2_LAUNCH), 1)
        self.assertEqual(
            self.launch.count(
                "fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>"
            ),
            1,
        )
        self.assertEqual(
            self.launch.count("fused_moe_i8_tn_mma_kernel<false, 0, 0, 0>"), 1
        )

    def test_all_case2_addresses_are_exact_and_bounded(self) -> None:
        _em, n, k = CASE2
        starts = 0
        max_local_start = 0
        for tile_n in range(n // TILE_N):
            for tid in range(THREADS):
                for load_index in range(LOADS_B):
                    raw_row, load_k = row_and_k_offset(tid, load_index)
                    row = min(raw_row, TILE_N - 1)
                    local_base_u32 = (tile_n * TILE_N + row) * k + load_k
                    self.assertLess(local_base_u32, INT32_LIMIT)
                    recovered_u32 = int(local_base_u32) & (UINT32_LIMIT - 1)
                    self.assertEqual(recovered_u32, local_base_u32)
                    for tile_k in range(k // TILE_K):
                        candidate_local = recovered_u32 + tile_k * TILE_K
                        baseline_local = (
                            (tile_n * TILE_N + row) * k
                            + tile_k * TILE_K
                            + load_k
                        )
                        self.assertEqual(candidate_local, baseline_local)
                        self.assertEqual(candidate_local % VECTOR_BYTES, 0)
                        self.assertLess(candidate_local + VECTOR_BYTES - 1, n * k)
                        max_local_start = max(max_local_start, candidate_local)
                        starts += 1

        self.assertEqual(starts, 1_835_008)
        self.assertEqual(max_local_start, n * k - VECTOR_BYTES)
        expert_255_base = (NUM_EXPERTS - 1) * n * k
        self.assertEqual(
            expert_255_base + max_local_start + VECTOR_BYTES - 1,
            NUM_EXPERTS * n * k - 1,
        )
        self.assertLess(n * k, INT32_LIMIT)
        self.assertLess(n * k, UINT32_LIMIT)

    def test_no_extra_carrier_or_schedule_change(self) -> None:
        self.assertEqual(self.kernel.count("int load_b_row[kMmaLoadsB];"), 1)
        self.assertEqual(
            self.kernel.count(
                "if constexpr (kUseCase2FixedNkU32BLocalOffsets)"
            ),
            4,
        )
        for forbidden in (
            "load_b_local_offset",
            "load_b_ptr",
            "scale_b_local_offset",
            "output_local_offset",
            "ldg_b128_bsm",
            "kMmaTileK = 64",
        ):
            self.assertNotIn(forbidden, self.source)
        self.assertEqual(self.kernel.count("__syncthreadshared();"), 3)
        self.assertEqual(self.kernel.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(self.kernel.count("XH_LDG_B_STAGE_I("), 5)
        self.assertEqual(self.kernel.count("__builtin_mxc_ldg_b128_predicator("), 5)

    def test_reverse_projection_protects_case4_unsorted_and_abi(self) -> None:
        self.assertEqual(digest(self.source), CANDIDATE_SOURCE_SHA256)
        self.assertEqual(digest(self.formal), BASELINE_SOURCE_SHA256)
        self.assertNotIn("kUseCase2FixedNkU32BLocalOffsets", self.formal)
        self.assertNotIn(CANDIDATE_CASE2_LAUNCH, self.formal)
        signature_start = self.source.index('extern "C" void run_kernel(')
        signature_end = self.source.index("{", signature_start)
        signature = self.source[signature_start:signature_end]
        for name in (
            "a",
            "b_col_major",
            "scale_a",
            "scale_b",
            "moe_weights",
            "token_ids",
            "expert_ids",
        ):
            self.assertRegex(signature, rf"const [^,]+\b{name}\b")
        self.assertNotIn("cudaDeviceSynchronize", self.source)
        self.assertNotIn("cudaMemcpy", self.source)

    def test_remote_job_is_byte_identical_to_template(self) -> None:
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
