#!/usr/bin/env python3
"""No-device proof for exact-case2 bounded-u32 A pointer offsets."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators" / "fused_moe_i8_tn" / "cuda_maca" / "submission.cu"
TEMPLATE_JOB = ROOT / "templates" / "remote-job.sh"
EXPERIMENT_JOB = ROOT / "remote-jobs" / "exp-20260830-236.sh"

EM = 32768
K = 7168
TILE_M = 128
TILE_K = 128
THREADS = 256
LOADS_A = 4
ROWS_PER_LOAD = 32
VECTOR_BYTES = 16
UINT32_LIMIT = 1 << 32

BASELINE_SOURCE_SHA256 = (
    "083eb1262dbe220aea0c2b324a00f2cf9b14720dc2b8c1701b261f794eaf8cf1"
)
CANDIDATE_SOURCE_SHA256 = (
    "b770a63ce5b363c9a6b804e295aed967e9fdccf7bc184d4eb4acb1d056b2aef6"
)

A_POLICY = """    constexpr bool kUseCase2FixedNkU32AOffsets =
        kUseOutputScratchExpertSort && kFixedEm == 0
        && kFixedN == 4096 && kFixedK == 7168;
"""

BASELINE_A_MACRO_ADDRESS = (
    "        a_base + load_a_row_offset[ldgi] + load_k,"
    "                                                \\\n"
)
CANDIDATE_A_MACRO_ADDRESS = """        kUseCase2FixedNkU32AOffsets                                                               \\
            ? a_base + static_cast<uint32_t>(load_a_row_offset[ldgi])                            \\
            : a_base + load_a_row_offset[ldgi] + load_k,                                         \\
"""

BASELINE_A_ROW_OFFSET = "        load_a_row_offset[i] = routed_row * k;\n"
CANDIDATE_A_ROW_OFFSET = """        if constexpr (kUseCase2FixedNkU32AOffsets) {
            load_a_row_offset[i] = static_cast<int>(
                static_cast<uint32_t>(routed_row) * static_cast<uint32_t>(k)
                + static_cast<uint32_t>(load_k));
        } else {
            load_a_row_offset[i] = routed_row * k;
        }
"""

A_BASE_INIT = (
    "    int8_t* a_base = const_cast<int8_t*>(a_ptr) "
    "+ (num_k_tiles - 1) * kMmaTileK;\n"
)
A_BASE_RESET = "    a_base = const_cast<int8_t*>(a_ptr);\n"
A_BASE_INCREMENT = "        a_base += kMmaTileK;\n"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"expected one occurrence, got {count}: {old!r}")
    return source.replace(old, new, 1)


def candidate_preheader_address(load_index: int) -> str:
    return f"""        kUseCase2FixedNkU32AOffsets
            ? a_base + static_cast<uint32_t>(load_a_row_offset[{load_index}])
            : a_base + load_a_row_offset[{load_index}] + load_k,
"""


def baseline_preheader_address(load_index: int) -> str:
    return f"        a_base + load_a_row_offset[{load_index}] + load_k,\n"


def reverse_to_formal_best(source: str) -> str:
    reverted = replace_once(source, A_POLICY, "")
    reverted = replace_once(
        reverted, CANDIDATE_A_MACRO_ADDRESS, BASELINE_A_MACRO_ADDRESS
    )
    reverted = replace_once(
        reverted, CANDIDATE_A_ROW_OFFSET, BASELINE_A_ROW_OFFSET
    )
    for load_index in range(LOADS_A):
        reverted = replace_once(
            reverted,
            candidate_preheader_address(load_index),
            baseline_preheader_address(load_index),
        )
    return reverted


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


class Case2AU32PointerInductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")
        cls.formal = reverse_to_formal_best(cls.source)
        cls.kernel = function_body(cls.source, "fused_moe_i8_tn_mma_kernel(")

    def test_all_case2_a_vector_starts_are_exact_bounded_and_aligned(self) -> None:
        starts = 0
        maximum = 0
        for tile_m in range(EM // TILE_M):
            row_base = tile_m * TILE_M
            for tid in range(THREADS):
                lane = tid % 64
                load_k = (lane % 8) * VECTOR_BYTES
                for load_index in range(LOADS_A):
                    routed_row = row_base + tid // 8 + ROWS_PER_LOAD * load_index
                    folded_offset = routed_row * K + load_k
                    self.assertLess(folded_offset, UINT32_LIMIT)
                    recovered = folded_offset & (UINT32_LIMIT - 1)
                    self.assertEqual(recovered, folded_offset)
                    for tile_k in range(K // TILE_K):
                        candidate = tile_k * TILE_K + recovered
                        baseline = routed_row * K + tile_k * TILE_K + load_k
                        self.assertEqual(candidate, baseline)
                        self.assertEqual(candidate % VECTOR_BYTES, 0)
                        self.assertLess(candidate + VECTOR_BYTES - 1, EM * K)
                        maximum = max(maximum, candidate)
                        starts += 1

        self.assertEqual(starts, 14_680_064)
        self.assertEqual(maximum, EM * K - VECTOR_BYTES)
        self.assertEqual(maximum, 234_881_008)
        self.assertLess(maximum, 1 << 28)

    def test_exact_case2_policy_and_u32_address_forms(self) -> None:
        self.assertEqual(self.source.count(A_POLICY), 1)
        self.assertEqual(self.source.count(CANDIDATE_A_MACRO_ADDRESS), 1)
        for load_index in range(LOADS_A):
            self.assertEqual(
                self.source.count(candidate_preheader_address(load_index)), 1
            )
        self.assertEqual(self.source.count(CANDIDATE_A_ROW_OFFSET), 1)
        self.assertEqual(self.source.count(A_BASE_INIT), 1)
        self.assertEqual(self.source.count(A_BASE_RESET), 1)
        self.assertEqual(self.source.count(A_BASE_INCREMENT), 1)
        self.assertIn(
            "for (uint32_t tile_k = 0; tile_k < loop_k_tiles; ++tile_k)",
            self.kernel,
        )
        self.assertNotIn("tile_k * static_cast<uint32_t>(kMmaTileK)", CANDIDATE_A_MACRO_ADDRESS)

    def test_reverse_projection_and_protected_schedule(self) -> None:
        self.assertEqual(digest(self.source), CANDIDATE_SOURCE_SHA256)
        self.assertEqual(digest(self.formal), BASELINE_SOURCE_SHA256)
        self.assertNotIn("kUseCase2FixedNkU32AOffsets", self.formal)
        self.assertEqual(self.kernel.count("__syncthreadshared();"), 3)
        self.assertEqual(self.kernel.count("XH_MMA_STAGE_MNKX2("), 129)
        self.assertEqual(self.kernel.count("XH_LDG_A_STAGE_I("), 5)
        self.assertEqual(self.kernel.count("XH_LDG_B_STAGE_I("), 5)
        self.assertEqual(self.kernel.count("__builtin_mxc_ldg_b128("), 6)
        self.assertEqual(self.kernel.count("__builtin_mxc_ldg_b128_predicator("), 5)

    def test_read_only_abi_and_remote_job_are_protected(self) -> None:
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
        self.assertEqual(EXPERIMENT_JOB.read_bytes(), TEMPLATE_JOB.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
