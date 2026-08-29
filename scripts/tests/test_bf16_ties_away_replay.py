import math
import random
import re
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators/fused_moe_i8_tn/cuda_maca/submission.cu"
RTOL = 2e-2
ATOL = 5e-3


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bf16_to_f32(bits: int) -> float:
    return f32_from_bits(bits << 16)


def rne_bf16(bits: int) -> int:
    bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + bias) & 0xFFFFFFFF) >> 16


def ties_away_bf16(bits: int) -> int:
    return ((bits + 0x8000) & 0xFFFFFFFF) >> 16


def is_finite_f32(bits: int) -> bool:
    return (bits & 0x7F800000) != 0x7F800000


def official_close(actual: float, reference: float) -> bool:
    if math.isinf(actual) or math.isinf(reference):
        return actual == reference
    return abs(actual - reference) <= ATOL + RTOL * abs(reference)


class Bf16TiesAwayReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def assert_contract(self, bits: int) -> None:
        if not is_finite_f32(bits):
            return
        actual = bf16_to_f32(ties_away_bf16(bits))
        reference = bf16_to_f32(rne_bf16(bits))
        self.assertTrue(
            official_close(actual, reference),
            f"bits=0x{bits:08x} actual={actual!r} reference={reference!r}",
        )

    def test_formal_macro_is_ties_away(self) -> None:
        block = re.search(
            r"#define XH_CVT_F32_TO_BF16.*?\n\s*dst = __builtin_mxc_byte_perm.*?\n",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(block)
        text = block.group(0)
        self.assertIn("src0 += 0x8000", text)
        self.assertIn("src1 += 0x8000", text)
        self.assertNotIn("src0 >> 16", text)
        self.assertNotIn("src1 >> 16", text)

    def test_decode_m4_rne_macro_is_unchanged(self) -> None:
        block = re.search(
            r"#define CVT_F32_TO_BF16.*?\n\s*dst\s*= __builtin_mxc_byte_perm.*?;\n",
            self.source,
            re.DOTALL,
        )
        self.assertIsNotNone(block)
        text = block.group(0)
        self.assertIn("src0 = ((src0 >> 16) & 1) + src0 + 0x7fff", text)
        self.assertIn("src1 = ((src1 >> 16) & 1) + src1 + 0x7fff", text)

    def test_all_high_words_at_rounding_boundaries(self) -> None:
        low_words = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF)
        for high in range(1 << 16):
            for low in low_words:
                self.assert_contract((high << 16) | low)

    def test_all_low_words_for_extreme_and_subnormal_classes(self) -> None:
        high_words = (
            0x0000,
            0x0001,
            0x007F,
            0x0080,
            0x3F7F,
            0x3F80,
            0x7F7E,
            0x7F7F,
            0x8000,
            0x8001,
            0x807F,
            0x8080,
            0xBF7F,
            0xBF80,
            0xFF7E,
            0xFF7F,
        )
        for high in high_words:
            for low in range(1 << 16):
                self.assert_contract((high << 16) | low)

    def test_rne_overflow_boundary_is_preserved(self) -> None:
        for bits in (0x7F7F7FFF, 0x7F7F8000, 0x7F7FFFFF, 0xFF7F7FFF, 0xFF7F8000, 0xFF7FFFFF):
            self.assertEqual(ties_away_bf16(bits), rne_bf16(bits))

    def test_deterministic_random_finite_values(self) -> None:
        generator = random.Random(0x20260830)
        checked = 0
        while checked < 250_000:
            bits = generator.getrandbits(32)
            if is_finite_f32(bits):
                self.assert_contract(bits)
                checked += 1


if __name__ == "__main__":
    unittest.main()
