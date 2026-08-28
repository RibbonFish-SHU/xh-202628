import hashlib
import re
import unittest
from pathlib import Path

from scripts.tests.test_case2_fixed_nk_u32_brow import reverse_to_formal_best


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators/fused_moe_i8_tn/cuda_maca/submission.cu"
FORMAL_SHA256 = "0edbf3ba0efea172fa1958e25c6904ec4314ff86b9249deaaac56a98092341b3"
M4_CORE_SHA256 = "8fdb8ea26dc0b9b05ed76a2d761138cc8cc8041cd60ec42ad2e9858337174cdf"


FORWARD_DECLARATION = """

#if XH_FUSED_MOE_MACA
namespace sync_m4 {
static inline void launch_decode(
    const int8_t* a,
    const int8_t* b_col_major,
    const float* scale_a,
    const float* scale_b,
    const float* moe_weights,
    const int32_t* expert_ids,
    __nv_bfloat16* out,
    const KernelConfig& config);
}  // namespace sync_m4
#endif
"""

DISPATCH = """    if (config.em == 4096
        && ((config.n == 4096 && config.k == 7168)
            || (config.n == 7168 && config.k == 2048))) {
        sync_m4::launch_decode(
            a, b_col_major, scale_a, scale_b, moe_weights, expert_ids, out, config);
        return;
    }
"""

COMPONENT_MARKER = "\n#if XH_FUSED_MOE_MACA\n\n// This component incorporates a modified mcTlass fused-MoE kernel."
CORE_BEGIN = "// ---- types (mirrors the 2stage/895 kernel) ----"
CORE_END = "\n\nstatic inline void launch_decode("


def text() -> str:
    return SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")


class SyncM4HybridTest(unittest.TestCase):
    def test_restricted_projection_recovers_formal_best(self):
        candidate = reverse_to_formal_best(text())
        formal, component = candidate.split(COMPONENT_MARKER, 1)
        formal = formal.replace(FORWARD_DECLARATION, "\n", 1)
        formal = formal.replace(DISPATCH, "", 1)
        self.assertEqual(hashlib.sha256(formal.encode()).hexdigest(), FORMAL_SHA256)
        self.assertIn("Provenance: exp-20260828-199 tested commit", component)

    def test_imported_runtime_core_is_exact_exp199(self):
        candidate = text()
        core = candidate.split(CORE_BEGIN, 1)[1].split(CORE_END, 1)[0]
        core = CORE_BEGIN + core + "\n"
        self.assertEqual(hashlib.sha256(core.encode()).hexdigest(), M4_CORE_SHA256)
        self.assertIn("constexpr int kTileK = 256;", core)
        self.assertIn("constexpr int kStage = 4;", core)
        self.assertIn("__shared__ T smem[(kABSize + kABSize)]", core)

    def test_exact_shape_dispatch_and_formal_fallback(self):
        candidate = text()
        self.assertEqual(candidate.count(DISPATCH), 1)

        def uses_m4(em: int, n: int, k: int) -> bool:
            return em == 4096 and ((n == 4096 and k == 7168) or (n == 7168 and k == 2048))

        self.assertTrue(uses_m4(4096, 4096, 7168))
        self.assertFalse(uses_m4(32768, 4096, 7168))
        self.assertTrue(uses_m4(4096, 7168, 2048))
        self.assertFalse(uses_m4(32768, 7168, 2048))
        self.assertFalse(uses_m4(128, 128, 128))
        self.assertIn(
            "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168>", candidate
        )
        self.assertNotIn("fused_moe_i8_tn_mma_kernel<true, 0, 0, 0>", candidate)
        self.assertIn("fused_moe_i8_tn_mma_kernel<true, 32768, 7168, 2048>", candidate)

    def test_single_abi_sync_bsm_and_macro_cleanup(self):
        candidate = text()
        self.assertEqual(candidate.count('extern "C" void run_kernel('), 1)
        self.assertEqual(candidate.count("struct KernelConfig"), 1)
        self.assertEqual(candidate.count("__builtin_mxc_ldg_b128_bsm"), 4)
        core = candidate.split(CORE_BEGIN, 1)[1].split(CORE_END, 1)[0]
        self.assertGreaterEqual(core.count("false, false"), 4)
        self.assertNotIn("false, true);", core)
        self.assertNotIn("is_async=true", candidate)
        self.assertNotIn("mcMemcpy", candidate)
        self.assertNotIn("mcStream_t", candidate)
        for macro in (
            "LDG_BSM_B_TILE_STAGE_I",
            "LDG_BSM_A_TILE_STAGE_I",
            "MMA_STAGE_MNKx2",
            "LDS_OFS",
            "LDS",
            "ARRIVE_GVM_BSM_BARRIER",
            "CVT_F32_TO_BF16",
            "arrive_bsmcnt",
            "arrive_gvmcnt",
        ):
            self.assertEqual(len(re.findall(rf"^#define {macro}(?:\(|\b)", candidate, re.M)), 1)
            self.assertEqual(len(re.findall(rf"^#undef {macro}$", candidate, re.M)), 1)

    def test_tested_dummy_token_source_and_runtime_launch_geometry(self):
        candidate = text()
        wrapper = candidate.split(CORE_END, 1)[1]
        self.assertIn("reinterpret_cast<int*>(const_cast<int8_t*>(a))", wrapper)
        self.assertIn("cfg.em, 8, true", wrapper)
        self.assertIn("const dim3 grid(1, grid_y, grid_m);", wrapper)
        self.assertIn("direct_moe_kernel_m4<<<grid, block>>>(args);", wrapper)
        self.assertNotIn("pragma max_work_group", candidate)
        self.assertNotIn("kFixedEm", wrapper)


if __name__ == "__main__":
    unittest.main()
