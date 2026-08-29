import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operators/fused_moe_i8_tn/cuda_maca/submission.cu"


def function_body(source: str, signature: str) -> str:
    start = source.rindex(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


class Case4SortedSyncM4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_only_exact_case4_routes_to_sorted_m4(self) -> None:
        predicate = function_body(
            self.source,
            "static inline bool use_prefill_down_module_full_expert_sort(",
        )
        self.assertIn(
            "config.em == 32768 && config.n == 7168 && config.k == 2048",
            predicate,
        )

        launch = function_body(self.source, "static inline void launch(")
        decode = launch.index("if (config.em == 4096")
        case4 = launch.index("if (use_prefill_down_module_full_expert_sort(config))")
        generic = launch.index("const dim3 block(kMmaThreads)")
        self.assertLess(decode, case4)
        self.assertLess(case4, generic)
        case4_dispatch = launch[case4:generic]
        self.assertIn("build_case2_full_expert_sort_map_kernel", case4_dispatch)
        self.assertIn("sync_m4::launch_sorted_prefill(", case4_dispatch)
        self.assertIn("return;", case4_dispatch)
        self.assertEqual(launch.count("sync_m4::launch_sorted_prefill("), 1)

    def test_kernel_maps_all_row_and_expert_consumers(self) -> None:
        kernel = function_body(
            self.source,
            "__global__ void direct_moe_kernel_m4(Arguments args)",
        )
        self.assertIn("template <bool kUseFullExpertSort>", self.source)
        self.assertIn(
            "int physical_bidx = blockIdx.x + blockIdx.z * gridDim.x;",
            kernel,
        )
        self.assertRegex(
            kernel,
            re.compile(
                r"const int bidx = kUseFullExpertSort\s*"
                r"\? g_case2_full_expert_sort_map\[physical_bidx\]\s*"
                r": physical_bidx;"
            ),
        )
        self.assertIn("int group_idx = expert_ids_ptr[bidx];", kernel)
        self.assertIn("int prev_m    = bidx * kTileM;", kernel)
        self.assertIn("token_ids_ptr + idx_row_a + prev_m", kernel)
        self.assertIn("int token_row_m = prev_m +", kernel)
        self.assertIn("group_idx * N + bidy * kTileN + colC", kernel)
        self.assertIn("Caddr + rowC_[i * 4 + j] * N + colC", kernel)

    def test_launch_geometry_and_case_isolation(self) -> None:
        decode = function_body(self.source, "static inline void launch_decode(")
        sorted_prefill = function_body(
            self.source,
            "static inline void launch_sorted_prefill(",
        )
        self.assertIn("direct_moe_kernel_m4<false><<<grid, block>>>(args);", decode)
        self.assertIn("const dim3 grid(8, 56, 32);", sorted_prefill)
        self.assertIn(
            "direct_moe_kernel_m4<true><<<grid, block>>>(args);",
            sorted_prefill,
        )

        launch = function_body(self.source, "static inline void launch(")
        generic = launch[launch.index("const dim3 block(kMmaThreads)") :]
        self.assertIn("use_case2_output_scratch_expert_sort(config)", generic)
        self.assertIn(
            "fused_moe_i8_tn_mma_kernel<true, 0, 4096, 7168>",
            generic,
        )

    def test_stable_rank_and_grid_cover_are_bijective(self) -> None:
        expert_ids = [(index * 73 + index // 7 * 19) % 256 for index in range(256)]
        ranks = []
        for logical_tile, expert in enumerate(expert_ids):
            rank = sum(
                candidate_expert < expert
                or (candidate_expert == expert and candidate < logical_tile)
                for candidate, candidate_expert in enumerate(expert_ids)
            )
            ranks.append(rank)
        self.assertEqual(sorted(ranks), list(range(256)))
        self.assertEqual(
            sorted(x + z * 8 for z in range(32) for x in range(8)),
            list(range(256)),
        )
        self.assertEqual(56 * 128, 7168)


if __name__ == "__main__":
    unittest.main()
