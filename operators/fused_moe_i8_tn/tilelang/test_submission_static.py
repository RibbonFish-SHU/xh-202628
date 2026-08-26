import ast
import random
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).with_name("submission.py")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def function(name):
    return next(
        node
        for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def stable_rank_map(experts):
    rank_map = [None] * len(experts)
    for logical, expert in enumerate(experts):
        rank = sum(
            candidate_expert < expert
            or (candidate_expert == expert and candidate < logical)
            for candidate, candidate_expert in enumerate(experts)
        )
        rank_map[rank] = logical
    return rank_map


class TileLangSubmissionStaticTest(unittest.TestCase):
    def test_public_abi_and_cache_key(self):
        run_kernel = function("run_kernel")
        self.assertEqual(
            [arg.arg for arg in run_kernel.args.args],
            [
                "a",
                "b_col_major",
                "scale_a",
                "scale_b",
                "moe_weights",
                "token_ids",
                "expert_ids",
                "topk",
                "out",
            ],
        )
        used_names = {
            node.id
            for node in ast.walk(run_kernel)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertNotIn("token_ids", used_names)
        self.assertNotIn("topk", used_names)
        self.assertIn("expert_ids", used_names)
        self.assertIn("out", used_names)

        cached_kernel = function("_cached_kernel")
        tuple_values = [
            [elt.id for elt in node.elts]
            for node in ast.walk(cached_kernel)
            if isinstance(node, ast.Tuple)
            and all(isinstance(elt, ast.Name) for elt in node.elts)
        ]
        self.assertIn(["EM", "N", "K", "E"], tuple_values)
        self.assertIn(
            "fused_moe_i8_tn_prefill_sorted_kernel if EM == 32768 else fused_moe_i8_tn_kernel",
            SOURCE,
        )

    def test_bk128_geometry_and_resource_model(self):
        for name in ("fused_moe_i8_tn_kernel", "fused_moe_i8_tn_prefill_sorted_kernel"):
            kernel = function(name)
            defaults = [ast.literal_eval(node) for node in kernel.args.defaults]
            self.assertEqual(defaults, [128, 128, 1, 256])

        shapes = [
            (4096, 4096, 7168, 56),
            (32768, 4096, 7168, 56),
            (4096, 7168, 2048, 16),
            (32768, 7168, 2048, 16),
        ]
        for em, n, k, expected_k_steps in shapes:
            self.assertEqual(em % 128, 0)
            self.assertEqual(n % 128, 0)
            self.assertEqual(k % 128, 0)
            self.assertEqual(k // 128, expected_k_steps)

        self.assertEqual(2 * 128 * 128, 32 * 1024)
        self.assertEqual(128 * 128 // 256, 64)
        self.assertEqual(SOURCE.count("T.Pipelined(K // block_K, num_stages=num_stages)"), 2)
        self.assertEqual(SOURCE.count("T.gemm(A_shared, B_shared, C_local, transpose_B=True)"), 2)

    def test_stable_rank_is_a_permutation(self):
        rng = random.Random(20260826)
        distributions = [
            [0] * 256,
            list(range(256)),
            list(reversed(range(256))),
            [rng.randrange(256) for _ in range(256)],
            [rng.randrange(8) for _ in range(256)],
        ]
        for experts in distributions:
            rank_map = stable_rank_map(experts)
            expected = sorted(range(256), key=lambda logical: (experts[logical], logical))
            self.assertEqual(rank_map, expected)
            self.assertEqual(sorted(rank_map), list(range(256)))

    def test_prefill_grid_exactly_covers_all_tiles(self):
        rank_map = stable_rank_map([(logical * 73) % 19 for logical in range(256)])
        physical_tiles = [bx + 8 * bz for bz in range(32) for bx in range(8)]
        self.assertEqual(sorted(physical_tiles), list(range(256)))

        for n in (4096, 7168):
            work = {
                (rank_map[bx + 8 * bz], bn)
                for bz in range(32)
                for bx in range(8)
                for bn in range(n // 128)
            }
            self.assertEqual(len(work), 256 * (n // 128))
            self.assertEqual(
                work,
                {(logical, bn) for logical in range(256) for bn in range(n // 128)},
            )


if __name__ == "__main__":
    unittest.main()
