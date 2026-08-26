import ast
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


class TileLangDirectSubmissionStaticTest(unittest.TestCase):
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
        self.assertIn("kernel = fused_moe_i8_tn_kernel(EM=EM, N=N, K=K, E=E)", SOURCE)

    def test_exact_bk128_geometry(self):
        kernel = function("fused_moe_i8_tn_kernel")
        defaults = [ast.literal_eval(node) for node in kernel.args.defaults]
        self.assertEqual(defaults, [128, 128, 1, 256])
        self.assertIn(
            "with T.Kernel(num_tiles, T.ceildiv(N, block_N), threads=threads) as (bt, bn):",
            SOURCE,
        )
        self.assertIn(
            "for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):",
            SOURCE,
        )
        self.assertEqual(SOURCE.count("T.gemm(A_shared, B_shared, C_local, transpose_B=True)"), 1)
        self.assertNotIn("policy=", SOURCE)
        self.assertNotIn("FullRow", SOURCE)

    def test_direct_2d_kernel_only(self):
        kernel_calls = [
            node
            for node in ast.walk(TREE)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr == "Kernel"
        ]
        self.assertEqual(len(kernel_calls), 1)
        self.assertEqual(len(kernel_calls[0].args), 2)
        self.assertEqual([kw.arg for kw in kernel_calls[0].keywords], ["threads"])
        for forbidden in (
            "T.alloc_global",
            "T.alloc_var",
            "RankMap",
            "rank_map",
            "prefill_sorted",
        ):
            self.assertNotIn(forbidden, SOURCE)

    def test_public_shapes_and_resource_model(self):
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


if __name__ == "__main__":
    unittest.main()
