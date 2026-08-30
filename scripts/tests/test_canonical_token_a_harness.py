"""Workflow proof for candidates that reuse a canonical routed A row per token."""

from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = (
    ROOT
    / "operators"
    / "fused_moe_i8_tn"
    / "cuda_maca"
    / "test_fused_moe_i8_tn.cu"
).read_text(encoding="utf-8")


class CanonicalTokenAHarnessTests(unittest.TestCase):
    def test_decode_token_ids_remain_a_complete_permutation(self) -> None:
        em = 4096
        token_ids = [(row * 8191) & (em - 1) for row in range(em)]
        self.assertEqual(sorted(token_ids), list(range(em)))

    def test_prefill_routes_align_token_rows_across_eight_experts(self) -> None:
        tile_count = 256
        experts = [(tile * tile + 3 * tile) % 16 for tile in range(tile_count)]
        active_experts = sorted(set(experts))
        self.assertEqual(active_experts, list(range(0, 16, 2)))

        expert_slots = {expert: slot for slot, expert in enumerate(active_experts)}
        expert_tile_ranks = {expert: 0 for expert in active_experts}
        token_ids = [0] * 32768
        tiles_by_expert: dict[int, list[int]] = {expert: [] for expert in active_experts}
        for tile, expert in enumerate(experts):
            expert_tile_rank = expert_tile_ranks[expert]
            expert_tile_ranks[expert] += 1
            tiles_by_expert[expert].append(tile)
            for local_row in range(128):
                token = expert_tile_rank * 128 + local_row
                token_ids[tile * 128 + local_row] = token * 8 + expert_slots[expert]

        self.assertEqual(sorted(token_ids), list(range(32768)))
        self.assertEqual(set(expert_tile_ranks.values()), {32})
        for expert_tile_rank in range(32):
            base_tile = tiles_by_expert[active_experts[0]][expert_tile_rank]
            base_tokens = [
                token_ids[base_tile * 128 + local_row] >> 3
                for local_row in range(128)
            ]
            for expert in active_experts[1:]:
                tile = tiles_by_expert[expert][expert_tile_rank]
                self.assertEqual(
                    [
                        token_ids[tile * 128 + local_row] >> 3
                        for local_row in range(128)
                    ],
                    base_tokens,
                )

    def test_candidate_capability_is_compile_time_isolated(self) -> None:
        self.assertIn("#if defined(XH_FUSED_MOE_CANONICAL_TOKEN_A)", HARNESS)
        self.assertIn("out, config, token_ids);", HARNESS)
        self.assertIn("(void)token_ids;", HARNESS)
        self.assertIn("out, config);", HARNESS)

    def test_public_benchmark_supplies_read_only_token_ids(self) -> None:
        self.assertIn("DeviceBuffer<int32_t> dev_token_ids(config.em);", HARNESS)
        self.assertIn("dev_token_ids.copy_from(token_ids);", HARNESS)
        self.assertIn("fill_routed_a(dev_a.get(), dev_token_ids.get(), config.em, config.k);", HARNESS)
        self.assertIn("if (dev_token_ids.copy_to_host() != token_ids)", HARNESS)
        self.assertIn("token_ids[tile * 128 + local_row] = token * 8 + expert_slot;", HARNESS)


if __name__ == "__main__":
    unittest.main()
