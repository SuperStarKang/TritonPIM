import unittest
from math import prod

from .constraints import ConstraintEngine
from .space import (
    build_config_space, heuristic_seeds, legal_partitions, make_config,
    rank_block_groups, tile_ladder,
)
from .targets import target_spec
from .types import PIMConfig, PIMProblem, pim_hierarchy


class AiMXFrontierTest(unittest.TestCase):
    def setUp(self):
        self.problem = PIMProblem(
            4096, 4096, 4096, input_dtype="float16", output_dtype="float32")
        self.target = target_spec("aimx", num_units=4, batch=8)
        self.space = build_config_space(self.problem, self.target)

    def test_block_group_saturates_native_width(self):
        blocks, groups = rank_block_groups(
            self.problem, self.target, self.space)[0]

        # extent (1024,4096,4096): no compute padding, grid (4,1,1) fills 4 units
        self.assertEqual(blocks, (128, 128, 128))
        self.assertEqual(groups, (8, 32, 32))

    def test_tile_ladder_runs_from_one_device_to_all_devices(self):
        ladder = tile_ladder(
            self.problem, self.target, (128, 128, 128), (8, 32, 32),
            self.space)

        self.assertEqual(ladder, [(4, 1, 1), (2, 1, 1), (1, 1, 1)])

    def test_frontier_seeds_most_parallel_first(self):
        seeds = heuristic_seeds(self.problem, self.target, self.space)

        self.assertEqual(seeds[:3], [
            make_config((128, 128, 128), (8, 32, 32), (1, 1, 1), (4, 1, 1)),
            make_config((128, 128, 128), (8, 32, 32), (2, 1, 1), (2, 1, 1)),
            make_config((128, 128, 128), (8, 32, 32), (4, 1, 1), (1, 1, 1)),
        ])

    def test_unlowerable_mappings_are_never_generated(self):
        partitions = legal_partitions(self.problem, self.target, self.space)
        seeds = heuristic_seeds(self.problem, self.target, self.space)
        engine = ConstraintEngine(self.problem, self.target)

        self.assertTrue(all(p[2] == 1 for p in partitions))
        self.assertTrue(all(engine.valid(config) for config in seeds))


class UPMEMFrontierTest(unittest.TestCase):
    def setUp(self):
        self.problem = PIMProblem(4096, 4096, 4096)
        self.target = target_spec("upmem", num_units=2048)
        self.space = build_config_space(self.problem, self.target)

    def test_block_group_fits_local_memory(self):
        blocks, groups = rank_block_groups(
            self.problem, self.target, self.space)[0]

        # extent (64,64,256): A 16 KiB + B 16 KiB + C 16 KiB = 48 KiB local
        self.assertEqual(blocks, (64, 64, 128))
        self.assertEqual(groups, (1, 1, 2))

    def test_ladder_ends_when_every_unit_can_work(self):
        ladder = tile_ladder(
            self.problem, self.target, (64, 64, 128), (1, 1, 2), self.space)
        last = make_config((64, 64, 128), (1, 1, 2), ladder[-1], (1, 1, 1))

        self.assertGreaterEqual(
            pim_hierarchy(self.problem, last).useful_execution_tiles, 2048)

    def test_first_seed_uses_every_unit(self):
        seeds = heuristic_seeds(self.problem, self.target, self.space)
        engine = ConstraintEngine(self.problem, self.target)
        first = seeds[0]

        self.assertEqual(prod((first.p_m, first.p_n, first.p_k)), 2048)
        self.assertTrue(all(engine.valid(config) for config in seeds))

    def test_group_holds_no_empty_block(self):
        problem = PIMProblem(1024, 4096, 4096)
        engine = ConstraintEngine(problem, self.target)
        # m: 1024 / 16 = Block 64개. Group extent 64는 local 24 KiB로 들어간다
        fits = PIMConfig(16, 64, 64, 4, 1, 1, 1, 1, 1, 1, 1, 1)
        empty = PIMConfig(16, 64, 64, 128, 1, 1, 1, 1, 1, 1, 1, 1)

        self.assertIsNone(engine.check(fits))
        self.assertEqual(engine.check(empty), "m group(128)이 Block 수(64)보다 크다")

    def test_block_group_rows_split_over_tasklets(self):
        config = PIMConfig(8, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1)

        self.assertTrue(ConstraintEngine(self.problem, self.target).check(
            config).startswith("Block Group m extent(8)"))


if __name__ == "__main__":
    unittest.main()
