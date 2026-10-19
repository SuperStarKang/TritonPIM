import csv
import json
import tempfile
import unittest
from dataclasses import replace
from math import prod
from pathlib import Path
from unittest.mock import PropertyMock, patch

from .artifacts import RESULTS_ROOT
from .constraints import ConstraintEngine
from .model import UPMEMModel, target_model
from .performance import PIMUnitTiming, calibrate_performance, performance_context, profile_single_unit
from .search import SearchBudget, tune
from .space import build_config_space
from .targets import PIMGemvProfileSpec, target_spec
from .types import PIMBenchmarkResult, PIMProblem, pim_hierarchy


class GemvCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.target = target_spec(
            "upmem", num_units=64, local_capacity_limit_bytes=20000,
            bank_capacity_limit_bytes=1024 * 1024)
        self.problem = PIMProblem.matvec(1024, 4096, input_dtype="int32", output_dtype="int32")
        self.space = build_config_space(self.problem, self.target)
        self.measurements = []

    def measure(self, problem, target, config):
        self.assertEqual(target.common_profile.num_units, 1)
        self.assertEqual((config.p_m, config.p_n, config.p_k), (1, 1, 1))
        self.assertEqual((config.block_n, config.group_n, config.tile_n), (1, 1, 1))
        self.assertEqual(problem.M, config.block_m)
        self.assertEqual(problem.N, 1)
        self.assertEqual((config.group_m, config.tile_m), (1, 1))
        self.assertEqual(pim_hierarchy(problem, config).execution_tiles, (1, 1, 1))
        self.assertTrue(ConstraintEngine(problem, target).valid(config))
        self.measurements.append(config)
        if config.tile_k == 1:
            rate = {1: 100, 2: 200, 4: 150}[config.group_k]
        else:
            rate = 250 if config.tile_k == 2 else 300
        return PIMUnitTiming(prod(problem.axis_extents) / rate)

    def calibrate(self):
        return calibrate_performance(
            [(self.problem, self.space)], self.target, self.measure, kernel_version="test-gemv")

    def test_tile_sweep_keeps_the_fastest_group_and_fixed_output_rows(self):
        performance = self.calibrate()
        group = next(curve for curve in performance.group_curves if curve.blocks == (16, 1, 64))
        tile = next(curve for curve in performance.curves if curve.blocks == (16, 1, 64))

        self.assertEqual([point.groups[2] for point in group.samples], [1, 2, 4])
        self.assertIn("local memory overflow", group.stop_reason)
        self.assertEqual(tile.groups, (1, 1, 2))
        self.assertEqual([sample.extent for sample in tile.samples], [
            (16, 1, length) for length in (128, 256, 512, 1024, 2048, 4096)])
        self.assertEqual(tile.stop_reason, "saturation")
        self.assertTrue(all(config.group_k == 2 for config in self.measurements
                            if (config.block_m, config.block_k) == (16, 64) and config.tile_k > 1))

    def test_profiling_is_not_limited_by_the_requested_gemv_size(self):
        problem = replace(self.problem, M=16, K=64)
        performance = calibrate_performance(
            [(problem, build_config_space(problem, self.target))], self.target,
            self.measure, kernel_version="test-gemv")

        self.assertTrue(any(sample.extent[2] > problem.K
                            for curve in performance.curves for sample in curve.samples))
        self.assertTrue(all(curve.stop_reason == "saturation" for curve in performance.curves))

    def test_tuning_profiles_before_using_multiple_dpus_and_keeps_columns_at_one(self):
        events = []

        def profile(problem, target, space):
            events.append("profile")
            return self.calibrate()

        def benchmark(problem, target, config):
            events.append("benchmark")
            self.assertEqual((config.block_n, config.group_n, config.tile_n, config.p_n),
                             (1, 1, 1, 1))
            self.assertTrue(ConstraintEngine(problem, target).valid(config))
            return PIMBenchmarkResult(1)

        result = tune(
            self.problem, self.target, benchmark, performance=None, kernel_version="test-gemv",
            profile_fn=profile, space=self.space, initial_random=0,
            budget=SearchBudget(max_benchmarks=1, max_generations=0))

        self.assertEqual(events, ["profile", "benchmark"])
        self.assertEqual(result.best.config.p_m * result.best.config.p_k, 64)

    def test_profile_policy_changes_do_not_invalidate_gemm_measurements(self):
        gemm = PIMProblem(1024, 1024, 1024)
        gemm_context = performance_context(gemm, self.target)
        gemv_context = performance_context(self.problem, self.target)
        self.assertNotIn("gemv_profile", json.loads(gemm_context))
        self.assertIn("gemv_profile", json.loads(gemv_context))

        with patch.object(UPMEMModel, "gemv_profile_spec", new_callable=PropertyMock) as spec:
            spec.return_value = PIMGemvProfileSpec(
                (16, 1, 32), "int32", "int32", (1, 1, 1))
            self.assertEqual(performance_context(gemm, self.target), gemm_context)
            self.assertNotEqual(performance_context(self.problem, self.target), gemv_context)

    def test_missing_plateau_preserves_raw_samples_and_raises(self):
        (RESULTS_ROOT / "tests").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RESULTS_ROOT / "tests") as directory:
            with self.assertRaisesRegex(ValueError, "has not stabilized"):
                profile_single_unit(
                    [(self.problem, self.space)], self.target,
                    lambda problem, target, config: PIMUnitTiming(1),
                    kernel_version="test-gemv", output_dir=directory)
            with open(Path(directory) / "single_unit.csv") as handle:
                samples = list(csv.DictReader(handle))
            self.assertEqual({sample["stage"] for sample in samples}, {"group", "tile"})
            self.assertTrue(all(sample["num_units"] == "1" for sample in samples))
            self.assertFalse((Path(directory) / "performance.json").exists())


class GemvTargetSpecTest(unittest.TestCase):
    def test_blocks_cover_tasklets_without_using_a_column_axis(self):
        for tasklets in (8, 11, 16):
            with self.subTest(tasklets=tasklets):
                spec = target_model(target_spec("upmem", nr_tasklets=tasklets)).gemv_profile_spec
                self.assertEqual((spec.input_dtype, spec.output_dtype), ("int32", "int32"))
                for blocks in spec.block_candidates:
                    self.assertEqual((blocks[1], spec.group_shape[1]), (1, 1))
                    self.assertEqual(blocks[0] * spec.group_shape[0] % tasklets, 0)
                    self.assertTrue(all(value & (value - 1) == 0 for value in blocks))


if __name__ == "__main__":
    unittest.main()
