import csv
import json
import tempfile
import unittest
from dataclasses import replace
from math import prod
from pathlib import Path
from unittest.mock import PropertyMock, patch

from .artifacts import RESULTS_ROOT
from .cache import PIMConfigCache
from .constraints import ConstraintEngine
from .model import UPMEMModel, target_model
from .performance import (
    PIMPerformanceProfile, PIMSaturationCurve, PIMSaturationSample, PIMUnitTiming,
    calibrate_performance, performance_context, profile_single_unit,
)
from .search import SearchBudget, tune
from .space import (
    _saturation_points, build_config_space, heuristic_seeds, legal_partitions, make_config,
)
from .targets import PIMGemmProfileSpec, target_spec
from .types import PIMBenchmarkResult, PIMProblem, pim_hierarchy


def unit_count(config):
    return config.p_m * config.p_n * config.p_k


class SaturationSelectionTest(unittest.TestCase):
    def setUp(self):
        self.problem = PIMProblem.elementwise(4096)
        self.target = target_spec("upmem", num_units=64)
        self.space = build_config_space(
            self.problem, self.target, block_values=(16,), group_values=(1,), block_group_top_k=1)
        self.candidates = [make_config((16, 1, 1), (1, 1, 1), (256 // units, 1, 1), (units, 1, 1))
                           for units in (1, 2, 4, 8, 16, 32, 64)]
        self.performance = PIMPerformanceProfile("test-kernel", (PIMSaturationCurve(
            performance_context(self.problem, self.target), (16, 1, 1), (1, 1, 1),
            tuple(PIMSaturationSample((work, 1, 1), PIMUnitTiming(1.0 * work / 1000))
                  for work in (256, 512, 1024))),))

    def seeds(self, *, space=None, engine=None):
        return heuristic_seeds(
            self.problem, self.target, self.space if space is None else space,
            performance=self.performance, kernel_version="test-kernel", engine=engine)

    def test_center_neighbors_and_maximum(self):
        seeds = _saturation_points(self.problem, self.candidates, 256)
        self.assertEqual([unit_count(config) for config in seeds], [16, 32, 8, 64])

    def test_capacity_endpoint_is_not_mandatory(self):
        seeds = _saturation_points(self.problem, self.candidates, 1)
        self.assertEqual([unit_count(config) for config in seeds], [64, 32])

    def test_all_unsaturated_uses_most_work_and_keeps_parallel_comparison(self):
        seeds = _saturation_points(self.problem, self.candidates, 8192)
        self.assertEqual([unit_count(config) for config in seeds], [1, 2, 64])

    def test_only_one_unit_is_legal(self):
        seeds = self.seeds(space=replace(self.space, instance_units=1))
        self.assertEqual(len(seeds), 1)
        self.assertEqual(unit_count(seeds[0]), 1)

    def test_boundary_padding_is_not_counted_as_useful_work(self):
        problem = PIMProblem.elementwise(100)
        candidates = [make_config((16, 1, 1), (1, 1, 1), (tile, 1, 1), (units, 1, 1))
                      for tile, units in ((7, 1), (4, 2), (2, 4))]
        self.assertEqual(unit_count(_saturation_points(problem, candidates, 60)[0]), 1)

    def test_full_heuristic_uses_saturation_order(self):
        self.assertEqual([unit_count(config) for config in self.seeds()], [16, 32, 8, 64])

    def test_custom_constraint_is_applied_before_center_selection(self):
        engine = ConstraintEngine(
            self.problem, self.target,
            constraint=lambda problem, config: "reserved units" if unit_count(config) > 8 else None)
        seeds = self.seeds(engine=engine)
        self.assertEqual(unit_count(seeds[0]), 8)
        self.assertTrue(all(engine.valid(config) for config in seeds))

    def test_one_point_budget_keeps_center(self):
        self.assertEqual([unit_count(config) for config in self.seeds(
            space=replace(self.space, max_ladder_steps=1))], [16])

    def test_partition_override_cannot_exceed_hardware(self):
        space = replace(self.space, instance_units=256, partition_values=(1, 64, 128, 256))
        self.assertTrue(all(prod(p) <= 64 for p in legal_partitions(self.problem, self.target, space)))

    def test_missing_profile_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "require measured"):
            heuristic_seeds(self.problem, self.target, self.space,
                            performance=None, kernel_version="test-kernel")

    def test_external_one_unit_candidate_remains_available(self):
        result = tune(
            self.problem, self.target,
            lambda problem, target, config: PIMBenchmarkResult(100 / unit_count(config)),
            performance=self.performance, kernel_version="test-kernel", space=self.space,
            initial=(self.candidates[0],), initial_random=0,
            budget=SearchBudget(max_benchmarks=10, max_generations=0))
        self.assertEqual([unit_count(record.config) for record in result.records], [1, 16, 32, 8, 64])
        self.assertEqual(unit_count(result.best.config), 64)

    def test_cache_hit_does_not_regenerate_seeds(self):
        (RESULTS_ROOT / "tests").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RESULTS_ROOT / "tests") as directory:
            cache = PIMConfigCache(str(Path(directory) / "cache.json"))
            tune(self.problem, self.target, lambda *args: PIMBenchmarkResult(1),
                 performance=self.performance, kernel_version="test-kernel", space=self.space,
                 cache=cache, initial=(self.candidates[0],), initial_random=0,
                 budget=SearchBudget(max_benchmarks=1, max_generations=0))

            def unexpected_measurement(*args):
                self.fail("a complete cache hit must not measure a seed")

            result = tune(self.problem, self.target, unexpected_measurement,
                          performance=None, kernel_version="test-kernel", cache=cache,
                          profile_fn=unexpected_measurement)
            self.assertTrue(result.from_cache)
            self.assertEqual(unit_count(result.best.config), 1)

    def test_missing_data_profiles_once_before_benchmarking(self):
        events = []

        def profile(problem, target, space):
            events.append("profile")
            return self.performance

        def benchmark(*args):
            events.append("benchmark")
            return PIMBenchmarkResult(1)

        tune(self.problem, self.target, benchmark, performance=None, kernel_version="test-kernel",
             profile_fn=profile, space=self.space, initial_random=0,
             budget=SearchBudget(max_benchmarks=1, max_generations=0))
        self.assertEqual(events, ["profile", "benchmark"])

    def test_supplied_data_does_not_profile_again(self):
        def unexpected_profile(*args):
            self.fail("supplied performance must skip profiling")

        tune(self.problem, self.target, lambda *args: PIMBenchmarkResult(1),
             performance=self.performance, kernel_version="test-kernel", profile_fn=unexpected_profile,
             space=self.space, initial_random=0,
             budget=SearchBudget(max_benchmarks=1, max_generations=0))


class PerformanceProfileTest(unittest.TestCase):
    def setUp(self):
        self.problem = PIMProblem.elementwise(4096)
        self.target = target_spec("upmem", num_units=64)
        self.space = build_config_space(
            self.problem, self.target, block_values=(16,), group_values=(1,), block_group_top_k=1)

    def curve(self, throughputs):
        return PIMSaturationCurve(
            performance_context(self.problem, self.target), (16, 1, 1), (1, 1, 1),
            tuple(PIMSaturationSample((16 * 2**index, 1, 1), PIMUnitTiming(16 * 2**index / throughput))
                  for index, throughput in enumerate(throughputs)))

    def test_95_percent_requires_sustained_throughput(self):
        self.assertEqual(self.curve((40, 96, 90, 96, 100, 100)).saturation_work, 128)

    def test_unstable_tail_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not stabilized"):
            self.curve((10, 30, 60, 100))

    def test_two_measurements_do_not_establish_saturation(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            self.curve((100, 100))

    def test_declining_curve_is_not_a_plateau(self):
        with self.assertRaisesRegex(ValueError, "declines"):
            self.curve((110, 100, 100, 100))

    def test_timing_does_not_replace_missing_launch_time(self):
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PIMUnitTiming(invalid, device_ms=1)
        self.assertIsNone(PIMUnitTiming(1).device_ms)

    def test_round_trip_and_kernel_version_check(self):
        profile = PIMPerformanceProfile("test-kernel", (self.curve((50, 100, 100, 100)),))
        (RESULTS_ROOT / "tests").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RESULTS_ROOT / "tests") as directory:
            path = Path(directory) / "performance.json"
            profile.save(path)
            self.assertEqual(PIMPerformanceProfile.load(path, kernel_version="test-kernel"), profile)
            with self.assertRaisesRegex(ValueError, "kernel mismatch"):
                PIMPerformanceProfile.load(path, kernel_version="another-kernel")
            path.write_text(json.dumps({"schema": "old"}))
            with self.assertRaisesRegex(ValueError, "unsupported performance schema"):
                PIMPerformanceProfile.load(path, kernel_version="test-kernel")

    def test_unit_count_is_shareable_but_tasklets_and_dtype_are_not(self):
        profile = PIMPerformanceProfile("test-kernel", (self.curve((100, 100, 100)),))
        single_target = replace(self.target, common_profile=replace(self.target.common_profile, num_units=1))
        self.assertEqual(profile.saturation_work(self.problem, single_target, (16, 1, 1), (1, 1, 1)), 16)
        changed_targets = [target_spec("upmem", num_units=64, nr_tasklets=8),
                           target_spec("aimx", num_units=2)]
        for target in changed_targets:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "missing single-unit"):
                profile.saturation_work(self.problem, target, (16, 1, 1), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "missing single-unit"):
            profile.saturation_work(PIMProblem.elementwise(4096, input_dtypes=("int32",)),
                                    self.target, (16, 1, 1), (1, 1, 1))

    def test_calibration_uses_one_unit_one_tile_and_reuses_families_across_cases(self):
        measurements = []

        def measure(problem, target, config):
            self.assertEqual(target.common_profile.num_units, 1)
            self.assertEqual(unit_count(config), 1)
            self.assertEqual(pim_hierarchy(problem, config).execution_tiles, (1, 1, 1))
            self.assertTrue(ConstraintEngine(problem, target).valid(config))
            measurements.append(config)
            return PIMUnitTiming(max(prod(problem.axis_extents), 64) / 1000)

        profile = calibrate_performance(
            [(self.problem, self.space), (replace(self.problem, M=2048), self.space)],
            self.target, measure, kernel_version="test-kernel", sweep_extents=(4096, 1, 1))
        self.assertEqual(len(profile.curves), 1)
        self.assertEqual(profile.curves[0].saturation_work, 64)
        self.assertEqual(len(profile.curves[0].samples), len(measurements))

    def test_calibration_propagates_measurement_errors(self):
        def measure(*args):
            raise RuntimeError("device measurement failed")

        with self.assertRaisesRegex(RuntimeError, "device measurement failed"):
            calibrate_performance([(self.problem, self.space)], self.target, measure,
                                  kernel_version="test-kernel", sweep_extents=(4096, 1, 1))


class GemmCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.target = target_spec(
            "upmem", num_units=64, local_capacity_limit_bytes=4096,
            bank_capacity_limit_bytes=1024 * 1024)
        self.problem = PIMProblem(1024, 1024, 1024)
        self.space = build_config_space(self.problem, self.target)
        self.measurements = []
        self.profile_spec = patch.object(UPMEMModel, "gemm_profile_spec", new_callable=PropertyMock)
        self.profile_spec.start().return_value = PIMGemmProfileSpec(
            (16, 16, 16), "int8", "int32", (1, 1, 1))
        self.addCleanup(self.profile_spec.stop)

    def measure(self, problem, target, config):
        self.assertEqual(target.common_profile.num_units, 1)
        self.assertEqual(unit_count(config), 1)
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
            [(self.problem, self.space)], self.target, self.measure, kernel_version="test-gemm")

    def test_tile_sweep_uses_best_measured_group_instead_of_largest(self):
        profile = self.calibrate()
        group = next(curve for curve in profile.group_curves if curve.blocks == (16, 16, 16))
        tile = next(curve for curve in profile.curves if curve.blocks == (16, 16, 16))
        self.assertEqual([point.groups[2] for point in group.samples], [1, 2, 4])
        self.assertIn("local memory overflow", group.stop_reason)
        self.assertEqual(tile.groups, (1, 1, 2))
        self.assertTrue(all(config.group_k == 2 for config in self.measurements
                            if config.block_m == 16 and config.block_n == 16 and config.tile_k > 1))
        self.assertEqual(tile.stop_reason, "saturation")
        self.assertEqual([sample.extent[2] for sample in tile.samples], [32, 64, 128, 256, 512, 1024])

    def test_small_problem_selects_a_legal_measured_group(self):
        profile = self.calibrate()
        problem = replace(self.problem, K=16)
        families = profile.seed_families(problem, self.target, 6)
        self.assertTrue(families)
        self.assertTrue(all(groups[2] == 1 for _, groups, _ in families))
        self.assertTrue(all(ConstraintEngine(problem, self.target).valid(
            make_config(blocks, groups, (1, 1, 1), (1, 1, 1))) for blocks, groups, _ in families))

    def test_group_records_and_tile_curves_round_trip(self):
        profile = self.calibrate()
        (RESULTS_ROOT / "tests").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RESULTS_ROOT / "tests") as directory:
            path = Path(directory) / "performance.json"
            profile.save(path)
            self.assertEqual(PIMPerformanceProfile.load(path, kernel_version="test-gemm"), profile)

    def test_raw_csv_survives_missing_tile_saturation(self):
        (RESULTS_ROOT / "tests").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RESULTS_ROOT / "tests") as directory:
            with self.assertRaisesRegex(ValueError, "has not stabilized"):
                profile_single_unit(
                    [(self.problem, self.space)], self.target,
                    lambda problem, target, config: PIMUnitTiming(1),
                    kernel_version="test-gemm", output_dir=directory)
            with open(Path(directory) / "single_unit.csv") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["stage"] for row in rows}, {"group", "tile"})
            self.assertTrue(all(row["num_units"] == "1" for row in rows))
            self.assertFalse((Path(directory) / "performance.json").exists())

    def test_gemm_measurement_errors_are_not_swallowed(self):
        def fail(*args):
            raise RuntimeError("GEMM device measurement failed")

        with self.assertRaisesRegex(RuntimeError, "GEMM device measurement failed"):
            calibrate_performance([(self.problem, self.space)], self.target, fail,
                                  kernel_version="test-gemm")


class GemmTargetSpecTest(unittest.TestCase):
    def test_tuning_uses_a_measured_block_outside_the_old_candidate_list(self):
        target = target_spec("aimx", num_units=2, batch=8)
        problem = PIMProblem(4096, 4096, 4096, input_dtype="float16", output_dtype="float32")
        profile = PIMPerformanceProfile("test-gemm", (PIMSaturationCurve(
            performance_context(problem, target), (8, 512, 32), (1, 1, 1), tuple(
                PIMSaturationSample((8, 512, 32 * scale), PIMUnitTiming(8 * 512 * 32 * scale / 1000))
                for scale in (1, 2, 4))),))
        result = tune(
            problem, target, lambda *args: PIMBenchmarkResult(1),
            performance=profile, kernel_version="test-gemm", initial_random=0,
            budget=SearchBudget(max_benchmarks=1, max_generations=0))
        self.assertEqual(result.best.config.block_n, 512)

    def test_upmem_output_rows_cover_all_tasklets(self):
        for tasklets in (8, 11, 16):
            with self.subTest(tasklets=tasklets):
                spec = target_model(target_spec("upmem", nr_tasklets=tasklets)).gemm_profile_spec
                for blocks in spec.block_candidates:
                    self.assertEqual(blocks[0] * spec.group_shape[0] % tasklets, 0)
                    self.assertTrue(all(value & (value - 1) == 0 for value in blocks))

    def test_aimx_candidates_follow_batch_compute_shape(self):
        for batch in (1, 2, 4, 8):
            with self.subTest(batch=batch):
                spec = target_model(target_spec("aimx", num_units=1, batch=batch)).gemm_profile_spec
                self.assertTrue(all(blocks[0] % batch == 0 for blocks in spec.block_candidates))
                self.assertTrue(all(blocks[1] % (4096 // batch) == 0 for blocks in spec.block_candidates))
                self.assertTrue(all(blocks[2] == 256 // batch for blocks in spec.block_candidates))


if __name__ == "__main__":
    unittest.main()
