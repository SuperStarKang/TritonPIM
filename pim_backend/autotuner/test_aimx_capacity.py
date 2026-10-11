import unittest

from .constraints import ConstraintEngine, aimx_runtime_allocations
from .features import extract_features
from .targets import target_spec
from .types import PIMConfig, PIMProblem


class AiMXCapacityTest(unittest.TestCase):
    def test_equal_logical_shapes_have_distinct_storage(self):
        problem = PIMProblem(16, 16, 16, input_dtype="bfloat16", output_dtype="bfloat16")
        target = target_spec("aimx", num_units=1, batch=8)
        config = PIMConfig(8, 8, 8, 1, 1, 1, 2, 2, 2, 1, 1, 1)

        self.assertEqual(aimx_runtime_allocations(problem, target, config), {
            "PMM": 196608,
            "A": 262144,
            "Arel": 131072,
            "B": 8388608,
            "C": 262144,
            "descriptor chain": 43008,
        })
        self.assertIsNone(ConstraintEngine(problem, target).check(config))

    def test_group_storage_padding_does_not_add_mac_steps(self):
        problem = PIMProblem(8, 512, 256, input_dtype="float16", output_dtype="float32")
        target = target_spec("aimx", num_units=1, batch=8)
        config = PIMConfig(8, 512, 32, 1, 1, 1, 1, 1, 8, 1, 1, 1)
        allocations = aimx_runtime_allocations(problem, target, config)

        self.assertEqual(allocations["B"], 16777216)
        self.assertEqual(allocations["Arel"], 524288)
        self.assertEqual(allocations["descriptor chain"], 16896)

    def test_all_k_group_relayouts_must_fit_one_allocation(self):
        problem = PIMProblem(8192, 512, 16384, input_dtype="float16", output_dtype="float32")
        target = target_spec("aimx", num_units=1, batch=8)
        config = PIMConfig(128, 128, 64, 64, 4, 64, 1, 1, 4, 1, 1, 1)

        self.assertEqual(aimx_runtime_allocations(problem, target, config)["Arel"],
                         4 * 1024**3)
        self.assertTrue(ConstraintEngine(problem, target).check(config).startswith(
            "AiMX Arel allocation overflow:"))

    def test_groups_larger_than_native_width_keep_their_width(self):
        problem = PIMProblem(16, 2048, 512, input_dtype="float16", output_dtype="float32")
        target = target_spec("aimx", num_units=1, batch=8)
        config = PIMConfig(16, 1024, 512, 1, 1, 1, 1, 2, 1, 1, 1, 1)
        allocations = aimx_runtime_allocations(problem, target, config)
        features = extract_features(problem, target, config)

        self.assertEqual(allocations["B"], 8388608)
        self.assertEqual(allocations["C"], 262144)
        self.assertEqual(allocations["descriptor chain"], 143360)
        self.assertEqual(features.h2d_bytes, 8519680)
        self.assertEqual(features.d2h_bytes, 262144)
        self.assertEqual(features.padding_ratio, 0)

    def test_lower_bound_already_exceeding_capacity_is_rejected(self):
        problem = PIMProblem(16, 16, 16, input_dtype="bfloat16", output_dtype="bfloat16")
        target = target_spec("aimx", num_units=1, batch=8, bank_capacity_limit_bytes=1)
        config = PIMConfig(8, 8, 8, 1, 1, 1, 2, 2, 2, 1, 1, 1)

        self.assertTrue(ConstraintEngine(problem, target).check(config).startswith(
            "AiMX bank footprint overflow:"))


if __name__ == "__main__":
    unittest.main()
