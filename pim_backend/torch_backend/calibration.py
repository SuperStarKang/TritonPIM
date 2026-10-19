"""Collect single-unit curves using the same Torch GEMM template as tuning."""

from statistics import median

import torch
import triton.backends.pim.driver as pim_driver

from ._autotuner import PIMUnitTiming, calibrate_performance, target_model
from .backend import PIMInductorBackend
from .config import PIMTorchConfig
from .template import TORCH_GEMM_TEMPLATE_VERSION


def calibrate_gemms(cases, target, *, sweep_extents, repeat=3):
    if repeat < 1:
        raise ValueError("calibration repeat must be positive")

    def measure_unit(problem, single_target, config):
        if problem.compute_kind != "matmul":
            raise ValueError("Torch GEMM calibration requires a matmul problem")
        if problem.input_dtype == "int8":
            if problem.output_dtype != "int32":
                raise ValueError("INT8 GEMM calibration requires INT32 output")
            operation = torch._int_mm
        else:
            if problem.output_dtype != problem.input_dtype:
                raise ValueError("Torch floating-point GEMM returns its input dtype")
            operation = torch.mm
        left = torch.ones((problem.M, problem.K), dtype=getattr(torch, problem.input_dtype))
        right = torch.ones((problem.K, problem.N), dtype=getattr(torch, problem.input_dtype))
        torch._dynamo.reset()
        compiled = torch.compile(
            operation, backend=PIMInductorBackend(PIMTorchConfig(
                spec=single_target, config=config, pack=False)), fullgraph=True, dynamic=False)
        compiled(left, right)
        samples = []
        for _ in range(repeat):
            compiled(left, right)
            timings = dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))
            samples.append(timings[target_model(single_target).launch_timing_name])
        print(f"[single-unit] blocks={config.key()[:3]} groups={config.key()[3:6]} "
              f"extent={config.execution_extent} launch_ms={median(samples):.6f}", flush=True)
        return PIMUnitTiming(launch_ms=median(samples))

    return calibrate_performance(
        cases, target, measure_unit, kernel_version=TORCH_GEMM_TEMPLATE_VERSION,
        sweep_extents=sweep_extents)
