"""UPMEM GEMV execution and single-device Group/Tile profiling."""

from statistics import median

import torch
import triton
import triton.language as tl
import triton.backends.pim.driver as pim_driver

from .artifacts import prepare_run_directory
from .decorator import pim_launch_options
from .model import target_model
from .performance import PIMUnitTiming, profile_single_unit
from .space import build_config_space
from .types import PIMProblem


GEMV_KERNEL_VERSION = "pim-upmem-gemv-v2"


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, y_ptr, M, K,
    stride_am, stride_ak, stride_x, stride_y,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M,), dtype=tl.int32)
    for offset in range(0, tl.cdiv(K, BLOCK_K)):
        columns = offset * BLOCK_K + reduction
        matrix = tl.load(
            a_ptr + rows[:, None] * stride_am + columns[None, :] * stride_ak,
            mask=(rows[:, None] < M) & (columns[None, :] < K), other=0)
        vector = tl.load(x_ptr + columns * stride_x, mask=columns < K, other=0)
        # 곱하기 전에 INT32로 넓힌다. INT8 입력은 lowering이 INT8 matvec로 만든다.
        accumulator += tl.sum(matrix.to(tl.int32) * vector.to(tl.int32)[None, :], axis=1)
    tl.store(y_ptr + rows * stride_y, accumulator, mask=rows < M)


def run_gemv(a, x, target, config):
    if target.common_profile.target != "upmem":
        raise ValueError("the GEMV autotuning kernel currently supports UPMEM only")
    if a.ndim != 2 or x.ndim != 1:
        raise ValueError("GEMV requires a matrix and a vector")
    if a.dtype != x.dtype or a.dtype not in (torch.int8, torch.int32):
        raise ValueError("UPMEM GEMV requires INT8 or INT32 inputs of one dtype")
    m, k = a.shape
    if x.shape[0] != k:
        raise ValueError("GEMV inputs must have matching reduction sizes")
    output = torch.empty((m,), dtype=torch.int32)
    gemv_kernel[(triton.cdiv(m, config.block_m),)](
        a, x, output, m, k,
        a.stride(0), a.stride(1), x.stride(0), output.stride(0),
        BLOCK_M=config.block_m, BLOCK_K=config.block_k,
        **pim_launch_options(config, target))
    return output


def profile_gemv(target, *, output_dir, repeat=5, problem=None, sweep_extents=None):
    """Select a measured Group, then find its Tile plateau on one UPMEM DPU."""
    if repeat < 1:
        raise ValueError("GEMV profiling repeat must be positive")
    if target.common_profile.target != "upmem":
        raise ValueError("GEMV profiling currently supports UPMEM only")
    spec = target_model(target).gemv_profile_spec
    if problem is None:
        problem = PIMProblem.matvec(
            spec.block_shape[0], spec.block_shape[2],
            input_dtype=spec.input_dtype, output_dtype=spec.output_dtype)
    if problem.compute_kind != "matvec" or problem.N != 1:
        raise ValueError("the GEMV profiler requires a matvec problem with one column")
    if (problem.input_dtype, problem.output_dtype) != (spec.input_dtype, spec.output_dtype):
        raise ValueError("UPMEM GEMV profiling requires INT32 inputs and output")
    output_dir = prepare_run_directory(output_dir)
    triton.runtime.driver.set_active(pim_driver.PIMDriver())

    def measure(local_problem, single_target, config):
        matrix = torch.ones((local_problem.M, local_problem.K), dtype=torch.int32)
        vector = torch.ones((local_problem.K,), dtype=torch.int32)
        output = run_gemv(matrix, vector, single_target, config)
        if not torch.equal(output, torch.full_like(output, local_problem.K)):
            raise ValueError(f"single-device GEMV correctness failed: config={config.key()}")
        samples = []
        for _ in range(repeat):
            run_gemv(matrix, vector, single_target, config)
            timings = dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))
            samples.append(timings[target_model(single_target).launch_timing_name])
        return PIMUnitTiming(median(samples))

    return profile_single_unit(
        [(problem, build_config_space(problem, target))], target, measure,
        kernel_version=GEMV_KERNEL_VERSION, output_dir=output_dir, sweep_extents=sweep_extents)
