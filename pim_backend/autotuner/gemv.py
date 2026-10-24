"""GEMV execution and single-device Group/Tile profiling for UPMEM and AiMX."""

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
from .types import PIMProblem, ceil_div


GEMV_KERNEL_VERSION = "pim-upmem-gemv-v2"


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, y_ptr, M, K,
    stride_am, stride_ak, stride_x, stride_y,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, ACC_DTYPE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M,), dtype=ACC_DTYPE)
    for offset in range(0, tl.cdiv(K, BLOCK_K)):
        columns = offset * BLOCK_K + reduction
        matrix = tl.load(
            a_ptr + rows[:, None] * stride_am + columns[None, :] * stride_ak,
            mask=(rows[:, None] < M) & (columns[None, :] < K), other=0)
        vector = tl.load(x_ptr + columns * stride_x, mask=columns < K, other=0)
        # 곱하기 전에 누산 type으로 넓힌다. INT8 입력은 lowering이 INT8 matvec로 만든다.
        accumulator += tl.sum(matrix.to(ACC_DTYPE) * vector.to(ACC_DTYPE)[None, :], axis=1)
    tl.store(y_ptr + rows * stride_y, accumulator, mask=rows < M)


def run_gemv(a, x, target, config):
    if a.ndim != 2 or x.ndim != 1:
        raise ValueError("GEMV requires a matrix and a vector")
    if a.dtype != x.dtype:
        raise ValueError("GEMV inputs must have one dtype")
    kind = target.common_profile.target
    if kind == "upmem":
        if a.dtype not in (torch.int8, torch.int32):
            raise ValueError("UPMEM GEMV requires INT8 or INT32 inputs")
        output_dtype, accumulator_dtype = torch.int32, tl.int32
    elif kind == "aimx":
        if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("AiMX GEMV requires FP16, BF16 or FP32 inputs")
        output_dtype, accumulator_dtype = torch.float32, tl.float32
    else:
        raise ValueError(f"unsupported GEMV target: {kind}")
    m, k = a.shape
    if x.shape[0] != k:
        raise ValueError("GEMV inputs must have matching reduction sizes")
    output = torch.empty((m,), dtype=output_dtype)
    gemv_kernel[(triton.cdiv(m, config.block_m),)](
        a, x, output, m, k,
        a.stride(0), a.stride(1), x.stride(0), output.stride(0),
        BLOCK_M=config.block_m, BLOCK_K=config.block_k, ACC_DTYPE=accumulator_dtype,
        **pim_launch_options(config, target))
    return output


def profile_gemv(target, *, output_dir, repeat=5, problem=None, sweep_extents=None):
    """Select a measured Group, then find its Tile plateau on one device."""
    if repeat < 1:
        raise ValueError("GEMV profiling repeat must be positive")
    spec = target_model(target).gemv_profile_spec
    if problem is None:
        problem = PIMProblem.matvec(
            spec.block_shape[0], spec.block_shape[2],
            input_dtype=spec.input_dtype, output_dtype=spec.output_dtype)
    if problem.compute_kind != "matvec" or problem.N != 1:
        raise ValueError("the GEMV profiler requires a matvec problem with one column")
    if (problem.input_dtype, problem.output_dtype) != (spec.input_dtype, spec.output_dtype):
        raise ValueError(
            f"GEMV profiling requires {spec.input_dtype} inputs and {spec.output_dtype} output")
    output_dir = prepare_run_directory(output_dir)
    triton.runtime.driver.set_active(pim_driver.PIMDriver())
    aimx = target.common_profile.target == "aimx"

    def measure(local_problem, single_target, config):
        dtype = getattr(torch, local_problem.input_dtype)
        matrix = torch.ones((local_problem.M, local_problem.K), dtype=dtype)
        vector = torch.ones((local_problem.K,), dtype=dtype)
        expected = local_problem.K
        if aimx:
            # AiMX vACC는 같은 부호 부분합이 1024를 넘으면 작은 곱을 버린다(Silent-PIM_Design.md §11.3).
            # 합이 작게 유지되도록 K-group(256)마다 원소 하나만 1로 둔다.
            vector = torch.zeros((local_problem.K,), dtype=dtype)
            vector[::256] = 1
            expected = ceil_div(local_problem.K, 256)
        output = run_gemv(matrix, vector, single_target, config)
        if not torch.equal(output, torch.full_like(output, expected)):
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
