"""The same Triton GEMM and single-device profiler for every PIM target."""

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


GEMM_KERNEL_VERSION = "pim-common-gemm-v2"


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    left_ptrs = a_ptr + rows[:, None] * stride_am + reduction[None, :] * stride_ak
    right_ptrs = b_ptr + reduction[:, None] * stride_bk + columns[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for offset in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - offset * BLOCK_K
        left = tl.load(left_ptrs, mask=(rows[:, None] < M) & (reduction[None, :] < remaining),
                       other=0)
        right = tl.load(right_ptrs, mask=(reduction[:, None] < remaining) & (columns[None, :] < N),
                        other=0)
        accumulator += tl.dot(left, right, out_dtype=ACC_DTYPE)
        left_ptrs += BLOCK_K * stride_ak
        right_ptrs += BLOCK_K * stride_bk
    output = c_ptr + rows[:, None] * stride_cm + columns[None, :] * stride_cn
    tl.store(output, accumulator.to(c_ptr.dtype.element_ty),
             mask=(rows[:, None] < M) & (columns[None, :] < N))


def run_gemm(a, b, target, config, *, output_dtype):
    if output_dtype == "int32":
        accumulator_dtype = tl.int32
    elif output_dtype in ("float16", "bfloat16", "float32"):
        accumulator_dtype = tl.float32
    else:
        raise ValueError(f"unsupported GEMM profiling output dtype: {output_dtype}")
    m, k = a.shape
    if b.shape[0] != k or a.dtype != b.dtype:
        raise ValueError("GEMM inputs must have matching reduction sizes and dtypes")
    n = b.shape[1]
    output = torch.zeros((m, n), dtype=getattr(torch, output_dtype))
    gemm_kernel[(triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))](
        a, b, output, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0), output.stride(1),
        BLOCK_M=config.block_m, BLOCK_N=config.block_n, BLOCK_K=config.block_k,
        ACC_DTYPE=accumulator_dtype, **pim_launch_options(config, target))
    return output


def profile_gemm(target, *, output_dir, repeat=5, problem=None, sweep_extents=None):
    """Profile one device from target information, optionally for an explicit dtype pair."""
    if repeat < 1:
        raise ValueError("GEMM profiling repeat must be positive")
    output_dir = prepare_run_directory(output_dir)
    triton.runtime.driver.set_active(pim_driver.PIMDriver())
    if problem is None:
        spec = target_model(target).gemm_profile_spec
        problem = PIMProblem(*spec.block_shape,
                             input_dtype=spec.input_dtype, output_dtype=spec.output_dtype)
    if problem.compute_kind != "matmul":
        raise ValueError("the common GEMM profiler requires a matmul problem")

    def measure(local_problem, single_target, config):
        left = torch.ones((local_problem.M, local_problem.K),
                          dtype=getattr(torch, local_problem.input_dtype))
        right = torch.ones((local_problem.K, local_problem.N),
                           dtype=getattr(torch, local_problem.input_dtype))
        output = run_gemm(left, right, single_target, config, output_dtype=local_problem.output_dtype)
        if not torch.equal(output, torch.full_like(output, local_problem.K)):
            raise ValueError(f"single-device GEMM correctness failed: config={config.key()}")
        samples = []
        for _ in range(repeat):
            run_gemm(left, right, single_target, config, output_dtype=local_problem.output_dtype)
            timings = dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))
            samples.append(timings[target_model(single_target).launch_timing_name])
        return PIMUnitTiming(median(samples))

    return profile_single_unit(
        [(problem, build_config_space(problem, target))], target, measure,
        kernel_version=GEMM_KERNEL_VERSION, output_dir=output_dir, sweep_extents=sweep_extents)
