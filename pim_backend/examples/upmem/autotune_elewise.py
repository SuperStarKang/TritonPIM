import argparse
import statistics
import time
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

import torch
import triton
import triton.language as tl
from triton.backends.pim.driver import PIMDriver
import triton.backends.pim.driver as pim_driver

triton.runtime.driver.set_active(PIMDriver())

from autotuner import (
    PIMBenchmarkResult,
    PIMConfigCache,
    PIMProblem,
    PIMUnitTiming,
    SearchBudget,
    build_config_space,
    calibrate_performance,
    target_spec,
    tune,
)
from autotuner.decorator import pim_launch_options


@triton.jit
def add_kernel(
    x,
    y,
    out,
    size,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < size

    xv = tl.load(x + offsets, mask=mask, other=0.0)
    yv = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, xv + yv, mask=mask)


parser = argparse.ArgumentParser()
parser.add_argument("--performance", help="--calibrate-performance 결과를 저장할 JSON")
parser.add_argument("--calibrate-performance", type=int, metavar="M",
                    help="measure up to this extent and write --performance, then exit")
args = parser.parse_args()

size = 1 << 20
x = torch.randn(size)
y = torch.randn(size)

problem = PIMProblem.elementwise(
    M=size,
    input_dtypes=("float32", "float32"),
    output_dtype="float32",
)
target = target_spec("upmem")
space = build_config_space(problem, target)
cache = PIMConfigCache("/home/dlrkdals/TritonPIM/workspace/autotune_cache.json")


def run(x, y, target, config):
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), config.block_m),)

    add_kernel[grid](
        x,
        y,
        out,
        x.numel(),
        BLOCK_M=config.block_m,
        **pim_launch_options(config, target),
    )
    return out


def benchmark_fn(problem, target, config):
    run(x, y, target, config)

    samples = []
    out = None
    for _ in range(3):
        start = time.perf_counter_ns()
        out = run(x, y, target, config)
        samples.append((time.perf_counter_ns() - start) / 1e6)

    return PIMBenchmarkResult(
        total_ms=statistics.median(samples),
        correctness=bool(torch.allclose(out, x + y)),
    )


if args.calibrate_performance is not None:
    if args.performance is None:
        parser.error("--calibrate-performance requires --performance as its output path")

    def measure_unit(problem, single_target, config):
        left = torch.ones(problem.M)
        right = torch.ones(problem.M)
        run(left, right, single_target, config)
        samples = []
        for _ in range(3):
            run(left, right, single_target, config)
            samples.append(dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))["kernel"])
        return PIMUnitTiming(statistics.median(samples))

    calibrate_performance(
        [(problem, space)], target, measure_unit, kernel_version="elementwise-add-v1",
        sweep_extents=(args.calibrate_performance, 1, 1)).save(args.performance)
    raise SystemExit(0)


result = tune(
    problem,
    target,
    benchmark_fn,
    space=space,
    cache=cache,
    budget=SearchBudget(max_benchmarks=30),
    verbose=True,
)

best = result.best.config
