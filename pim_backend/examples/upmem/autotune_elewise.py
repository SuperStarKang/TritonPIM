import statistics
import time
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

import torch
import triton
import triton.language as tl
from triton.backends.pim.driver import PIMDriver

triton.runtime.driver.set_active(PIMDriver())

from autotuner import (
    PIMBenchmarkResult,
    PIMConfigCache,
    PIMProblem,
    SearchBudget,
    build_config_space,
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


def run(config):
    out = torch.empty_like(x)
    grid = (triton.cdiv(size, config.block_m),)

    add_kernel[grid](
        x,
        y,
        out,
        size,
        BLOCK_M=config.block_m,
        **pim_launch_options(config, target),
    )
    return out


def benchmark_fn(problem, target, config):
    run(config)

    samples = []
    out = None
    for _ in range(3):
        start = time.perf_counter_ns()
        out = run(config)
        samples.append((time.perf_counter_ns() - start) / 1e6)

    return PIMBenchmarkResult(
        total_ms=statistics.median(samples),
        correctness=bool(torch.allclose(out, x + y)),
    )


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