"""Torch-facing decorator for PIM Common IR autotuning."""

import functools
import os
import statistics
import time

import torch
import triton.backends.pim.driver as pim_driver

from .cache import PIMConfigCache
from .constraints import ConstraintEngine
from .search import tune
from .space import build_config_space, heuristic_seeds
from .targets import AiMXRuntimeConfig, UPMEMRuntimeConfig, target_spec
from .types import PIMBenchmarkResult, PIMProblem


DEFAULT_CACHE_PATH = os.path.join(
    os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton")),
    "pim_autotune_cache.json",
)

TORCH_DTYPE_NAMES = {
    torch.int8: "int8",
    torch.int32: "int32",
    torch.float16: "float16",
    torch.float32: "float32",
}
OUTPUT_DTYPES = {
    "int8": "int32",
    "float16": "float32",
    "float32": "float32",
}


def problem_from_tensors(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"GEMM 형상 불일치: A[{M},{K}] @ B[{K2},{N}]"
    assert a.dtype == b.dtype, "A와 B의 dtype이 같아야 합니다"
    input_dtype = TORCH_DTYPE_NAMES[a.dtype]
    return PIMProblem(
        M=M, N=N, K=K,
        input_dtype=input_dtype,
        output_dtype=OUTPUT_DTYPES[input_dtype],
    )


def default_config(problem, target, space):
    engine = ConstraintEngine(problem, target)
    for config in heuristic_seeds(problem, target, space):
        if engine.valid(config):
            return config
    raise ValueError("유효한 seed configuration이 없습니다")


def pim_launch_options(config, target):
    """Translate one config directly to the current compiler options."""
    profile = target.common_profile
    runtime = target.runtime_config
    options = {
        "pim_target": profile.target,
        "num_units": profile.num_units,
        "banks_per_unit": profile.banks_per_unit,
        "cores_per_bank": profile.cores_per_bank,
        "lanes_per_core": profile.lanes_per_core,
        "bank_capacity_limit_bytes": profile.bank_capacity_limit_bytes,
        "local_capacity_limit_bytes": profile.local_capacity_limit_bytes,
        "group_m": config.group_m,
        "group_n": config.group_n,
        "group_k": config.group_k,
        "tile_m": config.tile_m,
        "tile_n": config.tile_n,
        "tile_k": config.tile_k,
        "p_m": config.p_m,
        "p_n": config.p_n,
        "p_k": config.p_k,
    }
    if isinstance(runtime, UPMEMRuntimeConfig):
        options.update(
            nr_tasklets=runtime.nr_tasklets,
            host_threads=runtime.host_threads,
        )
    if isinstance(runtime, AiMXRuntimeConfig):
        options.update(
            batch=runtime.batch,
            base_fpga=runtime.base_fpga,
        )
    return options


def _sample_indices(size, limit=32):
    step = -(-size // limit)
    return torch.arange(0, size, step)


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value == "1"


def pim_autotune(target=None, spec=None, cache_path=DEFAULT_CACHE_PATH,
                 budget=None, space_overrides=None, ref_fn=None, reps=3,
                 seed=0, search=True, retune=False, verbose=True,
                 rtol=0.0, atol=0.0,
                 template_version="pim-common-ir-semantic-schedule-v1"):
    """Autotune ``fn(a, b, *, config)`` with Common IR configurations."""
    selected_target = spec if spec is not None else target_spec(target)
    overrides = space_overrides if space_overrides is not None else {}

    def decorate(fn):
        verify_kind = "custom" if ref_fn is not None else "sampled"
        cache = PIMConfigCache(
            cache_path,
            template_version=(
                f"{template_version};verify={verify_kind};"
                f"rtol={rtol};atol={atol}"),
        )

        @functools.wraps(fn)
        def wrapper(a, b):
            problem = problem_from_tensors(a, b)
            do_search = _env_flag("PIM_AUTOTUNE", search)
            do_retune = _env_flag("PIM_RETUNE", retune)

            if not do_retune:
                hit = cache.lookup_result(problem, selected_target)
                if hit is not None:
                    if verbose:
                        print(
                            f"[autotuner] cache hit: "
                            f"{hit[1].total_ms:.2f} ms "
                            f"config={hit[0].key()}")
                    return fn(a, b, config=hit[0])

            space = build_config_space(
                problem, selected_target, **overrides)
            if not (do_search or do_retune):
                config = default_config(problem, selected_target, space)
                if verbose:
                    print(
                        f"[autotuner] search disabled: "
                        f"config={config.key()}")
                return fn(a, b, config=config)

            out_dtype = getattr(torch, problem.output_dtype)
            if ref_fn is not None:
                reference = ref_fn(a, b)

                def verify(out):
                    return bool(torch.allclose(
                        out, reference, rtol=rtol, atol=atol))
            else:
                rows = _sample_indices(problem.M)
                cols = _sample_indices(problem.N)
                ref_block = (
                    a.index_select(0, rows).to(out_dtype)
                    @ b.index_select(1, cols).to(out_dtype)
                )

                def verify(out):
                    got = out.index_select(0, rows).index_select(1, cols)
                    return bool(torch.allclose(
                        got, ref_block, rtol=rtol, atol=atol))

            def benchmark_fn(problem, target, config):
                out = fn(a, b, config=config)
                samples = []
                stage_samples = []
                for _ in range(reps):
                    start = time.perf_counter_ns()
                    out = fn(a, b, config=config)
                    samples.append(
                        (time.perf_counter_ns() - start) / 1e6)
                    stage_samples.append(tuple(pim_driver.timing))
                timings = tuple(
                    (name, statistics.median(
                        sample[stage] for sample in stage_samples))
                    for stage, name in enumerate(pim_driver.TIMING_STAGES)
                )
                return PIMBenchmarkResult(
                    total_ms=statistics.median(samples),
                    correctness=verify(out),
                    timings=timings,
                )

            result = tune(
                problem, selected_target, benchmark_fn,
                space=space, cache=cache, budget=budget, seed=seed,
                retune=do_retune, verbose=verbose,
            )
            return fn(a, b, config=result.best.config)

        return wrapper

    return decorate
