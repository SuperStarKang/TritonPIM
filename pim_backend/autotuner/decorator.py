"""Torch-facing decorator for PIM Common IR autotuning."""

import functools
import os
import statistics
import time

import torch
import triton.backends.pim.driver as pim_driver

from .cache import PIMConfigCache
from .artifacts import DEFAULT_CACHE_PATH
from .constraints import ConstraintEngine
from .model import target_model
from .performance import PIMUnitTiming, calibrate_performance
from .search import tune
from .space import build_config_space, heuristic_seeds
from .targets import AiMXRuntimeConfig, UPMEMRuntimeConfig, target_spec
from .types import PIMBenchmarkResult, PIMProblem


TORCH_DTYPE_NAMES = {
    torch.int8: "int8",
    torch.int32: "int32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
OUTPUT_DTYPES = {
    "int8": "int32",
    "float16": "float32",
    "bfloat16": "float32",
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


def default_config(problem, target, space, *, performance, kernel_version):
    engine = ConstraintEngine(problem, target)
    for config in heuristic_seeds(problem, target, space, performance=performance,
                                  kernel_version=kernel_version, engine=engine):
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
            aimx_matmul_k_alignment=runtime.matmul_k_alignment,
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
                 template_version="pim-common-ir-semantic-schedule-v1",
                 check_correctness=True, performance=None, profile_fn=None):
    """Autotune ``fn(a, b, *, config)`` with Common IR configurations."""
    selected_target = spec if spec is not None else target_spec(target)
    overrides = space_overrides if space_overrides is not None else {}

    def decorate(fn):
        verify_kind = "custom" if ref_fn is not None else "sampled"
        if not check_correctness:
            verify_kind = "disabled"
        cache = PIMConfigCache(
            cache_path,
            template_version=(
                f"{template_version};verify={verify_kind};"
                f"rtol={rtol};atol={atol}"),
        )

        @functools.wraps(fn)
        def wrapper(a, b):
            problem = problem_from_tensors(a, b)
            if verbose and not check_correctness:
                print(
                    "[autotuner] correctness=None: verification SKIPPED",
                    flush=True,
                )
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
                seed_performance = performance
                if seed_performance is None and profile_fn is not None:
                    seed_performance = profile_fn(problem, selected_target, space)
                config = default_config(problem, selected_target, space,
                                        performance=seed_performance, kernel_version=template_version)
                if verbose:
                    print(
                        f"[autotuner] search disabled: "
                        f"config={config.key()}")
                return fn(a, b, config=config)

            out_dtype = getattr(torch, problem.output_dtype)
            if not check_correctness:
                def verify(out):
                    return None
            elif ref_fn is not None:
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
                if verbose and isinstance(
                        selected_target.runtime_config, AiMXRuntimeConfig):
                    # 입력 반올림만 분리하며 AiMX 내부 누산을 재현하지 않는다.
                    ref_bf16_block = (
                        a.index_select(0, rows).to(torch.bfloat16).float()
                        @ b.index_select(1, cols).to(torch.bfloat16).float()
                    )
                    input_rounding_error = (ref_bf16_block - ref_block).abs()
                    print(
                        "[verify-reference] BF16 inputs, FP32 matmul: "
                        f"input_rounding_max_abs={input_rounding_error.max().item():.6g} "
                        f"input_rounding_mean_abs={input_rounding_error.mean().item():.6g}",
                        flush=True,
                    )

                def verify(out):
                    got = out.index_select(0, rows).index_select(1, cols)
                    ok = bool(torch.allclose(
                        got, ref_block, rtol=rtol, atol=atol))
                    if verbose and isinstance(
                            selected_target.runtime_config, AiMXRuntimeConfig):
                        error = (got - ref_block).abs()
                        bf16_error = (got - ref_bf16_block).abs()
                        failed = ~torch.isclose(
                            got, ref_block, rtol=rtol, atol=atol)
                        print(
                            f"\n[verify] ok={ok} "
                            f"sampled={got.numel()}/{out.numel()} "
                            f"failed={failed.sum().item()}/{got.numel()} "
                            f"rtol={rtol:g} atol={atol:g} "
                            f"max_abs={error.max().item():.6g} "
                            f"mean_abs={error.mean().item():.6g}",
                            flush=True,
                        )
                        print(
                            f"[verify] sampled_nan={torch.isnan(got).sum().item()} "
                            f"sampled_inf={torch.isinf(got).sum().item()} "
                            f"reference_nonfinite={(~torch.isfinite(ref_block)).sum().item()} "
                            f"vs_bf16_input_ref_max_abs={bf16_error.max().item():.6g} "
                            f"vs_bf16_input_ref_mean_abs={bf16_error.mean().item():.6g}",
                            flush=True,
                        )
                        for i, j in failed.nonzero()[:5].tolist():
                            print(
                                f"[verify] C[{rows[i].item()},{cols[j].item()}] "
                                f"got={got[i, j].item():.9g} "
                                f"ref={ref_block[i, j].item():.9g} "
                                f"abs_error={error[i, j].item():.6g} "
                                f"allowed={(atol + rtol * ref_block[i, j].abs()).item():.6g} "
                                f"bf16_input_ref={ref_bf16_block[i, j].item():.9g} "
                                f"abs_error_vs_bf16={bf16_error[i, j].item():.6g}",
                                flush=True,
                            )
                    return ok

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
                    physical_allocations=pim_driver.aimx_allocations,
                )

            result = tune(
                problem, selected_target, benchmark_fn,
                performance=performance, kernel_version=template_version,
                profile_fn=profile_fn,
                space=space, cache=cache, budget=budget, seed=seed,
                retune=do_retune, verbose=verbose,
            )
            return fn(a, b, config=result.best.config)

        def calibrate(a, b, *, sweep_extents):
            if reps < 1:
                raise ValueError("calibration reps must be positive")
            problem = problem_from_tensors(a, b)

            def measure_unit(local_problem, single_target, config):
                left = torch.ones((local_problem.M, local_problem.K), dtype=a.dtype)
                right = torch.ones((local_problem.K, local_problem.N), dtype=b.dtype)
                fn(left, right, config=config)
                samples = []
                for _ in range(reps):
                    fn(left, right, config=config)
                    timings = dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))
                    samples.append(timings[target_model(single_target).launch_timing_name])
                print(f"[single-unit] extent={config.execution_extent} "
                      f"launch_ms={statistics.median(samples):.6f}", flush=True)
                return PIMUnitTiming(launch_ms=statistics.median(samples))

            return calibrate_performance(
                [(problem, build_config_space(problem, selected_target, **overrides))],
                selected_target, measure_unit, kernel_version=template_version,
                sweep_extents=sweep_extents)

        wrapper.calibrate = calibrate
        return wrapper

    return decorate
