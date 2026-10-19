# GEMV(y = A·x, int32)의 최적 config를 PIM autotuner로 찾고, 그 config로 다시 측정한다.
#
# 실행:
#   python autotune_gemv.py --m 4096 --k 4096 --max-benchmarks 30
#     cache에 이 shape가 있으면 탐색 없이 그 best를 측정한다. --retune이면 다시 탐색한다.
#
# 탐색 objective는 autotuner 기본값인 end-to-end wall time(매 호출 A 전송 포함, reps의 median)이다.
# 마지막 두 줄은 run_autotune_gemv.sh가 CSV로 모으는 형식이다:
#   HEADER,m,k,nr_tasklets,block_m,block_k,group_m,group_k,tile_m,tile_k,p_m,p_k,dpus,wall,<stage...>
#   RESULT,...

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # autotuner

import triton.backends.pim.driver as pim_driver  # noqa: E402
from triton.backends.pim.driver import PIMDriver  # noqa: E402

from gemv import gemv_kernel  # noqa: E402
from autotuner import (PIMBenchmarkResult, PIMConfigCache, PIMPerformanceProfile, PIMProblem,  # noqa: E402
                       PIMUnitTiming, SearchBudget, build_config_space, profile_single_unit, target_spec, tune)
from autotuner.artifacts import RESULTS_ROOT, prepare_run_directory, project_path  # noqa: E402
from autotuner.decorator import pim_launch_options  # noqa: E402
from autotuner.model import target_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--num-units", type=int, default=2048, help="사용 가능한 DPU 수")
    parser.add_argument("--nr-tasklets", type=int, default=0, help="DPU당 tasklet 수. 0이면 lanes_per_core(16)")
    parser.add_argument("--max-benchmarks", type=int, default=30)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--reps", type=int, default=3, help="탐색 중 config당 측정 횟수 (median)")
    parser.add_argument("--repeat", type=int, default=10, help="best config 최종 측정 횟수 (median)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retune", action="store_true", help="cache가 있어도 다시 탐색")
    parser.add_argument("--cache", type=str)
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "upmem_gemv"))
    parser.add_argument("--performance", help="single-unit saturation JSON for gemv-int32-v1")
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile-only", action="store_true")
    profiling.add_argument("--calibrate-performance", nargs=2, type=int, metavar=("M", "K"),
                           help="측정 상한을 지정해 profiling 후 종료")
    args = parser.parse_args()
    if args.reps < 1 or args.repeat < 1:
        parser.error("--reps and --repeat must be positive")
    output_dir = prepare_run_directory(args.output_dir)
    cache_path = project_path(args.cache) if args.cache is not None else output_dir / "autotune_cache.json"
    performance = None
    if not args.profile_only and args.calibrate_performance is None and args.performance is not None:
        performance = PIMPerformanceProfile.load(args.performance, kernel_version="gemv-int32-v1")

    triton.runtime.driver.set_active(PIMDriver())

    target = target_spec("upmem", num_units=args.num_units, nr_tasklets=args.nr_tasklets)
    problem = PIMProblem.matvec(args.m, args.k, input_dtype="int32", output_dtype="int32")

    def run(a, x, target, config):
        m, k = a.shape
        y = torch.empty((m,), dtype=torch.int32)
        gemv_kernel[(triton.cdiv(m, config.block_m),)](
            a,
            x, y,
            m, k,
            a.stride(0), a.stride(1),
            x.stride(0), y.stride(0),
            BLOCK_M=config.block_m, BLOCK_K=config.block_k,
            **pim_launch_options(config, target),
        )
        return y

    def measure_unit(problem, single_target, config):
        left = torch.ones((problem.M, problem.K), dtype=torch.int32)
        right = torch.ones((problem.K,), dtype=torch.int32)
        output = run(left, right, single_target, config)
        if not torch.equal(output, torch.full_like(output, problem.K)):
            raise ValueError(f"single-device GEMV correctness failed: config={config.key()}")
        samples = []
        for _ in range(args.reps):
            run(left, right, single_target, config)
            samples.append(dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))[
                target_model(single_target).launch_timing_name])
        return PIMUnitTiming(statistics.median(samples))

    def collect_performance(problem, target, space):
        if not args.profile_only and args.calibrate_performance is None:
            path = output_dir / "performance.json"
            if path.exists():
                measured = PIMPerformanceProfile.load(path, kernel_version="gemv-int32-v1")
                measured.seed_families(problem, target, space.block_group_top_k)
                print(f"[single-unit] loaded: {path}", flush=True)
                return measured
        limits = args.calibrate_performance
        return profile_single_unit(
            [(problem, space)], target, measure_unit, kernel_version="gemv-int32-v1",
            output_dir=output_dir,
            sweep_extents=(args.m, 1, args.k) if limits is None else (limits[0], 1, limits[1]))

    if args.profile_only or args.calibrate_performance is not None:
        measured = collect_performance(problem, target, build_config_space(problem, target))
        if args.performance is not None:
            measured.save(args.performance)
        return

    torch.manual_seed(0)
    a = torch.randint(0, 50, (args.m, args.k), dtype=torch.int32)
    x = torch.randint(0, 50, (args.k,), dtype=torch.int32)
    reference = torch.mv(a, x)

    def measure(config, reps):
        # 첫 호출에는 device.elf 컴파일과 DPU 예약이 섞이므로 측정에서 뺀다
        run(a, x, target, config)
        walls = []
        stages = []
        for _ in range(reps):
            start = time.perf_counter_ns()
            y = run(a, x, target, config)
            walls.append((time.perf_counter_ns() - start) / 1e6)
            stages.append(list(pim_driver.timing))
        return PIMBenchmarkResult(
            total_ms=statistics.median(walls),
            correctness=bool(torch.equal(y, reference)),
            timings=tuple((name, statistics.median(sample[i] for sample in stages))
                          for i, name in enumerate(pim_driver.TIMING_STAGES)),
        )

    result = tune(
        problem, target,
        lambda problem, target, config: measure(config, args.reps),
        performance=performance, kernel_version="gemv-int32-v1",
        profile_fn=collect_performance,
        space=build_config_space(problem, target),
        cache=PIMConfigCache(str(cache_path), template_version="gemv-int32-v1"),
        budget=SearchBudget(max_generations=args.max_generations,
                            max_benchmarks=args.max_benchmarks,
                            max_autotune_seconds=args.max_seconds),
        seed=args.seed,
        retune=args.retune,
        verbose=True,
    )

    best = result.best.config
    final = measure(best, args.repeat)
    assert final.correctness, f"결과 불일치: config={best.key()}"

    print(f"\nM={args.m} K={args.k} nr_tasklets={args.nr_tasklets} from_cache={result.from_cache}")
    print(f"best: B=({best.block_m},{best.block_k}) G=({best.group_m},{best.group_k}) "
          f"tile=({best.tile_m},{best.tile_k}) P=({best.p_m},{best.p_n},{best.p_k}) "
          f"dpus={best.p_m * best.p_n * best.p_k}")
    for name, ms in final.timings:
        print(f"{name:12s}: {ms:9.3f} ms")
    print(f"{'wall':12s}: {final.total_ms:9.3f} ms")

    print(",".join(["HEADER", "m", "k", "nr_tasklets", "block_m", "block_k", "group_m", "group_k",
                    "tile_m", "tile_k", "p_m", "p_k", "dpus", "wall",
                    *(name for name, _ in final.timings)]))
    print(",".join(["RESULT",
                    *map(str, [args.m, args.k, args.nr_tasklets, best.block_m, best.block_k,
                               best.group_m, best.group_k, best.tile_m, best.tile_k,
                               best.p_m, best.p_k, best.p_m * best.p_n * best.p_k]),
                    f"{final.total_ms:.4f}",
                    *(f"{ms:.4f}" for _, ms in final.timings)]))


if __name__ == "__main__":
    main()
