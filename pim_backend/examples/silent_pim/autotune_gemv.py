# GEMV(y = A·x, FP16/BF16/FP32 입력, FP32 출력)의 최적 config를 PIM autotuner로 찾고, 그 config로 다시 측정한다.
#
# 실행:
#   python autotune_gemv.py --m 4096 --k 4096 --max-benchmarks 30
#     cache에 이 shape가 있으면 탐색 없이 그 best를 측정한다. --retune이면 다시 탐색한다.
#
# AiMX는 GEMV를 yᵀ = xᵀ·Wᵀ (batch 1, 출력 lane = M 방향)으로 실행한다.
# 탐색 objective는 autotuner 기본값인 end-to-end wall time(매 호출 A 전송 포함, reps의 median)이다.
# 정확도는 FP32 기준 relative L2로 본다. vACC 양자화 바닥이 signed 약 0.78%, llm 분포 약 0.40%다
# (Silent-PIM_Design.md §11.3).

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

from autotuner import (PIMBenchmarkResult, PIMConfigCache, PIMProblem,  # noqa: E402
                       SearchBudget, build_config_space, target_spec, tune)
from autotuner.artifacts import RESULTS_ROOT, prepare_run_directory, project_path  # noqa: E402
from autotuner.gemv import GEMV_KERNEL_VERSION, profile_gemv, run_gemv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float32",
                        help="A와 x의 dtype (출력은 FP32)")
    parser.add_argument("--num-units", type=int, default=1, help="사용할 FPGA 수")
    parser.add_argument("--base-fpga", type=int, default=0)
    parser.add_argument("--max-benchmarks", type=int, default=30)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--reps", type=int, default=3, help="탐색 중 config당 측정 횟수 (median)")
    parser.add_argument("--repeat", type=int, default=10, help="best config 최종 측정 횟수 (median)")
    parser.add_argument("--rel-l2", type=float, default=0.02,
                        help="정답 판정 relative L2 상한")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retune", action="store_true", help="cache가 있어도 다시 탐색")
    parser.add_argument("--cache", type=str)
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "aimx_gemv"))
    parser.add_argument("--performance", help="profiling 결과를 저장할 JSON")
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile-only", action="store_true")
    profiling.add_argument("--calibrate-performance", nargs=2, type=int, metavar=("M", "K"),
                           help="측정 상한을 지정해 profiling 후 종료")
    args = parser.parse_args()
    if args.reps < 1 or args.repeat < 1:
        parser.error("--reps and --repeat must be positive")
    output_dir = prepare_run_directory(args.output_dir)
    cache_path = project_path(args.cache) if args.cache is not None else output_dir / "autotune_cache.json"

    triton.runtime.driver.set_active(PIMDriver())

    target = target_spec("aimx", num_units=args.num_units, base_fpga=args.base_fpga)
    print(f"AiMX profile: num_units={target.common_profile.num_units}, "
          f"bank_capacity={target.common_profile.bank_capacity_limit_bytes}")
    problem = PIMProblem.matvec(args.m, args.k, input_dtype=args.dtype, output_dtype="float32")

    if args.profile_only or args.calibrate_performance is not None:
        measured = profile_gemv(
            target, output_dir=output_dir, repeat=args.reps,
            sweep_extents=(None if args.calibrate_performance is None else
                           (args.calibrate_performance[0], 1, args.calibrate_performance[1])))
        if args.performance is not None:
            measured.save(args.performance)
        return

    torch.manual_seed(0)
    # LLM weight 형태(Silent-PIM_Design.md §11.3의 llm 분포): A ~ N(0, 0.02), x ~ N(0, 1).
    a = (torch.randn((args.m, args.k)) * 0.02).to(getattr(torch, args.dtype))
    x = torch.randn((args.k,)).to(getattr(torch, args.dtype))
    reference = torch.mv(a.to(torch.float32), x.to(torch.float32))

    def measure(config, reps):
        # 첫 호출에는 컴파일과 chain 생성이 섞이므로 측정에서 뺀다
        run_gemv(a, x, target, config)
        walls = []
        stages = []
        for _ in range(reps):
            start = time.perf_counter_ns()
            y = run_gemv(a, x, target, config)
            walls.append((time.perf_counter_ns() - start) / 1e6)
            stages.append(list(pim_driver.timing))
        rel_l2 = ((y - reference).norm() / reference.norm()).item()
        return PIMBenchmarkResult(
            total_ms=statistics.median(walls),
            correctness=rel_l2 <= args.rel_l2,
            timings=tuple((name, statistics.median(sample[i] for sample in stages))
                          for i, name in enumerate(pim_driver.TIMING_STAGES)),
        ), rel_l2

    result = tune(
        problem, target,
        lambda problem, target, config: measure(config, args.reps)[0],
        space=build_config_space(problem, target),
        cache=PIMConfigCache(str(cache_path), template_version=GEMV_KERNEL_VERSION),
        budget=SearchBudget(max_generations=args.max_generations,
                            max_benchmarks=args.max_benchmarks,
                            max_autotune_seconds=args.max_seconds),
        seed=args.seed,
        retune=args.retune,
        verbose=True,
    )

    best = result.best.config
    final, rel_l2 = measure(best, args.repeat)
    assert final.correctness, f"결과 불일치: config={best.key()} rel_l2={rel_l2:.4%}"

    print(f"\nM={args.m} K={args.k} dtype={args.dtype} from_cache={result.from_cache}")
    print(f"best: B=({best.block_m},{best.block_k}) G=({best.group_m},{best.group_k}) "
          f"tile=({best.tile_m},{best.tile_k}) P=({best.p_m},{best.p_n},{best.p_k}) "
          f"fpgas={best.p_m * best.p_n * best.p_k}")
    print(f"rel_l2: {rel_l2:.4%}")
    for name, ms in final.timings:
        print(f"{name:12s}: {ms:9.3f} ms")
    print(f"{'wall':12s}: {final.total_ms:9.3f} ms")


if __name__ == "__main__":
    main()
