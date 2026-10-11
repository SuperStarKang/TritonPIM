# GEMV(y = A·x, int32)의 최적 config를 PIM autotuner로 찾고, 그 config로 다시 측정한다.
#
# 실행:
#   ./cmd.sh autotune_gemv.py --m 4096 --k 4096 --max-benchmarks 30
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
from autotuner import (PIMBenchmarkResult, PIMConfigCache, PIMProblem,  # noqa: E402
                       SearchBudget, build_config_space, target_spec, tune)
from autotuner.decorator import pim_launch_options  # noqa: E402


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
    parser.add_argument("--cache", type=str,
                        default=str(Path(__file__).resolve().parent / "gemv_results" / "gemv_autotune_cache.json"))
    args = parser.parse_args()

    triton.runtime.driver.set_active(PIMDriver())
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)

    target =target_spec("upmem", num_units=args.num_units, nr_tasklets=args.nr_tasklets)
    problem = PIMProblem.matvec(args.m, args.k, input_dtype="int32", output_dtype="int32")

    torch.manual_seed(0)
    a = torch.randint(0, 50, (args.m, args.k), dtype=torch.int32)
    x = torch.randint(0, 50, (args.k,), dtype=torch.int32)
    reference = torch.mv(a, x)

    def run(config):
        y = torch.empty((args.m,), dtype=torch.int32)
        gemv_kernel[(triton.cdiv(args.m, config.block_m),)](
            a,
            x, y,
            args.m, args.k,
            a.stride(0), a.stride(1),
            x.stride(0), y.stride(0),
            BLOCK_M=config.block_m, BLOCK_K=config.block_k,
            **pim_launch_options(config, target),
        )
        return y

    def measure(config, reps):
        # 첫 호출에는 device.elf 컴파일과 DPU 예약이 섞이므로 측정에서 뺀다
        run(config)
        walls = []
        stages = []
        for _ in range(reps):
            start = time.perf_counter_ns()
            y = run(config)
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
        space=build_config_space(problem, target),
        cache=PIMConfigCache(args.cache, template_version="gemv-int32-v1"),
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
