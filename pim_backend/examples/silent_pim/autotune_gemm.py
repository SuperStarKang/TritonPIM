# @pim_autotune 데코레이터 사용 예시 — AiMX bfloat16 GEMM (FP32 출력).
#
# 실행:
#   python autotune_gemm.py --m 1024 --n 8192 --k 28672 --max-benchmarks 30
#   정확성 검사는 기본 OFF이며 --check-correctness로 켠다.
#   --retune이면 cache가 있어도 다시 탐색한다.
#
# 같은 target spec을 autotuner와 kernel launch에 전달한다.
# config → 커널 인자 매핑은 아래 함수 본문이 소유하고,
# backend 공통 launch 옵션은 pim_launch_options helper가 조립한다.

import argparse
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # autotuner

from triton.backends.pim.driver import PIMDriver  # noqa: E402

from autotuner import SearchBudget  # noqa: E402
from autotuner.artifacts import RESULTS_ROOT, prepare_run_directory, project_path  # noqa: E402
from autotuner.decorator import pim_autotune  # noqa: E402
from autotuner.gemm import GEMM_KERNEL_VERSION, profile_gemm, run_gemm  # noqa: E402
from autotuner.model import target_model  # noqa: E402
from autotuner.targets import target_spec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="PIM AutoTuner 데코레이터 예시 — AiMX bfloat16 GEMM")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=28672)
    parser.add_argument("--reps", type=int, default=5,
                        help="configuration당 측정 반복 수 (median, §32)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--max-benchmarks", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--cache", type=str)
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "aimx_gemm"))
    parser.add_argument("--num-units", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base-fpga", type=int, default=0)
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--performance", help="profiling 결과를 저장할 JSON")
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile-only", action="store_true")
    profiling.add_argument("--calibrate-performance", nargs=3, type=int, metavar=("M", "N", "K"),
                           help="측정 크기 상한을 지정해 profiling 후 종료")
    parser.add_argument("--check-correctness",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="정확성 검사 ON/OFF (기본 OFF)")
    parser.add_argument("--rel-l2", type=float, default=0.02,
                        help="correctness 판정: 표본의 ‖PIM − bf16 matmul‖ / ‖bf16 matmul‖ 상한")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    output_dir = prepare_run_directory(args.output_dir)
    cache_path = project_path(args.cache) if args.cache is not None else output_dir / "autotune_cache.json"

    triton.runtime.driver.set_active(PIMDriver())
    hardware = target_spec("aimx", num_units=args.num_units, batch=args.batch, base_fpga=args.base_fpga)

    if args.profile_only or args.calibrate_performance is not None:
        performance = profile_gemm(
            hardware, output_dir=output_dir, repeat=args.reps,
            sweep_extents=args.calibrate_performance)
        if args.performance is not None:
            performance.save(args.performance)
        return

    budget = SearchBudget(
        max_generations=args.max_generations,
        max_benchmarks=args.max_benchmarks,
        max_autotune_seconds=args.max_seconds,
    )

    gemm_spec = target_model(hardware).gemm_profile_spec

    @pim_autotune(spec=hardware, cache_path=str(cache_path), budget=budget,
                  reps=args.reps, seed=args.seed,
                  rel_l2=args.rel_l2,
                  check_correctness=args.check_correctness,
                  retune=args.retune, template_version=GEMM_KERNEL_VERSION)
    def gemm(a, b, *, config):
        return run_gemm(a, b, hardware, config, output_dtype=gemm_spec.output_dtype)

    torch.manual_seed(0)
    # PIM DRAM이 bf16이므로 입력을 bf16으로 넘겨 host 변환과 입력 반올림 차이를 없앤다.
    a = torch.randn((args.m, args.k)).to(torch.bfloat16)
    b = torch.randn((args.k, args.n)).to(torch.bfloat16)

    start_ns = time.perf_counter_ns()
    out = gemm(a, b)
    wall_ms = (time.perf_counter_ns() - start_ns) / 1e6

    print(f"\nwall  : {wall_ms:.2f} ms (탐색이 돌았다면 그 시간을 포함한다)")
    if args.check_correctness:
        # bf16 matmul 결과와 PIM 결과(bf16 drain 값)를 bf16 값끼리 비교한다.
        # 오차 계산만 float로 해서 지표에 반올림이 더해지지 않게 한다.
        ref = (a @ b).float()
        got = out.to(torch.bfloat16).float()
        print(f"오차  : max_abs={(got - ref).abs().max().item():.6g} "
              f"rel_l2={((got - ref).norm() / ref.norm()).item():.4%}")


if __name__ == "__main__":
    main()
