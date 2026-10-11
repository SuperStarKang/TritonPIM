# torch.compile 경로를 실측해 autotune cache를 채운다 (설계 §10 단계 B~C).
#
# [왜 lowering 밖에서 재는가]
# Inductor lowering은 fake tensor를 다루므로 그 안에서는 잴 입력이 없다. 하지만
# 튜닝이 lowering 안에서 일어날 이유가 없다 — 바깥에는 실제 텐서가 있다. config
# 하나마다 torch.compile을 새로 만들어 재고, 그 측정을 autotuner의
# benchmark_fn으로 넘기면 된다.
#
# 이 스크립트는 Inductor wrapper와 layout 처리까지 포함한 Torch 경로 자체를 잰다.
# 직접 Triton launch의 측정과 objective가 다르므로 같은 cache 파일 안에서도
# Torch template version으로 별도 context를 사용한다.
#
# 실행:
#   conda activate pim_triton
#   python autotune_torch.py --m 8192 --n 8192 --k 128      # attention QK (UPMEM, INT8)
#   python autotune_torch.py --m 8192 --n 128  --k 8192     # attention P.V
#   python autotune_torch.py --target aimx --num-units 1 --dtype float32 --m 256 --n 512 --k 512
#
# PIMSession 모델 경로는 PIMTorchConfig(autotune_cache_path=...)로 이 측정 기록을 읽는다.
# target spec(unit 수, 용량)이 측정 때와 같아야 같은 기록을 찾는다.
#
# 직접 예제와 cache 파일은 공유할 수 있지만 template version이 context key에
# 포함되므로 각 경로의 측정과 best는 서로 덮어쓰지 않는다.

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import triton.backends.pim.driver as pim_driver

from torch_backend import PIMInductorBackend, PIMTorchConfig
from torch_backend.template import TORCH_GEMM_TEMPLATE_VERSION
from torch_backend._autotuner import (
    DEFAULT_CACHE_PATH,
    PIMConfigCache,
    PIMProblem,
    build_config_space,
    target_spec,
)

from autotuner import PIMBenchmarkResult, SearchBudget, tune


def pim_int_mm(a, b):
    return torch._int_mm(a, b)


def pim_mm(a, b):
    return torch.mm(a, b)


def sampled_reference(a, b, limit=32):
    """정답 검증 기준 — 출력의 균등 표본만 작은 참조 GEMM으로 계산한다.

    전체 참조는 (M,N,K)=(8192,8192,128)에서 8.6 GMAC이라 config마다 돌리기에
    무겁다. 표본은 결정론적이라 config 사이 비교가 성립한다."""
    M, _ = a.shape
    _, N = b.shape
    rows = torch.arange(0, M, max(1, -(-M // limit)))
    cols = torch.arange(0, N, max(1, -(-N // limit)))
    block = (a.index_select(0, rows).to(torch.int32)
             @ b.index_select(1, cols).to(torch.int32))
    return rows, cols, block


def main():
    parser = argparse.ArgumentParser(
        description="torch.compile → PIM 경로를 실측해 autotune cache를 채운다")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--reps", type=int, default=3,
                        help="config당 측정 반복 수 (median)")
    parser.add_argument("--max-benchmarks", type=int, default=30,
                        help="실측할 config 수 상한")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--retune", action="store_true",
                        help="cache에 있어도 다시 탐색한다")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=("upmem", "aimx"), default="upmem")
    parser.add_argument("--num-units", type=int, default=0,
                        help="0이면 target 기본값 (UPMEM 2048, AiMX는 보이는 FPGA 수)")
    parser.add_argument("--dtype", choices=("int8", "float32"), default="int8",
                        help="int8: _int_mm -> int32, float32: mm -> float32")
    args = parser.parse_args()

    target = target_spec(args.target, num_units=args.num_units)
    torch.manual_seed(args.seed)
    if args.dtype == "int8":
        problem = PIMProblem(M=args.m, N=args.n, K=args.k,
                             input_dtype="int8", output_dtype="int32")
        operation = pim_int_mm
        a = torch.randint(-8, 8, (args.m, args.k), dtype=torch.int8)
        b = torch.randint(-8, 8, (args.k, args.n), dtype=torch.int8)
        rows, cols, ref_block = sampled_reference(a, b)
    else:
        # 실수 경로는 정확도를 검사하지 않는다 (correctness=None). 시간만 비교한다.
        problem = PIMProblem(M=args.m, N=args.n, K=args.k,
                             input_dtype="float32", output_dtype="float32")
        operation = pim_mm
        a = torch.randn(args.m, args.k)
        b = torch.randn(args.k, args.n)

    base = PIMTorchConfig(spec=target)

    def benchmark_fn(problem, target, config):
        # config마다 새로 컴파일한다. reset이 없으면 dynamo가 앞 config로
        # 컴파일해 둔 것을 그대로 돌려준다 — 입력 shape가 같기 때문이다.
        torch._dynamo.reset()
        compiled = torch.compile(
            operation,
            backend=PIMInductorBackend(replace(base, config=config)),
            fullgraph=True,
            dynamic=False,
        )
        # 첫 호출은 compile(TTIR→PIM artifact 빌드 포함)이라 측정에서 뺀다.
        out = compiled(a, b)
        samples = []
        stage_samples = []
        for _ in range(args.reps):
            start = time.perf_counter_ns()
            out = compiled(a, b)
            samples.append((time.perf_counter_ns() - start) / 1e6)
            stage_samples.append(tuple(pim_driver.timing))

        timings = tuple(
            (name, statistics.median(
                sample[stage] for sample in stage_samples))
            for stage, name in enumerate(pim_driver.TIMING_STAGES)
        )
        correctness = None
        if args.dtype == "int8":
            got = out.index_select(0, rows).index_select(1, cols)
            correctness = bool(torch.equal(got, ref_block))
        return PIMBenchmarkResult(
            total_ms=statistics.median(samples),
            correctness=correctness,
            timings=timings,
        )

    space = build_config_space(problem, target)
    cache = PIMConfigCache(
        args.cache,
        template_version=TORCH_GEMM_TEMPLATE_VERSION,
    )
    budget = SearchBudget(max_generations=args.max_generations,
                          max_benchmarks=args.max_benchmarks,
                          max_autotune_seconds=args.max_seconds)

    print(f"problem (M,N,K)=({args.m},{args.n},{args.k}) dtype={args.dtype} "
          f"target={args.target} num_units={target.common_profile.num_units}")
    print(f"cache: {args.cache}")

    result = tune(problem, target, benchmark_fn, space=space, cache=cache,
                  budget=budget, seed=args.seed, retune=args.retune,
                  verbose=True)

    best = result.best
    c = best.config
    print(f"\nbest {best.result.total_ms:.3f} ms")
    print(f"  block=({c.block_m},{c.block_n},{c.block_k}) "
          f"group=({c.group_m},{c.group_n},{c.group_k}) "
          f"tile=({c.tile_m},{c.tile_n},{c.tile_k})")
    print(f"  execution_extent={c.execution_extent} "
          f"devices=({c.p_m},{c.p_n},{c.p_k})")
    print(f"\ntorch 경로에서 이 결과를 쓰려면:")
    print(f"  PIMTorchConfig(autotune_cache_path={args.cache!r})")


if __name__ == "__main__":
    main()
