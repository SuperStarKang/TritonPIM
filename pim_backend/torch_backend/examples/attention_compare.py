# PIM-LLM(CGO'25)의 attention kernel과 같은 shape로 torch → PIM 경로를 잰다.
#
# 그쪽 scripts/run_attention_kernel.py가 head 하나를 GEMM 둘로 본다:
#     QK^T :  m=seq, k=head_dim, n=seq
#     P.V  :  m=seq, k=seq,      n=head_dim
# 같은 (b,m,k,n)으로 우리 경로를 재고, 같은 분할로 app2를 돌릴 명령을 함께 낸다.
#
# [왜 attention 전체가 아니라 GEMM 단위인가]
# 그쪽 total은 GEMM 하나다. 우리 attention e2e에는 softmax·quantization CPU 작업이
# 섞이므로 그대로 비교하면 다른 것을 비교하게 된다.
#
# [무엇이 측정에 들어가는가]
# 우리 쪽 e2e는 compiled(a,b) 한 번이고, 그 안에 partial zero 채우기(CPU),
# PIM launch, partial 합(CPU)이 모두 들어간다. PIM launch의 단계별 시간은
# pim_driver.timing이 준다 — 마지막 launch의 것이므로 GEMM 하나짜리 그래프에서만
# 의미가 있다(그래서 attention 전체가 아니라 GEMM 단위로 잰다).
#
# 실행:
#   python attention_compare.py                       # llama3-70b, seq=8192
#   python attention_compare.py --seq 2048 --op qk
#   python attention_compare.py --cache ~/.pim_autotune_cache.json

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import triton.backends.pim.driver as pim_driver  # noqa: E402

from torch_backend import PIMInductorBackend, PIMTorchConfig  # noqa: E402
from torch_backend._autotuner import (  # noqa: E402
    PIMConfigCache, PIMProblem, target_spec)
from torch_backend.execution import resolve_execution  # noqa: E402

# name: (q_heads, kv_heads, head_dim) — run_attention_kernel.py의 표와 같다
MODELS = {
    "llama3-8b":   (32, 8, 128),
    "llama3-70b":  (64, 8, 128),
    "llama2-70b":  (64, 8, 128),
    "qwen2.5-72b": (64, 8, 128),
    "bert-large":  (16, 16, 64),
}


def pim_mm(a, b):
    return torch._int_mm(a, b)


def measure(config, m, k, n, reps, verify):
    """(M,K)@(K,N) 하나를 torch 경로로 재고 breakdown을 돌려준다."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
    b = torch.randint(-8, 8, (k, n), dtype=torch.int8)

    compiled = torch.compile(pim_mm, backend=PIMInductorBackend(config),
                             fullgraph=True, dynamic=False)
    out = compiled(a, b)                    # compile + PIM artifact 빌드
    if verify:
        ref = a.to(torch.int32) @ b.to(torch.int32)
        ok = bool(torch.equal(out, ref))
    else:
        ok = None

    e2e = []
    stages = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        compiled(a, b)
        e2e.append((time.perf_counter_ns() - start) / 1e6)
        stages.append(list(pim_driver.timing))

    mean_e2e = sum(e2e) / len(e2e)
    mean_stage = [sum(s[i] for s in stages) / len(stages)
                  for i in range(len(pim_driver.TIMING_STAGES))]
    return mean_e2e, mean_stage, ok


def app2_command(m, k, n, p_m, p_n, p_k, reps):
    """같은 분할로 PIM-LLM app2를 돌리는 명령.

    그쪽 Tm/Tk/Tn은 tile 크기가 아니라 분할 수다(tiling_parameter_prediction.py의
    splitM/splitK/splitN). splitM은 그쪽 탐색이 1로 고정하고 M 병렬은
    dpus_per_Ntile = nr_dpus/(splitN*splitK*batch) 로 얻으므로, 우리 P_M이
    거기에 대응한다."""
    devices = p_m * p_n * p_k
    return (f"$PGEMMLIB_BIN -d {devices} -b 1 -m {m} -k {k} -n {n} "
            f"-x 1 -y {p_k} -z {p_n} -t 0 -r {reps} -v 0")


def main():
    parser = argparse.ArgumentParser(
        description="PIM-LLM attention kernel과 같은 shape로 torch → PIM 경로를 잰다")
    parser.add_argument("-m", "--model", default="llama3-70b")
    parser.add_argument("-s", "--seq-len", type=int, default=8192)
    parser.add_argument("--op", default="both", choices=["qk", "pv", "both"])
    parser.add_argument("-r", "--reps", type=int, default=3)
    parser.add_argument("--cache", default=None,
                        help="autotune cache 경로. 없으면 heuristic seed를 쓴다 "
                             "(성능 숫자가 아니다)")
    parser.add_argument("--verify", action="store_true",
                        help="전체 참조와 대조한다. seq=8192에서는 느리다")
    parser.add_argument("--dry-run", action="store_true",
                        help="실행 계획과 app2 명령만 낸다")
    args = parser.parse_args()

    if args.model not in MODELS:
        sys.exit(f"unknown model {args.model!r}; known: {sorted(MODELS)}")
    q_heads, kv_heads, head_dim = MODELS[args.model]
    seq = args.seq_len

    cases = [("QK^T", seq, head_dim, seq), ("P.V", seq, seq, head_dim)]
    cases = {"qk": cases[:1], "pv": cases[1:], "both": cases}[args.op]

    target = target_spec("upmem")
    config = PIMTorchConfig(spec=target,
                            autotune_cache_path=args.cache)
    print(f"model={args.model} seq={seq} head_dim={head_dim} "
          f"q_heads={q_heads} kv_heads={kv_heads}")
    print(f"benchmark scope: query head 하나. attention layer 전체 = {q_heads}배")
    print(f"config source: "
          f"{'autotune cache ' + args.cache if args.cache else 'heuristic seed (성능 숫자 아님)'}")

    for label, m, k, n in cases:
        print(f"\n=== {label}  (m,k,n)=({m},{k},{n}) ===")
        problem = PIMProblem(M=m, N=n, K=k)
        execution = resolve_execution(config, problem)
        if args.cache:
            # record가 1~2개면 튜닝이 아니라 smoke test 결과다. tune()은 끝까지
            # 돌아야 저장하므로 중단된 탐색은 아무것도 남기지 않는다.
            measured = len(PIMConfigCache(args.cache).measured_results(
                problem, target))
            note = "  ← 측정이 너무 적다. 튜닝이 끝까지 돌지 않았다" if measured <= 2 else ""
            print(f"  cache: 측정된 config {measured}개{note}")
        p_m = execution.config.p_m
        p_n = execution.config.p_n
        p_k = execution.config.p_k
        print(f"  plan: {execution.describe()}")
        partial_gib = p_k * m * n * 4 / 2**30
        print(f"  partial[{p_k},{m},{n}] int32 = {partial_gib:.2f} GiB")
        print(f"  PIM-LLM 같은 분할:\n    {app2_command(m, k, n, p_m, p_n, p_k, args.reps)}")

        if args.dry_run:
            continue

        e2e, stage, ok = measure(config, m, k, n, args.reps, args.verify)
        gops = 2 * m * k * n / e2e / 1e6

        print(f"  측정 {args.reps}회 평균"
              + (f", 정답 {'일치' if ok else '불일치'}" if ok is not None else ""))
        for name, ms in zip(pim_driver.TIMING_STAGES, stage):
            print(f"    {name:9s} {ms:9.3f} ms")
        launch = sum(stage)
        print(f"    {'--':9s} {'':>9}")
        print(f"    {'PIM 합':9s} {launch:9.3f} ms")
        print(f"    {'CPU+기타':9s} {e2e - launch:9.3f} ms   "
              f"(partial zero 채우기 + partial 합 + wrapper)")
        print(f"    {'e2e':9s} {e2e:9.3f} ms   {gops:8.2f} GOPS")


if __name__ == "__main__":
    main()
