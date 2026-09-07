# 실험 C: attention 층 전체. KV group(kv_heads개)마다 실험 B의 compiled 함수를
# Python loop로 부른다.
#
# 실행:
#   conda activate pim_triton
#   python torch_attention_layer.py --model llama3-70b --seq 1024 --causal --verify
#
# loop는 torch.compile 밖이다. group마다 입력 shape가 같으므로 Dynamo는 warm-up에서
# 컴파일한 것을 재사용하고, PIM 컴파일과 host.so 빌드도 한 번이다.
#
# 측정은 층 전체의 벽시계 시간이다. pim_driver.timing은 마지막 launch(마지막 group의
# PV)만 남기므로 stage/h2d/kernel/d2h/combine 분해는 여기서 내지 않는다. GEMM 하나의
# 단계별 시간이 필요하면 attention_compare.py --gqa 를 쓴다.

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from torch_backend import PIMTorchConfig  # noqa: E402

from torch_attention_gqa import (  # noqa: E402
    add_shape_args, causal_mask, compile_attention, resolve_shape, verify)


def make_layer_inputs(kv_heads, group, seq, head_dim, causal, seed=0):
    """층 전체 입력. 첫 차원이 KV group이라 q[g] 등은 contiguous view다."""
    torch.manual_seed(seed)
    q = torch.randint(-8, 8, (kv_heads, group * seq, head_dim), dtype=torch.int8)
    k = torch.randint(-8, 8, (kv_heads, seq, head_dim), dtype=torch.int8)
    v = torch.randint(-8, 8, (kv_heads, seq, head_dim), dtype=torch.int8)
    if causal:
        mask = causal_mask(group, seq)
    else:
        mask = torch.zeros(group * seq, seq, dtype=torch.bool)
    return q, k, v, mask


def run_layer(compiled, q, k, v, mask, head_dim):
    """kv_heads개 group을 순서대로 돈다. (전체 ms, group별 ms, 첫 group 결과)"""
    per_group = []
    first = None
    start = time.perf_counter()
    for g in range(q.shape[0]):
        t0 = time.perf_counter()
        out = compiled(q[g], k[g], v[g], mask, head_dim)
        per_group.append((time.perf_counter() - t0) * 1e3)
        if g == 0:
            first = out
    total = (time.perf_counter() - start) * 1e3
    return total, per_group, first


def main():
    parser = argparse.ArgumentParser(
        description="실험 C: attention 층 전체를 KV group 단위로 torch.compile → PIM으로")
    add_shape_args(parser)
    parser.add_argument("--reps", type=int, default=1,
                        help="층 전체를 몇 번 반복해 median을 낼지")
    parser.add_argument("--verify", action="store_true",
                        help="첫 group의 결과를 eager/float 기준과 대조한다")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("torch_backend").setLevel(logging.INFO)

    q_heads, kv_heads, group, head_dim = resolve_shape(args)
    seq = args.seq
    rows = group * seq
    print(f"model={args.model} q_heads={q_heads} kv_heads={kv_heads} "
          f"head_dim={head_dim} seq={seq} causal={args.causal}")
    print(f"KV group {kv_heads}개, group당 query head {group}개 → GEMM당 M={rows}")
    print(f"  QK^T (M,N,K) = ({rows}, {seq}, {head_dim})  × {kv_heads}")
    print(f"  P.V  (M,N,K) = ({rows}, {head_dim}, {seq})  × {kv_heads}")

    q, k, v, mask = make_layer_inputs(kv_heads, group, seq, head_dim, args.causal)
    compiled = compile_attention(PIMTorchConfig())

    # warm-up. 첫 호출에 Inductor 컴파일, PIM 컴파일, host.so 빌드, DPU 확보가 들어간다.
    t0 = time.perf_counter()
    compiled(q[0], k[0], v[0], mask, head_dim)
    print(f"\nwarm-up (compile 포함) {(time.perf_counter() - t0) * 1e3:.1f} ms")

    totals = []
    groups = []
    first = None
    for _ in range(args.reps):
        total, per_group, first = run_layer(compiled, q, k, v, mask, head_dim)
        totals.append(total)
        groups.append(per_group)

    total_ms = statistics.median(totals)
    # 층 전체 연산량: group마다 QK (rows·seq·D)와 PV (rows·D·seq) MAC.
    macs = kv_heads * 2 * rows * seq * head_dim
    gops = 2 * macs / total_ms / 1e6

    print(f"층 전체 {total_ms:.1f} ms (median of {args.reps}), "
          f"{2 * macs / 1e9:.2f} GOP, {gops:.2f} GOP/s")
    # reps가 짝수면 median은 두 값의 평균이라 totals에 없을 수 있다. 가장 가까운 회차를 고른다.
    rep = groups[min(range(len(totals)), key=lambda i: abs(totals[i] - total_ms))]
    print(f"group당  min {min(rep):.1f}  mean {statistics.mean(rep):.1f}  "
          f"max {max(rep):.1f} ms")

    if args.verify:
        print("\nverify (group 0)")
        verify(first, q[0], k[0], v[0], mask, head_dim)
    print("OK")


if __name__ == "__main__":
    main()
