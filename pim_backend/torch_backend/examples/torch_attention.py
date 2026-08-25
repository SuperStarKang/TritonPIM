# 두 번째 검증: 한 그래프 안의 _int_mm 두 개(QK, PV)를 PIM에서 돌린다.
# 나머지(transpose, scale, mask, softmax, quantization, partial 합)는 CPU Cpp다.
#
# 실행:
#   conda activate pim_triton
#   python torch_attention.py --seq 256 --kv 256 --head-dim 128 --column-partitions <p_n>
#
# 예상 배치:
#   transpose        CPU Cpp
#   QK _int_mm       PIM
#   QK partial sum   CPU Cpp
#   scale/mask       CPU Cpp
#   softmax          CPU Cpp
#   quantization     CPU Cpp
#   PV _int_mm       PIM
#   PV partial sum   CPU Cpp
#
# KV group loop는 아직 torch.compile 밖에 둔다.

import argparse
import math
import sys
from pathlib import Path

# pim_backend를 sys.path에 얹어 torch_backend를 package로 import한다
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from torch_backend import PIMInductorBackend, PIMTorchConfig  # noqa: E402

from torch_int_mm import build_config  # noqa: E402


def attention_group_inductor(q, k, v, causal_mask, head_dim):
    # PIM 데이터 배치가 row-major 연속을 전제로 하므로 transpose 결과를
    # 명시적으로 contiguous로 만든다.
    kt = k.transpose(0, 1).contiguous()

    scores = torch._int_mm(q, kt)

    logits = scores.float() / math.sqrt(head_dim)
    logits = logits.masked_fill(causal_mask, float("-inf"))

    p = torch.softmax(logits, dim=-1)

    row_scale = p.amax(dim=-1, keepdim=True)
    pq = (p / row_scale * 127).round().clamp(0, 127).to(torch.int8)

    output = torch._int_mm(pq, v)

    return output, scores, pq, row_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=256)       # M
    parser.add_argument("--kv", type=int, default=256)        # N (= PV의 K)
    parser.add_argument("--head-dim", type=int, default=128)  # QK의 K
    parser.add_argument("--column-partitions", type=int, required=True,
                        help="output_column_tile device 분할 수 (p_n). "
                             "QK와 PV의 N이 다르므로 둘 다 만족하는 값을 준다")
    args = parser.parse_args()

    config = build_config(args.column_partitions)
    divisor = config.split_k * config.block_k
    for name, k in (("head_dim (QK의 K)", args.head_dim), ("kv (PV의 K)", args.kv)):
        if k % divisor:
            raise SystemExit(
                f"{name}={k} must be divisible by SPLIT_K*BLOCK_K={divisor}")

    torch.manual_seed(0)
    q = torch.randint(-8, 8, (args.seq, args.head_dim), dtype=torch.int8)
    k = torch.randint(-8, 8, (args.kv, args.head_dim), dtype=torch.int8)
    v = torch.randint(-8, 8, (args.kv, args.head_dim), dtype=torch.int8)
    causal_mask = torch.zeros(args.seq, args.kv, dtype=torch.bool)

    compiled = torch.compile(
        attention_group_inductor,
        backend=PIMInductorBackend(config),
        fullgraph=True,
        dynamic=False,
    )

    got = compiled(q, k, v, causal_mask, args.head_dim)
    ref = attention_group_inductor(q, k, v, causal_mask, args.head_dim)

    for name, a, b in zip(("output", "scores", "pq", "row_scale"), got, ref):
        if a.dtype.is_floating_point:
            torch.testing.assert_close(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        print(f"{name:9s} ok  {tuple(a.shape)} {a.dtype}")
    print("OK")


if __name__ == "__main__":
    main()
