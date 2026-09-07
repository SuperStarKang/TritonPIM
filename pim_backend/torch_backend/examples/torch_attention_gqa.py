# 실험 B: KV group 하나의 GQA attention. G개의 query head를 M축에 쌓아
# 두 _int_mm(QKᵀ, PV)을 PIM에서 돌린다. 나머지(transpose, scale, mask, softmax,
# quantization)는 CPU Cpp다.
#
# 실행:
#   conda activate pim_triton
#   python torch_attention_gqa.py --model llama3-70b --seq 1024 --causal -v
#
# 그래프 모양은 torch_attention.py와 같다. 다른 점은 입력 shape뿐이다.
#
#   q    [G·seq, D]   G = q_heads // kv_heads (KV head 하나를 공유하는 query head 수)
#   k, v [seq, D]
#   mask [G·seq, seq] bool. 행 i는 head i//seq의 query 위치 i%seq이고,
#                          causal이면 그 위치보다 뒤의 key를 막는다.
#
# GEMM은 QKᵀ (M,N,K) = (G·seq, seq, D), PV = (G·seq, D, seq)가 되어 head 하나일
# 때보다 M축 tile이 G배 많다. head마다 따로 부르는 것보다 device가 찬다.
#
# 검증: 같은 함수의 eager 결과와 정수 출력은 정확 일치, float 출력은 assert_close.
# 추가로 float attention 대비 상대 오차(int8 양자화 오차)를 출력한다.
#
# 실행 계획(block/group/tile/p_*)은 여기서 정하지 않는다. QK와 PV는 shape가 달라
# 각각 다른 config를 받는다. -v로 두 GEMM에 선택된 계획이 로그로 찍힌다.

import argparse
import logging
import math
import sys
from pathlib import Path

# pim_backend를 sys.path에 얹어 torch_backend를 package로 import한다
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from torch_backend import PIMInductorBackend, PIMTorchConfig  # noqa: E402

# name: (q_heads, kv_heads, head_dim). attention_compare.py의 표에 MHA인
# llama2-7b를 더했다 (G = 1이면 실험 A와 같은 모양이다).
MODELS = {
    "llama2-7b":   (32, 32, 128),
    "llama3-8b":   (32, 8, 128),
    "llama3-70b":  (64, 8, 128),
    "llama2-70b":  (64, 8, 128),
    "qwen2.5-72b": (64, 8, 128),
    "bert-large":  (16, 16, 64),
}


def attention_group(q, k, v, mask, head_dim):
    """KV group 하나. q[G·seq, D], k[seq, D], v[seq, D], mask[G·seq, seq]."""
    # PIM 데이터 배치가 row-major 연속을 전제로 하므로 transpose 결과를
    # 명시적으로 contiguous로 만든다.
    kt = k.transpose(0, 1).contiguous()

    scores = torch._int_mm(q, kt)                       # PIM

    logits = scores.float() / math.sqrt(head_dim)
    logits = logits.masked_fill(mask, float("-inf"))
    p = torch.softmax(logits, dim=-1)

    # P는 확률이라 행마다 최댓값으로 정규화해 int8로 양자화한다. 결과는
    # row_scale / 127 을 곱해 되돌린다.
    row_scale = p.amax(dim=-1, keepdim=True)
    pq = (p / row_scale * 127).round().clamp(0, 127).to(torch.int8)

    output = torch._int_mm(pq, v)                       # PIM

    return output, scores, pq, row_scale


def attention_float(q, k, v, mask, head_dim):
    """float attention. 양자화 오차를 재는 기준."""
    logits = q.float() @ k.float().t() / math.sqrt(head_dim)
    logits = logits.masked_fill(mask, float("-inf"))
    return torch.softmax(logits, dim=-1) @ v.float()


def causal_mask(group, seq):
    """[G·seq, seq] bool. True = 막는다. head 블록마다 같은 상삼각이 반복된다."""
    pos = torch.arange(seq)
    block = pos[:, None] < pos[None, :]                 # [seq, seq]
    return block.repeat(group, 1)


def make_inputs(group, seq, head_dim, causal, seed=0):
    torch.manual_seed(seed)
    q = torch.randint(-8, 8, (group * seq, head_dim), dtype=torch.int8)
    k = torch.randint(-8, 8, (seq, head_dim), dtype=torch.int8)
    v = torch.randint(-8, 8, (seq, head_dim), dtype=torch.int8)
    if causal:
        mask = causal_mask(group, seq)
    else:
        mask = torch.zeros(group * seq, seq, dtype=torch.bool)
    return q, k, v, mask


def compile_attention(config):
    return torch.compile(
        attention_group,
        backend=PIMInductorBackend(config),
        fullgraph=True,
        dynamic=False,
    )


def verify(got, q, k, v, mask, head_dim):
    """compiled 결과를 eager 결과, float attention과 대조한다."""
    ref = attention_group(q, k, v, mask, head_dim)
    for name, a, b in zip(("output", "scores", "pq", "row_scale"), got, ref):
        if a.dtype.is_floating_point:
            torch.testing.assert_close(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        print(f"  {name:9s} ok  {tuple(a.shape)} {a.dtype}")

    output, _, _, row_scale = got
    fp = attention_float(q, k, v, mask, head_dim)
    dequant = output.float() * row_scale / 127
    rel = ((dequant - fp).abs().max() / fp.abs().max()).item()
    print(f"  float attention 대비 상대 오차 {rel:.4f} (int8 양자화)")


def resolve_shape(args):
    q_heads, kv_heads, head_dim = MODELS[args.model]
    group = args.group if args.group else q_heads // kv_heads
    head_dim = args.head_dim if args.head_dim else head_dim
    return q_heads, kv_heads, group, head_dim


def add_shape_args(parser):
    parser.add_argument("-m", "--model", default="llama3-70b",
                        choices=sorted(MODELS))
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--causal", action="store_true",
                        help="상삼각 causal mask. 없으면 전부 허용")
    parser.add_argument("--group", type=int, default=None,
                        help="M축에 쌓을 query head 수 G. 기본은 q_heads // kv_heads")
    parser.add_argument("--head-dim", type=int, default=None,
                        help="모델 표의 head_dim을 덮어쓴다")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="QK/PV 각각에 선택된 실행 계획을 찍는다")


def main():
    parser = argparse.ArgumentParser(
        description="실험 B: KV group 하나의 GQA attention을 torch.compile → PIM으로")
    add_shape_args(parser)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("torch_backend").setLevel(logging.INFO)

    q_heads, kv_heads, group, head_dim = resolve_shape(args)
    rows = group * args.seq
    print(f"model={args.model} q_heads={q_heads} kv_heads={kv_heads} "
          f"head_dim={head_dim} seq={args.seq} causal={args.causal}")
    print(f"KV group 하나: query head {group}개를 M축에 쌓는다 → M={rows}")
    print(f"  QK^T (M,N,K) = ({rows}, {args.seq}, {head_dim})")
    print(f"  P.V  (M,N,K) = ({rows}, {head_dim}, {args.seq})")

    q, k, v, mask = make_inputs(group, args.seq, head_dim, args.causal)
    compiled = compile_attention(PIMTorchConfig())

    got = compiled(q, k, v, mask, head_dim)
    verify(got, q, k, v, mask, head_dim)
    print("OK")


if __name__ == "__main__":
    main()
