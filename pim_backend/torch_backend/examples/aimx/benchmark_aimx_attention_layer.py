"""FP32 Q/K/V projections, RoPE, 32-head QK/PV and O projection on AiMX FPGAs (--fpgas).

--qk-heads / --pv-heads pack that many heads into one block-diagonal GEMM
(PackedAttentionLayer); 1 and 1 run the per-head AttentionLayer.
"""

import argparse
from pathlib import Path

import torch

from aimx_benchmark_common import add_benchmark_arguments, benchmark_model
from fp32_workloads import HEADS, HIDDEN, SEQUENCE, AttentionLayer, PackedAttentionLayer


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, Path(__file__).with_name("aimx_attention_layer_report"))
    parser.add_argument("--qk-heads", type=int, default=1, help="heads per QK^T GEMM")
    parser.add_argument("--pv-heads", type=int, default=1, help="heads per PV GEMM")
    args = parser.parse_args()
    torch.manual_seed(0)
    if (args.qk_heads, args.pv_heads) == (1, 1):
        model = AttentionLayer()
    else:
        model = PackedAttentionLayer(args.qk_heads, args.pv_heads)
    benchmark_model(
        model.eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args,
        scope="attention_layer",
        expected_gemms=4 + HEADS // args.qk_heads + HEADS // args.pv_heads, expected_weights=4,
    )


if __name__ == "__main__":
    main()
