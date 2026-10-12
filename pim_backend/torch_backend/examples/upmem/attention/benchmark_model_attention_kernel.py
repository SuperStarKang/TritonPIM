"""Compare prepared LayerPlan attention at one or 32 heads on 2048 UPMEM DPUs."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention.int8_workloads import HEAD_DIM, SEQUENCE, AttentionKernel
from model_benchmark_common import add_benchmark_arguments, benchmark_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, Path(__file__).with_name("model_attention_kernel_report"))
    parser.set_defaults(gemm_config="baseline")
    parser.add_argument("--heads", type=int, choices=(1, 32), default=1)
    args = parser.parse_args()
    torch.manual_seed(0)
    inputs = tuple(torch.randint(-8, 8, (args.heads, SEQUENCE, HEAD_DIM), dtype=torch.int8)
                   for _ in range(3))
    benchmark_model(AttentionKernel(args.heads).eval(), inputs, args, scope="attention_kernel",
                    heads=args.heads, expected_gemms=2 * args.heads, expected_weights=0)


if __name__ == "__main__":
    main()
