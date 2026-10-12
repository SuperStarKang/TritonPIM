"""FP32 RMSNorm, attention layer, residual and SwiGLU FFN on AiMX FPGAs (--fpgas)."""

import argparse
from pathlib import Path

import torch

from aimx_benchmark_common import add_benchmark_arguments, benchmark_model
from fp32_workloads import HIDDEN, SEQUENCE, TransformerBlock


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, Path(__file__).with_name("aimx_transformer_block_report"))
    args = parser.parse_args()
    torch.manual_seed(0)
    benchmark_model(
        TransformerBlock().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args,
        scope="transformer_block", expected_gemms=71, expected_weights=7,
    )


if __name__ == "__main__":
    main()
