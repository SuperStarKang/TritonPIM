"""FP32 Q/K/V projections, RoPE, 32-head QK/PV and O projection on AiMX FPGAs (--fpgas)."""

import argparse
from pathlib import Path

import torch

from aimx_benchmark_common import add_benchmark_arguments, benchmark_model
from fp32_workloads import HIDDEN, SEQUENCE, AttentionLayer


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, Path(__file__).with_name("aimx_attention_layer_report"))
    args = parser.parse_args()
    torch.manual_seed(0)
    benchmark_model(
        AttentionLayer().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args,
        scope="attention_layer", expected_gemms=68, expected_weights=4,
    )


if __name__ == "__main__":
    main()
