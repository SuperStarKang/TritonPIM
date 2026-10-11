"""INT8 Q/K/V projections, RoPE, batched QK/PV and O projection on UPMEM."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention.int8_workloads import HEADS, HIDDEN, SEQUENCE, AttentionLayer
from model_benchmark_common import add_benchmark_arguments, benchmark_model


def main():
    torch.set_num_threads(8)

    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, Path(__file__).with_name("model_attention_layer_report"))
    args = parser.parse_args()
    torch.manual_seed(0)
    benchmark_model(
        AttentionLayer().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args,
        scope="attention_layer", heads=HEADS, expected_gemms=68, expected_weights=4,
    )


if __name__ == "__main__":
    main()
