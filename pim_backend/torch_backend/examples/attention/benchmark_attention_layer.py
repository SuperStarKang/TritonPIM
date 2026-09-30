"""INT8 Q/K/V projections + RoPE + 32 attention heads + O projection, batch 1 / S=1024."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention.int8_workloads import HEADS, HIDDEN, SEQUENCE, AttentionLayer
from benchmark_common import benchmark, parse_args


def main():
    args = parse_args(__doc__, Path(__file__).with_name("attention_layer_report"))
    torch.manual_seed(0)
    benchmark(AttentionLayer().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args,
              scope="attention_layer", heads=HEADS, expected_gemms=68, expected_weights=4)


if __name__ == "__main__":
    main()
