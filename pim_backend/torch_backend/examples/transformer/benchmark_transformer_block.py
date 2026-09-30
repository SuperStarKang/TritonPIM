"""INT8 Llama-2-7B-shaped transformer block, batch 1 / S=1024 / hidden=4096 / FFN=11008."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention.int8_workloads import HEADS, HIDDEN, SEQUENCE
from benchmark_common import benchmark, parse_args
from transformer.int8_workloads import TransformerBlock


def main():
    args = parse_args(__doc__, Path(__file__).with_name("transformer_block_report"))
    torch.manual_seed(0)
    benchmark(TransformerBlock().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args,
              scope="transformer_block", heads=HEADS, expected_gemms=71, expected_weights=7)


if __name__ == "__main__":
    main()
