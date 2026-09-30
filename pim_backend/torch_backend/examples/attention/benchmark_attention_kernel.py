"""INT8 single-head QK^T -> causal softmax -> PV performance, batch 1 / S=1024 / D=128."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention.int8_workloads import HEAD_DIM, SEQUENCE, AttentionKernel
from benchmark_common import benchmark, parse_args


def main():
    args = parse_args(__doc__, Path(__file__).with_name("attention_kernel_report"))
    torch.manual_seed(0)
    inputs = tuple(torch.randint(-8, 8, (1, SEQUENCE, HEAD_DIM), dtype=torch.int8) for _ in range(3))
    benchmark(AttentionKernel(1).eval(), inputs, args, scope="attention_kernel", heads=1,
              expected_gemms=2, expected_weights=0)


if __name__ == "__main__":
    main()
