"""LayerPlan transformer benchmark with the baseline's seven resident projection weights."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention.int8_workloads import HEADS, HIDDEN, SEQUENCE
from model_benchmark_common import add_benchmark_arguments, benchmark_model
from transformer.int8_workloads import TransformerBlock


class PaddedFFNProjection(torch.nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.register_buffer("weight", torch.nn.functional.pad(projection.weight, (0, 256)))
        self.output_scale = projection.output_scale
        self.out_features = projection.out_features

    def forward(self, activation):
        output = torch._int_mm(activation.reshape(-1, activation.shape[-1]), self.weight)
        # 물리 11264열(zero padding)로 계산하고 논리 11008열만 반환한다.
        output = output[:, :self.out_features].contiguous().float() * self.output_scale
        return output.reshape(*activation.shape[:-1], self.out_features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, Path(__file__).with_name("model_transformer_block_report"))
    args = parser.parse_args()
    torch.manual_seed(0)
    model = TransformerBlock()
    model.gate_proj = PaddedFFNProjection(model.gate_proj)
    model.up_proj = PaddedFFNProjection(model.up_proj)
    model.eval()
    inputs = (torch.randn(1, SEQUENCE, HIDDEN),)
    benchmark_model(model, inputs, args, scope="transformer_block",
                    heads=HEADS, expected_gemms=71, expected_weights=7)


if __name__ == "__main__":
    main()
