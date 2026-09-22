"""Two projections share the whole DPU set and one activation/output workspace."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from autotuner import PIMConfig
from torch_backend import PIMTorchConfig, ResidentGemmConfig, prepare_block, target_spec


class Projections(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight_q", torch.arange(1024).reshape(32, 32).remainder(7).to(torch.int8))
        self.register_buffer("weight_k", torch.arange(1024).reshape(32, 32).remainder(11).to(torch.int8))

    def forward(self, activation):
        return (torch._int_mm(activation, self.weight_q),
                torch._int_mm(activation, self.weight_k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orientation", choices=("normal", "dual"), required=True)
    args = parser.parse_args()

    if args.orientation == "normal":
        hierarchy = PIMConfig(2, 16, 16, 1, 1, 1, 1, 1, 2, 1, 2, 1)
        padded_shape = (2, 32, 32)
        tasklets = 2
    else:
        hierarchy = PIMConfig(16, 2, 16, 1, 1, 1, 1, 2, 2, 2, 1, 1)
        padded_shape = (4, 32, 32)
        tasklets = 16

    model = Projections().eval()
    activation = torch.arange(32).reshape(1, 32).remainder(5).to(torch.int8)
    exported = torch.export.export(model, (activation,))
    config = ResidentGemmConfig(
        torch_config=PIMTorchConfig(
            spec=target_spec("upmem", num_units=2, nr_tasklets=tasklets), config=hierarchy),
        orientation=args.orientation,
        padded_shape=padded_shape,
    )
    configs = {node.name: config for node in exported.graph_module.graph.nodes
               if node.op == "call_function" and node.target == torch.ops.aten._int_mm.default}

    with prepare_block(exported, (activation,), configs=configs, dpu_count=2) as block:
        print(f"orientation={args.orientation} DPUs=2 tasklets={tasklets}")
        print(f"prepare_weight_bytes={block.stats()['weight_bytes']}")
        for region in block.layout:
            print(region)
        for step in range(2):
            inputs = (activation + step).remainder(5).contiguous()
            output = block.run(inputs)
            torch.testing.assert_close(output, model(inputs), rtol=0, atol=0)
            assert block.last_transfers["weight_bytes"] == 0
            assert block.last_transfers["launches"] == 2
            print(f"step={step} correctness=PASS "
                  f"weight_h2d={block.last_transfers['weight_bytes']} "
                  f"activation_h2d={block.last_transfers['activation_bytes']} "
                  f"activation_hits={block.last_transfers['activation_hits']}")


if __name__ == "__main__":
    main()
