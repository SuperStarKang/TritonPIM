"""Check FX-normalized FP32 GEMMs and weight residency on the full DPU set."""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from autotuner import PIMConfig
from torch_backend import PIMTorchConfig, ResidentGemmConfig, prepare_block, target_spec
from torch_backend.fx_analysis import analyze_fx_gemms


class FP32Gemms(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(64, 1024)
        self.register_buffer("weight", torch.randn(64, 1024) / 8)
        self.register_buffer("bias", torch.randn(1024))

    def forward(self, activation, right):
        return (
            self.projection(activation),
            torch.addmm(self.bias, activation, self.weight, alpha=0.5, beta=2.0),
            torch.mm(activation, right),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation", choices=("normal", "dual"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    config = ResidentGemmConfig(
        torch_config=PIMTorchConfig(
            spec=target_spec("upmem", num_units=2048, nr_tasklets=16),
            config=PIMConfig(
                block_m=16,
                block_n=16,
                block_k=32,
                group_m=1,
                group_n=1,
                group_k=1,
                tile_m=1,
                tile_n=1,
                tile_k=2,
                p_m=32 if args.orientation == "normal" else 64,
                p_n=64 if args.orientation == "normal" else 32,
                p_k=1,
            ),
        ),
        orientation=args.orientation,
        padded_shape=(512, 1024, 64),
    )

    with torch.no_grad():
        model = FP32Gemms().eval()
        activation = torch.randn(512, 64)
        right = torch.randn(64, 1024) / 8
        exported = torch.export.export(model, (activation, right))
        gemms = analyze_fx_gemms(exported)
        assert len(gemms) == 3
        assert [gemm.op for gemm in gemms] == [
            "aten.linear.default", "aten.addmm.default", "aten.mm.default",
        ]
        assert [gemm.right.weight_target for gemm in gemms] == [
            "projection.weight", "weight", None,
        ]
        for gemm in gemms:
            print(f"{gemm.node}: {gemm.op} mnk={gemm.mnk} "
                  f"B_weight={gemm.right.weight_target}")

        with prepare_block(
            exported, (activation, right),
            configs={gemm.node: config for gemm in gemms}, dpu_count=2048,
        ) as block:
            prepared = block.stats()
            weight_regions = [region for region in block.layout if region["role"] == "weight"]
            assert len(weight_regions) == 2
            assert {region["kernel"] for region in weight_regions} == {
                gemm.node for gemm in gemms if gemm.right.weight_target is not None
            }
            assert prepared["weight_bytes"] == 2048 * sum(
                region["bytes"] for region in weight_regions)
            assert prepared["weight_bytes"] > 0
            assert prepared["activation_bytes"] == 0
            assert prepared["output_bytes"] == 0
            assert prepared["launches"] == 0

            print(f"orientation={args.orientation} DPUs=2048 tasklets=16 dtype=float32")
            print(f"prepare_weight_bytes={prepared['weight_bytes']}")
            for region in block.layout:
                print(region)

            for step in range(2):
                inputs = (activation + step * 0.125, right - step * 0.25)
                output = block.run(*inputs)
                torch.testing.assert_close(output, model(*inputs), rtol=1e-4, atol=1e-4)
                assert block.last_transfers["weight_bytes"] == 0
                assert block.last_transfers["activation_bytes"] > 0
                assert block.last_transfers["output_bytes"] > 0
                assert block.last_transfers["launches"] == 3
                assert block.last_transfers["activation_hits"] == 0
                assert block.stats()["weight_bytes"] == prepared["weight_bytes"]
                print(f"step={step} correctness=PASS "
                      f"weight_h2d={block.last_transfers['weight_bytes']} "
                      f"activation_h2d={block.last_transfers['activation_bytes']} "
                      f"activation_hits={block.last_transfers['activation_hits']} "
                      f"launches={block.last_transfers['launches']}")


if __name__ == "__main__":
    main()
