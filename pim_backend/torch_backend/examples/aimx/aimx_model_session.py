"""AiMX model session: resident projection weights and FPGA-batched head GEMMs.

Two blocks, each with a stationary FP32 projection and two batched head stages
(QK^T and scores @ heads, four instances each). Runs on one or more AiMX FPGAs;
with one FPGA the four instances run as four batch rounds.
--capacity-mib lowers the per-FPGA capacity the planner sees, to exercise P-2 (one layer resident).
--autotune-cache selects each stage's config from measured records
(examples/autotune_torch.py --target aimx --dtype float32) instead of the fixed configs.
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from torch_backend import PIMSession, PIMTorchConfig, target_spec
from torch_backend._autotuner import PIMConfig


HEADS = 4


class HeadBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.randn(512, 512) * 0.05)

    def forward(self, hidden):
        projected = torch.mm(hidden, self.weight)
        heads = projected.reshape(256, HEADS, 128).transpose(0, 1).contiguous()
        scores = [torch.mm(heads[h], heads[h].t().contiguous()) for h in range(HEADS)]
        mixed = torch.stack([torch.mm(scores[h], heads[h]) for h in range(HEADS)]) * 0.25
        return mixed.transpose(0, 1).reshape(256, 512).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpgas", type=int, default=2)
    parser.add_argument("--capacity-mib", type=int, default=0,
                        help="per-FPGA capacity seen by the planner; 0 keeps the device heap")
    parser.add_argument("--autotune-cache", default=None,
                        help="autotune cache JSON; when given, configs come from its records")
    args = parser.parse_args()

    torch.manual_seed(0)
    blocks = [HeadBlock().eval(), HeadBlock().eval()]
    hidden = torch.randn(256, 512) * 0.1
    # projection은 한 round여야 한다. FPGA가 둘 이상이면 M을 둘로 나누고, 하나면 tile_m=2로 담는다.
    p_m = min(args.fpgas, 2)
    configs = {
        # projection: weight stage, one round
        (256, 512, 512): PIMConfig(128, 128, 128, 1, 4, 4, 2 // p_m, 1, 1, p_m, 1, 1),
        # QK^T: four instances, one FPGA each, p_b = min(4, fpgas)
        (256, 256, 128): PIMConfig(128, 128, 128, 2, 2, 1, 1, 1, 1, 1, 1, 1),
        # scores @ heads: four instances, N = 128 is padded to the AiMX width
        (256, 128, 256): PIMConfig(128, 128, 128, 2, 1, 2, 1, 1, 1, 1, 1, 1),
    }

    def configure(gemms, spec):
        if args.autotune_cache:
            return {gemm.node: PIMTorchConfig(spec=spec, autotune_cache_path=args.autotune_cache)
                    for gemm in gemms}
        return {gemm.node: PIMTorchConfig(spec=spec, config=configs[gemm.mnk])
                for gemm in gemms}

    options = {"num_units": args.fpgas}
    if args.capacity_mib:
        options["bank_capacity_limit_bytes"] = args.capacity_mib * (1 << 20) // 256
    with PIMSession(target_spec("aimx", **options)) as session:
        start = time.perf_counter()
        plans = session.prepare(blocks, (hidden,), configure=configure)
        print(f"prepare_ms={(time.perf_counter() - start) * 1000:.3f} "
              f"policy={session.residency.policy} "
              f"bytes_per_fpga={session.residency.bytes_per_unit} "
              f"capacity={session.residency.capacity_bytes}")
        for plan in plans:
            print(f"layer={plan.index} backend_plan={plan.backend_plan} stages=" + str([
                (stage.name, stage.instance_count, stage.p_b, stage.batch_rounds,
                 stage.config.key())
                for stage in plan.stages]))
        for invocation in range(2):
            if invocation:
                hidden.copy_(torch.randn(hidden.shape) * 0.1)
            reference = hidden
            for block in blocks:
                reference = block(reference)
            before = session.stats()
            start = time.perf_counter()
            output = session.run_model(hidden)
            elapsed = (time.perf_counter() - start) * 1000
            after = session.stats()
            error = (output - reference).abs().max().item()
            print(f"invocation={invocation} max_abs={error:.4g} latency_ms={elapsed:.3f} "
                  f"weight_uploads={after['weight_uploads'] - before['weight_uploads']} "
                  f"state_releases={after['state_releases'] - before['state_releases']} "
                  f"invocations={after['invocations'] - before['invocations']} "
                  f"live_bytes={after['live_bytes']}")
    print("aimx model session: executed (accuracy not checked)")


if __name__ == "__main__":
    main()
