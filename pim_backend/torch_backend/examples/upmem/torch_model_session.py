"""Two static blocks with stationary weights and five independent head GEMMs.

Run from the configured TritonPIM environment with at least eight UPMEM DPUs.
This is an integer execution test, not an attention accuracy benchmark.
"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from triton.backends.pim.residency import LAYER

from torch_backend import PIMSession, PIMTorchConfig, target_spec
from torch_backend._autotuner import PIMConfig


class HeadBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.randint(-2, 3, (32, 32), dtype=torch.int8))

    def forward(self, hidden):
        projected = torch._int_mm(hidden.reshape(160, 32), self.weight)
        heads = projected.clamp(-3, 3).to(torch.int8).reshape(5, 32, 32)
        scores = torch.stack([torch._int_mm(heads[h], heads[h].t().contiguous())
                              for h in range(5)])
        return scores.clamp(-8, 8).to(torch.int8)


def main():
    torch.manual_seed(42)
    blocks = [HeadBlock().eval(), HeadBlock().eval()]
    hidden = torch.randint(-3, 4, (5, 32, 32), dtype=torch.int8)
    configs = {
        (160, 32, 32): PIMConfig(16, 16, 32, 1, 1, 1, 5, 1, 1, 2, 2, 1),
        (32, 32, 32): PIMConfig(16, 16, 32, 1, 1, 1, 1, 1, 1, 2, 2, 1),
    }

    def configure(gemms, spec):
        return {gemm.node: PIMTorchConfig(spec=spec, config=configs[gemm.mnk])
                for gemm in gemms}

    with PIMSession(target_spec("upmem", num_units=8, nr_tasklets=16)) as session:
        start = time.perf_counter()
        plans = session.prepare(blocks, (hidden,), configure=configure)
        print(f"prepare_ms={(time.perf_counter() - start) * 1000:.3f} "
              f"policy={session.residency.policy}")
        for plan in plans:
            print(f"layer={plan.index} stages=" + str([
                (stage.instance_count, stage.p_b, stage.batch_rounds) for stage in plan.stages]))
        session.reserve_kv(0, (5, 8, 32), torch.int8)
        session.append_kv(0, hidden[:, :1], hidden[:, 1:2])
        preserved_key, preserved_value = (value.clone() for value in session.kv_view(0))
        prepared = session.stats()
        for invocation in range(2):
            if invocation:
                hidden.copy_(torch.randint(-3, 4, hidden.shape, dtype=torch.int8))
            reference = hidden
            for block in blocks:
                reference = block(reference)
            before = session.stats()
            start = time.perf_counter()
            output = session.run_model(hidden)
            elapsed = (time.perf_counter() - start) * 1000
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            after = session.stats()
            assert after["launches"] - before["launches"] == 8
            # P-1은 첫 요청에서만 weight를 올리고, P-2는 매 요청 올린다.
            uploads = invocation == 0 or session.residency.policy == LAYER
            assert after["weight_bytes"] - before["weight_bytes"] == (4096 if uploads else 0)
            assert after["device_compiles"] == prepared["device_compiles"]
            torch.testing.assert_close(session.kv_view(0)[0], preserved_key)
            torch.testing.assert_close(session.kv_view(0)[1], preserved_value)
            print(f"invocation={invocation} exact_match=True launches=8 latency_ms={elapsed:.3f}")
        print(session.stats())
    assert session.stats()["resource_opens"] == session.stats()["resource_closes"] == 1
    print("model session: exact outputs, changed inputs, KV retention and one alloc/free OK")


if __name__ == "__main__":
    main()
