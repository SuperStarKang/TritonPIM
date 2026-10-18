"""BF16 host GEMM on one AiMX FPGA against the FP32 host path: about a minute.

AiMX stores BF16 and drains BF16, so with the same BF16-representable inputs the BF16
host output must equal the FP32 host output rounded to BF16, bit for bit. The FP32 path
converts on the host in the stub; the BF16 path copies the words.

  python verify_bf16_gemm.py --base-fpga 0
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from torch_backend import PIMInductorBackend, PIMTorchConfig, target_spec
from torch_backend._autotuner import PIMConfig


# (M, N, K) -> FPGA 하나의 config. projection (weight stage)와 QK^T 2 head block-diagonal (activation stage)
CASES = {
    (1024, 4096, 4096): PIMConfig(64, 128, 128, 2, 32, 32, 8, 1, 1, 1, 1, 1),
    (1024, 2048, 256): PIMConfig(128, 128, 128, 1, 16, 2, 8, 1, 1, 1, 1, 1),
}


def pim_mm(a, b):
    return torch.mm(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-fpga", type=int, default=0)
    args = parser.parse_args()
    spec = target_spec("aimx", num_units=1, base_fpga=args.base_fpga)
    generator = torch.Generator().manual_seed(0)
    for (m, n, k), config in CASES.items():
        a = torch.randn((m, k), generator=generator).bfloat16()
        b = torch.randn((k, n), generator=generator).bfloat16()
        outputs = {}
        for dtype in (torch.float32, torch.bfloat16):
            # 같은 shape이면 dynamo가 앞 dtype의 컴파일 결과를 재사용하지 않도록 비운다.
            torch._dynamo.reset()
            compiled = torch.compile(
                pim_mm, backend=PIMInductorBackend(PIMTorchConfig(spec=spec, config=config)),
                fullgraph=True, dynamic=False,
            )
            outputs[dtype] = compiled(a.to(dtype), b.to(dtype))
        bf16, fp32 = outputs[torch.bfloat16], outputs[torch.float32]
        reference = a.double() @ b.double()
        mismatches = (bf16.view(torch.int16) != fp32.bfloat16().view(torch.int16)).sum().item()
        print(f"mnk=({m},{n},{k}) bf16_dtype={bf16.dtype} "
              f"bf16_vs_fp32_rounded_mismatches={mismatches} "
              f"rel_l2_bf16={((bf16.double() - reference).norm() / reference.norm()).item():.4%} "
              f"rel_l2_fp32={((fp32.double() - reference).norm() / reference.norm()).item():.4%} "
              f"{'pass' if mismatches == 0 and bf16.dtype == torch.bfloat16 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
