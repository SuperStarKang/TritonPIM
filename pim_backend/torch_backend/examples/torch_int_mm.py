# 첫 검증: torch.compile된 그래프의 aten._int_mm 하나를 PIM에서 돌린다.
#
# 실행:
#   conda activate pim_triton
#   python torch_int_mm.py
#
# 생성된 코드를 보려면:
#   TORCH_LOGS=output_code python torch_int_mm.py
#     - PIM 커널은 @triton_heuristics.template(...)로 정의되고, 그 triton_meta에
#       "compile_backend": "pim"과 "backend_options"가 들어 있어야 한다.
#     - 커널 본문이 split-K 커널(pid_k, K_PER, tl.dot 누산)이어야 한다.
#     - 그 앞뒤로 cpp_fused_* 커널 두 개(zeros 채우기 / partial 합)가 있어야 한다.
#
# FX 그래프를 보려면:
#   TORCH_LOGS=graph_code python torch_int_mm.py
#     - aten._int_mm.default 노드가 있어야 한다 (decompose되지 않았다).
#
# 환경변수는 하나도 필요 없다. 필요하면 아래를 확인용으로 넣고 돌려도 된다:
#   env -u PIM_TARGET -u TRITON_PIM_ACTIVE python torch_int_mm.py

import argparse
import logging
import sys
from pathlib import Path

# pim_backend를 sys.path에 얹어 torch_backend를 package로 import한다
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from torch_backend import PIMInductorBackend, PIMTorchConfig  # noqa: E402


def build_config():
    """실행 계획은 여기서 정하지 않는다.

    block/group/execution tile과 축별 device 분할 수는 autotuner가 shape와
    hardware profile(autotuner/targets.py)에서 유도하고, ConstraintEngine이
    검증한다. 여기서는 별도 분할 값을 만들지 않고 기본 seed를 쓴다.
    실제로 고른 config는 실행 시 로그로 찍힌다:

        python torch_int_mm.py -v
    """
    return PIMTorchConfig()


def pim_mm(a, b):
    return torch._int_mm(a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="고른 실행 계획(block/group/tile/device 분할)을 찍는다")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("torch_backend").setLevel(logging.INFO)

    config = build_config()

    torch.manual_seed(0)
    a = torch.randint(-8, 8, (args.m, args.k), dtype=torch.int8)
    b = torch.randint(-8, 8, (args.k, args.n), dtype=torch.int8)

    compiled = torch.compile(
        pim_mm,
        backend=PIMInductorBackend(config),
        fullgraph=True,
        dynamic=False,
    )

    got = compiled(a, b)
    ref = a.to(torch.int32) @ b.to(torch.int32)

    print("out dtype :", got.dtype, "(expected torch.int32)")
    print("out device:", got.device, "(expected cpu)")
    print("max error :", (got - ref).abs().max().item())
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    print("OK")


if __name__ == "__main__":
    main()
