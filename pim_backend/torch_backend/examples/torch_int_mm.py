# 첫 검증: torch.compile된 그래프의 aten._int_mm 하나를 PIM에서 돌린다.
#
# 실행:
#   conda activate pim_triton
#   python torch_int_mm.py --column-partitions <p_n>
#
# 생성된 코드를 보려면:
#   TORCH_LOGS=output_code python torch_int_mm.py --column-partitions <p_n>
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
#   env -u PIM_TARGET -u TRITON_PIM_ACTIVE python torch_int_mm.py --column-partitions <p_n>

import argparse
import sys
from pathlib import Path

# pim_backend를 sys.path에 얹어 torch_backend를 package로 import한다
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from torch_backend import PIMInductorBackend, PIMTorchConfig  # noqa: E402


def build_config(column_partitions):
    # 값의 근거는 설계 문서(단계 A: 고정 config)를 따른다. 하드웨어에 맞춰
    # 바꿀 때는 tile_* 가 block_* × group_* 의 배수라는 조건을 지켜야 한다.
    #
    # device 분할은 자동 인수분해를 쓸 수 없다. 자유 knob은 p_n(column_partitions)
    # 하나이고, p_m=1 / p_k=split_k 는 저장 주소 구조가 강제한다 (config.py의
    # device_axis_partitions 주석 참고). 제약:
    #   p_n <= ceil(N / BLOCK_N)          그보다 크면 device가 논다
    #   1 * p_n * split_k <= num_units
    return PIMTorchConfig(
        target="upmem",
        strict=True,
        block_m=32, block_n=32, block_k=16,
        split_k=4,
        num_units=256,
        local_lanes=8,
        host_threads=8,
        group_m=1, group_n=1, group_k=1,
        tile_m=32, tile_n=32, tile_k=256,
        column_partitions=column_partitions,
    )


def pim_mm(a, b):
    return torch._int_mm(a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--column-partitions", type=int, required=True,
                        help="output_column_tile device 분할 수 (p_n). "
                             "p_n <= ceil(N/BLOCK_N), p_n*split_k <= num_units")
    args = parser.parse_args()

    config = build_config(args.column_partitions)
    divisor = config.split_k * config.block_k
    if args.k % divisor:
        raise SystemExit(
            f"K={args.k} must be divisible by SPLIT_K*BLOCK_K={divisor}")

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
