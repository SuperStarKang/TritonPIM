"""Compare candidate 1's vACC reduction with original libpim on FPGA 0.

Build libaimx_native_grouped_gemm.so beside this file before running it.
The reference calls matmul across a Tile's whole K range, without Group ADD.
This compares all final outputs, without changing the original vendor SW.
"""

import ctypes
import os
from pathlib import Path

import torch
import triton

from compare_aimx_gemm import (
    PIMConfig,
    PIMDriver,
    run_gemm,
    target_spec,
)


def main():
    vendor_root = Path("/home/yonsei_park/pim_aimx_sw/pim_sw")
    assert Path(os.environ["AIMX_PIM_SW"]).resolve() == vendor_root.resolve()

    native = ctypes.CDLL(str(
        Path(__file__).resolve().parent / "libaimx_native_grouped_gemm.so"
    ))
    native.aimx_native_grouped_gemm.argtypes = [ctypes.c_void_p] * 3
    native.aimx_native_grouped_gemm.restype = None

    torch.manual_seed(0)
    M, N, K = 1024, 4096, 4096
    a = torch.randn((M, K)).to(torch.float16)
    b = torch.randn((K, N)).to(torch.float16)

    config = PIMConfig(
        128, 128, 16,
        4, 16, 8,
        2, 2, 8,
        1, 1, 1,
    )

    triton.runtime.driver.set_active(PIMDriver())
    hardware = target_spec("aimx", num_units=1, base_fpga=0)

    print("[triton] candidate=1 fpga=0", flush=True)
    actual = run_gemm(a, b, config, hardware)
    assert torch.isfinite(actual).all().item()

    a_words = a.to(torch.bfloat16).contiguous().view(torch.int16)
    b_words = b.to(torch.bfloat16).contiguous().view(torch.int16)
    reference = torch.empty((M, N), dtype=torch.float32)

    print("[native-reference] original libpim, matmul K=1024 per Tile, no Group ADD",
          flush=True)
    native.aimx_native_grouped_gemm(
        a_words.data_ptr(),
        b_words.data_ptr(),
        reference.data_ptr(),
    )
    assert torch.isfinite(reference).all().item()

    actual_bits = actual.view(torch.int32)
    reference_bits = reference.view(torch.int32)
    mismatch = actual_bits != reference_bits
    mismatch_count = mismatch.sum().item()

    print(
        f"[compare] native_bit_match={mismatch_count == 0} "
        f"bit_mismatches={mismatch_count}/{M * N} "
        f"max_abs={(actual - reference).abs().max().item():.9g}",
        flush=True,
    )

    for row, col in mismatch.nonzero()[:10].tolist():
        print(
            f"[mismatch] C[{row},{col}] "
            f"triton={actual[row, col].item():.9g} "
            f"native={reference[row, col].item():.9g} "
            f"triton_bits=0x{actual_bits[row, col].item() & 0xffffffff:08x} "
            f"native_bits=0x{reference_bits[row, col].item() & 0xffffffff:08x}",
            flush=True,
        )

    assert mismatch_count == 0, (
        "TritonPIM vACC reduction differs from the original libpim Tile matmul"
    )


if __name__ == "__main__":
    main()
