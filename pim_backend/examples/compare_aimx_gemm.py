"""Legacy diagnostic for the former per-Group MAC + ADD lowering.

Build pim_backend/autotuner/libaimx_reference.so before running this script.
For the vACC reduction, use compare_aimx_native_gemm.py instead: this model's
per-Group ADD sequence no longer describes the generated GEMM descriptors.
The legacy model uses FPGA 0, P=(1,1,1).
If candidate 1 differs, probe exact cancellation within a MAC, across Groups,
and across K tiles before reporting the comparison failure.
"""

import ctypes
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton.backends.pim.driver import PIMDriver  # noqa: E402

from autotuner.decorator import pim_launch_options  # noqa: E402
from autotuner.targets import target_spec  # noqa: E402
from autotuner.types import PIMConfig  # noqa: E402
from gemm import gemm_kernel  # noqa: E402


def sw_reference(model, a, b, group_k, tile_k):
    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.float16
    assert a.device.type == b.device.type == "cpu"
    a_words = a.to(torch.bfloat16).contiguous().view(torch.int16)
    b_words = b.to(torch.bfloat16).contiguous().view(torch.int16)
    out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32)
    model.aimx_gemm_reference(
        a_words.data_ptr(), b_words.data_ptr(), out.data_ptr(),
        a.shape[0], b.shape[1], a.shape[1], group_k, tile_k,
    )
    return out


def run_gemm(a, b, config, hardware):
    M, K = a.shape
    _, N = b.shape
    out = torch.zeros((M, N), dtype=torch.float32)
    gemm_kernel[(triton.cdiv(M, config.block_m),
                 triton.cdiv(N, config.block_n))](
        a, b, out, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=config.block_m, BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        **pim_launch_options(config, hardware),
    )
    return out


def main():
    model = ctypes.CDLL(str(
        Path(__file__).resolve().parent.parent
        / "autotuner" / "libaimx_reference.so"))
    model.aimx_gemm_reference.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 5)
    model.aimx_gemm_reference.restype = None

    for name, k, group_k, tile_k, terms, expected in (
        ("zero", 256, 128, 2, (), 0.0),
        ("group_padding", 256, 128, 2, ((127, 1.0), (128, 1.0)), 2.0),
        ("host_fp32_sum", 512, 256, 1,
         ((0, 1.0), (256, 1.0 / 256)), 1.0 + 1.0 / 256),
    ):
        a = torch.zeros((1, k), dtype=torch.float16)
        b = torch.zeros((k, 1), dtype=torch.float16)
        for index, value in terms:
            a[0, index] = value
            b[index, 0] = 1.0
        reference = sw_reference(model, a, b, group_k, tile_k)
        assert reference.item() == expected, (
            f"SW model self-check {name}: "
            f"got={reference.item()} expected={expected}")
        print(f"[model-check] {name}: PASS", flush=True)

    torch.manual_seed(0)
    M, N, K = 1024, 4096, 4096
    a = torch.randn((M, K)).to(torch.float16)
    b = torch.randn((K, N)).to(torch.float16)
    rows = torch.arange(0, M, 32)
    cols = torch.arange(0, N, 128)
    a_rows = a.index_select(0, rows)
    b_cols = b.index_select(1, cols)
    fp32_reference = a_rows.float() @ b_cols.float()

    triton.runtime.driver.set_active(PIMDriver())
    hardware = target_spec("aimx", num_units=1, base_fpga=0)
    configs = (
        PIMConfig(128, 128, 16, 4, 16, 8, 2, 2, 8, 1, 1, 1),
        PIMConfig(64, 128, 64, 16, 2, 16, 1, 8, 4, 1, 1, 1),
        PIMConfig(32, 16, 64, 16, 32, 16, 2, 4, 4, 1, 1, 1),
    )
    matches = []
    for index, config in enumerate(configs, 1):
        print(
            f"\n[compare] candidate={index} config={config.key()} "
            f"problem=({M},{N},{K}) fpga=0 sampled=1024/{M * N}",
            flush=True,
        )
        reference = sw_reference(
            model, a_rows, b_cols,
            config.block_k * config.group_k, config.tile_k,
        )
        out = run_gemm(a, b, config, hardware)
        got = out.index_select(0, rows).index_select(1, cols)
        got_bits = got.view(torch.int32)
        reference_bits = reference.view(torch.int32)
        mismatched = got_bits != reference_bits
        matches.append(not mismatched.any().item())
        print(
            f"[compare] sw_bit_match={matches[-1]} "
            f"bit_mismatches={mismatched.sum().item()}/{got.numel()} "
            f"max_abs_vs_sw={(got - reference).abs().max().item():.9g}",
            flush=True,
        )
        for name, values in (("triton", got), ("aimx_sw", reference)):
            error = (values - fp32_reference).abs()
            print(
                f"[accuracy:{name}] "
                f"nonfinite={(~torch.isfinite(values)).sum().item()} "
                f"max_abs={error.max().item():.9g} "
                f"mean_abs={error.mean().item():.9g} "
                f"relative_l2={(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(fp32_reference)).item():.9g} "
                f"fp32_failed={(~torch.isclose(values, fp32_reference, rtol=0.05, atol=0.05)).sum().item()}/{values.numel()}",
                flush=True,
            )
        for i, j in mismatched.nonzero()[:5].tolist():
            print(
                f"[mismatch] C[{rows[i].item()},{cols[j].item()}] "
                f"triton={got[i, j].item():.9g} "
                f"sw={reference[i, j].item():.9g} "
                f"fp32={fp32_reference[i, j].item():.9g} "
                f"triton_bits=0x{got_bits[i, j].item() & 0xffffffff:08x} "
                f"sw_bits=0x{reference_bits[i, j].item() & 0xffffffff:08x}",
                flush=True,
            )

    if not matches[0]:
        for name, second_k, first_value in (
            ("same_mac_positive_first", 1, 2.0),
            ("same_mac_negative_first", 1, -2.0),
            ("different_groups_positive_first", 128, 2.0),
            ("different_groups_negative_first", 128, -2.0),
            ("different_tiles", 1024, 2.0),
        ):
            print(
                f"\n[cancel-probe] {name} candidate=1 "
                f"k=(0,{second_k}) terms=({first_value},{-first_value})",
                flush=True,
            )
            probe_a = torch.zeros_like(a)
            probe_b = torch.zeros_like(b)
            probe_a[32, 0] = first_value
            probe_b[0, 384] = 1.0
            probe_a[32, second_k] = -first_value
            probe_b[second_k, 384] = 1.0
            reference = sw_reference(
                model, probe_a[32:33], probe_b[:, 384:385],
                configs[0].block_k * configs[0].group_k, configs[0].tile_k,
            )
            out = run_gemm(probe_a, probe_b, configs[0], hardware)
            got = out[32:33, 384:385].contiguous()
            print(
                f"[cancel-probe] C[32,384] mathematical=0 "
                f"triton={got.item():.9g} sw={reference.item():.9g} "
                f"sw_bit_match={torch.equal(got.view(torch.int32), reference.view(torch.int32))} "
                f"triton_bits=0x{got.view(torch.int32).item() & 0xffffffff:08x} "
                f"sw_bits=0x{reference.view(torch.int32).item() & 0xffffffff:08x} "
                f"output_nonzero={torch.count_nonzero(out).item()}",
                flush=True,
            )

    assert all(matches), (
        "AiMX SW model comparison failed; check the FPGA/model arithmetic "
        "and the lowered Group/Tile reduction sequence.")
    print("\n[compare] All three candidates match the SW model at sampled positions.")


if __name__ == "__main__":
    main()
