"""Separate GEMV physical row width from the computed N extent.

Run `bash run_aimx_gemv_pitch.sh` from this directory to build and execute.
Each case/scale runs in a fresh process, sequentially on FPGA 0. All paths use
M=128, K=256, four 32-row tiles, and x[96:128]=scale. A is all ones; the exact
answer is 32*scale. Native input/output transfers use the same datacopy APIs
as Triton, while original libpim generates and executes the native descriptors.

triton4096 computes all 4096 columns. triton4096_first512 keeps the same data
buffers, row pitches and copies, but computes/drains only the first 512 columns.
triton512 uses 512 for both. The new path still initializes the entire C buffer
to zero before MAC, so copies of uncomputed padded columns never read stale C.
Descriptor-chain size and execution time necessarily change with the N extent;
passing first512 alone cannot prove that row pitch is the only failing field.
Native and Triton also differ in descriptor/buffer lifetime and output copies.

Full outputs and Triton IR are saved under gemv_pitch_results beside this file.
Numerical failures are reported after all eight cases, without relaxing tolerance.
"""

import ctypes
import os
from pathlib import Path
import re
import subprocess
import sys

import torch
import triton

from triton.backends.pim.driver import PIMDriver
from gemv import gemv_kernel


def run_worker(case, scale, output_file):
    assert case in ("native4096", "triton4096", "triton4096_first512", "triton512")
    assert scale in (1.0, 0.5)
    print(f"[case] {case} scale={scale:g} fpga=0 M=128 K=256 tile_m=32",
          flush=True)
    a = torch.ones((128, 256), dtype=torch.float32)
    x = torch.zeros((256,), dtype=torch.float32)
    x[96:128] = scale
    out = torch.full((128,), float("nan"), dtype=torch.float32)

    if case == "native4096":
        native = ctypes.CDLL(str(
            Path(__file__).resolve().parent / "libaimx_native_gemv_pitch.so"))
        native.aimx_native_gemv_pitch.argtypes = [ctypes.c_void_p] * 3
        native.aimx_native_gemv_pitch.restype = None
        a_words = a.to(torch.bfloat16).contiguous().view(torch.int16)
        x_words = x.to(torch.bfloat16).contiguous().view(torch.int16)
        native.aimx_native_gemv_pitch(
            a_words.data_ptr(), x_words.data_ptr(), out.data_ptr())
    else:
        triton.runtime.driver.set_active(PIMDriver())
        # Fixed fixture: copies + Arel + MAC/drain groups, each 512 bytes.
        row_width, mac_columns, descriptor_bytes = {
            "triton512": (512, 512, 45056),
            "triton4096": (4096, 4096, 331776),
            "triton4096_first512": (4096, 512, 88064),
        }[case]
        arguments = (
            a, x, out, 128, 256,
            a.stride(0), a.stride(1), x.stride(0), out.stride(0),
        )
        options = dict(
            BLOCK_M=32, BLOCK_K=64, pim_target="aimx",
            num_units=1, base_fpga=0, batch=8,
            group_m=1, group_n=1, group_k=4,
            tile_m=1, tile_n=1, tile_k=1,
            p_m=1, p_n=1, p_k=1,
            aimx_persistent=True,
            aimx_matvec_row_alignment=row_width,
            aimx_matvec_first_n_only=(case == "triton4096_first512"),
        )
        compiled = gemv_kernel.warmup(*arguments, grid=(4,), **options)
        target_ir = compiled.asm["target_ir"]
        output_file.with_suffix(".pim_common").write_text(compiled.asm["pim_common"])
        output_file.with_suffix(".target_ir").write_text(target_ir)
        assert (
            f"group_layout = array<i64: 256, 1, 256, {row_width}, 1, 1>"
            in target_ir
        ), "X physical row width differs from the requested experiment"
        assert (
            f"group_layout = array<i64: 32, 1, 32, {row_width}, 1, 1>"
            in target_ir
        ), "Y physical row width differs from the requested experiment"
        assert "group_layout = array<i64: 32, 256, 32, 4096, 1, 1>" in target_ir
        assert f"aimx.descriptor_bytes = {descriptor_bytes} : i64" in target_ir, (
            "Unexpected descriptor extent; inspect saved target IR before running the FPGA")
        print(f"[lowering] row_width={row_width} mac_columns={mac_columns} "
              f"descriptor_bytes={descriptor_bytes}", flush=True)
        gemv_kernel[(4,)](*arguments, **options)

    torch.save(out, output_file)


def main():
    assert Path(os.environ["AIMX_PIM_SW"]).resolve() == Path(
        "/home/yonsei_park/pim_aimx_sw/pim_sw").resolve(), (
        "The native probe and Triton must use /home/yonsei_park/pim_aimx_sw/pim_sw")

    if len(sys.argv) != 1:
        assert len(sys.argv) == 4, "worker arguments: case scale output_file"
        run_worker(sys.argv[1], float(sys.argv[2]), Path(sys.argv[3]))
        return

    results_dir = Path(__file__).resolve().parent / "gemv_pitch_results"
    results_dir.mkdir(exist_ok=True)
    outputs = {}
    for scale in (1.0, 0.5):
        for case in ("native4096", "triton4096", "triton4096_first512", "triton512"):
            output_file = results_dir / f"{case}_scale{scale:g}.pt"
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 case, str(scale), str(output_file)],
                check=True,
            )
            outputs[scale, case] = torch.load(output_file, weights_only=True)
            assert outputs[scale, case].shape == (128,)
        for case in ("triton512", "triton4096_first512"):
            assert (
                (results_dir / f"{case}_scale{scale:g}.pim_common").read_text()
                == (results_dir / f"triton4096_scale{scale:g}.pim_common").read_text()
            ), "All three Triton cases must have identical Common IR"
        wide_ir = (results_dir / f"triton4096_scale{scale:g}.target_ir").read_text()
        first_ir = (results_dir / f"triton4096_first512_scale{scale:g}.target_ir").read_text()
        assert re.findall(r"bytes = (\d+) : i64, kind = #aimx<kind data>", wide_ir) == [
            "2097152", "262144", "262144", "262144", "262144", "262144", "262144",
        ], "Unexpected data allocation plan for the 4096-wide control"
        assert (
            re.findall(r"bytes = (\d+) : i64, kind = #aimx<kind data>", first_ir)
            == re.findall(r"bytes = (\d+) : i64, kind = #aimx<kind data>", wide_ir)
        ), "The first-512 experiment must preserve the wide control's data allocations"
        assert (
            re.findall(r"group_layout = array<i64: ([^>]+)>", first_ir)
            == re.findall(r"group_layout = array<i64: ([^>]+)>", wide_ir)
        ), "The first-512 experiment must preserve the wide control's transfer layouts"

    print("\n[summary] exact comparison; no rtol/atol relaxation")
    print("case | scale | expected | correct/128 | nonfinite | max_abs_error")
    for (scale, case), out in outputs.items():
        print(
            f"{case} | {scale:g} | {32 * scale:g} | "
            f"{(out == 32 * scale).sum().item()}/128 | "
            f"{(~torch.isfinite(out)).sum().item()} | "
            f"{(out - 32 * scale).abs().max().item():g}")

    for scale in (1.0, 0.5):
        print(f"[common-ir] scale={scale:g} identical=True cases=3")
        print(f"[layout-control] scale={scale:g} wide_vs_first512 "
              "data_allocations_equal=True transfer_layouts_equal=True")
        for case in ("triton4096", "triton4096_first512", "triton512"):
            print(
                f"[vs-native] {case} scale={scale:g} "
                f"bit_mismatches="
                f"{(outputs[scale, case].view(torch.int32) != outputs[scale, 'native4096'].view(torch.int32)).sum().item()}/128 "
                f"max_abs="
                f"{(outputs[scale, case] - outputs[scale, 'native4096']).abs().max().item():g}")

    print(f"[artifacts] {results_dir}", flush=True)
    assert all(
        torch.isfinite(out).all().item() and (out == 32 * scale).all().item()
        for (scale, _), out in outputs.items()
    ), "GEMV layout comparison found incorrect outputs; see the complete summary above"


if __name__ == "__main__":
    main()
