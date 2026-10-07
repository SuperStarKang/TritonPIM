"""Compare GEMV K-Group partitions using the passing 4096-pitch/512-MAC path.

Run with the existing TritonPIM environment and rebuilt default matvec lowering:
    python compare_aimx_gemv_groups.py

M=128, K=256, BLOCK_M=32, BLOCK_K=64, batch=8, FPGA=0. The configurations
(group_k, tile_k) are (4,1), (2,2), (1,4); each Tile covers all K. Each of the
20 inputs/configurations runs sequentially in a fresh process (60 workers).
The actual gemv.py kernel is used. No native helper or vendor edits are needed.
Matvec layout options are deliberately omitted to exercise the backend defaults;
the generated IR must still have 4096-element rows and one N microtile.

Sparse, all-ones and row-tag probes require exact CPU FP64 agreement. Random
inputs are rounded to BF16 first and stored as FP32 so all paths consume the
same representable values. Random FP64 errors are reported without a tolerance
gate; bitwise agreement with the single-Group configuration is required for
every input. Passing this experiment is not an LLM accuracy qualification.

Common IR should differ in Group/Tile sizes. Physical K padding and resident
buffer sizes also change with the partition; B/C row pitch stays 4096 elements.
Inputs, full outputs, Common/target IR and the summary are saved beside this
file under gemv_group_results. Numerical failures are asserted after reporting
all cases; compilation, IR-check and process failures propagate immediately.
"""

import os
from pathlib import Path
import re
import subprocess
import sys

import torch
import triton

from triton.backends.pim.driver import PIMDriver
from gemv import gemv_kernel


def run_worker(group_k, input_file, output_file):
    assert group_k in (4, 2, 1)
    tile_k = 4 // group_k
    inputs = torch.load(input_file, weights_only=True)
    a, x = inputs["a"], inputs["x"]
    assert a.shape == (128, 256) and x.shape == (256,)
    assert a.dtype == x.dtype == torch.float32
    assert a.is_contiguous() and x.is_contiguous()
    out = torch.full((128,), float("nan"), dtype=torch.float32)
    print(f"[case] {input_file.stem} group_k={group_k} tile_k={tile_k} "
          "fpga=0 row_width=4096 mac_columns=512", flush=True)

    triton.runtime.driver.set_active(PIMDriver())
    arguments = (
        a, x, out, 128, 256,
        a.stride(0), a.stride(1), x.stride(0), out.stride(0),
    )
    options = dict(
        BLOCK_M=32, BLOCK_K=64, pim_target="aimx",
        num_units=1, base_fpga=0, batch=8,
        group_m=1, group_n=1, group_k=group_k,
        tile_m=1, tile_n=1, tile_k=tile_k,
        p_m=1, p_n=1, p_k=1,
        aimx_persistent=True,
    )
    compiled = gemv_kernel.warmup(*arguments, grid=(4,), **options)
    common_ir = compiled.asm["pim_common"]
    target_ir = compiled.asm["target_ir"]
    output_file.with_suffix(".pim_common").write_text(common_ir)
    output_file.with_suffix(".target_ir").write_text(target_ir)
    assert f"#pim.block_group<blocks = [1, 1, {group_k}]>" in common_ir
    assert f"#pim.execution_tile<groups = [1, 1, {tile_k}]>" in common_ir
    assert (
        f"group_layout = array<i64: {64 * group_k}, 1, 256, 4096, {tile_k}, 1>"
        in target_ir
    ), "X layout must retain each Group's padded K rows and 4096-element row pitch"
    assert (
        f"group_layout = array<i64: 32, {64 * group_k}, 32, 4096, 1, {tile_k}>"
        in target_ir
    ), "A layout must contain the requested resident K Groups"
    assert "group_layout = array<i64: 32, 1, 32, 4096, 1, 1>" in target_ir

    host_ir = target_ir.split('"func.func"()', 1)[1]
    assert "elementwise_" not in host_ir, "Unexpected partial-result ADD in GEMV"
    assert host_ir.count("program = @gemv_kernel_device0_matmul_accumulate_") == 1
    assert host_ir.count("program = @gemv_kernel_device0_matmul_drain_") == 1
    assert (
        host_ir.index('aimx.matmul_phase = "relayout"')
        < host_ir.index("program = @gemv_kernel_device0_matmul_accumulate_")
        < host_ir.index('aimx.matmul_phase = "accumulate"')
        < host_ir.index("program = @gemv_kernel_device0_matmul_drain_")
    ), "Arel must precede MAC and the drain must follow the K-Group loop"
    descriptor_bytes = int(re.search(
        r"aimx.descriptor_bytes = (\d+) : i64", target_ir).group(1))
    # Includes conservative final-K-padding capacity for the smaller Groups.
    assert descriptor_bytes == {4: 88064, 2: 120832, 1: 129024}[group_k], (
        "Unexpected descriptor capacity; inspect the saved IR before FPGA execution")
    print(f"[ir-check] group_k={group_k} tile_k={tile_k} "
          f"drain_after_k_loop=True no_partial_add=True descriptor_bytes={descriptor_bytes}",
          flush=True)
    gemv_kernel[(4,)](*arguments, **options)
    torch.save(out, output_file)


def main():
    assert Path(os.environ["AIMX_PIM_SW"]).resolve() == Path(
        "/home/yonsei_park/pim_aimx_sw/pim_sw").resolve()
    if len(sys.argv) != 1:
        assert len(sys.argv) == 4, "worker arguments: group_k input_file output_file"
        run_worker(int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
        return

    results_dir = Path(__file__).resolve().parent / "gemv_group_results"
    results_dir.mkdir(exist_ok=True)
    probes = []
    a = torch.ones((128, 256), dtype=torch.float32)
    for scale in (1.0, 0.5):
        for k_start in range(0, 256, 32):
            x = torch.zeros((256,), dtype=torch.float32)
            x[k_start:k_start + 32] = scale
            probes.append((f"k{k_start}_{k_start + 32}_scale{scale:g}", a, x))
        probes.append((f"all_ones_scale{scale:g}", a,
                       torch.full((256,), scale, dtype=torch.float32)))
    a = torch.arange(1, 129, dtype=torch.float32)[:, None].expand(128, 256).contiguous()
    x = torch.zeros((256,), dtype=torch.float32)
    x[128] = 1.0
    probes.append(("row_tags_k128", a, x))
    torch.manual_seed(0)
    probes.append((
        "random_seed0",
        torch.randn((128, 256)).bfloat16().float(),
        torch.randn((256,)).bfloat16().float(),
    ))
    print("[experiment] M=128 K=256 BLOCK=(32,64) batch=8 fpga=0 "
          "row_width=4096 mac_columns=512 configs=(G4/T1,G2/T2,G1/T4) "
          f"probes={len(probes)} workers={3 * len(probes)}", flush=True)
    print("[reference] CPU FP64 on the same BF16-representable inputs; "
          "random error metrics are diagnostic, Group bit comparisons are strict",
          flush=True)
    summary = [
        "probe | exact_correct G4/T1,G2/T2,G1/T4 | bit_mismatches_vs_G4 G2,G1 "
        "| nonfinite G4,G2,G1",
    ]
    diagnostics = []
    passed = True
    for probe, a, x in probes:
        assert torch.equal(a, a.bfloat16().float())
        assert torch.equal(x, x.bfloat16().float())
        reference = torch.mv(a.double(), x.double())
        assert torch.isfinite(reference).all().item()
        input_file = results_dir / f"{probe}_input.pt"
        torch.save({"a": a, "x": x, "reference_fp64": reference}, input_file)
        outputs = {}
        for group_k in (4, 2, 1):
            output_file = results_dir / f"{probe}_g{group_k}_t{4 // group_k}.pt"
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 str(group_k), str(input_file), str(output_file)],
                check=True,
            )
            outputs[group_k] = torch.load(output_file, weights_only=True)
            assert outputs[group_k].shape == (128,)
            assert outputs[group_k].dtype == torch.float32

        exact_counts = []
        bit_mismatches = []
        nonfinite_counts = []
        for group_k in (4, 2, 1):
            out = outputs[group_k]
            nonfinite = ~torch.isfinite(out)
            nonfinite_counts.append(nonfinite.sum().item())
            different_bits = out.view(torch.int32) != outputs[4].view(torch.int32)
            if group_k != 4:
                bit_mismatches.append(different_bits.sum().item())
            failed = nonfinite | different_bits
            error = (out.double() - reference).abs()
            if probe == "random_seed0":
                exact_counts.append("n/a")
                diagnostics.append(
                    f"[random-accuracy] G{group_k}/T{4 // group_k} "
                    f"max_abs={error.max().item():.9g} mean_abs={error.mean().item():.9g} "
                    f"rmse={error.square().mean().sqrt().item():.9g} "
                    f"relative_l2="
                    f"{(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)).item():.9g}")
            else:
                exact_counts.append(f"{(out.double() == reference).sum().item()}/128")
                failed |= out.double() != reference
            if failed.any().item():
                passed = False
            for row in failed.nonzero().flatten()[:3].tolist():
                diagnostics.append(
                    f"[mismatch] {probe} G{group_k}/T{4 // group_k} row={row} "
                    f"actual={out[row].item():.9g} fp64={reference[row].item():.9g} "
                    f"single_group={outputs[4][row].item():.9g} "
                    f"abs_error={error[row].item():.9g}")
        summary.append(
            f"{probe} | {','.join(exact_counts)} | "
            f"{','.join(str(count) for count in bit_mismatches)} | "
            f"{','.join(str(count) for count in nonfinite_counts)}")

    summary.extend(diagnostics)
    summary.append(f"[result] exact_probes_and_group_invariance={'PASS' if passed else 'FAIL'}")
    summary.append("[scope] Random FP64 accuracy is reported, not certified by this result.")
    (results_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n[summary]\n" + "\n".join(summary), flush=True)
    print(f"[artifacts] {results_dir}", flush=True)
    assert passed, "GEMV Group comparison failed; see summary.txt and per-case IR/outputs"


if __name__ == "__main__":
    main()
