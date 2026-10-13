"""Smallest correct GEMM A row (K) padding on one AiMX FPGA: a few minutes.

A is stored with its K padded to aimx_matmul_k_alignment (default 0 = 4096/batch). The MAC reads
the 256-aligned relayout (Arel), so a smaller pitch must give the 4096 result bit for
bit. A smaller pitch also stores, zero-fills and uploads fewer A bytes.

One worker process per alignment runs every case twice. The parent compares each
candidate with 4096 and the two runs with each other, and reports the A bytes and
host staging time. Legal alignments with batch 8: 512, 1024, 2048, 4096.

  python verify_k_alignment.py                          # candidates 512 1024 2048
  python verify_k_alignment.py --alignments 512         # only 512
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import triton
import triton.language as tl
import triton.backends.pim.driver as pim_driver


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


BASELINE = 4096
BATCH = 8

# name: (M, N, K), BLOCK (m, n, k), group (m, n, k), tile (m, n, k)
# A의 저장 pitch = round_up(Block Group k, 정렬). Block Group k = BLOCK_K × group_k.
CASES = {
    # QKᵀ처럼 K 128, K Group 하나
    "k128": ((32, 512, 128), (32, 512, 64), (1, 1, 2), (1, 1, 1)),
    # Block Group k 64, Tile 안에 K Group 4개
    "k256_t4": ((32, 512, 256), (32, 512, 64), (1, 1, 1), (1, 1, 4)),
    # K가 256의 배수가 아님 (masked tail)
    "k384": ((32, 512, 384), (32, 512, 64), (1, 1, 2), (1, 1, 3)),
    # PV처럼 K 1024, Block Group k 256
    "k1024_t4": ((32, 512, 1024), (32, 512, 64), (1, 1, 4), (1, 1, 4)),
}


def round_up(value, unit):
    return (value + unit - 1) // unit * unit


def a_pitch(case, alignment):
    _, block, group, _ = CASES[case]
    return round_up(block[2] * group[2], alignment)


def make_inputs():
    generator = torch.Generator().manual_seed(0)
    inputs = {}
    for case, ((m, n, k), _, _, _) in CASES.items():
        a = torch.randn((m, k), generator=generator).bfloat16()
        b = torch.randn((k, n), generator=generator).bfloat16()
        inputs[case] = {"a": a, "b": b}
    return inputs


def run_worker(alignment, input_file, output_file):
    triton.runtime.driver.set_active(pim_driver.PIMDriver())
    inputs = torch.load(input_file, weights_only=True)
    results = {}
    for case, ((m, n, k), block, group, tile) in CASES.items():
        a, b = inputs[case]["a"], inputs[case]["b"]
        options = dict(
            BLOCK_M=block[0], BLOCK_N=block[1], BLOCK_K=block[2], pim_target="aimx",
            num_units=1, base_fpga=0, batch=BATCH,
            group_m=group[0], group_n=group[1], group_k=group[2],
            tile_m=tile[0], tile_n=tile[1], tile_k=tile[2],
            p_m=1, p_n=1, p_k=1,
            aimx_matmul_k_alignment=alignment,
        )
        grid = (triton.cdiv(m, block[0]), triton.cdiv(n, block[1]))
        outputs = []
        for _ in range(2):
            out = torch.full((m, n), float("nan"), dtype=torch.float32)
            arguments = (a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         out.stride(0), out.stride(1))
            gemm_kernel[grid](*arguments, **options)
            outputs.append(out)
        target_ir = gemm_kernel.warmup(*arguments, grid=grid, **options).asm["target_ir"]
        # A의 Group panel: useful (rows, cols) = Block Group (m, k), 저장 cols = pitch
        group_m, group_k = block[0] * group[0], block[2] * group[2]
        expected = (f"group_layout = array<i64: {group_m}, {group_k}, {round_up(group_m, BATCH)}, "
                    f"{a_pitch(case, alignment)},")
        assert expected in target_ir, f"{case}: A storage pitch is not {a_pitch(case, alignment)}"
        timing = dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))
        results[case] = {"outputs": outputs, "stage_ms": timing["stage"],
                         "invoke_ms": timing["invoke"]}
        print(f"[worker] alignment={alignment} {case} pitch={a_pitch(case, alignment)} "
              f"stage_ms={timing['stage']:.3f} invoke_ms={timing['invoke']:.3f}", flush=True)
    torch.save(results, output_file)


def run_alignment(alignment, input_file, results_dir):
    output_file = results_dir / f"a{alignment}.pt"
    worker = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                             str(alignment), str(input_file), str(output_file)])
    if worker.returncode != 0:
        print(f"[experiment] alignment={alignment} worker exit code {worker.returncode}", flush=True)
        return None
    return torch.load(output_file, weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", nargs=3, metavar=("ALIGNMENT", "INPUT", "OUTPUT"))
    parser.add_argument("--alignments", type=int, nargs="+", default=(512, 1024, 2048))
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().with_name("k_alignment_results"))
    args = parser.parse_args()
    if args.worker:
        alignment, input_file, output_file = args.worker
        run_worker(int(alignment), Path(input_file), Path(output_file))
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs()
    input_file = args.results_dir / "inputs.pt"
    torch.save(inputs, input_file)
    baseline = run_alignment(BASELINE, input_file, args.results_dir)
    if baseline is None:
        raise RuntimeError("the 4096 baseline worker failed")

    rows = []
    for alignment in sorted(args.alignments):
        measured = run_alignment(alignment, input_file, args.results_dir)
        for case, ((m, n, k), _, _, _) in CASES.items():
            base = baseline[case]
            reference = inputs[case]["a"].double() @ inputs[case]["b"].double()
            row = {
                "alignment": alignment, "case": case, "mnk": [m, n, k],
                "pitch": a_pitch(case, alignment), "baseline_pitch": a_pitch(case, BASELINE),
                # A 저장·전송량: 행 수(batch 단위로 올림) × pitch × bf16 2 byte
                "a_bytes": round_up(m, BATCH) * a_pitch(case, alignment) * 2,
                "baseline_a_bytes": round_up(m, BATCH) * a_pitch(case, BASELINE) * 2,
                "baseline_stage_ms": base["stage_ms"],
            }
            if measured is None:
                row["status"] = "worker_failed"
            else:
                first, second = measured[case]["outputs"]
                row.update(
                    mismatches_vs_4096=(first != base["outputs"][0]).sum().item(),
                    mismatches_between_runs=(first != second).sum().item(),
                    baseline_between_runs=(base["outputs"][0] != base["outputs"][1]).sum().item(),
                    relative_l2=((first.double() - reference).norm() / reference.norm()).item(),
                    stage_ms=measured[case]["stage_ms"],
                    invoke_ms=measured[case]["invoke_ms"],
                )
                row["status"] = ("pass" if row["mismatches_vs_4096"] == 0
                                 and row["mismatches_between_runs"] == 0 else "FAIL")
            rows.append(row)

    print("\nalignment | case     | pitch (4096→) | A bytes (4096→)      | stage ms (4096→)  | "
          "vs4096 | runs | rel_l2  | status")
    for row in rows:
        if row["status"] == "worker_failed":
            print(f"{row['alignment']:9d} | {row['case']:8s} | worker failed")
            continue
        print(f"{row['alignment']:9d} | {row['case']:8s} | {row['baseline_pitch']:5d}→{row['pitch']:5d} | "
              f"{row['baseline_a_bytes']:9d}→{row['a_bytes']:9d} | "
              f"{row['baseline_stage_ms']:7.3f}→{row['stage_ms']:7.3f} | "
              f"{row['mismatches_vs_4096']:6d} | {row['mismatches_between_runs']:4d} | "
              f"{row['relative_l2']:.4%} | {row['status']}")
    passing = [alignment for alignment in sorted(args.alignments)
               if all(row["status"] == "pass" for row in rows if row["alignment"] == alignment)]
    print(f"\n[summary] baseline runs agree: "
          f"{all(baseline[case]['outputs'][0].equal(baseline[case]['outputs'][1]) for case in CASES)}")
    print(f"[summary] alignments that match 4096 in every case: {passing}")
    print(f"[summary] smallest correct alignment: {passing[0] if passing else 'none (keep 4096)'}")
    (args.results_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
