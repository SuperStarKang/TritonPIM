import argparse
from pathlib import Path
import re
import subprocess
import sys
import time
import statistics

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
    # --- 나는 C의 몇 번째 조각을 담당하는가 ---
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
 
    # --- 내가 만질 행 번호 / 열 번호 목록 ---
    # 예: pid_m=3, BLOCK_M=64  ->  offs_m = [192, 193, ..., 255]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
 
    # --- 주소 표 만들기 ---
    # [:, None] = 세로 방향, [None, :] = 가로 방향
    # 두 방향을 더하면 2차원 주소 표가 자동으로 만들어짐
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
 
    # 논리 accumulator의 FP32 타입이 AiMX 하드웨어의 FP32 연산을 뜻하지는 않는다.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
    # --- K 루프: K를 BLOCK_K씩 잘라서 나눠 곱하고 더한다 ---
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
 
        # mask: 진짜 존재하는 칸만 읽기
        # other: 없는 칸은 0으로 채우기 (0을 곱해 더하면 결과에 영향 없음)
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
 
        # (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        # Common matmul을 거쳐 AiMX MAC descriptor로 내려간다.
        acc += tl.dot(a, b)
 
        # 다음 K 조각으로 주소 표 전체를 이동
        a_ptrs += BLOCK_K * stride_ak   # A는 가로로
        b_ptrs += BLOCK_K * stride_bk   # B는 세로로
 
    # --- 결과 쓰기 (읽기와 완전히 대칭) ---
    c = acc.to(tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.zeros((M, N), dtype=torch.float32)

    configs = [
        (
            (64, 4096, 256),    # BLOCK_M, BLOCK_N, BLOCK_K
            (8, 1, 16),         # group_m, group_n, group_k
            (2, 1, 1),          # tile_m, tile_n, tile_k
            (1, 0),             # num_units, base_fpga
            (1, 1, 1),          # p_m, p_n, p_k
        ),
    ]

    # 사용할 config 선택
    config = configs[0]
    block, group, tile, device_info, partition = config

    BLOCK_M, BLOCK_N, BLOCK_K = block
    group_m, group_n, group_k = group
    tile_m, tile_n, tile_k = tile
    num_units, base_fpga = device_info
    p_m, p_n, p_k = partition

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    gemm_kernel[grid](
        a, b, c,
        M, N, K,

        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),

        # target
        pim_target="aimx",

        # 계층 1: Triton Block: basic granularity
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,

        # 계층 2: Block Group: compute granularity
        group_m=group_m, group_n=group_n, group_k=group_k,

        # 계층 3: Execution Tile: transfer granularity
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,

        # Device planning: partitioning granularity
        num_units=num_units, base_fpga=base_fpga,
        p_m=p_m, p_n=p_n, p_k=p_k,
    )

    return c


def print_output_tensors(actual, reference):
    error = (actual.double() - reference.double()).abs()
    print("[tensor] actual[:8, :8]:", actual[:8, :8], sep="\n")
    print("[tensor] CPU reference[:8, :8]:", reference[:8, :8], sep="\n")
    print("[tensor] abs_error[:8, :8]:", error[:8, :8], sep="\n")
    row, col = divmod(error.argmax().item(), actual.shape[1])
    print(f"[max-error] C[{row},{col}] actual={actual[row, col].item():.9g} "
          f"reference={reference[row, col].item():.9g} "
          f"abs_error={error[row, col].item():.9g}", flush=True)


def test():
    torch.manual_seed(0)
    M, N, K = 1024, 4096, 4096
    # M, N, K = 1024, 28672, 8192

    a = torch.randn((M, K)).to(torch.float16)
    b = torch.randn((K, N)).to(torch.float16)

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    print(f"M={M}, N={N}, K={K}", flush=True)
    start = time.perf_counter_ns()
    gemm(a, b)  # Warm up compilation, initialization and the resident plan.
    print(f"warmup total = {(time.perf_counter_ns() - start) / 1e6:.3f} ms",
          flush=True)

    # Wall time includes output allocation and launcher overhead. The driver
    # module supplies accumulated runtime-stage times for each completed call.
    elapsed_ms = []
    stage_ms = []

    for _ in range(5):
        start = time.perf_counter_ns()
        out = gemm(a, b)
        elapsed_ms.append((time.perf_counter_ns() - start) / 1e6)
        stage_ms.append(tuple(pim_driver.timing))

    total_ms = statistics.median(elapsed_ms)
    print(f"total median = {total_ms:.3f} ms")
    print(f"total min/max = {min(elapsed_ms):.3f}/{max(elapsed_ms):.3f} ms")

    for i, name in enumerate(pim_driver.TIMING_STAGES):
        print(f"{name} median = "
              f"{statistics.median(row[i] for row in stage_ms):.3f} ms")

    print(f"pim_exec calls/launch = {pim_driver.invokes}")
    print(f"effective GFLOP/s = {2 * M * N * K / (total_ms * 1e6):.3f}")

    max_diff = torch.max(torch.abs(ref - out)).item()
    print(f"max_diff = {max_diff:.6f}")
    print_output_tensors(out, ref)


LAYOUT_CASES = {
    "compact512": (512, False),
    "wide4096": (4096, False),
    "wide4096_first512": (4096, True),
}


def run_layout_worker(case, input_file, output_file):
    row_width, useful_n_only = LAYOUT_CASES[case]
    inputs = torch.load(input_file, weights_only=True)
    a, b = inputs["a"], inputs["b"]
    assert a.shape == (128, 256) and b.shape == (256, 512)
    assert a.dtype == b.dtype == torch.float16
    assert a.is_contiguous() and b.is_contiguous()
    out = torch.full((128, 512), float("nan"), dtype=torch.float32)
    print(f"[case] {input_file.stem} {case} fpga=0 "
          f"row_width={row_width} mac_columns={512 if useful_n_only else row_width}",
          flush=True)
    triton.runtime.driver.set_active(pim_driver.PIMDriver())
    arguments = (
        a, b, out, 128, 512, 256,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )
    options = dict(
        BLOCK_M=32, BLOCK_N=512, BLOCK_K=64, pim_target="aimx",
        num_units=1, base_fpga=0, batch=8,
        group_m=1, group_n=1, group_k=1,
        tile_m=1, tile_n=1, tile_k=4,
        p_m=1, p_n=1, p_k=1,
        aimx_matmul_row_alignment=row_width,
        aimx_matmul_useful_n_only=useful_n_only,
    )
    compiled = gemm_kernel.warmup(*arguments, grid=(4, 1), **options)
    common_ir = compiled.asm["pim_common"]
    target_ir = compiled.asm["target_ir"]
    output_file.with_suffix(".pim_common").write_text(common_ir)
    output_file.with_suffix(".target_ir").write_text(target_ir)
    assert "#pim.block_group<blocks = [1, 1, 1]>" in common_ir
    assert "#pim.execution_tile<groups = [1, 1, 4]>" in common_ir
    assert "group_layout = array<i64: 32, 64, 32, 4096, 1, 4>" in target_ir
    assert (
        f"group_layout = array<i64: 64, 512, 256, {row_width}, 4, 1>"
        in target_ir
    ), "B storage pitch does not match the experiment"
    assert (
        f"group_layout = array<i64: 32, 512, 32, {row_width}, 1, 1>"
        in target_ir
    ), "C storage pitch does not match the experiment"

    host_ir = target_ir.split('"func.func"()', 1)[1]
    assert "elementwise_" not in host_ir, "Unexpected partial-result ADD in GEMM"
    assert host_ir.count("program = @gemm_kernel_device0_matmul_accumulate_") == 1
    assert host_ir.count("program = @gemm_kernel_device0_matmul_drain_") == 1
    assert (
        host_ir.index('aimx.matmul_phase = "relayout"')
        < host_ir.index("program = @gemm_kernel_device0_matmul_accumulate_")
        < host_ir.index('aimx.matmul_phase = "accumulate"')
        < host_ir.index("program = @gemm_kernel_device0_matmul_drain_")
    ), "Arel must precede MAC and the drain must follow the K-Group loop"
    print("[ir-check] groups_k=4 drain_after_k_loop=True no_partial_add=True "
          f"descriptor_bytes={re.search(r'aimx.descriptor_bytes = ([0-9]+) : i64', target_ir).group(1)}",
          flush=True)
    gemm_kernel[(4, 1)](*arguments, **options)
    print(f"[launch] pim_exec_calls={pim_driver.invokes}", flush=True)
    torch.save(out, output_file)


def run_layout_experiment():
    results_dir = Path(__file__).resolve().parent / "gemm_pitch_results"
    results_dir.mkdir(exist_ok=True)
    probes = [("zero", torch.zeros((128, 256), dtype=torch.float16),
               torch.ones((256, 512), dtype=torch.float16))]
    for scale in (1.0, 0.5):
        for k_start in range(0, 256, 32):
            b = torch.zeros((256, 512), dtype=torch.float16)
            b[k_start:k_start + 32] = scale
            probes.append((f"k{k_start}_{k_start + 32}_scale{scale:g}",
                           torch.ones((128, 256), dtype=torch.float16), b))
        probes.append((f"all_ones_scale{scale:g}",
                       torch.ones((128, 256), dtype=torch.float16),
                       torch.full((256, 512), scale, dtype=torch.float16)))
    a = torch.zeros((128, 256), dtype=torch.float16)
    a[:, 128] = torch.arange(1, 129, dtype=torch.float16)
    probes.append(("row_tags_k128", a, torch.ones((256, 512), dtype=torch.float16)))
    b = torch.zeros((256, 512), dtype=torch.float16)
    b[128] = (torch.arange(512) % 256 + 1).half()
    probes.append(("column_tags_k128", torch.ones((128, 256), dtype=torch.float16), b))
    torch.manual_seed(0)
    probes.append(("random_seed0", torch.randn((128, 256)).bfloat16().half(),
                   torch.randn((256, 512)).bfloat16().half()))
    print("[experiment] M=128 N=512 K=256 BLOCK=(32,512,64) "
          "G=(1,1,1) tile=(1,1,4) batch=8 fpga=0 "
          f"probes={len(probes)} sequential_workers={len(probes) * len(LAYOUT_CASES)}",
          flush=True)
    print("[reference] CPU FP64 on BF16-representable inputs; exact probes and "
          "layout bit comparisons are strict; random FP64 accuracy is diagnostic",
          flush=True)
    summary = [
        "probe | case | exact_correct/65536 | bit_mismatches_vs_wide4096 | "
        "nonfinite | max_abs_vs_fp64",
    ]
    diagnostics = []
    passed = True
    for probe, a, b in probes:
        assert torch.equal(a.float(), a.bfloat16().float())
        assert torch.equal(b.float(), b.bfloat16().float())
        reference = a.double() @ b.double()
        assert torch.isfinite(reference).all().item()
        input_file = results_dir / f"{probe}_input.pt"
        torch.save({"a": a, "b": b, "reference_fp64": reference}, input_file)
        outputs = {}
        common_irs = {}
        target_irs = {}
        data_allocations = {}
        descriptor_bytes = {}
        for case in LAYOUT_CASES:
            output_file = results_dir / f"{probe}_{case}.pt"
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--layout-worker",
                 case, str(input_file), str(output_file)],
                check=True,
            )
            outputs[case] = torch.load(output_file, weights_only=True)
            assert outputs[case].shape == (128, 512)
            assert outputs[case].dtype == torch.float32
            common_irs[case] = output_file.with_suffix(".pim_common").read_text()
            target_irs[case] = output_file.with_suffix(".target_ir").read_text()
            data_allocations[case] = re.findall(
                r"bytes = (\d+) : i64, kind = #aimx<kind data>", target_irs[case])
            descriptor_bytes[case] = int(re.search(
                r"aimx.descriptor_bytes = (\d+) : i64", target_irs[case]).group(1))
        assert len(set(common_irs.values())) == 1, "Layout cases must have identical Common IR"
        assert data_allocations["wide4096"], "No data allocations found in the wide target IR"
        assert (
            data_allocations["wide4096"] == data_allocations["wide4096_first512"]
        ), "Changing compute extent must preserve the wide data allocations"
        assert re.findall(
            r"group_layout = array<i64: ([^>]+)>", target_irs["wide4096"]
        ) == re.findall(
            r"group_layout = array<i64: ([^>]+)>", target_irs["wide4096_first512"]
        ), "Changing compute extent must preserve the wide transfer layouts"
        assert descriptor_bytes["wide4096_first512"] < descriptor_bytes["wide4096"], (
            "Computing only useful N must reduce the descriptor extent")
        diagnostics.append(f"[layout-control] {probe} common_ir_identical=True "
                           "wide_data_allocations_equal=True wide_transfer_layouts_equal=True "
                           f"descriptor_bytes={descriptor_bytes}")

        for case, out in outputs.items():
            error = (out.double() - reference).abs()
            nonfinite = ~torch.isfinite(out)
            different_bits = out.view(torch.int32) != outputs["wide4096"].view(torch.int32)
            failed = nonfinite | different_bits
            if probe == "random_seed0":
                exact_count = "n/a"
                diagnostics.append(
                    f"[random-accuracy] {case} "
                    f"max_abs={error.max().item():.9g} mean_abs={error.mean().item():.9g} "
                    f"rmse={error.square().mean().sqrt().item():.9g} "
                    f"relative_l2={(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)).item():.9g} "
                    f"failed_rtol0.016_atol1e-5="
                    f"{(~torch.isclose(out.double(), reference, rtol=0.016, atol=1e-5)).sum().item()}/65536")
            else:
                exact_count = f"{(out.double() == reference).sum().item()}/65536"
                failed |= out.double() != reference
            if failed.any().item():
                passed = False
            summary.append(
                f"{probe} | {case} | {exact_count} | {different_bits.sum().item()} | "
                f"{nonfinite.sum().item()} | {error.max().item():.9g}")
            print(f"[tensor-case] {probe} {case}", flush=True)
            print_output_tensors(out, reference)
            for row, col in failed.nonzero()[:3].tolist():
                diagnostics.append(
                    f"[mismatch] {probe} {case} C[{row},{col}] "
                    f"actual={out[row, col].item():.9g} fp64={reference[row, col].item():.9g} "
                    f"wide4096={outputs['wide4096'][row, col].item():.9g}")

    summary.extend(diagnostics)
    summary.append(f"[result] exact_probes_and_layout_invariance={'PASS' if passed else 'FAIL'}")
    summary.append("[scope] Random FP64 accuracy is reported, not certified by this result.")
    (results_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n[summary]\n" + "\n".join(summary), flush=True)
    print(f"[artifacts] {results_dir}", flush=True)
    assert passed, "GEMM layout comparison failed; see summary.txt and per-case IR/outputs"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AiMX GEMM benchmark and storage-pitch experiment")
    parser.add_argument("--layout-experiment", action="store_true")
    parser.add_argument("--layout-worker", nargs=3, metavar=("CASE", "INPUT", "OUTPUT"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert not (args.layout_experiment and args.layout_worker)
    if args.layout_worker:
        run_layout_worker(args.layout_worker[0], Path(args.layout_worker[1]),
                          Path(args.layout_worker[2]))
    elif args.layout_experiment:
        run_layout_experiment()
    else:
        triton.runtime.driver.set_active(pim_driver.PIMDriver())
        test()
