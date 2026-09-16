# 한 그래프 안의 _int_mm 다섯 개(Q, K, V projection, QK, PV)를 PIM에서 돌린다.
# 나머지(transpose, scale, mask, softmax, quantization)는 CPU Cpp다.
#
# 실행:
#   conda activate pim_triton
#   python torch_attention.py --seq 256 --kv 256 --head-dim 128
#
# 예상 배치:
#   Q/K/V projection PIM (각각 X @ W)
#   Q/K/V requantize CPU Cpp (int32 -> int8, scale 보존)
#   transpose        CPU Cpp
#   QK _int_mm       PIM
#   scale/mask       CPU Cpp
#   softmax          CPU Cpp
#   quantization     CPU Cpp
#   PV _int_mm       PIM
#   output dequant   CPU Cpp (float32 출력)
#
# 단일 head self-attention: X [seq,D], Wq/Wk/Wv [D,D], seq == kv.
# Q/K/V는 같은 X에서 생성한다. warm-up 뒤 커널별 profiling 1회와 일반 호출 1회를 잰다.

import argparse
from collections import Counter
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# pim_backend를 sys.path에 얹어 torch_backend를 package로 import한다
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from torch_backend import PIMInductorBackend  # noqa: E402

from torch_int_mm import build_config  # noqa: E402
from profile_attention import (  # noqa: E402
    describe_records, print_table, shape_text, trace_kernels,
)


def requantize_projection(raw):
    """Symmetric per-tensor int8 quantization; raw ~= quantized * scale."""
    value = raw.float()
    # All-zero projections remain zero, with a finite, nonzero scale.
    scale = value.abs().amax().clamp_min(1.0) / 127.0
    quantized = (value / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def attention_group_inductor(x, wq, wk, wv, mask, head_dim):
    # 입력과 가중치는 int8, projection 누산 결과는 int32다.
    # 다음 _int_mm에 넣기 위해 int8로 재양자화하고 scale은 아래에서 복원한다.
    q, q_scale = requantize_projection(torch._int_mm(x, wq))
    k, k_scale = requantize_projection(torch._int_mm(x, wk))
    v, v_scale = requantize_projection(torch._int_mm(x, wv))

    # PIM 데이터 배치가 row-major 연속을 전제로 하므로 transpose 결과를
    # 명시적으로 contiguous로 만든다.
    kt = k.transpose(0, 1).contiguous()

    scores = torch._int_mm(q, kt)

    logits = scores.float() * q_scale * k_scale / math.sqrt(head_dim)
    logits = logits.masked_fill(mask, float("-inf"))

    p = torch.softmax(logits, dim=-1)

    row_scale = p.amax(dim=-1, keepdim=True)
    pq = (p / row_scale * 127).round().clamp(0, 127).to(torch.int8)

    output_int = torch._int_mm(pq, v)
    output = output_int.float() * row_scale / 127.0 * v_scale

    return output, scores, pq, row_scale


def make_pim_labeler(x, wq, wk, wv):
    """Identify projections even when one kernel serves Q/K/V in any order.

    Inductor can schedule V after QK, and S == D makes GEMM shapes identical.
    Input buffer identities identify projections without assuming launch order.
    The remaining two calls have a dependency ordering: QK must precede PV.
    """
    x_pointer = x.data_ptr()
    weights = {weight.data_ptr(): f"{name} projection"
               for name, weight in zip(("Q", "K", "V"), (wq, wk, wv))}
    attention_calls = 0

    def label(args):
        nonlocal attention_calls
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if len(tensors) != 3:
            raise RuntimeError("Expected PIM GEMM tensor arguments A, B, C")
        a, b, c = tensors
        if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
                or c.shape != (a.shape[0], b.shape[1])):
            raise RuntimeError("Unexpected PIM GEMM argument shapes")
        if a.data_ptr() == x_pointer and b.data_ptr() in weights:
            return weights[b.data_ptr()]
        if attention_calls >= 2:
            raise RuntimeError("Could not identify the five attention GEMMs")
        name = ("QK^T", "PV")[attention_calls]
        attention_calls += 1
        return name

    return label


def print_attention_report(report, verbose=False):
    """One row per measured kernel call, preserving the compiler's fusion."""
    records = report["kernels"]
    total_ms = report["total_ms"]
    args = report["arguments"]
    print(f"\nAttention with QKV: seq={args['seq']}, kv={args['kv']}, "
          f"head_dim={args['head_dim']}, input_dim={args['head_dim']}")
    print(f"CPU threads={report['cpu_threads']}; UPMEM_PROFILE={report['upmem_profile']}")
    rows = [[r["operator"], r["shape"], r["where"], f"{r['time_ms']:.4f}",
             f"{r['percent_total']:.2f}", r["dtype"]] for r in records]
    rows.extend([
        ["wrapper/alloc/tracing/other", "-", "CPU", f"{report['other_ms']:.4f}",
         f"{100 * report['other_ms'] / total_ms:.2f}", "-"],
        ["Profiled call TOTAL", shape_text(report["output_shape"]), "CPU+UPMEM",
         f"{total_ms:.4f}", "100.00", "float32"],
    ])
    print_table(["operator", "shape", "where", "time (ms)", "% total", "dtype"], rows)

    pim_records = [r for r in records if r["pim_stages_ms"] is not None]
    stages = list(pim_records[0]["pim_stages_ms"])
    rows = []
    for record in pim_records:
        timings = record["pim_stages_ms"]
        options = record["pim_config"]["backend_options"]
        rows.append([record["operator"], *(f"{timings[name]:.4f}" for name in stages),
                     f"{record['time_ms'] - sum(timings.values()):.4f}",
                     options["p_m"] * options["p_n"] * options["p_k"], options["host_threads"]])
    print("\nPIM breakdown (ms; already included in the PIM rows above)")
    print_table(["operator", *stages, "launch other", "active DPUs", "host threads"], rows)
    print(f"\nExecuted kernels: {len(records)} = {len(pim_records)} PIM + {len(records) - len(pim_records)} CPU.")
    print("CPU shapes include outputs/scratch. Fused CPU operations share one measured time.")
    print("PIM times include host work and transfers; kernel includes DPU compute/internal DMA.")
    print("Profiled call TOTAL includes Python dispatch and tracing overhead.")
    print(f"Attention TOTAL (QKV + attention, one plain call): {report['wall_total_ms']:.4f} ms")
    print("Plain call is measured separately with tracing removed; compare with PIM-LLM's plain-call TOTAL.")
    print("Compilation, warm-ups, reference checks and printing are excluded from both measurements.")
    print("Checks: output, scores, pq, row_scale PASS (profiled and plain calls).")
    if verbose:
        for record in records:
            print(f"\n{record['operator']}: {record['kernel']}")
            if record["pim_config"] is not None:
                print(json.dumps(record["pim_config"], sort_keys=True))
    print("OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=256)       # M
    parser.add_argument("--kv", type=int, default=None,
                        help="defaults to --seq; must equal --seq for self-attention")
    parser.add_argument("--head-dim", type=int, default=128)  # QK의 K
    parser.add_argument("--warmup", type=int, default=3,
                        help="calls before profiling and the separate plain call (including compilation)")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="set CPU threads before compilation; otherwise keep Torch's default")
    parser.add_argument("--json", type=Path,
                        help="save kernel timings, PIM breakdowns, and separate plain-call time")
    parser.add_argument("--save-inputs", type=Path,
                        help="save X/weights/mask and CPU reference output as NPZ for PIM-LLM comparison")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="고른 실행 계획(block/group/tile/device 분할)을 찍는다")
    args = parser.parse_args()
    if args.kv is None:
        args.kv = args.seq
    if args.seq < 1 or args.head_dim < 1 or args.warmup < 1:
        parser.error("--seq, --head-dim and --warmup must be positive")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.kv != args.seq:
        parser.error("Q/K/V use the same X: self-attention requires --kv == --seq")

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("torch_backend").setLevel(logging.INFO)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    config = build_config()

    torch.manual_seed(0)
    x = torch.randint(-8, 8, (args.seq, args.head_dim), dtype=torch.int8)
    wq, wk, wv = (torch.randint(-8, 8, (args.head_dim, args.head_dim), dtype=torch.int8)
                  for _ in range(3))
    mask = torch.zeros(args.seq, args.kv, dtype=torch.bool)

    inputs = (x, wq, wk, wv, mask, args.head_dim)
    print(f"Preparing {args.warmup} call(s), including compilation ...", flush=True)
    from torch._inductor import config as inductor_config
    from torch._inductor.codecache import PyCodeCache

    # Time the generated wrapper's existing calls after compilation. No timers
    # inside the attention graph and no changes to CPU/PIM fusion boundaries.
    with torch.no_grad(), inductor_config.patch({"cpp_wrapper": False}):
        previous_modules = {id(module) for module in PyCodeCache.modules}
        compiled = torch.compile(attention_group_inductor, backend=PIMInductorBackend(config),
                                 fullgraph=True, dynamic=False)
        preparation_start = time.perf_counter_ns()
        for _ in range(args.warmup):
            compiled(*inputs)
        preparation_ms = (time.perf_counter_ns() - preparation_start) / 1e6
        modules = [module for module in PyCodeCache.modules if id(module) not in previous_modules]
        print(f"Preparation: {preparation_ms:.3f} ms (excluded). Profiling one call ...", flush=True)
        with trace_kernels(modules, pim_labeler=make_pim_labeler(x, wq, wk, wv)) as records:
            start = time.perf_counter_ns()
            profiled = compiled(*inputs)
            total_ms = (time.perf_counter_ns() - start) / 1e6
        # Match PIM-LLM's separate wall-time sample after removing tracing.
        start = time.perf_counter_ns()
        got = compiled(*inputs)
        wall_total_ms = (time.perf_counter_ns() - start) / 1e6
        ref = attention_group_inductor(*inputs)

    for measured in (profiled, got):
        for a, b in zip(measured, ref):
            if a.dtype.is_floating_point:
                torch.testing.assert_close(a, b)
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    pim_labels = [r["operator"] for r in records if r["pim_stages_ms"] is not None]
    if Counter(pim_labels) != Counter(("Q projection", "K projection", "V projection", "QK^T", "PV")):
        raise RuntimeError(f"Expected five attention GEMMs in the measured wrapper, found {pim_labels}")
    describe_records(records, total_ms, pim_labels=None)
    report = {
        "framework": "Torch / Triton PIM",
        "scope": "single-head QKV projection + attention, FP32 output",
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "upmem_profile": os.environ.get("UPMEM_PROFILE"),
        "preparation_ms": preparation_ms,
        "warmup_calls": args.warmup, "profiled_calls": 1, "plain_timed_calls": 1,
        "total_ms": total_ms,
        "total_scope": "compiled-call interval including Python dispatch and kernel tracing",
        "wall_total_ms": wall_total_ms,
        "wall_total_scope": "separate warmed compiled call with tracing removed",
        "other_ms": total_ms - sum(r["time_ms"] for r in records),
        "output_shape": list(got[0].shape),
        "checks": {name: "PASS" for name in ("output", "scores", "pq", "row_scale")},
        "kernels": records,
    }
    print_attention_report(report, args.verbose)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Timings saved: {args.json}")
    if args.save_inputs is not None:
        import numpy as np
        args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
        with args.save_inputs.open("wb") as handle:
            np.savez(handle, x=x.numpy(), wq=wq.numpy(), wk=wk.numpy(), wv=wv.numpy(),
                     mask=mask.numpy(), reference_output=ref[0].numpy())
        print(f"Inputs and CPU reference saved: {args.save_inputs}")


if __name__ == "__main__":
    main()
