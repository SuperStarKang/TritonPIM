"""Print the kernels executed by one compiled CPU/UPMEM attention call.

One command performs warm-up (including compilation), then ONE measured call.
Timers wrap the generated Python module's C++ functions and PIM .run methods
after warm-up. The attention graph and its fusion boundaries stay intact.
This uses the local Torch Inductor Python wrapper / PyCodeCache API.
"""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from torch._inductor import config as inductor_config  # noqa: E402
from torch._inductor.codecache import PyCodeCache  # noqa: E402
import triton  # noqa: E402
import triton.backends.pim.driver as pim_driver  # noqa: E402

from torch_backend import (  # noqa: E402
    PIMInductorBackend, PIMTorchConfig, target_spec,
)
from torch_attention_gqa import attention_group, make_inputs  # noqa: E402


def attention(q, k, v, mask, head_dim):
    # Reuse the existing quantized attention, returning only its final FP32
    # output. Output dequantization is part of the compiled, measured graph.
    output, _, _, row_scale = attention_group(q, k, v, mask, head_dim)
    return output.float() * row_scale / 127


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def tensor_metadata(args):
    # Keep metadata, not tensors: retaining intermediate buffers would change
    # memory lifetime across the rest of the attention call.
    return [
        {"argument": index, "shape": list(arg.shape),
         "stride": list(arg.stride()),
         "dtype": str(arg.dtype).removeprefix("torch.")}
        for index, arg in enumerate(args) if isinstance(arg, torch.Tensor)
    ]


def is_pim_kernel(name, value):
    return (name.startswith("triton_")
            and getattr(value, "triton_meta", {}).get("compile_backend") == "pim"
            and callable(getattr(value, "run", None)))


class KernelProxy:
    """Replace only the generated module's reference to a warmed PIM kernel."""

    def __init__(self, kernel, run):
        self._kernel = kernel
        self.run = run

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def timed_kernel(fn, name, module, records, pim_kernel=None, pim_labeler=None):
    options = (dict(pim_kernel.triton_meta["backend_options"])
               if pim_kernel is not None else None)
    blocks = (dict(pim_kernel.inductor_meta.get("config_args", {}))
              if pim_kernel is not None else None)

    def measured(*args, **kwargs):
        start = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        # Both C++ calls and the UPMEM ctypes launcher are synchronous. Copy
        # stages now: the next PIM launch overwrites the driver's global array.
        stages = (dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing))
                  if pim_kernel is not None else None)
        record = {
            "kernel": name,
            "module": module.__file__,
            "where": "CPU+UPMEM" if pim_kernel is not None else "CPU",
            "time_ms": elapsed_ms,
            "tensor_arguments": tensor_metadata(args),
            "pim_stages_ms": stages,
            "pim_config": ({"blocks": blocks, "backend_options": options}
                           if pim_kernel is not None else None),
        }
        if pim_kernel is not None and pim_labeler is not None:
            record["operator"] = pim_labeler(args)
        records.append(record)
        return result

    return measured


@contextmanager
def trace_kernels(modules, pim_labeler=None):
    records = []
    pim_count = 0
    with ExitStack() as stack:
        for module in modules:
            for name, value in list(vars(module).items()):
                if is_pim_kernel(name, value):
                    replacement = KernelProxy(value, timed_kernel(
                        value.run, name, module, records, pim_kernel=value,
                        pim_labeler=pim_labeler))
                    pim_count += 1
                elif name.startswith("cpp_") and callable(value):
                    replacement = timed_kernel(value, name, module, records)
                else:
                    continue
                stack.enter_context(patch.object(module, name, replacement))
        if pim_count == 0:
            raise RuntimeError(
                "No generated PIM Python wrapper found after warm-up. "
                "This profiler requires the local Inductor PyCodeCache API "
                "and generated triton_* / cpp_* kernel calls.")
        yield records


def shape_text(shape):
    return "[" + ",".join(map(str, shape)) + "]"


def cpu_operator(name):
    # Descriptive names describe the whole generated kernel, not separately
    # timed ATen operations. Retain the actual name in JSON / --verbose.
    body = re.sub(r"_\d+$", "", name.removeprefix("cpp_"))
    fused = body.startswith("fused_")
    body = body.removeprefix("fused_")
    if "softmax" in body:
        parts = []
        if "div" in body:
            parts.append("scale/normalize")
        if "masked_fill" in body:
            parts.append("mask")
        parts.append("softmax")
        if all(op in body for op in ("amax", "round", "clamp")):
            parts.append("quantize")
        label = "+".join(parts)
    else:
        body = (body.replace("_to_copy", "cast")
                .replace("masked_fill", "mask")
                .replace("clone", "contiguous"))
        label = "+".join(part for part in body.split("_") if part)
    return label + (" (fused)" if fused else "")


def describe_records(records, total_ms, pim_labels=("QK^T", "PV")):
    pim_index = 0
    for record in records:
        tensors = record["tensor_arguments"]
        shapes = [shape_text(t["shape"]) for t in tensors]
        dtypes = [t["dtype"] for t in tensors]
        if record["pim_stages_ms"] is not None:
            if len(tensors) != 3:
                raise RuntimeError("Expected PIM GEMM tensor arguments A, B, C")
            if "operator" not in record:
                if pim_labels is None:
                    raise RuntimeError("No semantic label supplied for PIM GEMM")
                record["operator"] = pim_labels[pim_index]
            pim_index += 1
            record["shape"] = f"{shapes[0]} @ {shapes[1]} -> {shapes[2]}"
            record["dtype"] = f"{dtypes[0]} @ {dtypes[1]} -> {dtypes[2]}"
        else:
            record["operator"] = cpu_operator(record["kernel"])
            # CPU arguments include outputs and reduction scratch buffers.
            # No input/output arrow: their ABI does not encode that distinction.
            record["shape"] = ", ".join(dict.fromkeys(shapes)) or "-"
            record["dtype"] = ", ".join(dict.fromkeys(dtypes)) or "-"
        record["percent_total"] = 100 * record["time_ms"] / total_ms


def print_table(headers, rows):
    strings = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in strings))
              for i, header in enumerate(headers)]
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in strings:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def print_report(report, verbose):
    records = report["kernels"]
    total_ms = report["total_ms"]
    other_ms = report["other_ms"]
    rows = [[r["operator"], r["shape"], r["where"], f'{r["time_ms"]:.4f}',
             f'{r["percent_total"]:.2f}', r["dtype"]] for r in records]
    rows.extend([
        ["wrapper/alloc/tracing/other", "-", "CPU", f"{other_ms:.4f}",
         f"{100 * other_ms / total_ms:.2f}", "-"],
        ["Attention TOTAL (one call)", shape_text(report["output"]["shape"]),
         "CPU+UPMEM", f"{total_ms:.4f}", "100.00", report["output"]["dtype"]],
    ])
    print()
    print_table(["operator", "shape", "where", "time (ms)", "% total", "dtype"], rows)

    stage_names = pim_driver.TIMING_STAGES
    breakdown = []
    for record in records:
        stages = record["pim_stages_ms"]
        if stages is None:
            continue
        options = record["pim_config"]["backend_options"]
        dpus = options["p_m"] * options["p_n"] * options["p_k"]
        breakdown.append([
            record["operator"], *(f"{stages[name]:.4f}" for name in stage_names),
            f'{record["time_ms"] - sum(stages.values()):.4f}', dpus,
        ])
    print("\nPIM breakdown (ms; already included in QK^T / PV above)")
    print_table(["operator", *stage_names, "launch other", "active DPUs"], breakdown)
    print("\nCPU shape: distinct tensor argument shapes, including outputs/scratch.")
    print("Fused CPU operations share one time; no per-ATen time is inferred.")
    print("PIM rows include host work and transfers; kernel includes DPU compute/internal DMA.")
    print("H2D/D2H measure prepare+push, matching PGEMM; kernel measures synchronous dpu_launch.")
    print("buffer_alloc/init/free separate storage management from layout_a/b/other packing.")
    print("layout_c writes the first partial; reduction adds later partials at output addresses.")
    print("TOTAL includes tracing overhead; preparation, input generation and printing are excluded.")
    if verbose:
        for record in records:
            print(f'\n{record["operator"]}: {record["kernel"]}')
            if record["pim_config"] is not None:
                print(json.dumps(record["pim_config"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=positive_int, default=256)
    parser.add_argument("--head-dim", type=positive_int, default=128)
    parser.add_argument("--group", type=positive_int, default=1,
                        help="query heads sharing one KV head; default: one head")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=positive_int, default=1,
                        help="preparation calls excluded from the one measured call")
    parser.add_argument("--num-units", type=positive_int, default=2048,
                        help="available DPU budget, not necessarily active DPU count")
    parser.add_argument("--host-threads", type=positive_int, default=8)
    parser.add_argument("--cpu-threads", type=positive_int, default=None)
    parser.add_argument("--cache", default=None, help="Torch GEMM autotune cache for QK and PV")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-inputs", type=Path, default=None,
                        help="save q/k/v/mask as .npz before timing, for the TVM profiler")
    parser.add_argument("--json", type=Path, default=None, help="save timings and configuration")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    target = target_spec("upmem", num_units=args.num_units, host_threads=args.host_threads)
    config = PIMTorchConfig(spec=target, autotune_cache_path=args.cache)
    q, k, v, mask = make_inputs(args.group, args.seq, args.head_dim, args.causal, args.seed)
    if args.save_inputs is not None:
        import numpy as np
        args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
        with args.save_inputs.open("wb") as handle:
            np.savez(handle, q=q.numpy(), k=k.numpy(), v=v.numpy(), mask=mask.numpy())
        print(f"Inputs saved: {args.save_inputs}", flush=True)
    print(f"One attention core: group={args.group}, seq={args.seq}, "
          f"head_dim={args.head_dim}, causal={args.causal}", flush=True)
    print(f"Q/K/V=int8, output=float32; config={args.cache or 'heuristic seed (untuned)'}", flush=True)
    print(f"Preparing {args.warmup} call(s), including compilation ...", flush=True)

    # Instrument module globals only after compilation. No timers inside the
    # source function, graph breaks, separate operator replays, or CPU defusion.
    with torch.no_grad(), inductor_config.patch({"cpp_wrapper": False}):
        previous_modules = {id(module) for module in PyCodeCache.modules}
        compiled = torch.compile(attention, backend=PIMInductorBackend(config),
                                 fullgraph=True, dynamic=False)
        start = time.perf_counter_ns()
        for _ in range(args.warmup):
            compiled(q, k, v, mask, args.head_dim)
        preparation_ms = (time.perf_counter_ns() - start) / 1e6
        modules = [module for module in PyCodeCache.modules
                   if id(module) not in previous_modules]
        print(f"Preparation: {preparation_ms:.3f} ms (excluded). Measuring one call ...", flush=True)
        with trace_kernels(modules) as records:
            start = time.perf_counter_ns()
            output = compiled(q, k, v, mask, args.head_dim)
            total_ms = (time.perf_counter_ns() - start) / 1e6

    if sum(r["pim_stages_ms"] is not None for r in records) != 2:
        raise RuntimeError(
            "Expected exactly two measured PIM calls (QK and PV). "
            "The generated wrapper may have changed or recompiled during measurement.")
    describe_records(records, total_ms)
    report = {
        "scope": "one KV group attention core, with FP32 output dequantization",
        "measured_calls": 1,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "versions": {"torch": torch.__version__, "triton": triton.__version__},
        "target": asdict(target),
        "upmem_profile": os.environ.get("UPMEM_PROFILE"),
        "cpu_threads": torch.get_num_threads(),
        "preparation_ms": preparation_ms,
        "total_ms": total_ms,
        "other_ms": total_ms - sum(r["time_ms"] for r in records),
        "pim_timing_version": pim_driver.UPMEM_TIMING_VERSION,
        "output": tensor_metadata((output,))[0],
        "kernels": records,
    }
    print(f"UPMEM_PROFILE={report['upmem_profile']}; CPU threads={report['cpu_threads']}; "
          f"host threads={args.host_threads}")
    print_report(report, args.verbose)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved: {args.json}")


if __name__ == "__main__":
    main()
