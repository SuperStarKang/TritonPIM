"""Shared performance measurement for the fixed batch-one INT8 experiments."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
import triton
import triton.backends.pim.driver as pim_driver

from autotuner import PIMConfig
from torch_backend import PIMSession, PIMTorchConfig, ResidentGemmConfig, compile_layer, target_spec
from torch_backend.fx_analysis import analyze_fx_gemms
from torch_backend.resident import _ResidentGemm

from fx_utils import write_fx_graph


def parse_args(description, report_dir):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--report-dir", type=Path, default=report_dir)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 1:
        parser.error("--warmup and --repeat must be at least 1")
    return args


def gemm_configs(gemms, spec):
    placements = {
        (1024, 4096, 4096): (16, 32, 64),
        (1024, 1024, 128): (16, 32, 64),
        (1024, 128, 1024): (4, 64, 32),
        (1024, 11008, 4096): (16, 32, 64),
        (1024, 4096, 11008): (16, 32, 64),
    }
    configs = {}
    for gemm in gemms:
        m, n, k = gemm.mnk
        block_n, p_m, p_n = placements[gemm.mnk]
        padded_n = ((n + block_n * p_n - 1) // (block_n * p_n)) * (block_n * p_n)
        configs[gemm.node] = ResidentGemmConfig(
            PIMTorchConfig(spec=spec, config=PIMConfig(
                block_m=16, block_n=block_n, block_k=32,
                group_m=1, group_n=1, group_k=1,
                tile_m=m // (16 * p_m), tile_n=padded_n // (block_n * p_n),
                tile_k=k // 32, p_m=p_m, p_n=p_n, p_k=1,
            )),
            orientation="normal", padded_shape=(m, padded_n, k),
        )
    return configs


def profile_partition(name, module, session, labels, records):
    forward = module.forward
    is_pim = isinstance(module, _ResidentGemm)
    operators = [labels[module.name]["operation"]] if is_pim else [
        str(node.target) for node in module._orig_mod.graph.nodes if node.op == "call_function"
    ]

    def measured(*args, **kwargs):
        before = session.stats() if is_pim else None
        start = time.perf_counter_ns()
        output = forward(*args, **kwargs)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        record = {
            "partition": name, "where": "CPU+PIM" if is_pim else "CPU",
            "operators": operators, "wall_ms": elapsed_ms,
        }
        if is_pim:
            record["gemm"] = module.name
            record.update(labels[module.name])
            record["native_ms"] = dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing, strict=True))
            record["h2d_ms"] = dict(pim_driver.resident_h2d_timing)
            after = session.stats()
            record["transfers"] = {key: after[key] - before[key] for key in before}
            record["outside_native_stages_ms"] = elapsed_ms - sum(record["native_ms"].values())
        records.append(record)
        return output

    return measured


def benchmark(model, inputs, args, *, scope, heads, expected_gemms, expected_weights):
    logging.basicConfig(level=logging.INFO)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        print(f"Exporting {scope}: batch=1 sequence=1024 heads={heads} head_dim=128 ...", flush=True)
        start = time.perf_counter_ns()
        exported = torch.export.export(model, inputs)
        gemms = analyze_fx_gemms(exported)
        export_ms = (time.perf_counter_ns() - start) / 1e6
        if len(gemms) != expected_gemms:
            raise ValueError(f"expected {expected_gemms} FX GEMMs, found {len(gemms)}")
        if sum(gemm.right.weight_target is not None for gemm in gemms) != expected_weights:
            raise ValueError(f"expected {expected_weights} resident projection weights")
        if any(gemm.op != "aten._int_mm.default" or gemm.left.weight_target is not None for gemm in gemms):
            raise ValueError("this benchmark requires INT8 GEMMs with activation on the left")

        spec = target_spec("upmem", num_units=2048, nr_tasklets=16)
        configs = gemm_configs(gemms, spec)
        labels = {}
        for gemm in gemms:
            if gemm.right.weight_target is not None:
                operation = gemm.right.weight_target.removesuffix(".weight")
            else:
                operation = {(1024, 1024, 128): "QK^T", (1024, 128, 1024): "PV"}[gemm.mnk]
            labels[gemm.node] = {
                "operation": operation, "mnk": list(gemm.mnk),
                "weight": gemm.right.weight_target,
                "padded_mnk": list(configs[gemm.node].padded_shape),
                "config": asdict(configs[gemm.node].torch_config.config),
            }
        write_fx_graph(exported.graph_module, args.report_dir / "exported_fx", {
            name: json.dumps(label) for name, label in labels.items()
        })
        print(f"Compiling {len(gemms)} INT8 GEMMs and CPU partitions (excluded from samples) ...", flush=True)
        plan = compile_layer(exported, inputs, configs=configs, index=0)
        write_fx_graph(plan.graph, args.report_dir / "partitioned_fx", {
            node.name: labels[plan.graph.get_submodule(node.target).name]["operation"]
            for node in plan.graph.graph.nodes
            if node.op == "call_module" and isinstance(plan.graph.get_submodule(node.target), _ResidentGemm)
        })
        print(f"Compile: {plan.compile_ms:.3f} ms. Opening 2048 units ...", flush=True)
        samples = []
        warmups = []
        records = []
        session = PIMSession(spec)
        with torch._dynamo.config.patch(error_on_recompile=True), session:
            print("Preparing layer placement and uploading weights once ...", flush=True)
            token = session.load_weights(plan, session.start())
            load_record = dict(session.records[-1])
            print(f"Weight load: {load_record['weight_load_ms']:.3f} ms; "
                  f"weight_bytes={load_record['load_transfers']['weight_bytes']} "
                  f"upload_h2d_ms={load_record['upload_breakdown_ms']['h2d']:.3f} "
                  f"upload_program_load_ms={load_record['upload_breakdown_ms']['program_load']:.3f}",
                  flush=True)
            for step in range(args.warmup):
                output, token = session.run_layer(plan, inputs, token)
                warmups.append({"run_ms": session.records[-1]["run_ms"],
                                "transfers": session.records[-1]["run_transfers"]})
                print(f"warmup={step} run_ms={warmups[-1]['run_ms']:.3f}", flush=True)
                del output
            for step in range(args.repeat):
                start = time.perf_counter_ns()
                output, token = session.run_layer(plan, inputs, token)
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                samples.append({"wall_ms": wall_ms, "run_ms": session.records[-1]["run_ms"],
                                "transfers": session.records[-1]["run_transfers"]})
                print(f"sample={step} wall_ms={wall_ms:.3f} "
                      f"weight_h2d={samples[-1]['transfers']['weight_bytes']} "
                      f"activation_h2d={samples[-1]['transfers']['activation_bytes']} "
                      f"activation_hits={samples[-1]['transfers']['activation_hits']} "
                      f"launches={samples[-1]['transfers']['launches']}", flush=True)
                del output

            print("Profiling one additional call (excluded from the sample median) ...", flush=True)
            with ExitStack() as instrumentation:
                for name, module in plan.graph.named_children():
                    instrumentation.enter_context(patch.object(
                        module, "forward", profile_partition(name, module, session, labels, records),
                    ))
                start = time.perf_counter_ns()
                output, token = session.run_layer(plan, inputs, token)
                profile_ms = (time.perf_counter_ns() - start) / 1e6
            profile_transfers = session.records[-1]["run_transfers"]

        operations = {}
        for record in records:
            if record["where"] != "CPU+PIM":
                continue
            operation = record["operation"]
            if operation not in operations:
                operations[operation] = {"calls": 0, "wall_ms": 0.0,
                                         "native_ms": {name: 0.0 for name in pim_driver.TIMING_STAGES},
                                         "h2d_ms": {role: 0.0 for role in ("weight", "activation", "metadata")}}
            operations[operation]["calls"] += 1
            operations[operation]["wall_ms"] += record["wall_ms"]
            for name, elapsed in record["native_ms"].items():
                operations[operation]["native_ms"][name] += elapsed
            for role, elapsed in record["h2d_ms"].items():
                operations[operation]["h2d_ms"][role] += elapsed
        median_ms = statistics.median(sample["wall_ms"] for sample in samples)
        gemm_operations = sum(2 * m * n * k for m, n, k in (gemm.mnk for gemm in gemms))
        report = {
            "scope": scope, "batch": 1, "sequence": 1024, "heads": heads, "head_dim": 128,
            "hidden_dim": None if scope == "attention_kernel" else 4096,
            "ffn_dim": 11008 if scope == "transformer_block" else None,
            "inputs": [{"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for tensor in inputs],
            "output": {"shape": list(output.shape), "dtype": str(output.dtype)},
            "gemm_dtype": "int8 x int8 -> int32", "cpu_dtype": "float32",
            "activation_transfer_policy": "always",
            "quantization": {"weights": "synthetic int8 [-8, 7], scale=1/(8*sqrt(K))",
                             "activation_scale": 1.0 / 32.0, "probability": "per-row max/127"},
            "causal": True, "seed": 0, "orientation": "normal", "accuracy_checked": False,
            "versions": {"torch": torch.__version__, "triton": triton.__version__},
            "target": asdict(spec), "cpu_threads": torch.get_num_threads(),
            "upmem_backend": os.environ.get("UPMEM_BACKEND"),
            "upmem_profile": os.environ.get("UPMEM_PROFILE"),
            "gemms": labels, "resident_weights": expected_weights,
            "preparation": {"export_ms": export_ms, "compile_ms": plan.compile_ms,
                            "session_open_ms": session.open_ms, "weight_load": load_record},
            "warmups": warmups, "samples": samples, "median_wall_ms": median_ms,
            "logical_gemm_operations": gemm_operations,
            "logical_gemm_gops_per_s": gemm_operations / (median_ms * 1e6),
            "profile": {"wall_ms": profile_ms, "transfers": profile_transfers,
                        "timing_version": pim_driver.TIMING_VERSION,
                        "partitions": records, "gemm_totals": operations,
                        "cpu_partitions_ms": sum(r["wall_ms"] for r in records if r["where"] == "CPU"),
                        "outside_partitions_ms": profile_ms - sum(r["wall_ms"] for r in records)},
            "session_close_ms": session.close_ms, "session_totals": session.stats(),
        }
        (args.report_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(f"{scope}: median_wall_ms={median_ms:.3f} "
              f"logical_GEMM_GOPS={report['logical_gemm_gops_per_s']:.3f}", flush=True)
        for operation, totals in operations.items():
            print(f"  {operation}: calls={totals['calls']} wall_ms={totals['wall_ms']:.3f} "
                  f"h2d_ms={totals['native_ms']['h2d']:.3f} "
                  f"program_load_ms={totals['native_ms']['program_load']:.3f} "
                  f"resident_host_ms={totals['native_ms']['resident_host']:.3f} "
                  f"kernel_ms={totals['native_ms']['kernel']:.3f}")
        print(f"resource_opens={report['session_totals']['resource_opens']} "
              f"resource_closes={report['session_totals']['resource_closes']}")
        print(f"Report and FX graphs: {args.report_dir.resolve()}", flush=True)
