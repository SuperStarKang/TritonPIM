"""Hardware measurements for one prepared model LayerPlan with resident weights."""

import ctypes
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch import fx
import triton
import triton.backends.pim.driver as pim_driver

from autotuner import PIMConfig
from torch_backend import PIMSession, PIMTorchConfig, target_spec
from torch_backend.model import _LayerGemm

from fx_utils import write_fx_graph


def add_benchmark_arguments(parser, report_dir):
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--report-dir", type=Path, default=report_dir)
    parser.add_argument("--gemm-config", choices=("baseline", "batched"), default="batched",
                        help="baseline uses 2048 DPUs per head; batched uses 64")


def model_configs(gemms, spec, attention_config):
    configs = {
        (1024, 4096, 4096): PIMConfig(16, 16, 32, 1, 1, 1, 2, 4, 128, 32, 64, 1),
        (1024, 11264, 4096): PIMConfig(16, 16, 32, 1, 1, 1, 2, 11, 128, 32, 64, 1),
        (1024, 4096, 11008): PIMConfig(16, 16, 32, 1, 1, 1, 2, 4, 344, 32, 64, 1),
    }
    if attention_config == "baseline":
        configs[(1024, 1024, 128)] = PIMConfig(16, 16, 32, 1, 1, 1, 2, 1, 4, 32, 64, 1)
        configs[(1024, 128, 1024)] = PIMConfig(16, 4, 32, 1, 1, 1, 1, 1, 32, 64, 32, 1)
    else:
        configs[(1024, 1024, 128)] = PIMConfig(16, 16, 32, 1, 1, 1, 8, 8, 4, 8, 8, 1)
        configs[(1024, 128, 1024)] = PIMConfig(16, 16, 32, 1, 1, 1, 8, 1, 32, 8, 8, 1)
    return {gemm.node: PIMTorchConfig(spec=spec, config=configs[gemm.mnk]) for gemm in gemms}


class ModelProfile(fx.Interpreter):
    def __init__(self, plan, session, labels):
        super().__init__(plan.graph)
        self.session = session
        self.labels = labels
        self.records = []

    def call_module(self, target, args, kwargs):
        module = self.module.get_submodule(target)
        stages = [child for child in module.modules() if isinstance(child, _LayerGemm)]
        before = self.session.stats() if stages else None
        start = time.perf_counter_ns()
        output = super().call_module(target, args, kwargs)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        record = {"partition": target, "where": "CPU+PIM" if stages else "CPU", "wall_ms": elapsed}
        if stages:
            stage, = stages
            record.update(self.labels[stage.name])
            record["stage"] = stage.name
            record["instances"] = stage.instance_count
            names = pim_driver._STAGE_NAMES["upmem"]
            values = (ctypes.c_double * len(names)).in_dll(stage.library, "pim_timing_ms")
            record["native_ms"] = dict(zip(names, values, strict=True))
            record["outside_native_stages_ms"] = elapsed - sum(values)
            after = self.session.stats()
            record["transfers"] = {name: after[name] - before[name] for name in before}
        else:
            record["operators"] = [str(node.target) for node in module._orig_mod.graph.nodes
                                   if node.op == "call_function"]
        self.records.append(record)
        return output


def benchmark_model(model, inputs, args, *, scope, heads, expected_gemms, expected_weights):
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError("--warmup and --repeat must be at least 1")
    logging.basicConfig(level=logging.INFO)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    labels = {}

    def configure(gemms, spec):
        if len(gemms) != expected_gemms:
            raise ValueError(f"expected {expected_gemms} FX GEMMs, found {len(gemms)}")
        if sum(gemm.right.weight_target is not None for gemm in gemms) != expected_weights:
            raise ValueError(f"expected {expected_weights} projection weights")
        configs = model_configs(gemms, spec, args.gemm_config)
        for gemm in gemms:
            if gemm.right.weight_target is not None:
                operation = gemm.right.weight_target.removesuffix(".weight")
            else:
                operation = {(1024, 1024, 128): "QK^T", (1024, 128, 1024): "PV"}[gemm.mnk]
            logical = (1024, 11008, 4096) if gemm.mnk == (1024, 11264, 4096) else gemm.mnk
            labels[gemm.node] = {
                "operation": operation, "mnk": list(logical),
                "padded_mnk": list(gemm.mnk), "weight": gemm.right.weight_target,
                "config": asdict(configs[gemm.node].config),
            }
        return configs

    with torch.no_grad():
        session = PIMSession(target_spec("upmem", num_units=2048, nr_tasklets=16))
        with session:
            start = time.perf_counter_ns()
            plan, = session.prepare([model], inputs, configure=configure)
            prepare_ms = (time.perf_counter_ns() - start) / 1e6
            print(f"scope={scope} heads={heads} gemm_config={args.gemm_config} "
                  f"logical_gemms={len(labels)} planned_stages={len(plan.stages)}", flush=True)
            write_fx_graph(plan.exported_graph, args.report_dir / "exported_fx", {
                name: json.dumps(label) for name, label in labels.items()
            })
            stage_records = []
            for stage in plan.stages:
                record = {
                    "name": stage.name, **labels[stage.name],
                    "instances": stage.instance_count, "p_b": stage.p_b,
                    "batch_rounds": stage.batch_rounds, "mnk_rounds": list(stage.mnk_rounds),
                    "execution_extent": list(stage.config.execution_extent),
                    "participating_units": stage.p_b * stage.config.p_m * stage.config.p_n * stage.config.p_k,
                    "expected_launches": stage.batch_rounds * math.prod(stage.mnk_rounds),
                    "program": stage.program,
                }
                stage_records.append(record)
                print(f"stage={stage.name} operation={record['operation']} "
                      f"instances={stage.instance_count} p_b={stage.p_b} "
                      f"batch_rounds={stage.batch_rounds} mnk_rounds={stage.mnk_rounds} "
                      f"extent={stage.config.execution_extent}", flush=True)
            write_fx_graph(plan.graph, args.report_dir / "partitioned_fx", {})
            expected_launches = sum(record["expected_launches"] for record in stage_records)
            before = session.stats()
            start = time.perf_counter_ns()
            # Use the session's weight loader once, matching the legacy layer benchmark boundary.
            session.backend.model.load_weights(plan)
            session.loaded_layer = plan.index
            weight_load_ms = (time.perf_counter_ns() - start) / 1e6
            after = session.stats()
            weight_load = {"weight_load_ms": weight_load_ms,
                           "load_transfers": {name: after[name] - before[name] for name in before}}
            print(f"prepare_ms={prepare_ms:.3f} weight_load_ms={weight_load_ms:.3f} "
                  f"weight_bytes={weight_load['load_transfers']['weight_bytes']} "
                  f"expected_launches={expected_launches}", flush=True)
            warmups = []
            samples = []
            with torch._dynamo.config.patch(error_on_recompile=True):
                for phase, count, records in (("warmup", args.warmup, warmups),
                                              ("sample", args.repeat, samples)):
                    for step in range(count):
                        before = session.stats()
                        start = time.perf_counter_ns()
                        output = plan.run_model(*inputs)
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        after = session.stats()
                        transfers = {name: after[name] - before[name] for name in before}
                        assert transfers["weight_bytes"] == transfers["device_compiles"] == 0
                        assert transfers["launches"] == expected_launches
                        records.append({"wall_ms": elapsed, "run_ms": elapsed, "transfers": transfers})
                        print(f"{phase}={step} wall_ms={elapsed:.3f} weight_h2d={transfers['weight_bytes']} "
                              f"activation_h2d={transfers['activation_bytes']} "
                              f"launches={transfers['launches']} program_loads={transfers['program_loads']}",
                              flush=True)
                        del output
                profiler = ModelProfile(plan, session, labels)
                before = session.stats()
                start = time.perf_counter_ns()
                profile_output, = profiler.run(*inputs)
                profile_ms = (time.perf_counter_ns() - start) / 1e6
                after = session.stats()
                profile_transfers = {name: after[name] - before[name] for name in before}
        median_ms = statistics.median(sample["wall_ms"] for sample in samples)
        logical_operations = sum(2 * math.prod(label["mnk"]) for label in labels.values())
        operations = {}
        for record in profiler.records:
            if record["where"] == "CPU+PIM":
                name = record["operation"]
                if name not in operations:
                    operations[name] = {"calls": 0, "instances": 0, "wall_ms": 0.0}
                operations[name]["calls"] += 1
                operations[name]["instances"] += record["instances"]
                operations[name]["wall_ms"] += record["wall_ms"]
        report = {
            "scope": scope, "runtime": "model", "gemm_config": args.gemm_config,
            "batch": 1, "sequence": 1024, "heads": heads, "head_dim": 128,
            "hidden_dim": 4096 if scope in ("attention_layer", "transformer_block") else None,
            "ffn_dim": 11008 if scope == "transformer_block" else None,
            "inputs": [{"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for tensor in inputs],
            "output": {"shape": list(profile_output.shape), "dtype": str(profile_output.dtype)},
            "gemm_dtype": "int8 x int8 -> int32", "cpu_dtype": "float32",
            "causal": True, "seed": 0, "orientation": "normal", "accuracy_checked": False,
            "activation_transfer_policy": "always", "resident_weights": expected_weights,
            "quantization": {"weights": "synthetic int8 [-8, 7], scale=1/(8*sqrt(K))",
                             "activation_scale": 1.0 / 32.0, "probability": "per-row max/127"},
            "measurement_scope": "CPU/PIM layer execution with weights already loaded; excludes prepare/upload",
            "versions": {"torch": torch.__version__, "triton": triton.__version__},
            "target": asdict(session.spec), "cpu_threads": torch.get_num_threads(),
            "upmem_backend": os.environ.get("UPMEM_BACKEND"),
            "upmem_profile": os.environ.get("UPMEM_PROFILE"),
            "gemms": labels, "stages": stage_records, "expected_launches": expected_launches,
            "unique_device_artifacts": len({stage.program for stage in plan.stages}),
            "memory": plan.backend_plan,
            "preparation": {"session_open_ms": session.open_ms, "prepare_ms": prepare_ms,
                            "prepare_scope": "FX export/analysis, config selection, CPU/PIM compilation and placement",
                            "weight_load": weight_load},
            "warmups": warmups, "samples": samples, "median_wall_ms": median_ms,
            "logical_gemm_operations": logical_operations,
            "logical_gemm_gops_per_s": logical_operations / (median_ms * 1e6),
            "profile": {"wall_ms": profile_ms, "transfers": profile_transfers,
                        "partitions": profiler.records, "gemm_totals": operations,
                        "native_timing_scope": "generated model host v3; kernel includes active metadata transfer; program load is outside native stages",
                        "cpu_partitions_ms": sum(r["wall_ms"] for r in profiler.records if r["where"] == "CPU"),
                        "outside_partitions_ms": profile_ms - sum(r["wall_ms"] for r in profiler.records)},
            "session_close_ms": session.close_ms, "session_totals": session.stats(),
        }
        assert report["session_totals"]["resource_opens"] == report["session_totals"]["resource_closes"] == 1
        (args.report_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{scope}: median_wall_ms={median_ms:.3f} launches={expected_launches} "
              f"logical_GEMM_GOPS={report['logical_gemm_gops_per_s']:.3f}", flush=True)
        for operation, totals in operations.items():
            print(f"  {operation}: calls={totals['calls']} instances={totals['instances']} "
                  f"wall_ms={totals['wall_ms']:.3f}", flush=True)
        print(f"resource_opens=1 resource_closes=1 report={args.report_dir.resolve() / 'report.json'}", flush=True)
