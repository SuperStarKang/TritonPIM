"""One FP32 model layer on AiMX FPGAs: GEMMs on PIM, everything else on CPU."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import fx

from torch_backend import PIMSession, PIMTorchConfig, target_spec
from torch_backend._autotuner import PIMConfig
from torch_backend.model import _LayerGemm

from fx_utils import write_fx_graph


# (M, N, K) -> FPGA 하나일 때의 config. FPGA가 여럿이면 stage_config가 나눈다.
# Projection은 weight 상주 stage라 Execution Tile 하나가 문제 전체를 덮어야 한다(round 1).
# B/C는 Block Group의 n을 4096 단위로 저장하므로, Block Group은 n·k를 문제 전체로 잡는다.
# (1024, 4096, 4096)은 autotune best(2 FPGA, p_m=2: Block Group (128,4096,4096), tile_m 4)다.
# Execution Tile의 n·k가 1이라 FPGA 분할은 --split m만 된다.
CONFIGS = {
    (1024, 4096, 4096): PIMConfig(64, 128, 128, 2, 32, 32, 8, 1, 1, 1, 1, 1),
    (1024, 11264, 4096): PIMConfig(64, 128, 128, 2, 88, 32, 8, 1, 1, 1, 1, 1),
    (1024, 4096, 11008): PIMConfig(64, 128, 128, 2, 32, 86, 8, 1, 1, 1, 1, 1),
    # QK^T, PV: 32 heads가 instance 32개인 stage 하나가 되고 batch round 32번으로 실행된다.
    (1024, 1024, 128): PIMConfig(128, 128, 128, 1, 8, 1, 8, 1, 1, 1, 1, 1),
    (1024, 128, 1024): PIMConfig(128, 128, 128, 1, 1, 8, 8, 1, 1, 1, 1, 1),
    # head 묶음 block-diagonal (PackedAttentionLayer). Block Group이 n·k 전체, tile_m이 m 전체다.
    # QK^T 2 head: (1024 x 256) @ (256 x 2048)
    (1024, 2048, 256): PIMConfig(128, 128, 128, 1, 16, 2, 8, 1, 1, 1, 1, 1),
    # PV 4 head: (1024 x 4096) @ (4096 x 512)
    (1024, 512, 4096): PIMConfig(128, 128, 128, 1, 4, 32, 8, 1, 1, 1, 1, 1),
}
# head GEMM shape -> (이름, 0 블록을 뺀 유효 계산의 mnk)
HEAD_OPERATIONS = {
    (1024, 1024, 128): ("QK^T", (1024, 1024, 128)),
    (1024, 128, 1024): ("PV", (1024, 128, 1024)),
    (1024, 2048, 256): ("QK^T", (1024, 2048, 128)),
    (1024, 512, 4096): ("PV", (1024, 512, 1024)),
}
HEAD_SHAPES = tuple(HEAD_OPERATIONS)


def stage_config(mnk, fpgas, split):
    """FPGA 수에 맞춘 config.

    head stage는 instance 하나가 FPGA 하나를 쓰므로 그대로 두면 p_b = FPGA 수가 된다.
    weight stage는 round 1을 지키며 split 축의 tile을 FPGA 수로 나눈다. n으로 나누면
    weight가 FPGA마다 절반씩, m으로 나누면 weight가 FPGA마다 통째로 복제된다.
    """
    config = CONFIGS[mnk]
    if mnk in HEAD_SHAPES:
        return config
    if split == "m":
        if config.tile_m % fpgas:
            raise ValueError(f"{mnk}: tile_m {config.tile_m} is not divisible by {fpgas} FPGAs")
        return replace(config, tile_m=config.tile_m // fpgas, p_m=fpgas)
    if config.tile_n % fpgas:
        raise ValueError(f"{mnk}: tile_n {config.tile_n} is not divisible by {fpgas} FPGAs")
    return replace(config, tile_n=config.tile_n // fpgas, p_n=fpgas)


def add_benchmark_arguments(parser, report_dir):
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--report-dir", type=Path, default=report_dir)
    parser.add_argument("--fpgas", type=int, default=1)
    parser.add_argument("--base-fpga", type=int, default=0)
    parser.add_argument("--split", choices=("n", "m"), default="n",
                        help="axis that divides the weight stages over FPGAs")


class StageProfile(fx.Interpreter):
    """Wall time of each CPU/PIM partition in one layer call."""

    def __init__(self, plan, labels):
        super().__init__(plan.graph)
        self.labels = labels
        self.records = []

    def call_module(self, target, args, kwargs):
        module = self.module.get_submodule(target)
        stages = [child for child in module.modules() if isinstance(child, _LayerGemm)]
        start = time.perf_counter_ns()
        output = super().call_module(target, args, kwargs)
        record = {"partition": target, "where": "CPU+PIM" if stages else "CPU",
                  "wall_ms": (time.perf_counter_ns() - start) / 1e6}
        if stages:
            stage, = stages
            record.update(self.labels[stage.name])
            record["instances"] = stage.instance_count
            record["native_ms"] = dict(stage.last_native_ms)
            record["pim_exec"] = stage.last_pim_exec
        else:
            record["operators"] = [str(node.target) for node in module._orig_mod.graph.nodes
                                   if node.op == "call_function"]
        self.records.append(record)
        return output


def benchmark_model(model, inputs, args, *, scope, expected_gemms, expected_weights):
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError("--warmup and --repeat must be at least 1")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    labels = {}
    configs = {mnk: stage_config(mnk, args.fpgas, args.split) for mnk in CONFIGS}

    def configure(gemms, spec):
        if len(gemms) != expected_gemms:
            raise ValueError(f"expected {expected_gemms} FX GEMMs, found {len(gemms)}")
        if sum(gemm.right.weight_target is not None for gemm in gemms) != expected_weights:
            raise ValueError(f"expected {expected_weights} projection weights")
        for gemm in gemms:
            if gemm.right.weight_target is not None:
                operation = gemm.right.weight_target.removesuffix(".weight")
                logical = (1024, 11008, 4096) if gemm.mnk == (1024, 11264, 4096) else gemm.mnk
            else:
                operation, logical = HEAD_OPERATIONS[gemm.mnk]
            labels[gemm.node] = {
                "operation": operation, "mnk": list(logical), "padded_mnk": list(gemm.mnk),
                "weight": gemm.right.weight_target, "config": asdict(configs[gemm.mnk]),
            }
        return {gemm.node: PIMTorchConfig(spec=spec, config=configs[gemm.mnk]) for gemm in gemms}

    with torch.no_grad():
        reference = model(*inputs)
        with PIMSession(target_spec("aimx", num_units=args.fpgas, base_fpga=args.base_fpga)) as session:
            start = time.perf_counter_ns()
            plan, = session.prepare([model], inputs, configure=configure)
            prepare_ms = (time.perf_counter_ns() - start) / 1e6
            residency = asdict(session.residency)
            print(f"scope={scope} logical_gemms={len(labels)} planned_stages={len(plan.stages)} "
                  f"prepare_ms={prepare_ms:.3f}", flush=True)
            print(f"residency policy={residency['policy']} bytes_per_fpga={residency['bytes_per_unit']} "
                  f"capacity={residency['capacity_bytes']}", flush=True)
            write_fx_graph(plan.exported_graph, args.report_dir / "exported_fx", {
                name: json.dumps(label) for name, label in labels.items()
            })
            write_fx_graph(plan.graph, args.report_dir / "partitioned_fx", {})
            stage_records = []
            for stage in plan.stages:
                record = {
                    "name": stage.name, **labels[stage.name],
                    "instances": stage.instance_count, "p_b": stage.p_b,
                    "batch_rounds": stage.batch_rounds, "mnk_rounds": list(stage.mnk_rounds),
                    "execution_extent": list(stage.config.execution_extent),
                    "weight_slot_bytes": sum(size for size, weight in zip(stage.slot_bytes, stage.slot_weights) if weight),
                    "state_slot_bytes": sum(size for size, weight in zip(stage.slot_bytes, stage.slot_weights) if not weight),
                }
                stage_records.append(record)
                print(f"stage={stage.name} operation={record['operation']} mnk={stage.shape} "
                      f"instances={stage.instance_count} p_b={stage.p_b} "
                      f"batch_rounds={stage.batch_rounds} mnk_rounds={stage.mnk_rounds} "
                      f"extent={stage.config.execution_extent} "
                      f"partitions=({stage.config.p_m},{stage.config.p_n},{stage.config.p_k})", flush=True)

            # 첫 호출은 layer 진입 시 weight upload를 포함한다. warm-up과 sample은 이후 호출이다.
            calls = []
            for phase, count in (("first", 1), ("warmup", args.warmup), ("sample", args.repeat)):
                for step in range(count):
                    before = session.stats()
                    start = time.perf_counter_ns()
                    output = session.run_model(*inputs)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    after = session.stats()
                    counters = {name: after[name] - before[name]
                                for name in ("weight_uploads", "weight_bytes", "invocations")}
                    max_abs = (output - reference).abs().max().item()
                    calls.append({"phase": phase, "wall_ms": elapsed, "max_abs": max_abs, **counters})
                    print(f"{phase}={step} wall_ms={elapsed:.3f} max_abs={max_abs:.4g} "
                          f"weight_uploads={counters['weight_uploads']} "
                          f"weight_bytes={counters['weight_bytes']} "
                          f"invocations={counters['invocations']}", flush=True)

            # session.run_model과 같은 layer 진입/퇴장 사이에서 partition별 시간을 잰다.
            runtime = session.backend.model
            profiler = StageProfile(plan, labels)
            runtime.enter_layer(plan)
            start = time.perf_counter_ns()
            profiler.run(*inputs)
            profile_ms = (time.perf_counter_ns() - start) / 1e6
            runtime.exit_layer(plan)
            session_totals = session.stats()

        samples = [call["wall_ms"] for call in calls if call["phase"] == "sample"]
        median_ms = statistics.median(samples)
        logical_operations = sum(2 * math.prod(label["mnk"]) for label in labels.values())
        operations = {}
        for record in profiler.records:
            if record["where"] == "CPU+PIM":
                name = record["operation"]
                if name not in operations:
                    operations[name] = {"calls": 0, "instances": 0, "wall_ms": 0.0, "pim_exec": 0,
                                        "native_ms": dict.fromkeys(record["native_ms"], 0.0)}
                operations[name]["calls"] += 1
                operations[name]["instances"] += record["instances"]
                operations[name]["wall_ms"] += record["wall_ms"]
                operations[name]["pim_exec"] += record["pim_exec"]
                for stage_name, milliseconds in record["native_ms"].items():
                    operations[name]["native_ms"][stage_name] += milliseconds
        cpu_ms = sum(record["wall_ms"] for record in profiler.records if record["where"] == "CPU")
        report = {
            "scope": scope, "target": "aimx", "fpgas": args.fpgas, "base_fpga": args.base_fpga,
            "split": args.split,
            "batch": 1, "sequence": 1024, "heads": 32, "head_dim": 128, "hidden_dim": 4096,
            "ffn_dim": 11008 if scope == "transformer_block" else None,
            "gemm_dtype": "float32 host, bfloat16 on AiMX", "cpu_dtype": "float32", "causal": True,
            "seed": 0, "cpu_threads": torch.get_num_threads(),
            "spec": asdict(session.spec), "residency": residency,
            "gemms": labels, "stages": stage_records,
            "prepare_ms": prepare_ms, "session_open_ms": session.open_ms,
            "session_close_ms": session.close_ms, "calls": calls,
            "median_wall_ms": median_ms,
            "measurement_scope": "samples exclude the first call, which uploads the weights",
            "logical_gemm_operations": logical_operations,
            "logical_gemm_gops_per_s": logical_operations / (median_ms * 1e6),
            "native_ms_scope": "per FPGA thread, summed over FPGAs; can exceed wall time",
            "profile": {"wall_ms": profile_ms, "partitions": profiler.records,
                        "gemm_totals": operations, "cpu_partitions_ms": cpu_ms},
            "session_totals": session_totals,
        }
        (args.report_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{scope}: median_wall_ms={median_ms:.3f} "
              f"logical_GEMM_GOPS={report['logical_gemm_gops_per_s']:.3f}", flush=True)
        for operation, totals in operations.items():
            native = " ".join(f"{stage_name}={milliseconds:.1f}"
                              for stage_name, milliseconds in totals["native_ms"].items())
            print(f"  {operation}: calls={totals['calls']} instances={totals['instances']} "
                  f"wall_ms={totals['wall_ms']:.3f} pim_exec={totals['pim_exec']} native_ms[{native}]",
                  flush=True)
        print(f"  CPU partitions: wall_ms={cpu_ms:.3f}", flush=True)
        print(f"report={args.report_dir.resolve() / 'report.json'}", flush=True)
