"""Policy-first model compile and runtime of one FP32 layer on AiMX.

Compile time: stage analysis -> residency policy -> policy-constrained autotuning
-> config selection -> PIM/CPU compilation, without allocating the FPGAs.
Runtime: open (allocate, bind stages) -> run the partitioned FX graph -> close.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from aimx_benchmark_common import StageProfile
from fp32_workloads import HIDDEN, SEQUENCE, AttentionLayer, TransformerBlock
from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("attention", "transformer"), required=True)
    parser.add_argument("--units", type=int, default=1, help="AiMX FPGAs")
    parser.add_argument("--base-fpga", type=int, default=0)
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--max-benchmarks", type=int, default=20, help="per stage class")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--tune-warmup", type=int, default=1)
    parser.add_argument("--tune-repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError("--warmup and --repeat must be at least 1")
    report_dir = args.report_dir or Path(__file__).with_name(f"aimx_{args.scope}_model_compile_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = (AttentionLayer() if args.scope == "attention" else TransformerBlock()).eval()
    inputs = (torch.randn(1, SEQUENCE, HIDDEN),)

    compiled = compile_model(
        [model], inputs, target="aimx", units=args.units,
        runtime_options={"base_fpga": args.base_fpga}, cache_path=args.cache,
        max_benchmarks=args.max_benchmarks, max_generations=args.max_generations,
        warmup=args.tune_warmup, repeat=args.tune_repeat,
    )
    report = compiled.report
    print(f"compile: policy={report['policy']} selection={report['selection']} "
          f"needed_bytes={report['needed_bytes']} compiled_bytes={report['compiled_bytes_per_unit']} "
          f"capacity={report['capacity_bytes']} analysis_ms={report['analysis_ms']:.1f} "
          f"tuning_ms={report['tuning_ms']:.1f} compile_ms={report['compile_ms']:.1f}", flush=True)
    for stage_class in report["stage_classes"]:
        selected = stage_class["selected"]
        print(f"  {stage_class['label']}: stages/layer={stage_class['stages_per_layer']} "
              f"measured={stage_class['measured']} E_ms={selected['execution_ms']:.3f} "
              f"U_ms={selected['upload_ms']:.3f} config={tuple(selected['config'].values())}",
              flush=True)

    with torch.no_grad():
        reference = model(*inputs)
        with compiled.session as session:
            plan, = session.layers
            labels = {}
            for stage in plan.stages:
                gemm = stage.gemms[0]
                weight = gemm.right.weight_target or gemm.left.weight_target
                labels[stage.name] = {
                    "operation": weight.removesuffix(".weight") if weight else f"{gemm.op} {stage.shape}",
                    "mnk": list(stage.shape), "config": list(stage.config.key()),
                }
            # 첫 호출은 layer 진입 시 weight upload를 포함한다.
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
                          f"invocations={counters['invocations']}", flush=True)

            # session.run_model과 같은 layer 진입/퇴장 사이에서 partition별 시간을 잰다.
            runtime = session.backend.model
            profiler = StageProfile(plan, labels)
            runtime.enter_layer(plan)
            profiler.run(*inputs)
            runtime.exit_layer(plan)
            session_totals = session.stats()

    median_ms = statistics.median(call["wall_ms"] for call in calls if call["phase"] == "sample")
    report.update({
        "scope": args.scope, "open_ms": compiled.session.open_ms,
        "close_ms": compiled.session.close_ms, "calls": calls, "median_wall_ms": median_ms,
        "estimate_error_ms": median_ms - report["estimated_request_ms"],
        "profile": profiler.records, "session_totals": session_totals,
    })
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{args.scope}: median_wall_ms={median_ms:.3f} "
          f"estimated_request_ms={report['estimated_request_ms']:.3f}", flush=True)
    print(f"report={report_dir.resolve() / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
