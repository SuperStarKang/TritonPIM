"""Shared by the gemm/, layer/ and e2e/ experiments: config selection and the request breakdown.

Models (layer/, e2e/) compile with compile_model: one layer is tuned with its weights
resident and every layer uses those configs. A GEMM alone (gemm/) chooses its config by
  analytical  the first heuristic seed (autotuner/space.py) that passes the constraints; no measurement
  autotune    measured search over the torch.compile path (A, B and C staged every launch)
"""

from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_backend import PIMPerformanceProfile, compile_model
from torch_backend.template import TORCH_GEMM_TEMPLATE_VERSION
from torch_backend._autotuner import ConstraintEngine, heuristic_seeds
from triton.backends.pim.driver import _STAGE_NAMES


# gemm/ 의 config 선택 방법
CONFIG_SOURCES = ("analytical", "autotune")
SLOTS = _STAGE_NAMES["aimx"]
# native 칸을 묶은 시간. setup은 device buffer 할당과 descriptor 생성이다.
SLOT_GROUPS = {
    "transfer": ("desc_h2d", "h2d", "d2h"),
    "layout": ("layout_in", "layout_out"),
    "compute": ("invoke",),
    "setup": ("alloc", "build"),
}


def analytical_config(problem, spec, space, performance):
    engine = ConstraintEngine(problem, spec)
    for config in heuristic_seeds(problem, spec, space, performance=performance,
                                  kernel_version=TORCH_GEMM_TEMPLATE_VERSION, engine=engine):
        if engine.valid(config):
            return config
    raise ValueError(f"({problem.M},{problem.N},{problem.K}): no heuristic seed passes the constraints")


def compile_session(layers, example_inputs, args):
    """compile_model on the experiment's FPGAs: the unopened session and its config report."""
    performance = None
    if args.performance is not None:
        performance = PIMPerformanceProfile.load(args.performance, kernel_version=TORCH_GEMM_TEMPLATE_VERSION)
    compiled = compile_model(
        layers, example_inputs, target="aimx", units=args.fpgas,
        performance=performance,
        runtime_options={"base_fpga": args.base_fpga},
        cache_path=args.cache, max_benchmarks=args.max_benchmarks,
    )
    return compiled.session, compiled.report


def memory_record(memory):
    """The session's memory plan for a report: slots, layers kept in their own slot, bytes."""
    return {"slots": memory.slots, "pinned": memory.pinned, "slot_of_layer": memory.slot_of_layer,
            "bytes_per_fpga": memory.bytes_per_unit, "capacity_bytes": memory.capacity_bytes}


def add_device_arguments(parser, cache_path):
    parser.add_argument("--fpgas", type=int, default=2)
    parser.add_argument("--base-fpga", type=int, default=0)
    parser.add_argument("--cache", default=cache_path, help="autotune cache JSON")
    parser.add_argument("--performance", help="single-unit saturation JSON")
    parser.add_argument("--max-benchmarks", type=int, default=20, help="autotune: per GEMM or stage class")


def stage_records(plan):
    return [{
        "name": stage.name, "kind": "weight" if any(stage.weight_mask) else "activation",
        "mnk": list(stage.shape), "instances": stage.instance_count, "p_b": stage.p_b,
        "batch_rounds": stage.batch_rounds, "mnk_rounds": list(stage.mnk_rounds),
        "config": asdict(stage.config),
        # 0 블록까지 포함해 PIM이 실제로 계산하는 연산 수
        "executed_operations": 2 * stage.shape[0] * stage.shape[1] * stage.shape[2] * stage.instance_count,
    } for stage in plan.stages]


def slot_totals(records):
    totals = {field: sum(record[field] for record in records) for field in ("wall_ms", *SLOTS)}
    totals["other"] = totals["wall_ms"] - sum(totals[name] for name in SLOTS)
    for group, names in SLOT_GROUPS.items():
        totals[group] = sum(totals[name] for name in names)
    return totals


def request_breakdown(before, after, weight_stages, layers_ms):
    """One request from two session.stats(): counters, PIM wall, slot sums and the weight/activation/CPU split."""
    breakdown = {key: {field: value - before["breakdown"][key][field]
                       for field, value in record.items() if field != "jobs"}
                 for key, record in after["breakdown"].items()}
    wall = {name: after[name] - before[name] for name in ("upload_wall_ms", "run_wall_ms")}
    phases = {kind: slot_totals([record for key, record in breakdown.items() if key.startswith(kind + ":")])
              for kind in ("upload", "run")}
    weight_run_ms = sum(record["wall_ms"] for key, record in breakdown.items()
                        if key.startswith("run:") and key.removeprefix("run:") in weight_stages)
    # 업로드는 weight를 올리는 비용이므로 weight GEMM에 넣는다.
    kinds = {
        "weight_gemm_ms": weight_run_ms + wall["upload_wall_ms"],
        "activation_gemm_ms": wall["run_wall_ms"] - weight_run_ms,
        "cpu_ms": layers_ms - sum(wall.values()),
    }
    return {
        **{name: after[name] - before[name] for name in ("weight_uploads", "weight_bytes", "invocations")},
        "layers_ms": layers_ms, **wall, "kinds": kinds, "phases": phases, "breakdown": breakdown,
    }


def print_request(label, record):
    print(f"{label} layers_ms={record['layers_ms']:.1f} weight_uploads={record['weight_uploads']} "
          f"weight_bytes={record['weight_bytes']} invocations={record['invocations']}", flush=True)
    print("  kinds   " + " ".join(f"{name}={value:.1f}" for name, value in record["kinds"].items()), flush=True)
    for kind, totals in record["phases"].items():
        print(f"  {kind:<7} " + " ".join(f"{name}={value:.1f}" for name, value in totals.items()), flush=True)
