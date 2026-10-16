"""Llama-2-7b prefill GEMMs alone on AiMX: the per-launch breakdown when every operand moves.

Each GEMM is one torch.compile(torch.mm) on BF16 operands. The plain stub stages A and B
and drains C at every launch, so nothing stays on the device between calls. The config
comes from --config-source:
  analytical  the first heuristic seed that passes the target constraints
  autotune    measured search over this same launch (cache: --cache, shared with autotune_torch.py)

  python benchmark_gemm.py --config-source analytical --fpgas 2
  python benchmark_gemm.py --config-source autotune --fpgas 2 --shapes q_proj qk_2head
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_common import SLOT_GROUPS, SLOTS, add_config_arguments, analytical_config

import torch
import triton.backends.pim.driver as pim_driver

from torch_backend import PIMInductorBackend, PIMTorchConfig, target_spec
from torch_backend._autotuner import (
    DEFAULT_CACHE_PATH, PIMBenchmarkResult, PIMConfigCache, PIMProblem, SearchBudget,
    build_config_space, tune,
)
from torch_backend.template import TORCH_GEMM_TEMPLATE_VERSION


# 이름 -> (m, n, k). q/k/v/o는 shape이 같아 하나만 잰다. head GEMM은 head 하나와 block-diagonal 묶음.
SHAPES = {
    "q_proj": (1024, 4096, 4096),
    "gate_proj": (1024, 11264, 4096),
    "down_proj": (1024, 4096, 11008),
    "qk_head": (1024, 1024, 128),
    "pv_head": (1024, 128, 1024),
    "qk_2head": (1024, 2048, 256),
    "pv_4head": (1024, 512, 4096),
}


def pim_mm(a, b):
    return torch.mm(a, b)


def measure(spec, config, a, b, *, warmup, repeat):
    """Median wall and native slots of `repeat` launches after the compile launch and `warmup`."""
    # config마다 새로 컴파일한다. reset이 없으면 dynamo가 앞 config의 컴파일 결과를 돌려준다.
    torch._dynamo.reset()
    compiled = torch.compile(pim_mm, backend=PIMInductorBackend(PIMTorchConfig(spec=spec, config=config)),
                             fullgraph=True, dynamic=False)
    for _ in range(1 + warmup):
        compiled(a, b)
    walls = []
    slots = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        compiled(a, b)
        walls.append((time.perf_counter_ns() - start) / 1e6)
        slots.append(dict(zip(pim_driver.TIMING_STAGES, pim_driver.timing)))
    return statistics.median(walls), {name: statistics.median(slot[name] for slot in slots) for name in SLOTS}


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--shapes", nargs="+", choices=tuple(SHAPES), default=tuple(SHAPES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--tune-repeat", type=int, default=3)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or Path(__file__).resolve().with_name(
        f"report_{args.config_source}_f{args.fpgas}")
    report_dir.mkdir(parents=True, exist_ok=True)
    spec = target_spec("aimx", num_units=args.fpgas, base_fpga=args.base_fpga)
    generator = torch.Generator().manual_seed(0)

    rows = []
    for name in args.shapes:
        m, n, k = SHAPES[name]
        problem = PIMProblem(M=m, N=n, K=k, input_dtype="bfloat16", output_dtype="bfloat16")
        a = torch.randn((m, k), generator=generator).bfloat16()
        b = torch.randn((k, n), generator=generator).bfloat16()
        if args.config_source == "autotune":
            def benchmark_fn(problem, target, config):
                wall_ms, slots = measure(target, config, a, b, warmup=0, repeat=args.tune_repeat)
                return PIMBenchmarkResult(total_ms=wall_ms, correctness=None, timings=tuple(slots.items()))

            config = tune(problem, spec, benchmark_fn, space=build_config_space(problem, spec),
                          cache=PIMConfigCache(args.cache, template_version=TORCH_GEMM_TEMPLATE_VERSION),
                          budget=SearchBudget(max_generations=args.max_generations,
                                              max_benchmarks=args.max_benchmarks),
                          verbose=True).best.config
        else:
            config = analytical_config(problem, spec, build_config_space(problem, spec))
        wall_ms, native = measure(spec, config, a, b, warmup=args.warmup, repeat=args.repeat)
        # native는 FPGA thread마다 잰 시간의 합이므로 device 수로 나눠 wall과 같은 척도로 둔다.
        devices = config.p_m * config.p_n * config.p_k
        slots = {slot: milliseconds / devices for slot, milliseconds in native.items()}
        slots["other"] = wall_ms - sum(slots.values())
        groups = {group: sum(slots[slot] for slot in members) for group, members in SLOT_GROUPS.items()}
        row = {
            "name": name, "mnk": [m, n, k], "config": asdict(config), "devices": devices,
            "a_bytes": m * k * 2, "b_bytes": k * n * 2, "c_bytes": m * n * 2,
            "operations": 2 * m * n * k, "wall_ms": wall_ms, "native_sum_ms": native,
            "slots_ms": slots, "groups_ms": groups,
            "groups_percent": {group: 100 * value / wall_ms for group, value in groups.items()},
        }
        rows.append(row)
        print(f"{name} mnk=({m},{n},{k}) config={config.key()} wall_ms={wall_ms:.2f} "
              + " ".join(f"{slot}={value:.2f}" for slot, value in slots.items()), flush=True)
        print("  " + " ".join(f"{group}={value:.2f}ms({row['groups_percent'][group]:.1f}%)"
                              for group, value in groups.items()), flush=True)

    report = {"config_source": args.config_source, "fpgas": args.fpgas, "base_fpga": args.base_fpga,
              "dtype": "bfloat16", "warmup": args.warmup, "repeat": args.repeat,
              "slots_scope": "per FPGA thread average; other = wall - slots", "gemms": rows}
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
