"""Checkpoint Llama-family (Llama-2-7b, Llama-3-8B) generation through model compile.

One request: a 1024-token prefill, then --max-new-tokens - 1 decode steps, greedy, on
2048 DPUs with 16 tasklets (e2e/llama_generation.py). Both phases are compiled
together: the decode configs are chosen for that many steps per request, and the
memory table reserves the key/value cache for 1024 + --max-new-tokens positions at the
bottom of every DPU. The cache itself still stays on the CPU here, so every decode
step sends the cache of every layer to the DPUs. The first decode step includes the
decode weight upload, which overwrites the prefill weights. The token IDs go to the
report; e2e/check_llama_generate_cpu.py compares them with the CPU layers. Compile and
resource open/close are excluded.

  ../cmd.sh benchmark_model_llama_generate.py --checkpoint <dir> --prompt-file prompt.txt \\
      --report-dir <dir> --local-capacity-limit-bytes <bytes>
"""

import argparse
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
from transformers import AutoModelForCausalLM, AutoTokenizer
from triton.backends.pim.residency import KVRequirement

from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH
from e2e.llama_generation import PROMPT_TOKENS, decode_example, generate, generation_layers
from e2e.llama_workloads import KV_MODES


def plan_launches(plan):
    return sum(stage.batch_rounds * math.prod(stage.mnk_rounds) for stage in plan.stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--kv", choices=KV_MODES, default="share", help="how a prefill query head reads its KV head")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="tokens the request generates; the key/value cache holds 1024 + this many positions")
    parser.add_argument("--max-benchmarks", type=int, default=20, help="per stage class")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--local-capacity-limit-bytes", type=int, required=True,
                        help="WRAM bytes per DPU for Block Group buffers; the rest holds tasklet stacks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    steps = args.max_new_tokens - 1

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.checkpoint), local_files_only=True, dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)
        config = model.config
        layers = config.num_hidden_layers
        capacity = PROMPT_TOKENS + args.max_new_tokens
        prefill_layers, decode_layers = generation_layers(model, args.kv)

        print(f"Compiling {layers} prefill and {layers} decode layers ({steps} decode steps per request) ...",
              flush=True)
        kv = KVRequirement(layers, config.num_key_value_heads, config.head_dim, capacity, 1)
        prefill_example = (torch.zeros((1, PROMPT_TOKENS, config.hidden_size), dtype=torch.float32), ())
        compiled = compile_model(
            {"prefill": (prefill_layers, prefill_example),
             "decode": (decode_layers, decode_example(model, capacity))},
            kv=kv, repeats={"prefill": 1, "decode": steps}, target="upmem", units=2048,
            runtime_options={"nr_tasklets": 16,
                             "local_capacity_limit_bytes": args.local_capacity_limit_bytes},
            cache_path=args.cache,
            max_benchmarks=args.max_benchmarks, max_generations=args.max_generations,
        )
        compile_report = compiled.report
        memory = compiled.session.memory
        prefill = memory.phases["prefill"]
        decode = memory.phases["decode"]
        for phase in ("prefill", "decode"):
            for stage_class in compile_report["phases"][phase]["stage_classes"]:
                selected = stage_class["selected"]
                print(f"  {phase} {stage_class['label']}: measured={stage_class['measured']} "
                      f"E_ms={selected['execution_ms']:.3f} U_ms={selected['first_upload_ms']:.3f} "
                      f"config={tuple(selected['config'].values())}", flush=True)
        print(f"prefill groups={prefill.groups} decode groups={decode.groups} kv_bytes={memory.kv_bytes} "
              f"bytes_per_unit={memory.bytes_per_unit} capacity={memory.capacity_bytes}", flush=True)

        with compiled.session as session:
            prefill_launches = sum(plan_launches(plan) for plan in session.plans["prefill"])
            decode_launches = sum(plan_launches(plan) for plan in session.plans["decode"])
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=PROMPT_TOKENS)["input_ids"]
            decode_stages = {(stage.name, tuple(stage.shape)): stage
                             for plan in session.plans["decode"] for stage in plan.stages}
            # decode layer 전체(session.run_model)의 step별 시간
            decode_layers_ms = []

            def run_decode(*inputs):
                start = time.perf_counter_ns()
                outputs = session.run_model("decode", *inputs)
                decode_layers_ms.append((time.perf_counter_ns() - start) / 1e6)
                return outputs

            before = session.stats()
            with torch._dynamo.config.patch(error_on_recompile=True):
                print(f"Executing one request: prefill -> {steps} decode steps", flush=True)
                result = generate(
                    model, input_ids, args.max_new_tokens,
                    lambda *inputs: session.run_model("prefill", *inputs), run_decode)
            after = session.stats()
            transfers = {name: after[name] - before[name] for name in before}
            stage_times = {key: dict(times) for key, times in session.backend.model.stage_times.items()
                           if key in decode_stages}

    # decode가 group 하나면 첫 step만 decode weight를 올리고, 아니면 step마다 모든 group을 올린다.
    expected_pushes = prefill.groups + (decode.groups if decode.groups == 1 else decode.groups * steps)
    expected_launches = prefill_launches + steps * decode_launches
    assert transfers["weight_pushes"] == expected_pushes, (transfers["weight_pushes"], expected_pushes)
    assert transfers["launches"] == expected_launches, (transfers["launches"], expected_launches)
    assert transfers["device_compiles"] == 0

    step_ms = result["step_ms"]
    # 첫 step은 decode weight 업로드를 포함하므로 step과 layer 시간은 두 번째 step부터 평균한다.
    # stage 시간은 step마다 같은 일을 하므로 전체 합을 step 수로 나눈다.
    layers_ms = statistics.mean(decode_layers_ms[1:])
    program_load_ms = sum(times["program_load_ms"] for times in stage_times.values()) / steps
    invoke_ms = sum(times["invoke_ms"] for times in stage_times.values()) / steps
    breakdown = {
        "step_ms": statistics.mean(step_ms[1:]),
        "decode_layers_ms": layers_ms,
        "program_load_ms": program_load_ms,
        "invoke_ms": invoke_ms,
        # layer 안에서 stage 실행 밖의 시간: CPU partition, stage 입력 준비, graph 해석
        "cpu_in_layers_ms": layers_ms - program_load_ms - invoke_ms,
        # layer 밖: embedding, cache 쓰기, norm/lm_head/argmax
        "host_outside_layers_ms": statistics.mean(step_ms[1:]) - layers_ms,
        "stages": [{"name": name, "mnk": list(mnk), "instances": decode_stages[name, mnk].instance_count,
                    "calls_per_step": times["calls"] / steps,
                    "program_load_ms": times["program_load_ms"] / steps,
                    "invoke_ms": times["invoke_ms"] / steps}
                   for (name, mnk), times in stage_times.items()],
    }
    report = {
        "status": "execution_complete", "runtime": "model_compile",
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt_file": str(args.prompt_file.resolve()),
        "scope": f"one request: {PROMPT_TOKENS}-token prefill, then {steps} greedy decode steps",
        "measurement_scope": (
            "first token: embedding, prefill layers, key/value cache write, norm/lm_head/argmax; "
            "decode step: embedding, decode layers (the CPU key/value cache is sent every step), "
            "cache write, norm/lm_head/argmax; excludes tokenization, compile and resource open/close"
        ),
        "layers": layers, "heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
        "kv": args.kv, "max_new_tokens": args.max_new_tokens, "cache_positions": capacity,
        "kv_cache_location": "cpu", "kv_cache_dtype": "int8",
        "device_kv_bytes_reserved": memory.kv_bytes,
        "prefill_groups": prefill.groups, "decode_groups": decode.groups,
        "cpu_threads": torch.get_num_threads(),
        "versions": {"torch": torch.__version__, "triton": triton.__version__},
        "target": asdict(session.spec),
        "compile": compile_report,
        "expected_weight_pushes": expected_pushes, "expected_launches": expected_launches,
        "first_token_ms": result["first_token_ms"],
        "first_step_ms": step_ms[0],
        # 첫 step은 decode weight 업로드를 포함하므로 뺀다.
        "step_ms_median_after_first": statistics.median(step_ms[1:]),
        "step_ms": step_ms,
        "total_ms": result["first_token_ms"] + sum(step_ms),
        "token_ids": result["tokens"],
        "text": tokenizer.decode(result["tokens"]),
        "transfers": transfers,
        "decode_step_breakdown": breakdown,
    }
    (args.report_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"first_token_ms={report['first_token_ms']:.1f} first_step_ms={report['first_step_ms']:.1f} "
          f"step_ms_median={report['step_ms_median_after_first']:.1f} total_ms={report['total_ms']:.1f} "
          f"weight_pushes={transfers['weight_pushes']} launches={transfers['launches']}", flush=True)
    print(f"decode step (mean after the first): step={breakdown['step_ms']:.1f} "
          f"layers={layers_ms:.1f} program_load={program_load_ms:.1f} invoke={invoke_ms:.1f} "
          f"cpu_in_layers={breakdown['cpu_in_layers_ms']:.1f} "
          f"host_outside_layers={breakdown['host_outside_layers_ms']:.1f} ms", flush=True)
    for stage in breakdown["stages"]:
        print(f"  {stage['name']} mnk={tuple(stage['mnk'])} x{stage['instances']}: "
              f"calls/step={stage['calls_per_step']:.0f} program_load={stage['program_load_ms']:.1f} "
              f"invoke={stage['invoke_ms']:.1f} ms/step", flush=True)
    print(f"token_ids={result['tokens']}", flush=True)
    print(f"text={report['text']!r}", flush=True)
    print(f"Execution report saved to {args.report_dir.resolve() / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
