"""Checkpoint Llama-family (Llama-2-7b, Llama-3-8B) generation through model compile.

One request: a --prompt-tokens prefill (1024 by default), then --max-new-tokens - 1 decode steps, greedy, on
2048 DPUs with 16 tasklets (e2e/llama_generation.py). Both phases are compiled
together: the decode configs are chosen for that many steps per request. The decode
GEMMs that read the INT8 key/value cache (key cache x query, probability x value cache)
read it in place from the top of every DPU's MRAM; the prefill weights may use that range,
since the cache is written only when decode starts. After each prefill layer its prompt
positions, and every TAIL decode steps the last TAIL positions of every layer, are queued;
in between, the decode layers read those positions from the host cache on the CPU. Right
before the first cache stage of the next step runs, every queued position goes into the
host image of the cache and the whole cache range goes out in one transfer, at the same
offset on every DPU. The cache scales stay on the host. The
first decode step includes the decode weight upload, which overwrites the prefill
weights, the prefill cache transfer and a one-time check of the host cache layout
against the generated pack. Every decode step is split like a run_upmem.py request
(upmem_workloads.PartitionTimer): CPU, weight upload, program load, cache push, the native
stages of every PIM call (h2d, kernel, d2h, reduction, host layout), the rest of the
invoke, and other; the report keeps the first step and the mean of the others, and
<report-dir>/step_breakdown.csv has every step (also in PIM-LLM's items h2d, d2h, pim, cpu)
and mean_after_first, <report-dir>/gemm.csv every PIM partition per step. The token
IDs go to the report; e2e/check_llama_generate_cpu.py compares them with the CPU layers.
Compile and resource open/close are excluded.

  ../cmd.sh benchmark_model_llama_generate.py --checkpoint <dir> --prompt-file prompt.txt \\
      --report-dir <dir> --local-capacity-limit-bytes <bytes>
"""

import argparse
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
from transformers import AutoModelForCausalLM, AutoTokenizer
from triton.backends.pim.residency import KEY, VALUE

from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH
from e2e.llama_generation import PROMPT_TOKENS, TAIL, decode_example, decode_kv_inputs, generate, generation_layers
from e2e.llama_workloads import KV_MODES
from upmem_workloads import (
    HOST_LAYOUT_STAGES, NATIVE_STAGES, PartitionTimer, gemm_rows, mean_row, pimllm_items, write_csv,
)


def plan_launches(plan):
    return sum(stage.batch_rounds * math.prod(stage.mnk_rounds) for stage in plan.stages)


def step_breakdown(step_ms, layers_ms, totals, partitions):
    """One decode step in the items of a run_upmem.py request; totals are the runtime counters
    of the step and partitions the PartitionTimer records of its decode layers."""
    native = {name: sum(record["native_ms"][name] for record in partitions.values() if record["where"] == "PIM")
              for name in NATIVE_STAGES}
    breakdown = {
        # decode layer 밖(embedding, host cache 쓰기와 전송 대기열, norm/lm_head/argmax)과 CPU partition
        "cpu_ms": step_ms - layers_ms + sum(record["wall_ms"] for record in partitions.values()
                                            if record["where"] == "CPU"),
        "weight_upload_ms": totals["pack_wall_ms"] + totals["upload_wall_ms"],
        "program_load_ms": totals["program_load_ms"],
        "kv_push_ms": totals["kv_push_ms"],
        "h2d_ms": native["h2d"],
        "kernel_ms": native["kernel"],
        "d2h_ms": native["d2h"],
        "reduction_ms": native["reduction"],
        "host_layout_ms": sum(native[name] for name in HOST_LAYOUT_STAGES),
        "invoke_outside_native_ms": totals["invoke_ms"] - sum(native.values()),
    }
    # partition 사이 FX 처리, _LayerGemm의 입력 contiguous·출력 할당, exit_layer
    breakdown["other_ms"] = step_ms - sum(breakdown.values())
    return breakdown


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--kv", choices=KV_MODES, default="share", help="how a prefill query head reads its KV head")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS, help="prefill length")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="tokens the request generates; the key/value cache holds the prompt + this many positions")
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
        capacity = args.prompt_tokens + args.max_new_tokens
        prefill_layers, decode_layers = generation_layers(model, args.kv, args.prompt_tokens)

        print(f"Compiling {layers} prefill and {layers} decode layers ({steps} decode steps per request) ...",
              flush=True)
        prefill_example = (torch.zeros((1, args.prompt_tokens, config.hidden_size), dtype=torch.float32), ())
        compiled = compile_model(
            {"prefill": (prefill_layers, prefill_example),
             "decode": (decode_layers, decode_example(model, args.prompt_tokens, capacity))},
            kv_inputs={"decode": decode_kv_inputs}, repeats={"prefill": 1, "decode": steps},
            target="upmem", units=2048,
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
              f"kv_base={memory.kv_base} prefill_bytes={prefill.bytes} decode_bytes={decode.bytes} "
              f"bytes_per_unit={memory.bytes_per_unit} capacity={memory.capacity_bytes}", flush=True)

        with compiled.session as session:
            prefill_launches = sum(plan_launches(plan) for plan in session.plans["prefill"])
            decode_launches = sum(plan_launches(plan) for plan in session.plans["decode"])
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=args.prompt_tokens)["input_ids"]
            if tuple(input_ids.shape) != (1, args.prompt_tokens):
                raise ValueError(f"the prompt must contain at least {args.prompt_tokens} tokens including "
                                 f"special tokens")
            decode_stages = {(stage.name, tuple(stage.shape)): stage
                             for plan in session.plans["decode"] for stage in plan.stages}
            # step별 decode layer 전체(session.run_model) 시간, 그 step의 runtime 계수 증가분,
            # partition 기록. 계수는 Python dict만 읽으므로 step 시간에 native 호출을 더하지 않는다.
            decode_layers_ms = []
            decode_totals = []
            decode_partitions = []
            runtime = session.backend.model
            timer = PartitionTimer(session.plans["decode"])

            def counters():
                return {**runtime.totals, **runtime.loader.totals}

            def run_decode(*inputs):
                timer.reset()
                before_step = counters()
                start = time.perf_counter_ns()
                outputs = session.run_model("decode", *inputs)
                decode_layers_ms.append((time.perf_counter_ns() - start) / 1e6)
                after_step = counters()
                decode_totals.append({name: after_step[name] - before_step[name] for name in after_step})
                decode_partitions.append(timer.partitions)
                return outputs

            # (시작 위치, ms): store 호출마다 key와 value 위치를 전송 대기열에 넣는 시간
            kv_writes = []

            def store(layer, cache, start, stop):
                begin = time.perf_counter_ns()
                key_cache, _, value_cache, _ = cache
                session.write_kv(layer, KEY, key_cache, start, stop)
                session.write_kv(layer, VALUE, value_cache, start, stop)
                kv_writes.append((start, (time.perf_counter_ns() - begin) / 1e6))

            before = session.stats()
            with torch._dynamo.config.patch(error_on_recompile=True):
                print(f"Executing one request: prefill -> {steps} decode steps", flush=True)
                result = generate(
                    model, input_ids, args.max_new_tokens,
                    lambda hidden, cache, after_layer: session.run_model(
                        "prefill", hidden, cache, after_layer=after_layer),
                    run_decode, store)
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
    breakdowns = [step_breakdown(*step) for step in zip(step_ms, decode_layers_ms, decode_totals, decode_partitions,
                                                         strict=True)]
    decode_kv_push_ms = [totals["kv_push_ms"] for totals in decode_totals]
    # 첫 step은 decode weight 업로드, prefill cache 전송, 배치 대조를 포함하므로 따로 두고 나머지를 평균한다.
    # stage 시간은 step마다 같은 일을 하므로 전체 합을 step 수로 나눈다.
    breakdown = {
        "step_ms": statistics.mean(step_ms[1:]),
        "decode_layers_ms": statistics.mean(decode_layers_ms[1:]),
        **{name: statistics.mean(step[name] for step in breakdowns[1:]) for name in breakdowns[0]},
        # 보낸 step 하나의 cache 전송 시간: step i 끝에 store한 위치는 step i + 1에서 보낸다 (TAIL, 2·TAIL, ...).
        "kv_push_ms_per_flush": statistics.mean(decode_kv_push_ms[TAIL::TAIL]),
        "kv_queue_ms_per_store": statistics.mean(ms for start, ms in kv_writes if start >= args.prompt_tokens),
        "stages": [{"name": name, "mnk": list(mnk), "instances": decode_stages[name, mnk].instance_count,
                    "calls_per_step": times["calls"] / steps,
                    "program_load_ms": times["program_load_ms"] / steps,
                    "invoke_ms": times["invoke_ms"] / steps,
                    "kv_push_ms": times["kv_push_ms"] / steps}
                   for (name, mnk), times in stage_times.items()],
    }
    report = {
        "status": "execution_complete", "runtime": "model_compile",
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt_file": str(args.prompt_file.resolve()),
        "scope": f"one request: {args.prompt_tokens}-token prefill, then {steps} greedy decode steps",
        "measurement_scope": (
            "first token: embedding, prefill layers, host key/value cache write and transfer queue, "
            "norm/lm_head/argmax; decode step: embedding, decode layers (the first cache stage sends "
            "the queued positions, if any, then the stages read the cache in place; the tail positions "
            "join on the CPU), host cache write of the new position and, every TAIL steps, the transfer "
            "queue, norm/lm_head/argmax; the first step also sends the prefill cache and checks "
            "the host layout once; excludes tokenization, compile and resource open/close"
        ),
        "layers": layers, "heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
        "kv": args.kv, "prompt_tokens": args.prompt_tokens, "max_new_tokens": args.max_new_tokens,
        "cache_positions": capacity, "kv_tail": TAIL,
        "kv_cache_location": "device (scales on the host)", "kv_cache_dtype": "int8",
        "device_kv_bytes": memory.kv_bytes,
        "first_step_kv_push_ms": decode_kv_push_ms[0],
        "kv_layout_check_ms": after["kv_layout_check_ms"] - before["kv_layout_check_ms"],
        "prefill_groups": prefill.groups, "decode_groups": decode.groups,
        "cpu_threads": torch.get_num_threads(), "omp_num_threads": os.getenv("OMP_NUM_THREADS", ""),
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
        "first_step_breakdown": breakdowns[0],
        "decode_step_breakdown": breakdown,
    }
    (args.report_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = [{"step": index, "position": args.prompt_tokens + index, "step_ms": ms, "layers_ms": layers_ms,
             **pimllm_items(ms, step), **{"detail_" + name: value for name, value in step.items()}}
            for index, (ms, layers_ms, step) in enumerate(zip(step_ms, decode_layers_ms, breakdowns, strict=True))]
    write_csv(args.report_dir / "step_breakdown.csv", rows + [mean_row("mean_after_first", rows[1:])])
    # 첫 step은 decode weight 업로드와 prefill cache 전송을 포함하므로 뺀 step 평균이다.
    write_csv(args.report_dir / "gemm.csv", gemm_rows(decode_partitions[1:]))
    print(f"first_token_ms={report['first_token_ms']:.1f} first_step_ms={report['first_step_ms']:.1f} "
          f"step_ms_median={report['step_ms_median_after_first']:.1f} total_ms={report['total_ms']:.1f} "
          f"weight_pushes={transfers['weight_pushes']} launches={transfers['launches']} "
          f"kv_bytes={transfers['kv_bytes']} first_step_kv_push_ms={decode_kv_push_ms[0]:.1f} "
          f"kv_layout_check_ms={report['kv_layout_check_ms']:.1f}", flush=True)
    print(f"decode step (mean after the first): step={breakdown['step_ms']:.1f} "
          f"layers={breakdown['decode_layers_ms']:.1f} kv_push per flush={breakdown['kv_push_ms_per_flush']:.1f} ms",
          flush=True)
    print("  " + " ".join(f"{name}={breakdown[name]:.1f}" for name in breakdowns[0]), flush=True)
    for stage in breakdown["stages"]:
        print(f"  {stage['name']} mnk={tuple(stage['mnk'])} x{stage['instances']}: "
              f"calls/step={stage['calls_per_step']:.0f} program_load={stage['program_load_ms']:.1f} "
              f"invoke={stage['invoke_ms']:.1f} kv_push={stage['kv_push_ms']:.1f} ms/step", flush=True)
    print(f"token_ids={result['tokens']}", flush=True)
    print(f"text={report['text']!r}", flush=True)
    print(f"Execution report saved to {args.report_dir.resolve() / 'report.json'} "
          f"(+ step_breakdown.csv, gemm.csv)", flush=True)


if __name__ == "__main__":
    main()
