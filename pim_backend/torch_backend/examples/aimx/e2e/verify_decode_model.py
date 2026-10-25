"""Whole-model Llama-family decode on AiMX with the key/value cache resident: a few minutes per step budget.

Every decoder layer runs on AiMX (llama_decode.LlamaAimxDecodeLayer); the prompt is prefilled
on the CPU by the Hugging Face model and its cache is written to the FPGAs once. Three
references, all from the same prompt:
  hf     Hugging Face greedy generation (BF16, its own attention): is the text sensible
  cpu    the same decode layers eagerly on the CPU (BF16 GEMMs): what AiMX must reproduce
  forced AiMX fed the cpu tokens at every step: argmax agreement and logits relative L2
The report also gives the memory plan per FPGA (weights, key/value cache) and, per stage,
the wall time of one call with its native breakdown (layout, transfer, invoke).

  python verify_decode_model.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../../upmem/e2e/prompt.txt \
      --prompt-tokens 1024 --new-tokens 33 --cache ./decode_cache.json --report decode_llama2.json
"""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_common import add_device_arguments
from llama_decode import LlamaAimxDecodeLayer, decode_inputs, decode_kv_inputs, generate, prefill_caches
from llama_workloads import load_llama

import torch
from triton.backends.pim.residency import KEY, VALUE

from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH


def run_layers(layers, *inputs):
    for layer in layers:
        inputs = layer(*inputs)
    return inputs


def agreement(tokens, reference):
    matches = [left == right for left, right in zip(tokens, reference)]
    return {"match": sum(matches), "of": len(matches),
            "first_mismatch": matches.index(False) if False in matches else None}


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=33, help="generated tokens, the prefill's included")
    parser.add_argument("--report", type=Path, help="JSON report path")
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must include at least one decode step")

    tokenizer, model = load_llama(args.checkpoint)
    input_ids = tokenizer(args.prompt_file.read_text(), return_tensors="pt", truncation=True,
                          max_length=args.prompt_tokens)["input_ids"]
    if input_ids.shape[1] != args.prompt_tokens:
        parser.error(f"the prompt has {input_ids.shape[1]} tokens; --prompt-tokens is {args.prompt_tokens}")
    capacity = args.prompt_tokens + args.new_tokens
    norm = copy.deepcopy(model.model.norm).float()
    lm_head = copy.deepcopy(model.lm_head).float()

    def head(hidden):
        return lm_head(norm(hidden))

    with torch.no_grad():
        hf_tokens = model.generate(input_ids, max_new_tokens=args.new_tokens, do_sample=False)[
            0, args.prompt_tokens:].tolist()
        token, caches = prefill_caches(model, input_ids, capacity)
        layers = [LlamaAimxDecodeLayer(layer, index).eval()
                  for index, layer in enumerate(model.model.layers)]
        cpu_tokens, cpu_logits, cpu_ms = generate(
            model, token, copy.deepcopy(caches), args.prompt_tokens, args.new_tokens,
            lambda *inputs: run_layers(layers, *inputs), lambda *store: None, head)
        example = decode_inputs(model, model.model.embed_tokens(token.reshape(1, 1)).float(),
                                args.prompt_tokens, args.prompt_tokens, caches)
    print(f"layers={len(layers)} capacity={capacity} cpu vs hf: {agreement(cpu_tokens, hf_tokens)} "
          f"cpu step ms median={statistics.median(cpu_ms):.1f}", flush=True)

    compiled = compile_model(
        {"decode": (layers, example)}, target="aimx", units=args.fpgas,
        kv_inputs={"decode": decode_kv_inputs}, repeats={"decode": args.new_tokens - 1},
        runtime_options={"base_fpga": args.base_fpga},
        cache_path=args.cache, max_benchmarks=args.max_benchmarks,
    )
    memory = compiled.session.memory
    layout = memory.phases["decode"]
    memory_report = {
        "slots": layout.slots, "groups": layout.groups, "phase_bytes": layout.bytes,
        "weight_bytes": sum(layout.slot_weight_bytes), "state_bytes": sum(layout.slot_state_bytes),
        "scratch_bytes": layout.scratch_bytes, "kv_bytes": memory.kv_bytes,
        "bytes_per_fpga": memory.bytes_per_unit, "capacity_bytes": memory.capacity_bytes,
    }
    print(f"memory per FPGA: {memory_report}", flush=True)
    stages = [{"name": stage.name, "mnk": list(stage.shape), "instances": stage.instance_count,
               "kv": None if stage.kv is None else list(stage.kv), "p_b": stage.p_b,
               "rounds": stage.batch_rounds, "config": list(stage.config.key())}
              for stage in compiled.session.plans["decode"][0].stages]
    for stage in stages:
        print(f"  stage {stage}", flush=True)

    with compiled.session as session:
        prompt_caches = copy.deepcopy(caches)

        def write_all(cache_pairs, stop):
            for layer, (key_cache, value_cache) in enumerate(cache_pairs):
                session.write_kv(layer, KEY, key_cache, 0, stop)
                session.write_kv(layer, VALUE, value_cache, 0, stop)

        def store(layer, cache, start, stop):
            session.write_kv(layer, KEY, cache[0], start, stop)
            session.write_kv(layer, VALUE, cache[1], start, stop)

        def run_decode(*inputs):
            return session.run_model("decode", *inputs)

        write_all(caches, args.prompt_tokens)
        with torch.no_grad():
            aimx_tokens, _, aimx_ms = generate(model, token, caches, args.prompt_tokens, args.new_tokens,
                                               run_decode, store, head)
        stats = session.stats()
        # 같은 session으로 다시: prompt 위치만 다시 보내고 cpu 토큰을 입력으로 쓴다.
        write_all(prompt_caches, capacity)
        with torch.no_grad():
            forced_tokens, forced_logits, _ = generate(model, token, prompt_caches, args.prompt_tokens,
                                                       args.new_tokens, run_decode, store, head,
                                                       forced=cpu_tokens)

    errors = [((got - reference).norm() / reference.norm()).item()
              for got, reference in zip(forced_logits, cpu_logits)]
    # 첫 step은 cache 전체 전송과 배치 대조를 포함한다.
    steps = aimx_ms[1:] or aimx_ms
    stage_ms = {name.removeprefix("run:"): {key: value / record["calls"] for key, value in record.items()
                                            if key not in ("calls", "jobs")}
                for name, record in stats["breakdown"].items() if record["calls"]}
    report = {
        "checkpoint": str(args.checkpoint), "layers": len(layers), "fpgas": args.fpgas,
        "prompt_tokens": args.prompt_tokens, "new_tokens": args.new_tokens, "capacity": capacity,
        "tokens": {"hf": hf_tokens, "cpu": cpu_tokens, "aimx": aimx_tokens, "forced": forced_tokens},
        "agreement": {"cpu_vs_hf": agreement(cpu_tokens, hf_tokens),
                      "aimx_vs_cpu": agreement(aimx_tokens, cpu_tokens),
                      "aimx_vs_hf": agreement(aimx_tokens, hf_tokens),
                      "forced_vs_cpu": agreement(forced_tokens, cpu_tokens)},
        "forced_logits_rel_l2": {"median": statistics.median(errors), "max": max(errors)},
        "step_ms": {"cpu_median": statistics.median(cpu_ms), "aimx_first": aimx_ms[0],
                    "aimx_median": statistics.median(steps), "aimx_all": aimx_ms},
        "kv": {name: stats[name] for name in ("kv_bytes", "kv_push_ms", "kv_layout_check_ms")},
        "memory": memory_report, "stages": stages, "stage_ms_per_call": stage_ms,
    }
    for name, value in report["agreement"].items():
        print(f"{name}: {value}", flush=True)
    print(f"forced logits rel_l2 median={statistics.median(errors):.4%} max={max(errors):.4%}", flush=True)
    print(f"hf   text: {tokenizer.decode(hf_tokens)!r}", flush=True)
    print(f"cpu  text: {tokenizer.decode(cpu_tokens)!r}", flush=True)
    print(f"aimx text: {tokenizer.decode(aimx_tokens)!r}", flush=True)
    print(f"step ms: cpu median={report['step_ms']['cpu_median']:.1f} aimx first={aimx_ms[0]:.1f} "
          f"aimx median={report['step_ms']['aimx_median']:.1f} kv={report['kv']}", flush=True)
    for name, record in stage_ms.items():
        print(f"  {name}: " + " ".join(f"{key}={value:.3f}" for key, value in record.items()), flush=True)
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
