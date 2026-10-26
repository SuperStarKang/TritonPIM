"""Llama-family greedy decode on AiMX with the key/value cache resident: tokens and step time.

The Hugging Face model prefills the prompt on the CPU; its keys after RoPE and its values
fill the host caches, which are then written to the FPGAs once. Every decode step runs the
first --layers decoder layers (llama_decode.LlamaAimxDecodeLayer) on AiMX while the cache
stays there; new positions go to the FPGAs TAIL at a time. The same layers on the CPU give
the reference tokens. The final norm and lm_head run on the CPU in FP32.

Two AiMX runs: free (its own argmax feeds the next step) and teacher-forced (the CPU tokens
feed every step). With few layers the logits are nearly flat, so one small difference flips
the argmax and a free run diverges for good; the forced run compares every step on the same
inputs: argmax agreement and the relative L2 of the logits.

Every step of the free run is split as a UPMEM benchmark_model_llama_generate.py step into
<report-dir>/step_breakdown.csv: one row per step and mean_after_first (the first step also
sends the prompt cache and checks the cache layout once), step_ms in PIM-LLM's items h2d
(weight pack + push, desc_h2d, h2d), pim (invoke), d2h and cpu (the rest, the cache push
included as on UPMEM), their shares, then every breakdown item as detail_* (they add up to
step_ms). gemm.csv (weight GEMVs) and attention.csv (QK and PV on the resident cache) have
one row per PIM stage, summed over the layers and averaged over the steps after the first.

  python benchmark_decode.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../../upmem/e2e/prompt.txt \
      --prompt-tokens 1024 --new-tokens 33 --layers 4 --report-dir decode_report
"""

import argparse
import copy
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_common import SLOTS, add_device_arguments, slot_totals, stage_records, stage_rows, write_csv
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


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=33, help="generated tokens, the prefill's included")
    parser.add_argument("--layers", type=int, default=0, help="first decoder layers to run (0: all)")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.new_tokens < 3:
        parser.error("--new-tokens must include at least two decode steps: the first is reported apart")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_llama(args.checkpoint)
    if args.layers:
        model.model.layers = model.model.layers[:args.layers]
        model.config.num_hidden_layers = args.layers
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
        token, caches = prefill_caches(model, input_ids, capacity)
        layers = [LlamaAimxDecodeLayer(layer, index).eval()
                  for index, layer in enumerate(model.model.layers)]
        # CPU 기준: 같은 layer를 eager로 돌린다. store는 할 일이 없다.
        reference_tokens, reference_logits, reference_ms = generate(
            model, token, copy.deepcopy(caches), args.prompt_tokens, args.new_tokens,
            lambda *inputs: run_layers(layers, *inputs), lambda *store: None, head)
        example = decode_inputs(model, model.model.embed_tokens(token.reshape(1, 1)).float(),
                                args.prompt_tokens, args.prompt_tokens, caches)
    print(f"layers={len(layers)} capacity={capacity} cpu tokens={reference_tokens} "
          f"cpu step ms median={statistics.median(reference_ms):.1f}", flush=True)

    compiled = compile_model(
        {"decode": (layers, example)}, target="aimx", units=args.fpgas,
        kv_inputs={"decode": decode_kv_inputs}, repeats={"decode": args.new_tokens - 1},
        runtime_options={"base_fpga": args.base_fpga},
        cache_path=args.cache, max_benchmarks=args.max_benchmarks,
    )
    memory = compiled.session.memory
    print(f"kv_bytes_per_fpga={memory.kv_bytes} bytes_per_fpga={memory.bytes_per_unit} "
          f"capacity={memory.capacity_bytes}", flush=True)
    for stage in compiled.session.plans["decode"][0].stages:
        print(f"  stage {stage.name} mnk={tuple(stage.shape)} x{stage.instance_count} kv={stage.kv} "
              f"p_b={stage.p_b} rounds={stage.batch_rounds} config={stage.config.key()}", flush=True)

    with compiled.session as session:
        for layer, (key_cache, value_cache) in enumerate(caches):
            session.write_kv(layer, KEY, key_cache, 0, args.prompt_tokens)
            session.write_kv(layer, VALUE, value_cache, 0, args.prompt_tokens)

        def store(layer, cache, start, stop):
            session.write_kv(layer, KEY, cache[0], start, stop)
            session.write_kv(layer, VALUE, cache[1], start, stop)

        stages = stage_records(session.plans["decode"][0])
        runtime = session.backend.model
        step_layers_ms = []
        step_counters = []

        def counters():
            # Python dict만 읽으므로 step 시간에 native 호출을 더하지 않는다.
            return {**runtime.totals, **runtime.loader.totals,
                    "breakdown": {key: dict(record) for key, record in runtime.breakdown.items()}}

        def run_decode(*inputs):
            before = counters()
            start = time.perf_counter_ns()
            outputs = session.run_model("decode", *inputs)
            step_layers_ms.append((time.perf_counter_ns() - start) / 1e6)
            after = counters()
            # kv_push_ms: 이전 step 끝에 store한 위치를 이 step의 첫 cache stage 앞에서 보낸 시간
            step_counters.append({
                **{name: after[name] - before[name]
                   for name in ("pack_wall_ms", "upload_wall_ms", "kv_push_ms", "run_wall_ms")},
                "breakdown": {key: {field: value - before["breakdown"][key][field]
                                    for field, value in record.items() if field != "jobs"}
                              for key, record in after["breakdown"].items()},
            })
            return outputs

        prompt_caches = copy.deepcopy(caches)
        with torch.no_grad():
            tokens, _, step_ms = generate(
                model, token, caches, args.prompt_tokens, args.new_tokens, run_decode, store, head)
        totals = dict(session.backend.model.totals)
        # 같은 session으로 처음부터 다시: prompt 위치만 다시 보내고 CPU 토큰을 입력으로 쓴다.
        for layer, (key_cache, value_cache) in enumerate(prompt_caches):
            session.write_kv(layer, KEY, key_cache, 0, capacity)
            session.write_kv(layer, VALUE, value_cache, 0, capacity)
        with torch.no_grad():
            forced_tokens, forced_logits, _ = generate(
                model, token, prompt_caches, args.prompt_tokens, args.new_tokens,
                lambda *inputs: session.run_model("decode", *inputs), store, head, forced=reference_tokens)

    matches = [left == right for left, right in zip(tokens, reference_tokens)]
    print(f"aimx tokens={tokens}", flush=True)
    print(f"free: match={sum(matches)}/{len(matches)} first_mismatch="
          f"{matches.index(False) if False in matches else None}", flush=True)
    forced_matches = [left == right for left, right in zip(forced_tokens, reference_tokens)]
    errors = [((got - reference).norm() / reference.norm()).item()
              for got, reference in zip(forced_logits, reference_logits)]
    print(f"forced: argmax match={sum(forced_matches)}/{len(forced_matches)} logits rel_l2 "
          f"median={statistics.median(errors):.4%} max={max(errors):.4%}", flush=True)
    print(f"aimx text: {tokenizer.decode(tokens)!r}", flush=True)
    print(f"cpu  text: {tokenizer.decode(reference_tokens)!r}", flush=True)
    # 첫 step은 cache 전체 전송과 배치 대조를 포함한다.
    print(f"aimx step ms: first={step_ms[0]:.1f} median_after={statistics.median(step_ms[1:] or step_ms):.1f} "
          f"kv_bytes={totals['kv_bytes']} kv_push_ms={totals['kv_push_ms']:.1f} "
          f"kv_layout_check_ms={totals['kv_layout_check_ms']:.1f}", flush=True)

    rows = []
    for index, (ms, layers_ms, counter) in enumerate(zip(step_ms, step_layers_ms, step_counters, strict=True)):
        run = slot_totals(list(counter["breakdown"].values()))
        # 더하면 step_ms다. cpu_outside_layers는 embedding, host cache 쓰기와 store, norm/lm_head/argmax.
        detail = {
            "cpu_outside_layers_ms": ms - layers_ms,
            "weight_upload_ms": counter["pack_wall_ms"] + counter["upload_wall_ms"],
            "kv_push_ms": counter["kv_push_ms"],
            **{f"run_{name}_ms": run[name] for name in SLOTS},
            "run_other_ms": run["other"],
        }
        detail["cpu_layers_ms"] = (layers_ms - detail["weight_upload_ms"] - detail["kv_push_ms"]
                                   - counter["run_wall_ms"])
        items = {"h2d_ms": detail["weight_upload_ms"] + run["desc_h2d"] + run["h2d"],
                 "pim_ms": run["invoke"], "d2h_ms": run["d2h"]}
        items["cpu_ms"] = ms - sum(items.values())
        items.update({name.replace("_ms", "_pct"): 100 * value / ms for name, value in list(items.items())})
        rows.append({"step": index, "position": args.prompt_tokens + index, "step_ms": ms, "layers_ms": layers_ms,
                     **items, **{"detail_" + name: value for name, value in detail.items()}})
    names = [name for name in rows[0] if name not in ("step", "position")]
    rows.append({"step": "mean_after_first", "position": "",
                 **{name: statistics.mean(row[name] for row in rows[1:]) for name in names}})
    write_csv(args.report_dir / "step_breakdown.csv", rows)
    # 첫 step은 prompt cache 전송과 배치 대조를 포함하므로 뺀 step 평균이다.
    breakdowns = [counter["breakdown"] for counter in step_counters[1:]]
    write_csv(args.report_dir / "gemm.csv", stage_rows(stages, "weight", breakdowns))
    write_csv(args.report_dir / "attention.csv", stage_rows(stages, "activation", breakdowns))
    print(f"csv={args.report_dir} (step_breakdown.csv, gemm.csv, attention.csv)", flush=True)


if __name__ == "__main__":
    main()
