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

  python benchmark_decode.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../../upmem/e2e/prompt.txt \
      --prompt-tokens 1024 --new-tokens 33 --layers 4
"""

import argparse
import copy
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


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=33, help="generated tokens, the prefill's included")
    parser.add_argument("--layers", type=int, default=0, help="first decoder layers to run (0: all)")
    args = parser.parse_args()
    if args.new_tokens < 2:
        parser.error("--new-tokens must include at least one decode step")

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
        layers = [LlamaAimxDecodeLayer(layer, index, args.fpgas).eval()
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
        # AiMX cache stage는 instance마다 모든 위치를 가지므로 위치 순서를 바꾸지 않는다.
        if not torch.equal(session.kv_rows("decode"), torch.arange(capacity)):
            raise ValueError("the AiMX cache rows must be the positions in order")
        for layer, (key_cache, value_cache) in enumerate(caches):
            session.write_kv(layer, KEY, key_cache, 0, args.prompt_tokens)
            session.write_kv(layer, VALUE, value_cache, 0, args.prompt_tokens)

        def store(layer, cache, start, stop):
            session.write_kv(layer, KEY, cache[0], start, stop)
            session.write_kv(layer, VALUE, cache[1], start, stop)

        prompt_caches = copy.deepcopy(caches)
        with torch.no_grad():
            tokens, _, step_ms = generate(
                model, token, caches, args.prompt_tokens, args.new_tokens,
                lambda *inputs: session.run_model("decode", *inputs), store, head)
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


if __name__ == "__main__":
    main()
