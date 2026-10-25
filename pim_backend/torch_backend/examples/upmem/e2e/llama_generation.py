"""Greedy generation of one Llama-family request: a 1024-token prefill, then decode steps.

The prefill layers return the INT8 key and value of their 1024 positions with one scale
per (head, position); after each prefill layer the host writes them into that layer's
cache, [kv_heads, 1024 + new tokens, head_dim] (the scales [kv_heads, positions]), the
layout the AiMX decode layer uses too. Each decode step feeds the last token at the next
position through the decode layers, which read the cache and return the new key and value
with their scales; the host writes them at that position after the step. Positions are
token positions: each PIM target places them on its units in its own layout. After every
write, store(layer, cache, start, stop) is told which positions changed: on PIM it sends
them to the device cache the decode layers read in place (decode_kv_inputs), on the CPU it
does nothing. The scales stay on the host.
run_prefill and run_decode chain the layers of a phase, so a PIMSession phase and the
CPU layers go through the same loop.
"""

import time

import torch
from triton.backends.pim.residency import KEY, VALUE

from e2e.llama_workloads import LlamaDecodeLayer, LlamaPrefillLayer, quantize_rows


PROMPT_TOKENS = 1024


class GenerationPrefillLayer(LlamaPrefillLayer):
    def forward(self, hidden, cache):
        hidden, key, value = super().forward(hidden)
        # decode가 읽는 cache: prefill attention이 쓴 양자화 값 [kv_heads, 1024, head_dim]과
        # scale [kv_heads, 1024]
        key, key_scale = quantize_rows(key[0])
        value, value_scale = quantize_rows(value[0])
        return hidden, cache + ((key, key_scale[..., 0], value, value_scale[..., 0]),)


def generation_layers(model, kv):
    """Prefill and decode layers of every checkpoint layer; a decode layer shares its weights."""
    config = model.config
    example = torch.zeros((1, PROMPT_TOKENS, config.hidden_size), dtype=torch.float32)
    cosine, sine = model.model.rotary_emb(example, position_ids=torch.arange(PROMPT_TOKENS).unsqueeze(0))
    causal_mask = torch.full(
        (PROMPT_TOKENS, PROMPT_TOKENS), torch.finfo(torch.float32).min, dtype=torch.float32,
    ).triu(1)
    prefill = [GenerationPrefillLayer(layer, config, cosine, sine, causal_mask, kv).eval()
               for layer in model.model.layers]
    decode = [LlamaDecodeLayer(layer, index).eval() for index, layer in enumerate(prefill)]
    return prefill, decode


def empty_caches(config, capacity):
    """Per layer: INT8 key [kv_heads, capacity, head_dim], its scale [kv_heads, capacity], INT8 value, its scale."""
    shape = (config.num_key_value_heads, capacity, config.head_dim)
    scale_shape = (config.num_key_value_heads, capacity)
    return tuple((torch.zeros(shape, dtype=torch.int8), torch.zeros(scale_shape, dtype=torch.float32),
                  torch.zeros(shape, dtype=torch.int8), torch.zeros(scale_shape, dtype=torch.float32))
                 for _ in range(config.num_hidden_layers))


def decode_inputs(model, hidden, position, caches):
    """First decode layer inputs for the token at position; the cache holds positions 0..position-1."""
    cosine, sine = model.model.rotary_emb(hidden, position_ids=torch.tensor([[position]]))
    capacity = caches[0][0].shape[1]
    return hidden, cosine, sine, (torch.arange(capacity) < position).unsqueeze(0), caches, ()


def decode_kv_inputs(index, inputs):
    """The host INT8 key and value cache of decode layer index among its inputs (decode_inputs)."""
    key_cache, _, value_cache, _ = inputs[4][index]
    return {KEY: key_cache, VALUE: value_cache}


def decode_example(model, capacity):
    """Example inputs of the first decode layer at the first decode position."""
    hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float32)
    return decode_inputs(model, hidden, PROMPT_TOKENS, empty_caches(model.config, capacity))


def run_layers(layers, *inputs, after_layer=None):
    """Chain layers like PIMSession.run_model; after_layer(index, outputs) runs after each."""
    for index, layer in enumerate(layers):
        inputs = layer(*inputs)
        if after_layer is not None:
            after_layer(index, inputs)
    return inputs


def generate(model, input_ids, new_tokens, run_prefill, run_decode, store):
    """Greedy token ids of one request and the wall time of the first token and of every step.

    run_prefill(hidden, cache, after_layer) runs the prefill layers; run_decode(*inputs) the
    decode layers. store(layer, cache, start, stop) follows every cache write of positions
    [start, stop) of a layer; both times include it. The cache holds PROMPT_TOKENS +
    new_tokens positions; the last token is never fed back.
    """
    if tuple(input_ids.shape) != (1, PROMPT_TOKENS):
        raise ValueError(f"the prompt must have {PROMPT_TOKENS} tokens, got {tuple(input_ids.shape)}")
    capacity = PROMPT_TOKENS + new_tokens
    caches = empty_caches(model.config, capacity)

    def write_prefill(index, outputs):
        # layer index가 막 쓴 cache는 출력 cache tuple의 마지막 항목이다.
        _, prefill_cache = outputs
        for part, written in zip(caches[index], prefill_cache[index], strict=True):
            part[:, :PROMPT_TOKENS] = written
        store(index, caches[index], 0, PROMPT_TOKENS)

    start = time.perf_counter_ns()
    hidden, _ = run_prefill(model.model.embed_tokens(input_ids), (), write_prefill)
    token = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1)
    tokens = [token.item()]
    first_token_ms = (time.perf_counter_ns() - start) / 1e6
    step_ms = []
    for position in range(PROMPT_TOKENS, PROMPT_TOKENS + new_tokens - 1):
        start = time.perf_counter_ns()
        hidden = model.model.embed_tokens(token.reshape(1, 1))
        hidden, *_, new = run_decode(*decode_inputs(model, hidden, position, caches))
        for layer, (cache, new_parts) in enumerate(zip(caches, new, strict=True)):
            for part, written in zip(cache, new_parts, strict=True):
                part[:, position] = written
            store(layer, cache, position, position + 1)
        token = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1)
        tokens.append(token.item())
        step_ms.append((time.perf_counter_ns() - start) / 1e6)
    return {"tokens": tokens, "first_token_ms": first_token_ms, "step_ms": step_ms}
