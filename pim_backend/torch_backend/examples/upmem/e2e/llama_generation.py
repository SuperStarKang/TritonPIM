"""Greedy generation of one Llama-family request: a 1024-token prefill, then decode steps.

The prefill layers return the INT8 key and value of their 1024 positions with one scale
per (head, position); after each prefill layer the host writes them into that layer's
cache, which holds 1024 + new tokens positions. Each decode step feeds the last token at
the next position through the decode layers, which read the cache and return the new key
and value with their scales; the host writes them after the step. Position p sits at cache
row rows[p] (on PIM the cyclic order of PIMSession.kv_rows, so the written positions fill
every unit evenly); the key, the value, their scales and the valid mask share that order.
After every write, store(layer, cache, start, stop) is told which rows changed: on PIM it
sends them to the device cache the decode layers read in place (decode_kv_inputs), on the
CPU it does nothing. The scales stay on the host.
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
        # decode가 읽는 cache: prefill attention이 쓴 양자화 값과 scale이다. [kv_heads, 1024, head_dim], [kv_heads, 1024, 1]
        return hidden, cache + ((*quantize_rows(key[0]), *quantize_rows(value[0])),)


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
    """Per layer: INT8 key, its scale, INT8 value, its scale."""
    shape = (config.num_key_value_heads, capacity, config.head_dim)
    scale_shape = (config.num_key_value_heads, capacity, 1)
    return tuple((torch.zeros(shape, dtype=torch.int8), torch.zeros(scale_shape, dtype=torch.float32),
                  torch.zeros(shape, dtype=torch.int8), torch.zeros(scale_shape, dtype=torch.float32))
                 for _ in range(config.num_hidden_layers))


def decode_inputs(model, hidden, position, caches, rows):
    """First decode layer inputs for the token at position; the cache holds positions 0..position-1.

    rows[p] is the cache row of position p, so the rows of the earlier positions are valid.
    """
    cosine, sine = model.model.rotary_emb(hidden, position_ids=torch.tensor([[position]]))
    capacity = caches[0][0].shape[1]
    valid = torch.zeros(capacity, dtype=torch.bool)
    valid[rows[:position]] = True
    return hidden, cosine, sine, valid.unsqueeze(0), caches, ()


def decode_kv_inputs(index, inputs):
    """The host INT8 key and value cache of decode layer index among its inputs (decode_inputs)."""
    key_cache, _, value_cache, _ = inputs[4][index]
    return {KEY: key_cache, VALUE: value_cache}


def decode_example(model, capacity):
    """Example inputs of the first decode layer at the first decode position."""
    hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float32)
    # 행 순서는 compile 뒤에 정해진다. 예시 입력은 shape만 쓰이므로 위치 순서로 둔다.
    return decode_inputs(model, hidden, PROMPT_TOKENS, empty_caches(model.config, capacity),
                         torch.arange(capacity))


def run_layers(layers, *inputs, after_layer=None):
    """Chain layers like PIMSession.run_model; after_layer(index, outputs) runs after each."""
    for index, layer in enumerate(layers):
        inputs = layer(*inputs)
        if after_layer is not None:
            after_layer(index, inputs)
    return inputs


def generate(model, input_ids, new_tokens, rows, run_prefill, run_decode, store):
    """Greedy token ids of one request and the wall time of the first token and of every step.

    rows[p] is the cache row of position p (PIMSession.kv_rows, or torch.arange for position
    order). run_prefill(hidden, cache, after_layer) runs the prefill layers; run_decode(*inputs)
    the decode layers. store(layer, cache, start, stop) follows every cache write of rows
    [start, stop) of a layer; both times include it. The cache holds PROMPT_TOKENS + new_tokens
    positions; the last token is never fed back.
    """
    if tuple(input_ids.shape) != (1, PROMPT_TOKENS):
        raise ValueError(f"the prompt must have {PROMPT_TOKENS} tokens, got {tuple(input_ids.shape)}")
    capacity = PROMPT_TOKENS + new_tokens
    if tuple(rows.shape) != (capacity,):
        raise ValueError(f"rows order {tuple(rows.shape)} positions; the cache holds {capacity}")
    caches = empty_caches(model.config, capacity)

    def write_prefill(index, outputs):
        # layer index가 막 쓴 cache는 출력 cache tuple의 마지막 항목이다.
        _, prefill_cache = outputs
        for part, written in zip(caches[index], prefill_cache[index], strict=True):
            part[:, rows[:PROMPT_TOKENS]] = written
        # prefill 위치의 행은 연속이 아닐 수 있으므로 cache 전체를 알린다.
        store(index, caches[index], 0, capacity)

    start = time.perf_counter_ns()
    hidden, _ = run_prefill(model.model.embed_tokens(input_ids), (), write_prefill)
    token = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1)
    tokens = [token.item()]
    first_token_ms = (time.perf_counter_ns() - start) / 1e6
    step_ms = []
    for position in range(PROMPT_TOKENS, PROMPT_TOKENS + new_tokens - 1):
        start = time.perf_counter_ns()
        hidden = model.model.embed_tokens(token.reshape(1, 1))
        hidden, *_, new = run_decode(*decode_inputs(model, hidden, position, caches, rows))
        row = rows[position].item()
        for layer, (cache, new_parts) in enumerate(zip(caches, new, strict=True)):
            for part, written in zip(cache, new_parts, strict=True):
                part[:, row:row + 1].copy_(written)
            store(layer, cache, row, row + 1)
        token = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1)
        tokens.append(token.item())
        step_ms.append((time.perf_counter_ns() - start) / 1e6)
    return {"tokens": tokens, "first_token_ms": first_token_ms, "step_ms": step_ms}
