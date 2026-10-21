"""Greedy generation of one Llama-family request: a 1024-token prefill, then decode steps.

The prefill layers return the INT8 key and value of their 1024 positions with one scale
per (head, position); the host writes them into one cache per layer that holds
1024 + new tokens positions. Each decode step feeds the last token at the next position
through the decode layers, which read the cache and return the new key and value with
their scales; the host writes them after the step.
run_prefill and run_decode chain the layers of a phase, so a PIMSession phase and the
CPU layers go through the same loop.
"""

import time

import torch

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


def decode_inputs(model, hidden, position, caches):
    """First decode layer inputs for the token at position; the cache holds positions 0..position-1."""
    cosine, sine = model.model.rotary_emb(hidden, position_ids=torch.tensor([[position]]))
    capacity = caches[0][0].shape[1]
    valid = (torch.arange(capacity) < position).unsqueeze(0)
    return hidden, cosine, sine, valid, caches, ()


def decode_example(model, capacity):
    """Example inputs of the first decode layer at the first decode position."""
    hidden = torch.zeros((1, 1, model.config.hidden_size), dtype=torch.float32)
    return decode_inputs(model, hidden, PROMPT_TOKENS, empty_caches(model.config, capacity))


def run_layers(layers, *inputs):
    for layer in layers:
        inputs = layer(*inputs)
    return inputs


def generate(model, input_ids, new_tokens, run_prefill, run_decode):
    """Greedy token ids of one request and the wall time of the first token and of every step.

    The cache holds PROMPT_TOKENS + new_tokens positions; the last token is never fed back.
    """
    if tuple(input_ids.shape) != (1, PROMPT_TOKENS):
        raise ValueError(f"the prompt must have {PROMPT_TOKENS} tokens, got {tuple(input_ids.shape)}")
    caches = empty_caches(model.config, PROMPT_TOKENS + new_tokens)
    start = time.perf_counter_ns()
    hidden, prefill_cache = run_prefill(model.model.embed_tokens(input_ids), ())
    for cache, prefill_parts in zip(caches, prefill_cache, strict=True):
        for part, written in zip(cache, prefill_parts, strict=True):
            part[:, :PROMPT_TOKENS].copy_(written)
    token = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1)
    tokens = [token.item()]
    first_token_ms = (time.perf_counter_ns() - start) / 1e6
    step_ms = []
    for position in range(PROMPT_TOKENS, PROMPT_TOKENS + new_tokens - 1):
        start = time.perf_counter_ns()
        hidden = model.model.embed_tokens(token.reshape(1, 1))
        hidden, *_, new = run_decode(*decode_inputs(model, hidden, position, caches))
        for cache, new_parts in zip(caches, new, strict=True):
            for part, written in zip(cache, new_parts, strict=True):
                part[:, position:position + 1].copy_(written)
        token = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1)
        tokens.append(token.item())
        step_ms.append((time.perf_counter_ns() - start) / 1e6)
    return {"tokens": tokens, "first_token_ms": first_token_ms, "step_ms": step_ms}
