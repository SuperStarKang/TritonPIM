"""Llama-family decode on AiMX with the key/value cache resident on the FPGAs.

The host cache of a layer is key, value [capacity, kv_heads * head_dim] BF16, one row per
position, after RoPE. FPGA f owns the KV heads of column window f (kv_heads / fpgas heads,
W = kv_heads * head_dim / fpgas columns) and the query heads that read them (rows = heads /
fpgas). Every attention GEMM is one instance per FPGA:
  QK  [rows, W] @ key window^T  -> [rows, capacity]   row r holds its query only in the
                                                      columns of its KV head (zeros elsewhere)
  PV  [rows, capacity] @ value window -> [rows, W]    row r keeps the columns of its KV head
MHA (rows 16 x 1 KV head each) and GQA (group query heads per KV head) differ only in
which columns a row uses.

The device holds positions [0, flushed). A new token's key and value go into the host cache
right after its step; positions [flushed, position) are the tail, at most TAIL - 1, and
join the attention on the CPU together with the token's own key and value. When the tail
reaches TAIL positions, store sends them (one burst of 32 B per key row on the device).
"""

import time

import torch
from triton.backends.pim.residency import KEY, VALUE

from llama_workloads import CheckpointLinear, CheckpointRMSNorm, rotate_half


TAIL = 16


class LlamaAimxDecodeLayer(torch.nn.Module):
    def __init__(self, layer, index, fpgas):
        super().__init__()
        config = layer.self_attn.config
        self.index = index
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group = self.heads // self.kv_heads
        if self.heads % fpgas or self.kv_heads % fpgas:
            raise ValueError(f"{self.heads} query and {self.kv_heads} KV heads must split over {fpgas} FPGAs")
        self.fpgas = fpgas
        self.rows = self.heads // fpgas
        self.width = self.kv_heads * self.head_dim // fpgas
        self.input_layernorm = CheckpointRMSNorm(layer.input_layernorm)
        self.post_attention_layernorm = CheckpointRMSNorm(layer.post_attention_layernorm)
        self.q_proj = CheckpointLinear(layer.self_attn.q_proj)
        self.k_proj = CheckpointLinear(layer.self_attn.k_proj)
        self.v_proj = CheckpointLinear(layer.self_attn.v_proj)
        self.o_proj = CheckpointLinear(layer.self_attn.o_proj)
        # prefill layer(LlamaAimxLayer)와 같이 Gate/Up은 N을 512 단위로 올려 계산하고 앞 열만 쓴다.
        padded = -(-config.intermediate_size // 512) * 512
        self.gate_proj = CheckpointLinear(layer.mlp.gate_proj, padded)
        self.up_proj = CheckpointLinear(layer.mlp.up_proj, padded)
        self.down_proj = CheckpointLinear(layer.mlp.down_proj)
        self.scaling = layer.self_attn.scaling
        # FPGA f의 행 r(query head f·rows + r)이 자기 window 안에서 읽는 KV head: one-hot [fpgas, rows, window head 수]
        heads_per_window = self.kv_heads // fpgas
        local = ((torch.arange(self.heads) // self.group).reshape(fpgas, self.rows)
                 - (torch.arange(fpgas) * heads_per_window).reshape(fpgas, 1))
        self.register_buffer("select", torch.nn.functional.one_hot(local, heads_per_window).to(torch.bfloat16))

    def forward(self, hidden, cosine, sine, device_valid, tail_rows, tail_valid, caches, new):
        key_cache, value_cache = caches[self.index]
        capacity = key_cache.shape[0]
        normalized = self.input_layernorm(hidden)
        query = self.q_proj(normalized).reshape(self.heads, self.head_dim)
        key = self.k_proj(normalized).reshape(self.kv_heads, self.head_dim)
        value = self.v_proj(normalized).reshape(self.kv_heads, self.head_dim).bfloat16()
        rotation = cosine.reshape(1, self.head_dim), sine.reshape(1, self.head_dim)
        query = (query * rotation[0] + rotate_half(query) * rotation[1]).bfloat16()
        key = (key * rotation[0] + rotate_half(key) * rotation[1]).bfloat16()

        # QK on PIM: 행마다 자기 KV head 열에만 q를 두고 나머지는 0이다.
        spread = (query.reshape(self.fpgas, self.rows, 1, self.head_dim) * self.select.unsqueeze(-1)
                  ).reshape(self.fpgas, self.rows, self.width)
        device_scores = torch.cat([
            torch.mm(spread[f], key_cache.narrow(1, f * self.width, self.width).t())
            for f in range(self.fpgas)]).float()
        # tail과 자기 위치는 CPU. query head h는 KV head h // group을 읽는다.
        tail_keys = key_cache[tail_rows].reshape(TAIL, self.kv_heads, self.head_dim).repeat_interleave(self.group, dim=1)
        tail_scores = torch.einsum("hd,thd->ht", query.float(), tail_keys.float())
        own_scores = (query.float() * key.repeat_interleave(self.group, dim=0).float()).sum(dim=-1, keepdim=True)
        scores = torch.cat((device_scores.masked_fill(~device_valid, float("-inf")),
                            tail_scores.masked_fill(~tail_valid.reshape(1, TAIL), float("-inf")),
                            own_scores), dim=1) * self.scaling
        probability = torch.softmax(scores, dim=-1)

        # PV on PIM: 결과 행마다 자기 KV head 열만 남긴다.
        device_probability = probability[:, :capacity].bfloat16().reshape(self.fpgas, self.rows, capacity)
        device_context = torch.cat([
            (torch.mm(device_probability[f], value_cache.narrow(1, f * self.width, self.width))
             .reshape(self.rows, -1, self.head_dim) * self.select[f].unsqueeze(-1)).sum(dim=1)
            for f in range(self.fpgas)]).float()
        tail_values = value_cache[tail_rows].reshape(TAIL, self.kv_heads, self.head_dim).repeat_interleave(self.group, dim=1)
        tail_context = torch.einsum("ht,thd->hd", probability[:, capacity:capacity + TAIL], tail_values.float())
        own_context = probability[:, -1:] * value.repeat_interleave(self.group, dim=0).float()
        context = (device_context + tail_context + own_context).reshape(1, 1, self.heads * self.head_dim)

        hidden = hidden + self.o_proj(context)
        normalized = self.post_attention_layernorm(hidden)
        gated = torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        output = hidden + self.down_proj(gated)
        # Session은 tuple 출력을 다음 layer의 위치 인자로 넘긴다. cache는 읽기만 하고 그대로 넘긴다.
        return (output, cosine, sine, device_valid, tail_rows, tail_valid, caches,
                new + ((key.reshape(1, -1), value.reshape(1, -1)),))


def empty_caches(config, capacity):
    """Per layer: BF16 key and value [capacity, kv_heads * head_dim]."""
    shape = (capacity, config.num_key_value_heads * config.head_dim)
    return tuple((torch.zeros(shape, dtype=torch.bfloat16), torch.zeros(shape, dtype=torch.bfloat16))
                 for _ in range(config.num_hidden_layers))


def decode_inputs(model, hidden, position, flushed, caches):
    """First decode layer inputs for the token at position: the device holds [0, flushed),
    the tail [flushed, position) is read from the host cache."""
    cosine, sine = model.model.rotary_emb(hidden, position_ids=torch.tensor([[position]]))
    capacity = caches[0][0].shape[0]
    tail = torch.arange(flushed, flushed + TAIL)
    return (hidden, cosine, sine, (torch.arange(capacity) < flushed).unsqueeze(0),
            tail.clamp(max=capacity - 1), tail < position, caches, ())


def decode_kv_inputs(index, inputs):
    """The host key and value cache of decode layer index among its inputs (decode_inputs)."""
    key_cache, value_cache = inputs[6][index]
    return {KEY: key_cache, VALUE: value_cache}


def prefill_caches(model, input_ids, capacity):
    """CPU prefill with the Hugging Face model: the first token and the caches holding the prompt."""
    output = model(input_ids, use_cache=True)
    caches = empty_caches(model.config, capacity)
    prompt = input_ids.shape[1]
    for (key_cache, value_cache), layer in zip(caches, output.past_key_values.layers, strict=True):
        # [1, kv_heads, prompt, head_dim] -> 위치마다 한 행
        key_cache[:prompt] = layer.keys[0].transpose(0, 1).reshape(prompt, -1)
        value_cache[:prompt] = layer.values[0].transpose(0, 1).reshape(prompt, -1)
    return output.logits[:, -1].argmax(dim=-1), caches


def generate(model, token, caches, prompt, new_tokens, run_decode, store, head, forced=None):
    """Greedy decode from the prompt's next token; store(layer, cache pair, start, stop) sends rows.

    head(hidden) -> logits. forced, a token list like the result, feeds its tokens instead of
    the argmax (teacher forcing), so two runs see the same inputs at every step. Returns the
    token ids (the first one included), the logits of every step and step times in ms.
    """
    capacity = caches[0][0].shape[0]
    if prompt + new_tokens > capacity:
        raise ValueError(f"{prompt} + {new_tokens} positions exceed the cache capacity {capacity}")
    tokens = [token.item()]
    flushed = prompt
    step_logits = []
    step_ms = []
    for position in range(prompt, prompt + new_tokens - 1):
        start = time.perf_counter_ns()
        hidden = model.model.embed_tokens(token.reshape(1, 1)).float()
        hidden, *_, new = run_decode(*decode_inputs(model, hidden, position, flushed, caches))
        for (key_cache, value_cache), (key, value) in zip(caches, new, strict=True):
            key_cache[position] = key[0]
            value_cache[position] = value[0]
        if position + 1 - flushed == TAIL:
            for layer, cache in enumerate(caches):
                store(layer, cache, flushed, position + 1)
            flushed = position + 1
        logits = head(hidden[:, -1, :])
        tokens.append(logits.argmax(dim=-1).item())
        token = logits.argmax(dim=-1) if forced is None else torch.tensor([forced[len(tokens) - 1]])
        step_logits.append(logits[0])
        step_ms.append((time.perf_counter_ns() - start) / 1e6)
    return tokens, step_logits, step_ms
