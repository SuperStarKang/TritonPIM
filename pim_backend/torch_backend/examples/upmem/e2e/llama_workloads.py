"""Llama-family W8A8 prefill and decode layers built from a checkpoint layer.

Every GEMM is INT8 x INT8 -> INT32; the scales are applied to the INT32 result on the
CPU, so the GEMM shapes do not depend on them:
  projection  weight: one scale per output channel (absmax/127);
              activation: one scale per token (absmax/127, computed at run time)
  QK          query and key: one scale per (head, token)
  PV          value: one scale per (head, token), multiplied into the probabilities
              before they are quantized per row, so one scale per row remains
Norms, RoPE, softmax, SwiGLU and residuals stay FP32. No calibration is performed.

Sizes come from the checkpoint config, so MHA (Llama-2-7b) and GQA (Llama-3-8B) layers
share this code. kv selects how a prefill query head reads its KV head:
  share   head h reads KV head h // group, so heads of one group read the same tensor
  repeat  the KV heads are copied to every query head first (Hugging Face repeat_kv)
A decode layer always reads KV head h // group; both modes compute the same values.
"""

import math

import torch

from attention.int8_workloads import rotate_half


KV_MODES = ("share", "repeat")


def quantize_rows(activation):
    """INT8 values and the FP32 scale [..., 1] of every row (last axis): absmax / 127."""
    # 모두 0인 행(compile 예제 입력)은 scale이 0이 되지 않게 해서 0으로 양자화한다.
    scale = activation.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
    return (activation / scale).round().clamp(-127, 127).to(torch.int8), scale


class CheckpointInt8Linear(torch.nn.Module):
    """W8A8 projection; pad_to rounds the physical output columns up (zero weight columns)."""

    def __init__(self, linear, pad_to=1):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("this experiment requires bias-free projections")
        weight = linear.weight.detach()
        weight_scale = weight.abs().amax(dim=1) / 127.0
        if (weight_scale == 0).any():
            raise ValueError("a projection output channel has zero weights")
        quantized = (weight / weight_scale[:, None]).round().clamp(-127, 127).to(torch.int8).t()
        # 물리 열 수를 pad_to 단위로 올린다 (Llama-2 gate/up 11008 → 11264).
        padding = -(-linear.out_features // pad_to) * pad_to - linear.out_features
        self.register_buffer("weight", torch.nn.functional.pad(quantized, (0, padding)).contiguous())
        self.register_buffer("weight_scale", weight_scale)
        self.out_features = linear.out_features

    def forward(self, activation, scale):
        output = torch._int_mm(activation.reshape(-1, activation.shape[-1]).contiguous(), self.weight)
        # 물리 padding 열로 계산하고 논리 out_features 열만 남긴다.
        output = output[:, :self.out_features].float() * scale.reshape(-1, 1) * self.weight_scale
        return output.reshape(*activation.shape[:-1], self.out_features)


class LlamaPrefillLayer(torch.nn.Module):
    def __init__(self, layer, config, cosine, sine, causal_mask, kv="share"):
        super().__init__()
        if config.model_type != "llama" or config.num_attention_heads % config.num_key_value_heads:
            raise ValueError(f"this experiment requires a Llama-family checkpoint with grouped KV heads; "
                             f"got {config.model_type} with {config.num_attention_heads} query and "
                             f"{config.num_key_value_heads} KV heads")
        if kv not in KV_MODES:
            raise ValueError(f"kv must be one of {KV_MODES}, got {kv!r}")
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.q_proj = CheckpointInt8Linear(layer.self_attn.q_proj)
        self.k_proj = CheckpointInt8Linear(layer.self_attn.k_proj)
        self.v_proj = CheckpointInt8Linear(layer.self_attn.v_proj)
        self.o_proj = CheckpointInt8Linear(layer.self_attn.o_proj)
        self.gate_proj = CheckpointInt8Linear(layer.mlp.gate_proj, pad_to=1024)
        self.up_proj = CheckpointInt8Linear(layer.mlp.up_proj, pad_to=1024)
        self.down_proj = CheckpointInt8Linear(layer.mlp.down_proj)
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.group = self.heads // self.kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.kv = kv
        self.register_buffer("mask", causal_mask != 0)
        self.register_buffer("cosine", cosine.contiguous())
        self.register_buffer("sine", sine.contiguous())

    def forward(self, activation):
        sequence = activation.shape[1]
        normalized, normalized_scale = quantize_rows(self.input_layernorm(activation))
        query = self.q_proj(normalized, normalized_scale).reshape(
            sequence, self.heads, self.head_dim,
        ).transpose(0, 1)
        key = self.k_proj(normalized, normalized_scale).reshape(
            sequence, self.kv_heads, self.head_dim,
        ).transpose(0, 1)
        value = self.v_proj(normalized, normalized_scale).reshape(
            sequence, self.kv_heads, self.head_dim,
        ).transpose(0, 1)
        query = query * self.cosine + rotate_half(query) * self.sine
        key = key * self.cosine + rotate_half(key) * self.sine
        key_cache = key.unsqueeze(0).contiguous()
        value_cache = value.unsqueeze(0).contiguous()

        query, query_scale = quantize_rows(query)
        key, key_scale = quantize_rows(key)
        value, value_scale = quantize_rows(value)
        group = self.group
        if self.kv == "repeat":
            # Hugging Face repeat_kv와 같이 KV head를 query head 수만큼 복사한다. head마다 다른 tensor가 된다.
            key, key_scale, value, value_scale = (
                tensor.repeat_interleave(self.group, dim=0) for tensor in (key, key_scale, value, value_scale))
            group = 1
        outputs = []
        for head in range(self.heads):
            # group개의 query head가 같은 KV head를 읽는다 (MHA는 group 1).
            kv_head = head // group
            scores = torch._int_mm(query[head].contiguous(), key[kv_head].t().contiguous())
            scores = (scores.float() * query_scale[head] * key_scale[kv_head].reshape(1, sequence)
                      / math.sqrt(self.head_dim))
            probability = torch.softmax(scores.masked_fill(self.mask, float("-inf")), dim=-1)
            # value scale을 확률에 먼저 곱하면 PV 결과에는 행 scale 하나만 남는다.
            weighted, weighted_scale = quantize_rows(probability * value_scale[kv_head].reshape(1, sequence))
            outputs.append(torch._int_mm(weighted, value[kv_head].contiguous()).float() * weighted_scale)
        context = torch.stack(outputs).transpose(0, 1).reshape(1, sequence, self.hidden_size)
        hidden = activation + self.o_proj(*quantize_rows(context))
        normalized, normalized_scale = quantize_rows(self.post_attention_layernorm(hidden))
        gated = (torch.nn.functional.silu(self.gate_proj(normalized, normalized_scale))
                 * self.up_proj(normalized, normalized_scale))
        output = hidden + self.down_proj(*quantize_rows(gated))
        return output, key_cache, value_cache


class LlamaDecodeLayer(torch.nn.Module):
    """One new token through the projections of a prefill layer, reading the key/value cache.

    caches holds, per layer, the INT8 key [kv_heads, capacity, head_dim] after RoPE, its scale
    [kv_heads, capacity], the INT8 value and its scale, quantized as the prefill layer
    quantizes them; positions are token positions. device_valid [1, capacity] marks the
    positions the device cache holds. The positions written to the host cache but not yet
    sent are the tail: tail_rows indexes them and tail_valid marks the real ones. The tail
    and the new token's own key and value join the attention on the CPU, as in the AiMX
    decode layer (examples/aimx/llama_decode.py), so this layer only reads the cache. A query
    head reads its KV head h // group. The caller writes the returned (key [kv_heads,
    head_dim], key scale [kv_heads], value, value scale) at the token position after the
    step. Every GEMM has one output row (decode GEMV): the projections, query x key[h]^T and
    probability x value[h], the cache operand on the right.
    """

    def __init__(self, prefill, index):
        super().__init__()
        for name in ("input_layernorm", "post_attention_layernorm", "q_proj", "k_proj", "v_proj",
                     "o_proj", "gate_proj", "up_proj", "down_proj"):
            setattr(self, name, getattr(prefill, name))
        self.index = index
        self.heads = prefill.heads
        self.kv_heads = prefill.kv_heads
        self.group = prefill.group
        self.head_dim = prefill.head_dim
        self.hidden_size = prefill.hidden_size

    def window(self, cache, head):
        """The cache [capacity, head_dim] of the KV head that query head reads."""
        return cache[head // self.group]

    def forward(self, activation, cosine, sine, device_valid, tail_rows, tail_valid, caches, new):
        key_cache, key_cache_scale, value_cache, value_cache_scale = caches[self.index]
        capacity = key_cache.shape[1]
        normalized, normalized_scale = quantize_rows(self.input_layernorm(activation))
        query = self.q_proj(normalized, normalized_scale).reshape(self.heads, 1, self.head_dim)
        key = self.k_proj(normalized, normalized_scale).reshape(self.kv_heads, 1, self.head_dim)
        value = self.v_proj(normalized, normalized_scale).reshape(self.kv_heads, 1, self.head_dim)
        query, query_scale = quantize_rows(query * cosine + rotate_half(query) * sine)
        key, key_scale = quantize_rows(key * cosine + rotate_half(key) * sine)
        value, value_scale = quantize_rows(value)

        # CPU가 맡는 위치: 장치에 아직 보내지 않은 tail 위치들과 새 token 자신 [heads, tail + 1, head_dim]
        host_keys = torch.cat((key_cache[:, tail_rows], key), dim=1).repeat_interleave(self.group, dim=0)
        host_values = torch.cat((value_cache[:, tail_rows], value), dim=1).repeat_interleave(self.group, dim=0)
        cache_scores = torch.stack([
            torch._int_mm(query[head], self.window(key_cache, head).t()).reshape(capacity)
            for head in range(self.heads)])
        host_scores = (query.to(torch.int32) * host_keys.to(torch.int32)).sum(dim=-1)
        # 장치 위치 0..capacity-1 다음에 CPU 위치가 붙는다. scale과 mask도 같은 순서로 잇는다.
        scores = torch.cat((cache_scores, host_scores), dim=1)
        key_scales = torch.cat((key_cache_scale, key_cache_scale[:, tail_rows], key_scale.reshape(self.kv_heads, 1)),
                               dim=1).repeat_interleave(self.group, dim=0)
        value_scales = torch.cat((value_cache_scale, value_cache_scale[:, tail_rows],
                                  value_scale.reshape(self.kv_heads, 1)), dim=1).repeat_interleave(self.group, dim=0)
        scores = scores.float() * query_scale.reshape(self.heads, 1) * key_scales / math.sqrt(self.head_dim)
        visible = torch.cat((device_valid, tail_valid.reshape(1, -1), torch.ones((1, 1), dtype=torch.bool)), dim=1)
        probability = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
        weighted, weighted_scale = quantize_rows(probability * value_scales)

        cache_outputs = torch.stack([
            torch._int_mm(weighted[head, :capacity].reshape(1, capacity).contiguous(),
                          self.window(value_cache, head)
                          ).reshape(self.head_dim)
            for head in range(self.heads)])
        host_outputs = (weighted[:, capacity:, None].to(torch.int32) * host_values.to(torch.int32)).sum(dim=1)
        context = ((cache_outputs + host_outputs).float() * weighted_scale).reshape(1, 1, self.hidden_size)

        hidden = activation + self.o_proj(*quantize_rows(context))
        normalized, normalized_scale = quantize_rows(self.post_attention_layernorm(hidden))
        gated = (torch.nn.functional.silu(self.gate_proj(normalized, normalized_scale))
                 * self.up_proj(normalized, normalized_scale))
        output = hidden + self.down_proj(*quantize_rows(gated))
        # Session은 tuple 출력을 다음 layer의 위치 인자로 넘긴다. cache는 읽기만 하고 그대로 넘긴다.
        # key/value [kv_heads, 1, head_dim], scale [kv_heads, 1, 1] -> 위치 하나의 값
        return output, cosine, sine, device_valid, tail_rows, tail_valid, caches, new + (
            (key[:, 0], key_scale[:, 0, 0], value[:, 0], value_scale[:, 0, 0]),)
