"""Llama-family W8A8 prefill layer built from a checkpoint layer.

The seven projections are quantized to INT8 with a per-tensor absmax/128 scale and
run as INT8 x INT8 -> INT32 GEMMs, as are the per-head QK/PV GEMMs. Norms, RoPE,
softmax, SwiGLU and residuals stay FP32. No calibration is performed.

Sizes come from the checkpoint config, so MHA (Llama-2-7b) and GQA (Llama-3-8B) layers
share this code. kv selects how a query head reads its KV head:
  share   head h reads KV head h // group, so heads of one group read the same tensor
  repeat  the KV heads are copied to every query head first (Hugging Face repeat_kv)
"""

import torch

from attention.int8_workloads import (
    ACTIVATION_SCALE, AttentionKernel, quantize_activation, rotate_half,
)


KV_MODES = ("share", "repeat")


class CheckpointInt8Linear(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("this experiment requires bias-free projections")

        weight_scale = linear.weight.detach().abs().amax() / 128.0
        if weight_scale.item() == 0:
            raise ValueError("projection weight scale is zero")
        self.register_buffer(
            "weight",
            (linear.weight.detach() / weight_scale).round().clamp(-127, 127)
            .to(torch.int8).t().contiguous(),
        )
        self.register_buffer("output_scale", weight_scale * ACTIVATION_SCALE)
        self.out_features = linear.out_features

    def forward(self, activation):
        output = torch._int_mm(
            activation.reshape(-1, activation.shape[-1]).contiguous(), self.weight,
        )
        output = output.float() * self.output_scale
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
        self.gate_proj = CheckpointInt8Linear(layer.mlp.gate_proj)
        self.up_proj = CheckpointInt8Linear(layer.mlp.up_proj)
        self.down_proj = CheckpointInt8Linear(layer.mlp.down_proj)
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.group = self.heads // self.kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.kv = kv
        self.core = AttentionKernel(self.heads, 1 if kv == "repeat" else self.group)
        self.core.mask = causal_mask != 0
        self.register_buffer("cosine", cosine.contiguous())
        self.register_buffer("sine", sine.contiguous())

    def forward(self, activation):
        sequence = activation.shape[1]
        normalized = quantize_activation(self.input_layernorm(activation))
        query = self.q_proj(normalized).reshape(
            sequence, self.heads, self.head_dim,
        ).transpose(0, 1)
        key = self.k_proj(normalized).reshape(
            sequence, self.kv_heads, self.head_dim,
        ).transpose(0, 1)
        value = self.v_proj(normalized).reshape(
            sequence, self.kv_heads, self.head_dim,
        ).transpose(0, 1)
        query = query * self.cosine + rotate_half(query) * self.sine
        key = key * self.cosine + rotate_half(key) * self.sine
        key_cache = key.unsqueeze(0).contiguous()
        value_cache = value.unsqueeze(0).contiguous()

        key = quantize_activation(key)
        value = quantize_activation(value)
        if self.kv == "repeat":
            # Hugging Face repeat_kv와 같이 KV head를 query head 수만큼 복사한다. head마다 다른 tensor가 된다.
            key = key.repeat_interleave(self.group, dim=0)
            value = value.repeat_interleave(self.group, dim=0)
        context = self.core(
            quantize_activation(query).contiguous(), key.contiguous(), value.contiguous(),
        ).transpose(0, 1).reshape(1, sequence, self.hidden_size)
        hidden = activation + self.o_proj(quantize_activation(context))
        normalized = quantize_activation(self.post_attention_layernorm(hidden))
        gated = torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        output = hidden + self.down_proj(quantize_activation(gated))
        return output, key_cache, value_cache
