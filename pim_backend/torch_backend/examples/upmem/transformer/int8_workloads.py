"""One synthetic Llama-2-7B-shaped transformer block for performance measurement."""

import torch

from attention.int8_workloads import HIDDEN, AttentionLayer, Int8Linear, quantize_activation


INTERMEDIATE = 11008


class RMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(HIDDEN, dtype=torch.float32))

    def forward(self, activation):
        return activation * torch.rsqrt(activation.square().mean(dim=-1, keepdim=True) + 1e-5) * self.weight


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm()
        self.attention = AttentionLayer()
        self.post_attention_norm = RMSNorm()
        self.gate_proj = Int8Linear(HIDDEN, INTERMEDIATE)
        self.up_proj = Int8Linear(HIDDEN, INTERMEDIATE)
        self.down_proj = Int8Linear(INTERMEDIATE, HIDDEN)

    def forward(self, activation):
        hidden = activation + self.attention(self.input_norm(activation))
        normalized = quantize_activation(self.post_attention_norm(hidden))
        gated = torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return hidden + self.down_proj(quantize_activation(gated))
