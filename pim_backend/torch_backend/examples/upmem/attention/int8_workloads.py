"""Synthetic batch-one W8A8 prefill workloads; all GEMMs accumulate in INT32."""

import math

import torch


SEQUENCE = 1024
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
ACTIVATION_SCALE = 1.0 / 32.0


def quantize_activation(activation):
    return (activation / ACTIVATION_SCALE).round().clamp(-127, 127).to(torch.int8)


class Int8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer("weight", torch.randint(
            -8, 8, (in_features, out_features), dtype=torch.int8,
        ))
        self.out_features = out_features
        self.output_scale = ACTIVATION_SCALE / (8.0 * math.sqrt(in_features))

    def forward(self, activation):
        output = torch._int_mm(
            activation.reshape(-1, activation.shape[-1]), self.weight,
        ).float() * self.output_scale
        return output.reshape(*activation.shape[:-1], self.out_features)


class AttentionKernel(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        self.register_buffer("mask", torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool).triu(1))

    def forward(self, query, key, value):
        outputs = []
        for head in range(self.heads):
            scores = torch._int_mm(query[head], key[head].t().contiguous())
            scores = scores.float() * (ACTIVATION_SCALE ** 2 / math.sqrt(HEAD_DIM))
            probability = torch.softmax(scores.masked_fill(self.mask, float("-inf")), dim=-1)
            # Row scaling preserves small probabilities at sequence length 1024.
            probability_scale = probability.amax(dim=-1, keepdim=True) / 127.0
            quantized = (probability / probability_scale).round().clamp(0, 127).to(torch.int8)
            output = torch._int_mm(quantized, value[head])
            outputs.append(output.float() * probability_scale * ACTIVATION_SCALE)
        return torch.stack(outputs)


def rotate_half(activation):
    first, second = activation.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Int8Linear(HIDDEN, HIDDEN)
        self.k_proj = Int8Linear(HIDDEN, HIDDEN)
        self.v_proj = Int8Linear(HIDDEN, HIDDEN)
        self.o_proj = Int8Linear(HIDDEN, HIDDEN)
        self.core = AttentionKernel(HEADS)
        frequencies = torch.outer(
            torch.arange(SEQUENCE, dtype=torch.float32),
            1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)),
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", angles.cos())
        self.register_buffer("sine", angles.sin())

    def forward(self, activation):
        quantized = quantize_activation(activation)
        query = self.q_proj(quantized).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1)
        key = self.k_proj(quantized).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1)
        value = self.v_proj(quantized).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1)
        query = query * self.cosine + rotate_half(query) * self.sine
        key = key * self.cosine + rotate_half(key) * self.sine
        context = self.core(
            quantize_activation(query).contiguous(),
            quantize_activation(key).contiguous(),
            quantize_activation(value).contiguous(),
        ).transpose(0, 1).reshape(1, SEQUENCE, HIDDEN)
        return self.o_proj(quantize_activation(context))
