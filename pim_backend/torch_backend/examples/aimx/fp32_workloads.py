"""Synthetic batch-one FP32 prefill workloads with Llama-2-7B shapes for AiMX.

AiMX accepts only bfloat16/float16/float32 host buffers, so every projection is an
FP32 torch.mm against a [K, N] weight buffer. Only the GEMMs go to PIM.
"""

import math

import torch


SEQUENCE = 1024
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
INTERMEDIATE = 11008
# Gate/Up는 UPMEM baseline과 같게 N=11264(=22*512)로 계산하고 11008열만 쓴다.
PADDED_INTERMEDIATE = 11264


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, padded_features=None):
        super().__init__()
        weight = torch.randn(in_features, out_features) / math.sqrt(in_features)
        padding = (padded_features or out_features) - out_features
        self.register_buffer("weight", torch.nn.functional.pad(weight, (0, padding)))
        self.out_features = out_features

    def forward(self, activation):
        output = torch.mm(activation.reshape(-1, activation.shape[-1]), self.weight)
        if output.shape[-1] != self.out_features:
            output = output[:, :self.out_features].contiguous()
        return output.reshape(*activation.shape[:-1], self.out_features)


def rotate_half(activation):
    first, second = activation.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Linear(HIDDEN, HIDDEN)
        self.k_proj = Linear(HIDDEN, HIDDEN)
        self.v_proj = Linear(HIDDEN, HIDDEN)
        self.o_proj = Linear(HIDDEN, HIDDEN)
        frequencies = torch.outer(
            torch.arange(SEQUENCE, dtype=torch.float32),
            1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)),
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", angles.cos())
        self.register_buffer("sine", angles.sin())
        self.register_buffer("mask", torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool).triu(1))

    def project(self, activation):
        """Q/K/V projections and RoPE; each result is [HEADS, SEQUENCE, HEAD_DIM]."""
        query = self.q_proj(activation).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1)
        key = self.k_proj(activation).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1)
        value = self.v_proj(activation).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1).contiguous()
        query = (query * self.cosine + rotate_half(query) * self.sine).contiguous()
        key = (key * self.cosine + rotate_half(key) * self.sine).contiguous()
        return query, key, value

    def forward(self, activation):
        query, key, value = self.project(activation)
        outputs = []
        for head in range(HEADS):
            scores = torch.mm(query[head], key[head].t().contiguous()) / math.sqrt(HEAD_DIM)
            probability = torch.softmax(scores.masked_fill(self.mask, float("-inf")), dim=-1)
            outputs.append(torch.mm(probability, value[head]))
        context = torch.stack(outputs).transpose(0, 1).reshape(1, SEQUENCE, HIDDEN)
        return self.o_proj(context)


def block_diagonal(blocks):
    """Block i at rows/columns i of the result, zeros elsewhere (pad + cat)."""
    columns = sum(block.shape[1] for block in blocks)
    rows = []
    left = 0
    for block in blocks:
        rows.append(torch.nn.functional.pad(block, (left, columns - left - block.shape[1])))
        left += block.shape[1]
    return torch.cat(rows, dim=0)


class PackedAttentionLayer(AttentionLayer):
    """AttentionLayer with qk_heads / pv_heads heads per block-diagonal GEMM.

    QK^T: [Q_h | Q_h+1 ...] (SEQ x H*D)   @ blockdiag(K_h^T ...) (H*D x H*SEQ)   = [S_h | S_h+1 ...]
    PV:   [P_h | P_h+1 ...] (SEQ x H*SEQ) @ blockdiag(V_h ...)   (H*SEQ x H*D)   = [O_h | O_h+1 ...]
    Weights and the input order match AttentionLayer, so the same seed gives the same layer.
    """

    def __init__(self, qk_heads, pv_heads):
        super().__init__()
        if HEADS % qk_heads or HEADS % pv_heads:
            raise ValueError(f"heads per GEMM must divide {HEADS}: qk {qk_heads}, pv {pv_heads}")
        self.qk_heads = qk_heads
        self.pv_heads = pv_heads

    def forward(self, activation):
        query, key, value = self.project(activation)
        scores = []
        for first in range(0, HEADS, self.qk_heads):
            heads = range(first, first + self.qk_heads)
            left = torch.cat([query[head] for head in heads], dim=1)
            right = block_diagonal([key[head].t() for head in heads])
            scores.append(torch.mm(left, right))
        # 열 순서가 head, key 위치이므로 [SEQ, HEADS, SEQ]로 보고 head마다 softmax한다.
        scores = torch.cat(scores, dim=1).reshape(SEQUENCE, HEADS, SEQUENCE) / math.sqrt(HEAD_DIM)
        probability = torch.softmax(scores.masked_fill(self.mask[:, None, :], float("-inf")), dim=-1)
        probability = probability.reshape(SEQUENCE, HEADS * SEQUENCE)
        outputs = []
        for first in range(0, HEADS, self.pv_heads):
            heads = range(first, first + self.pv_heads)
            left = probability[:, first * SEQUENCE:(first + self.pv_heads) * SEQUENCE]
            right = block_diagonal([value[head] for head in heads])
            outputs.append(torch.mm(left, right))
        # [O_h | O_h+1 ...]의 열 순서가 곧 hidden 열 순서 head * HEAD_DIM + d다.
        context = torch.cat(outputs, dim=1).reshape(1, SEQUENCE, HIDDEN)
        return self.o_proj(context)


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
        self.gate_proj = Linear(HIDDEN, INTERMEDIATE, PADDED_INTERMEDIATE)
        self.up_proj = Linear(HIDDEN, INTERMEDIATE, PADDED_INTERMEDIATE)
        self.down_proj = Linear(INTERMEDIATE, HIDDEN)

    def forward(self, activation):
        hidden = activation + self.attention(self.input_norm(activation))
        normalized = self.post_attention_norm(hidden)
        gated = torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return hidden + self.down_proj(gated)
