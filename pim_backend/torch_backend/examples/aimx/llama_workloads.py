"""Llama-2-7b-hf prefill decoder layer on AiMX, built from a Hugging Face checkpoint layer.

The seven projections and the QK^T/PV GEMMs are BF16 x BF16 -> BF16, which is what AiMX
stores and computes; BF16 host weights also keep the 32 layers (13 GB) inside host memory.
RMSNorm, RoPE, softmax, SwiGLU and residuals stay FP32 on the CPU. The attention is written
with one QK^T and one PV GEMM per head; the compiler packs them (torch_backend/fx_packing.py).
qk_heads / pv_heads > 1 pack heads by hand into one block-diagonal GEMM instead.
"""

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEQUENCE = 1024
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
INTERMEDIATE = 11008
# Gate/Up는 UPMEM baseline과 같게 N=11264(=22*512)로 계산하고 11008열만 쓴다.
PADDED_INTERMEDIATE = 11264
QK_HEADS = 1
PV_HEADS = 1


def rotate_half(activation):
    first, second = activation.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def block_diagonal(blocks):
    """Block i at rows/columns i of the result, zeros elsewhere (pad + cat)."""
    columns = sum(block.shape[1] for block in blocks)
    rows = []
    left = 0
    for block in blocks:
        rows.append(torch.nn.functional.pad(block, (left, columns - left - block.shape[1])))
        left += block.shape[1]
    return torch.cat(rows, dim=0)


def load_llama(checkpoint):
    """Tokenizer and BF16 model; FP32 would need 27 GB of host memory."""
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cpu().eval().requires_grad_(False)
    config = model.config
    if config.model_type != "llama" or (
        config.num_hidden_layers, config.hidden_size, config.intermediate_size,
        config.num_attention_heads, config.num_key_value_heads, config.head_dim,
    ) != (32, HIDDEN, INTERMEDIATE, HEADS, HEADS, HEAD_DIM):
        raise ValueError("this experiment requires Llama-2-7b-hf")
    return tokenizer, model


def prompt_ids(tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SEQUENCE)["input_ids"]
    if tuple(input_ids.shape) != (1, SEQUENCE):
        raise ValueError(f"the prompt must contain at least {SEQUENCE} tokens including special tokens")
    return input_ids


def position_embeddings(model):
    """RoPE cos/sin of positions 0..SEQUENCE-1, each [1, SEQUENCE, HEAD_DIM] FP32."""
    return model.model.rotary_emb(torch.zeros(1, SEQUENCE, HIDDEN),
                                  position_ids=torch.arange(SEQUENCE).unsqueeze(0))


def reference_layer(layer, hidden, cosine, sine):
    """The Hugging Face decoder layer in FP32 (eager attention, additive causal mask)."""
    mask = torch.full((SEQUENCE, SEQUENCE), torch.finfo(torch.float32).min).triu(1)
    return copy.deepcopy(layer).float()(
        hidden, attention_mask=mask[None, None], position_embeddings=(cosine, sine))


class CheckpointLinear(torch.nn.Module):
    """HF Linear weight [out, in] stored as a BF16 [in, padded out] GEMM operand."""

    def __init__(self, linear, padded_features=None):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("this experiment requires bias-free projections")
        weight = linear.weight.detach().t().to(torch.bfloat16).contiguous()
        padding = (padded_features or linear.out_features) - linear.out_features
        self.register_buffer("weight", torch.nn.functional.pad(weight, (0, padding)))
        self.out_features = linear.out_features

    def forward(self, activation):
        output = torch.mm(activation.reshape(-1, activation.shape[-1]).bfloat16(), self.weight).float()
        if output.shape[-1] != self.out_features:
            output = output[:, :self.out_features].contiguous()
        return output.reshape(*activation.shape[:-1], self.out_features)


class CheckpointRMSNorm(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.register_buffer("weight", norm.weight.detach().float())
        self.eps = norm.variance_epsilon

    def forward(self, activation):
        return self.weight * (activation * torch.rsqrt(activation.square().mean(dim=-1, keepdim=True) + self.eps))


class LlamaAimxLayer(torch.nn.Module):
    def __init__(self, layer, cosine, sine, qk_heads=QK_HEADS, pv_heads=PV_HEADS):
        super().__init__()
        if HEADS % qk_heads or HEADS % pv_heads:
            raise ValueError(f"heads per GEMM must divide {HEADS}: qk {qk_heads}, pv {pv_heads}")
        self.qk_heads = qk_heads
        self.pv_heads = pv_heads
        self.input_layernorm = CheckpointRMSNorm(layer.input_layernorm)
        self.post_attention_layernorm = CheckpointRMSNorm(layer.post_attention_layernorm)
        self.q_proj = CheckpointLinear(layer.self_attn.q_proj)
        self.k_proj = CheckpointLinear(layer.self_attn.k_proj)
        self.v_proj = CheckpointLinear(layer.self_attn.v_proj)
        self.o_proj = CheckpointLinear(layer.self_attn.o_proj)
        # Gate/Up는 N=11264로 계산하고 11008열만 쓴다.
        self.gate_proj = CheckpointLinear(layer.mlp.gate_proj, PADDED_INTERMEDIATE)
        self.up_proj = CheckpointLinear(layer.mlp.up_proj, PADDED_INTERMEDIATE)
        self.down_proj = CheckpointLinear(layer.mlp.down_proj)
        self.scaling = layer.self_attn.scaling
        self.register_buffer("cosine", cosine[0].float().contiguous())
        self.register_buffer("sine", sine[0].float().contiguous())
        self.register_buffer("mask", torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool).triu(1))

    def attention(self, normalized):
        query = self.q_proj(normalized).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1)
        key = self.k_proj(normalized).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1)
        value = self.v_proj(normalized).reshape(SEQUENCE, HEADS, HEAD_DIM).transpose(0, 1).bfloat16()
        query = (query * self.cosine + rotate_half(query) * self.sine).bfloat16()
        key = (key * self.cosine + rotate_half(key) * self.sine).bfloat16()
        scores = []
        for first in range(0, HEADS, self.qk_heads):
            heads = range(first, first + self.qk_heads)
            left = torch.cat([query[head] for head in heads], dim=1)
            right = block_diagonal([key[head].t() for head in heads])
            scores.append(torch.mm(left, right))
        # 열 순서가 head, key 위치이므로 [SEQ, HEADS, SEQ]로 보고 head마다 softmax한다.
        scores = torch.cat(scores, dim=1).float().reshape(SEQUENCE, HEADS, SEQUENCE) * self.scaling
        probability = torch.softmax(scores.masked_fill(self.mask[:, None, :], float("-inf")), dim=-1)
        probability = probability.reshape(SEQUENCE, HEADS * SEQUENCE).bfloat16()
        outputs = []
        for first in range(0, HEADS, self.pv_heads):
            heads = range(first, first + self.pv_heads)
            left = probability[:, first * SEQUENCE:(first + self.pv_heads) * SEQUENCE]
            right = block_diagonal([value[head] for head in heads])
            outputs.append(torch.mm(left, right))
        # [O_h | O_h+1 ...]의 열 순서가 곧 hidden 열 순서 head * HEAD_DIM + d다.
        context = torch.cat(outputs, dim=1).float().reshape(1, SEQUENCE, HIDDEN)
        return self.o_proj(context)

    def forward(self, hidden):
        hidden = hidden + self.attention(self.input_layernorm(hidden))
        normalized = self.post_attention_layernorm(hidden)
        gated = torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return hidden + self.down_proj(gated)
