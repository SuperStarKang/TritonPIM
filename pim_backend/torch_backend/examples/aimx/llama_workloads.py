"""Llama-family prefill decoder layer on AiMX, built from a Hugging Face checkpoint layer.

Sizes come from the checkpoint config, so MHA (Llama-2-7b) and GQA (Llama-3-8B) layers
share this code. The seven projections and the QK^T/PV GEMMs are BF16 x BF16 -> BF16,
which is what AiMX stores and computes; BF16 host weights keep the 32 layers inside
host memory. RMSNorm, RoPE, softmax, SwiGLU and residuals stay FP32 on the CPU.

The attention is written with one QK^T and one PV GEMM per query head; the compiler
packs them (torch_backend/fx_packing.py). kv selects how a query head reads its KV head:
  share   head h reads KV head h // group, so heads of one group read the same tensor
  repeat  the KV heads are copied to every query head first (Hugging Face repeat_kv)
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
QK_HEADS = 1
PV_HEADS = 1
KV_MODES = ("share", "repeat")


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
    """Tokenizer and BF16 Llama-family model; FP32 would not fit host memory."""
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cpu().eval().requires_grad_(False)
    config = model.config
    if config.model_type != "llama" or config.num_attention_heads % config.num_key_value_heads:
        raise ValueError(f"this experiment requires a Llama-family checkpoint with grouped KV heads; "
                         f"got {config.model_type} with {config.num_attention_heads} query and "
                         f"{config.num_key_value_heads} KV heads")
    return tokenizer, model


def prompt_ids(tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SEQUENCE)["input_ids"]
    if tuple(input_ids.shape) != (1, SEQUENCE):
        raise ValueError(f"the prompt must contain at least {SEQUENCE} tokens including special tokens")
    return input_ids


def position_embeddings(model):
    """RoPE cos/sin of positions 0..SEQUENCE-1, each [1, SEQUENCE, head_dim] FP32."""
    return model.model.rotary_emb(torch.zeros(1, SEQUENCE, model.config.hidden_size),
                                  position_ids=torch.arange(SEQUENCE).unsqueeze(0))


def reference_layer(layer, hidden, cosine, sine):
    """The Hugging Face decoder layer in FP32 (eager attention, additive causal mask)."""
    mask = torch.full((SEQUENCE, SEQUENCE), torch.finfo(torch.float32).min).triu(1)
    return copy.deepcopy(layer).float()(
        hidden, attention_mask=mask[None, None], position_embeddings=(cosine, sine))


class CheckpointLinear(torch.nn.Module):
    """HF Linear weight [out, in] BF16; the GEMM operand is its transpose view [in, out].

    The buffer is the checkpoint tensor itself (a BF16 checkpoint is not copied); the PIM
    stage reads the view through its strides. padded_features adds zero output rows, a
    padded copy, for the prefill layer's N in 512 columns.
    """

    def __init__(self, linear, padded_features=None):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("this experiment requires bias-free projections")
        weight = linear.weight.detach().to(torch.bfloat16)
        padding = (padded_features or linear.out_features) - linear.out_features
        self.register_buffer("weight", torch.nn.functional.pad(weight, (0, 0, 0, padding)) if padding else weight)
        self.out_features = linear.out_features

    def forward(self, activation):
        output = torch.mm(activation.reshape(-1, activation.shape[-1]).bfloat16(), self.weight.t()).float()
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
    def __init__(self, layer, cosine, sine, qk_heads=QK_HEADS, pv_heads=PV_HEADS, kv="share"):
        super().__init__()
        config = layer.self_attn.config
        self.hidden = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.heads % qk_heads or self.heads % pv_heads:
            raise ValueError(f"heads per GEMM must divide {self.heads}: qk {qk_heads}, pv {pv_heads}")
        if kv not in KV_MODES:
            raise ValueError(f"kv must be one of {KV_MODES}, got {kv!r}")
        self.qk_heads = qk_heads
        self.pv_heads = pv_heads
        self.kv = kv
        self.input_layernorm = CheckpointRMSNorm(layer.input_layernorm)
        self.post_attention_layernorm = CheckpointRMSNorm(layer.post_attention_layernorm)
        self.q_proj = CheckpointLinear(layer.self_attn.q_proj)
        self.k_proj = CheckpointLinear(layer.self_attn.k_proj)
        self.v_proj = CheckpointLinear(layer.self_attn.v_proj)
        self.o_proj = CheckpointLinear(layer.self_attn.o_proj)
        # Gate/Up는 N을 512 단위로 올려 계산하고 앞 열만 쓴다 (Llama-2 11008 → 11264, Llama-3 14336 그대로).
        padded = -(-config.intermediate_size // 512) * 512
        self.gate_proj = CheckpointLinear(layer.mlp.gate_proj, padded)
        self.up_proj = CheckpointLinear(layer.mlp.up_proj, padded)
        self.down_proj = CheckpointLinear(layer.mlp.down_proj)
        self.scaling = layer.self_attn.scaling
        self.register_buffer("cosine", cosine[0].float().contiguous())
        self.register_buffer("sine", sine[0].float().contiguous())
        self.register_buffer("mask", torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool).triu(1))

    def attention(self, normalized):
        group = self.heads // self.kv_heads
        query = self.q_proj(normalized).reshape(SEQUENCE, self.heads, self.head_dim).transpose(0, 1)
        key = self.k_proj(normalized).reshape(SEQUENCE, self.kv_heads, self.head_dim).transpose(0, 1)
        value = self.v_proj(normalized).reshape(SEQUENCE, self.kv_heads, self.head_dim).transpose(0, 1).bfloat16()
        query = (query * self.cosine + rotate_half(query) * self.sine).bfloat16()
        key = (key * self.cosine + rotate_half(key) * self.sine).bfloat16()
        if self.kv == "repeat":
            # Hugging Face repeat_kv와 같이 KV head를 query head 수만큼 복사한다. head마다 다른 tensor가 된다.
            key = key.repeat_interleave(group, dim=0)
            value = value.repeat_interleave(group, dim=0)
            group = 1
        scores = []
        for first in range(0, self.heads, self.qk_heads):
            heads = range(first, first + self.qk_heads)
            left = torch.cat([query[head] for head in heads], dim=1)
            right = block_diagonal([key[head // group].t() for head in heads])
            scores.append(torch.mm(left, right))
        # 열 순서가 head, key 위치이므로 [SEQ, heads, SEQ]로 보고 head마다 softmax한다.
        scores = torch.cat(scores, dim=1).float().reshape(SEQUENCE, self.heads, SEQUENCE) * self.scaling
        probability = torch.softmax(scores.masked_fill(self.mask[:, None, :], float("-inf")), dim=-1)
        probability = probability.reshape(SEQUENCE, self.heads * SEQUENCE).bfloat16()
        outputs = []
        for first in range(0, self.heads, self.pv_heads):
            heads = range(first, first + self.pv_heads)
            left = probability[:, first * SEQUENCE:(first + self.pv_heads) * SEQUENCE]
            right = block_diagonal([value[head // group] for head in heads])
            outputs.append(torch.mm(left, right))
        # [O_h | O_h+1 ...]의 열 순서가 곧 hidden 열 순서 head * head_dim + d다.
        context = torch.cat(outputs, dim=1).float().reshape(1, SEQUENCE, self.heads * self.head_dim)
        return self.o_proj(context)

    def forward(self, hidden):
        hidden = hidden + self.attention(self.input_layernorm(hidden))
        normalized = self.post_attention_layernorm(hidden)
        gated = torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        return hidden + self.down_proj(gated)
