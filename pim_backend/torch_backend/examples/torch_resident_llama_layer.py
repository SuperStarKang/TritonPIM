"""Validate the first Llama-2-7b-hf decoder layer with resident FP32 weights.

Batch size 1, sequence length 1024, 2048 DPUs and 16 tasklets.
Pass the local snapshot directory containing config.json as --checkpoint.
The checkpoint and tokenizer must already be downloaded.
Reported times cover the first decoder layer, not model TTFT.
"""

import argparse
from collections import Counter
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from autotuner import ConstraintEngine, PIMConfig, PIMProblem, pim_hierarchy
from torch_backend import (
    PIMTorchConfig,
    ResidentGemmConfig,
    prepare_block,
    target_spec,
)
from torch_backend.fx_analysis import analyze_fx_gemms


class LlamaFirstLayer(torch.nn.Module):
    def __init__(self, layer, config, cosine, sine, causal_mask):
        super().__init__()
        self.self_attn = layer.self_attn
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

        self.hidden_size = config.hidden_size
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group_size = self.query_heads // self.kv_heads

        self.register_buffer("cosine", cosine.contiguous())
        self.register_buffer("sine", sine.contiguous())
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, activation):
        sequence = activation.shape[1]
        normalized = self.input_layernorm(activation)

        query = self.self_attn.q_proj(normalized).reshape(
            1, sequence, self.query_heads, self.head_dim
        ).transpose(1, 2)
        key = self.self_attn.k_proj(normalized).reshape(
            1, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.self_attn.v_proj(normalized).reshape(
            1, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        query, key = apply_rotary_pos_emb(
            query, key, self.cosine, self.sine
        )

        # 같은 KV head를 공유하는 query head만 GEMM의 M축으로 묶는다.
        query = query.reshape(
            self.kv_heads, self.group_size * sequence, self.head_dim
        ).contiguous()
        key = key[0].contiguous()
        value = value[0].contiguous()

        outputs = []
        for group in range(self.kv_heads):
            scores = torch.mm(query[group], key[group].t())
            scores = scores * self.self_attn.scaling
            scores = scores.reshape(self.group_size, sequence, sequence)
            probability = torch.softmax(
                scores + self.causal_mask,
                dim=-1,
                dtype=torch.float32,
            ).reshape(self.group_size * sequence, sequence)
            outputs.append(torch.mm(probability, value[group]))

        context = torch.cat(outputs, dim=0).reshape(
            self.query_heads, sequence, self.head_dim
        ).transpose(0, 1).reshape(
            1, sequence, self.hidden_size
        ).contiguous()

        hidden = activation + self.self_attn.o_proj(context)
        normalized = self.post_attention_layernorm(hidden)
        return hidden + self.mlp(normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.checkpoint),
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(args.checkpoint),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)

        if model.config.model_type != "llama" or (
            model.config.num_hidden_layers,
            model.config.hidden_size,
            model.config.intermediate_size,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            model.config.head_dim,
        ) != (32, 4096, 11008, 32, 32, 128):
            raise ValueError("This example requires Llama-2-7b-hf.")

        input_ids = tokenizer(
            args.prompt_file.read_text(encoding="utf-8"),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )["input_ids"]
        if tuple(input_ids.shape) != (1, 1024):
            raise ValueError(
                "The prompt must contain at least 1024 tokens "
                "including tokenizer-added special tokens."
            )

        activation = model.model.embed_tokens(input_ids).contiguous()
        position_ids = torch.arange(1024).unsqueeze(0)
        cosine, sine = model.model.rotary_emb(
            activation, position_ids=position_ids
        )
        causal_mask = torch.full(
            (1024, 1024),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        ).triu(1)

        layer = model.model.layers[0]
        adapter = LlamaFirstLayer(
            layer, model.config, cosine, sine, causal_mask
        ).eval()

        reference = layer(
            activation,
            attention_mask=causal_mask[None, None, :, :],
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cosine, sine),
        )
        adapter_output = adapter(activation)
        torch.testing.assert_close(
            adapter_output, reference, rtol=1e-4, atol=1e-4
        )
        print(
            "HF_vs_adapter correctness=PASS "
            f"max_abs_error={(adapter_output - reference).abs().max().item():.8g}",
            flush=True,
        )

        exported = torch.export.export(adapter, (activation,))
        gemms = analyze_fx_gemms(exported)

        assert Counter(gemm.mnk for gemm in gemms) == Counter({
            (1024, 4096, 4096): 4,
            (1024, 1024, 128): 32,
            (1024, 128, 1024): 32,
            (1024, 11008, 4096): 2,
            (1024, 4096, 11008): 1,
        })
        assert {
            gemm.right.weight_target: gemm.mnk
            for gemm in gemms
            if gemm.right.weight_target is not None
        } == {
            "self_attn.q_proj.weight": (1024, 4096, 4096),
            "self_attn.k_proj.weight": (1024, 4096, 4096),
            "self_attn.v_proj.weight": (1024, 4096, 4096),
            "self_attn.o_proj.weight": (1024, 4096, 4096),
            "mlp.gate_proj.weight": (1024, 11008, 4096),
            "mlp.up_proj.weight": (1024, 11008, 4096),
            "mlp.down_proj.weight": (1024, 4096, 11008),
        }

        # 각 값은 (block_n, p_m, p_n)이며 p_k는 모두 1이다.
        placements = {
            (1024, 4096, 4096): (16, 32, 64),
            (1024, 1024, 128): (16, 32, 64),
            (1024, 128, 1024): (4, 64, 32),
            (1024, 11008, 4096): (16, 32, 64),
            (1024, 4096, 11008): (16, 32, 64),
        }
        target = target_spec("upmem", num_units=2048, nr_tasklets=16)
        configs = {}
        weight_sizes = {}
        input_scratch_bytes = 0
        output_scratch_bytes = 0

        for gemm in gemms:
            assert gemm.left.weight_target is None
            assert gemm.op == (
                "aten.mm.default"
                if gemm.right.weight_target is None
                else "aten.linear.default"
            )

            m, n, k = gemm.mnk
            block_n, p_m, p_n = placements[gemm.mnk]
            # Gate/up의 N=11008을 DPU별 N 타일에 맞춰 11264로 확장한다.
            padded_n = ((n + block_n * p_n - 1) // (block_n * p_n)) * (block_n * p_n)
            assert p_m * p_n == 2048
            assert m % (16 * p_m) == 0
            assert padded_n % (block_n * p_n) == 0
            assert k % 32 == 0

            hierarchy = PIMConfig(
                block_m=16, block_n=block_n, block_k=32,
                group_m=1, group_n=1, group_k=1,
                tile_m=m // (16 * p_m),
                tile_n=padded_n // (block_n * p_n),
                tile_k=k // 32,
                p_m=p_m, p_n=p_n, p_k=1,
            )
            problem = PIMProblem(
                m, padded_n, k, input_dtype="float32", output_dtype="float32"
            )
            reason = ConstraintEngine(problem, target).check(hierarchy)
            if reason is not None:
                raise ValueError(f"{gemm.node}: {reason}")
            assert pim_hierarchy(problem, hierarchy).rounds == (1, 1, 1)

            configs[gemm.node] = ResidentGemmConfig(
                torch_config=PIMTorchConfig(spec=target, config=hierarchy),
                orientation="normal",
                padded_shape=(m, padded_n, k),
            )
            extent_m, extent_n, extent_k = hierarchy.execution_extent
            if gemm.right.weight_target is not None:
                weight_sizes[gemm.node] = extent_k * extent_n * 4

            input_scratch_bytes = max(
                input_scratch_bytes,
                extent_m * extent_k * 4
                + (
                    extent_k * extent_n * 4
                    if gemm.right.weight_target is None else 0
                ),
            )
            output_scratch_bytes = max(
                output_scratch_bytes, extent_m * extent_n * 4
            )
            print(
                f"{gemm.node}: {gemm.op} mnk={gemm.mnk} "
                f"padded={(m, padded_n, k)} "
                f"B_weight={gemm.right.weight_target} "
                f"partitions=({p_m},{p_n},1)",
                flush=True,
            )

        assert len(weight_sizes) == 7
        weight_bytes_per_dpu = sum(weight_sizes.values())
        mram_bytes_per_dpu = (
            weight_bytes_per_dpu
            + input_scratch_bytes
            + output_scratch_bytes
        )
        if mram_bytes_per_dpu > target.common_profile.bank_capacity_limit_bytes:
            raise ValueError(
                f"MRAM capacity exceeded: {mram_bytes_per_dpu} bytes per DPU"
            )

        print(
            f"FX analysis=PASS gemms={len(gemms)} weights={len(weight_sizes)}\n"
            f"preflight MRAM_bytes_per_DPU={mram_bytes_per_dpu} "
            f"captured_weight_host_bytes={2048 * weight_bytes_per_dpu}",
            flush=True,
        )

        with prepare_block(
            exported,
            (activation,),
            configs=configs,
            dpu_count=2048,
        ) as block:
            prepared = block.stats()
            assert prepared["weight_bytes"] == 2048 * weight_bytes_per_dpu
            assert prepared["launches"] == 0
            assert {
                region["kernel"]: region["bytes"]
                for region in block.layout
                if region["role"] == "weight"
            } == weight_sizes
            assert all(
                region["window"] == 0
                for region in block.layout
                if region["role"] == "weight"
            )

            print(
                f"prepare_ms={block.prepare_ms:.3f} "
                f"prepare_weight_bytes={prepared['weight_bytes']}",
                flush=True,
            )

            for step in range(3):
                start = time.perf_counter_ns()
                output = block.run(activation)
                wall_ms = (time.perf_counter_ns() - start) / 1e6

                torch.testing.assert_close(
                    output, reference, rtol=1e-4, atol=1e-4
                )
                assert block.last_transfers["weight_bytes"] == 0
                assert block.last_transfers["activation_bytes"] > 0
                assert block.last_transfers["output_bytes"] > 0
                assert block.last_transfers["launches"] == 71
                assert block.stats()["weight_bytes"] == prepared["weight_bytes"]

                phase = "warmup" if step == 0 else f"sample={step - 1}"
                print(
                    f"{phase} correctness=PASS wall_ms={wall_ms:.3f} "
                    f"max_abs_error={(output - reference).abs().max().item():.8g} "
                    f"weight_h2d={block.last_transfers['weight_bytes']} "
                    f"activation_h2d={block.last_transfers['activation_bytes']} "
                    f"activation_hits={block.last_transfers['activation_hits']} "
                    f"launches={block.last_transfers['launches']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
