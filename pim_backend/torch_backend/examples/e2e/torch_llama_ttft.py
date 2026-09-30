"""Llama-2-7b-hf W8A8 layer GEMMs through the first greedy token.

Fixed experiment: batch 1, 1024 prompt tokens, 32 layers, 2048 units,
16 UPMEM tasklets. Compile before the request; allocate once, replace weights
at each layer's runtime FX node, and free immediately after the first token.
Projection weights are quantized before the request. CPU operations, including
embedding/norm/last-token lm_head and CPU-resident KV caches, remain FP32.
No calibration or CPU reference accuracy run is performed.
Reports and FX graphs are written to llama_int8_ttft_report beside this script.
"""

import argparse
from collections import Counter
import json
import logging
import operator
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import fx
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotuner import PIMConfig
from torch_backend import (
    LoadLayerWeights, PIMSession, PIMTorchConfig, ResidentGemmConfig, RunLayer,
    compile_layer, target_spec,
)
from torch_backend.fx_analysis import analyze_fx_gemms
from attention.int8_workloads import (
    ACTIVATION_SCALE, AttentionKernel, quantize_activation, rotate_half,
)
from fx_utils import write_fx_graph


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
    def __init__(self, layer, config, cosine, sine, causal_mask):
        super().__init__()
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
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.core = AttentionKernel(self.heads)
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
            sequence, self.heads, self.head_dim,
        ).transpose(0, 1)
        value = self.v_proj(normalized).reshape(
            sequence, self.heads, self.head_dim,
        ).transpose(0, 1)
        query = query * self.cosine + rotate_half(query) * self.sine
        key = key * self.cosine + rotate_half(key) * self.sine
        key_cache = key.unsqueeze(0).contiguous()
        value_cache = value.unsqueeze(0).contiguous()

        context = self.core(
            quantize_activation(query).contiguous(),
            quantize_activation(key).contiguous(),
            quantize_activation(value).contiguous(),
        ).transpose(0, 1).reshape(1, sequence, self.hidden_size)
        hidden = activation + self.o_proj(quantize_activation(context))
        normalized = quantize_activation(self.post_attention_layernorm(hidden))
        gated = torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        output = hidden + self.down_proj(quantize_activation(gated))
        return output, key_cache, value_cache


def layer_configs(gemms, spec):
    if any(
        gemm.op != "aten._int_mm.default"
        or gemm.left.dtype != torch.int8
        or gemm.right.dtype != torch.int8
        or gemm.output_dtype != torch.int32
        for gemm in gemms
    ):
        raise ValueError("all layer GEMMs must use int8 x int8 -> int32")
    if Counter(gemm.mnk for gemm in gemms) != Counter({
        (1024, 4096, 4096): 4,
        (1024, 1024, 128): 32,
        (1024, 128, 1024): 32,
        (1024, 11008, 4096): 2,
        (1024, 4096, 11008): 1,
    }):
        raise ValueError("expected 7 projection and 64 attention GEMMs per Llama layer")
    if sum(gemm.right.weight_target is not None for gemm in gemms) != 7:
        raise ValueError("FX analysis must identify seven layer weights")
    placements = {
        (1024, 4096, 4096): (16, 32, 64),
        (1024, 1024, 128): (16, 32, 64),
        (1024, 128, 1024): (4, 64, 32),
        (1024, 11008, 4096): (16, 32, 64),
        (1024, 4096, 11008): (16, 32, 64),
    }
    configs = {}
    for gemm in gemms:
        m, n, k = gemm.mnk
        block_n, p_m, p_n = placements[gemm.mnk]
        padded_n = ((n + block_n * p_n - 1) // (block_n * p_n)) * (block_n * p_n)
        configs[gemm.node] = ResidentGemmConfig(
            PIMTorchConfig(spec=spec, config=PIMConfig(
                block_m=16, block_n=block_n, block_k=32,
                group_m=1, group_n=1, group_k=1,
                tile_m=m // (16 * p_m), tile_n=padded_n // (block_n * p_n),
                tile_k=k // 32, p_m=p_m, p_n=p_n, p_k=1,
            )),
            orientation="normal", padded_shape=(m, padded_n, k),
        )
    return configs


def request_graph(model, plans):
    root = torch.nn.Module()
    root.add_module("embedding", model.model.embed_tokens)
    root.add_module("norm", model.model.norm)
    root.add_module("lm_head", model.lm_head)
    graph = fx.Graph()
    session = graph.placeholder("session")
    input_ids = graph.placeholder("input_ids")
    done = graph.call_method("start", (session,))
    hidden = graph.call_module("embedding", (input_ids,))
    cache = []
    for plan in plans:
        load_name = f"load_layer_{plan.index}_weights"
        run_name = f"run_layer_{plan.index}"
        root.add_module(load_name, LoadLayerWeights(plan))
        root.add_module(run_name, RunLayer(plan))
        ready = graph.call_module(load_name, (session, done))
        execution = graph.call_module(run_name, (session, hidden, ready))
        outputs = graph.call_function(operator.getitem, (execution, 0))
        done = graph.call_function(operator.getitem, (execution, 1))
        hidden = graph.call_function(operator.getitem, (outputs, 0))
        cache.append((graph.call_function(operator.getitem, (outputs, 1)),
                      graph.call_function(operator.getitem, (outputs, 2))))
    last_hidden = graph.call_function(operator.getitem, (hidden, (slice(None), -1, slice(None))))
    normalized = graph.call_module("norm", (last_hidden,))
    logits = graph.call_module("lm_head", (normalized,))
    first_token = graph.call_function(torch.argmax, (logits,), {"dim": -1})
    graph.output((first_token, logits, tuple(cache), done))
    return fx.GraphModule(root, graph, "LlamaPrefillRequest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report_dir = Path(__file__).with_name("llama_int8_ttft_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")

    with torch.no_grad():
        preparation_start = time.perf_counter_ns()
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.checkpoint), local_files_only=True, dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)
        if model.config.model_type != "llama" or (
            model.config.num_hidden_layers, model.config.hidden_size,
            model.config.intermediate_size, model.config.num_attention_heads,
            model.config.num_key_value_heads, model.config.head_dim,
        ) != (32, 4096, 11008, 32, 32, 128):
            raise ValueError("this experiment requires Llama-2-7b-hf")

        example = torch.zeros((1, 1024, 4096), dtype=torch.float32)
        cosine, sine = model.model.rotary_emb(
            example, position_ids=torch.arange(1024).unsqueeze(0),
        )
        causal_mask = torch.full(
            (1024, 1024), torch.finfo(torch.float32).min, dtype=torch.float32,
        ).triu(1)
        spec = target_spec("upmem", num_units=2048, nr_tasklets=16)
        plans = []
        for index, layer in enumerate(model.model.layers):
            print(f"Compiling INT8 layer {index}/31 without device allocation ...", flush=True)
            exported = torch.export.export(
                LlamaPrefillLayer(layer, model.config, cosine, sine, causal_mask).eval(),
                (example,),
            )
            gemms = analyze_fx_gemms(exported)
            write_fx_graph(exported.graph_module, report_dir / f"layer_{index}_export", {
                gemm.node: f"{gemm.op} mnk={gemm.mnk} B_weight={gemm.right.weight_target}"
                for gemm in gemms
            })
            plan = compile_layer(exported, (example,), configs=layer_configs(gemms, spec), index=index)
            plans.append(plan)
            write_fx_graph(plan.graph, report_dir / f"layer_{index}_partitions", {})
            print(
                f"layer={index} GEMMs={len(plan.runtime.kernels)} "
                f"gemm_dtype=int8->int32 "
                f"MRAM_bytes_per_unit={plan.backend_plan.bytes_per_unit} "
                f"compile_ms={plan.compile_ms:.3f}", flush=True,
            )
        graph = request_graph(model, plans)
        write_fx_graph(graph, report_dir / "request_fx", {
            f"load_layer_{index}_weights": "LoadLayerWeights: replace current layer weights"
            for index in range(32)
        })
        preparation_ms = (time.perf_counter_ns() - preparation_start) / 1e6

        print("Executing one request: tokenize -> allocate -> 32 layers -> first token -> free", flush=True)
        # Reject guard failures that would trigger recompilation inside TTFT.
        with torch._dynamo.config.patch(error_on_recompile=True):
            request_start = time.perf_counter_ns()
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024,
            )["input_ids"]
            if tuple(input_ids.shape) != (1, 1024):
                raise ValueError("the prompt must contain at least 1024 tokens including special tokens")
            tokenization_ms = (time.perf_counter_ns() - request_start) / 1e6
            with PIMSession(spec) as session:
                model_start = time.perf_counter_ns()
                token, logits, cache, done = graph(session, input_ids)
                token_id = token.item()
                token_ready = time.perf_counter_ns()
                if done is not session.token or not done.completed:
                    raise RuntimeError("the final layer did not return its completion token")
            request_ms = (time.perf_counter_ns() - request_start) / 1e6

        report = {
            "status": "execution_complete",
            "checkpoint": str(args.checkpoint.resolve()),
            "scope": "single request prefill to first greedy token ID; no decode steps",
            "dtype": "mixed", "batch_size": 1, "sequence_length": 1024,
            "gemm_dtype": "int8 x int8 -> int32", "cpu_dtype": "float32",
            "lm_head_dtype": "float32", "lm_head_scope": "last_token_only",
            "accuracy_checked": False,
            "quantization": {
                "weight": "per-tensor absmax/128, round, clip [-127, 127]",
                "activation_scale": ACTIVATION_SCALE,
                "probability": "per-row max/127",
                "calibration": "none",
            },
            "layers": 32, "num_units": 2048, "tasklets": 16,
            "weight_policy": "one compiled layer resident at a time; all units shared",
            "activation_transfer_policy": "always",
            "kv_cache_location": "cpu",
            "kv_cache_dtype": "float32",
            "kv_cache_bytes": sum(tensor.numel() * tensor.element_size()
                                  for pair in cache for tensor in pair),
            "preparation_ms": preparation_ms,
            "compile_ms_by_layer": [plan.compile_ms for plan in plans],
            "tokenization_ms": tokenization_ms,
            "resource_open_ms": session.open_ms,
            "model_to_token_ms": (token_ready - model_start) / 1e6,
            "ttft_ms": (token_ready - request_start) / 1e6,
            "resource_close_ms": session.close_ms,
            "request_ms_including_close": request_ms,
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id]),
            "transfers": session.stats(),
            "layer_records": session.records,
            "memory_by_layer": [{
                "layer": plan.index,
                "bytes_per_unit": plan.backend_plan.bytes_per_unit,
                "weight_bytes_per_unit": plan.backend_plan.weight_bytes_per_unit,
            } for plan in plans],
        }
        (report_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"TTFT_ms={report['ttft_ms']:.3f} first_token_id={token_id}", flush=True)
        print(f"Execution report saved to {report_dir.resolve()}", flush=True)

        assert session.stats()["resource_opens"] == session.stats()["resource_closes"] == 1
        assert session.stats()["launches"] == 32 * 71
        assert len(session.records) == 32
        assert report["kv_cache_bytes"] == 1024 ** 3
        for index, record in enumerate(session.records):
            assert record["layer"] == index
            assert record["load_transfers"]["weight_bytes"] == (
                plans[index].backend_plan.weight_bytes_per_unit * 2048
            )
            assert record["load_transfers"]["launches"] == 0
            assert record["run_transfers"]["weight_bytes"] == 0
            assert record["run_transfers"]["launches"] == 71
            assert record["run_transfers"]["activation_hits"] == 0
        print("TTFT execution complete: resources=1/1 launches=2272 "
              "run_weight_h2d=0 activation_hits=0 gemm_dtype=int8->int32", flush=True)


if __name__ == "__main__":
    main()
