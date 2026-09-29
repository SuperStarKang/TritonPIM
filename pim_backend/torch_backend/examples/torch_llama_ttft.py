"""Llama-2-7b-hf FP32 prefill through the first greedy token.

Fixed experiment: batch 1, 1024 prompt tokens, 32 layers, 2048 units,
16 UPMEM tasklets. Compile before the request; allocate once, replace weights
at each layer's runtime FX node, and free immediately after the first token.
CPU embedding/norm/lm_head and CPU-resident KV caches are included.
Reports and FX graphs are written to llama_ttft_report in the working directory.
"""

import argparse
from collections import Counter
import json
import logging
import operator
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import fx
from transformers import AutoModelForCausalLM, AutoTokenizer

from autotuner import PIMConfig
from torch_backend import (
    LoadLayerWeights, PIMSession, PIMTorchConfig, ResidentGemmConfig, RunLayer,
    compile_layer, target_spec,
)
from torch_backend.fx_analysis import analyze_fx_gemms
from torch_resident_llama_layer import LlamaFirstLayer, write_fx_graph


class LlamaPrefillLayer(LlamaFirstLayer):
    def forward(self, activation):
        return self.prefill(activation)


def layer_configs(gemms, spec):
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
    report_dir = Path("llama_ttft_report")
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
            print(f"Compiling layer {index}/31 without device allocation ...", flush=True)
            exported = torch.export.export(
                LlamaPrefillLayer(layer, model.config, cosine, sine, causal_mask).eval(),
                (example,),
            )
            gemms = analyze_fx_gemms(exported)
            write_fx_graph(exported.graph_module, report_dir / f"layer_{index}_export", {
                gemm.node: f"mnk={gemm.mnk} B_weight={gemm.right.weight_target}"
                for gemm in gemms
            })
            plan = compile_layer(exported, (example,), configs=layer_configs(gemms, spec), index=index)
            plans.append(plan)
            write_fx_graph(plan.graph, report_dir / f"layer_{index}_partitions", {})
            print(
                f"layer={index} GEMMs={len(plan.runtime.kernels)} "
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
            "status": "execution_complete_validation_pending",
            "checkpoint": str(args.checkpoint.resolve()),
            "scope": "single request prefill to first greedy token ID; no decode steps",
            "dtype": "float32", "batch_size": 1, "sequence_length": 1024,
            "layers": 32, "num_units": 2048, "tasklets": 16,
            "weight_policy": "one compiled layer resident at a time; all units shared",
            "kv_cache_location": "cpu",
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
        print(f"Execution report saved to {report_dir.resolve()}; validating against HF CPU ...", flush=True)

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

        reference_start = time.perf_counter_ns()
        reference = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        reference_logits = reference.logits[:, -1, :]
        report["reference_ms"] = (time.perf_counter_ns() - reference_start) / 1e6
        report["reference_token_id"] = reference_logits.argmax(dim=-1).item()
        report["logits_max_abs_error"] = (logits - reference_logits).abs().max().item()
        report["kv_max_abs_error_by_layer"] = [{
            "layer": index,
            "key": (key - reference.past_key_values.layers[index].keys).abs().max().item(),
            "value": (value - reference.past_key_values.layers[index].values).abs().max().item(),
        } for index, (key, value) in enumerate(cache)]
        (report_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        torch.testing.assert_close(logits, reference_logits, rtol=1e-4, atol=1e-4)
        assert token_id == report["reference_token_id"]
        assert len(reference.past_key_values.layers) == len(cache) == 32
        for index, (key, value) in enumerate(cache):
            torch.testing.assert_close(key, reference.past_key_values.layers[index].keys,
                                       rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(value, reference.past_key_values.layers[index].values,
                                       rtol=1e-4, atol=1e-4)
        report["status"] = "PASS"
        (report_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("TTFT correctness=PASS resources=1/1 launches=2272 run_weight_h2d=0", flush=True)


if __name__ == "__main__":
    main()
