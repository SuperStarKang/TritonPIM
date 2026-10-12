"""Checkpoint Llama-2-7B prefill through the session-first model runtime.

Run one request with 1024 tokens, 32 layers, 2048 DPUs and 16 tasklets.
Compare baseline and batched attention configs on the same model runtime.
TTFT includes per-layer weight upload, but excludes resource allocation and
model preparation. CPU FP32 KV tensors are retained through the first token.
No calibration or model accuracy comparison is performed.
"""

import argparse
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_backend import PIMSession, target_spec
from attention.int8_workloads import ACTIVATION_SCALE
from e2e.llama_workloads import LlamaPrefillLayer
from fx_utils import write_fx_graph
from model_benchmark_common import model_configs
from transformer.benchmark_model_transformer_block import PaddedFFNProjection


class ModelPrefillLayer(LlamaPrefillLayer):
    def __init__(self, layer, config, cosine, sine, causal_mask):
        super().__init__(layer, config, cosine, sine, causal_mask)
        self.gate_proj = PaddedFFNProjection(self.gate_proj)
        self.up_proj = PaddedFFNProjection(self.up_proj)

    def forward(self, hidden, cache):
        hidden, key, value = super().forward(hidden)
        # Session forwards tuple outputs as the next layer's positional inputs.
        return hidden, cache + ((key, value),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--gemm-config", choices=("baseline", "batched"), required=True,
                        help="baseline uses 2048 DPUs per head; batched uses 64")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")

    def configure(gemms, spec):
        return model_configs(gemms, spec, args.gemm_config)

    with torch.no_grad():
        setup_start = time.perf_counter_ns()
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
        blocks = [
            ModelPrefillLayer(layer, model.config, cosine, sine, causal_mask).eval()
            for layer in model.model.layers
        ]

        with PIMSession(target_spec("upmem", num_units=2048, nr_tasklets=16)) as session:
            print(f"Preparing 32 INT8 layers on {session.num_units} allocated DPUs ...",
                  flush=True)
            prepare_start = time.perf_counter_ns()
            plans = session.prepare(blocks, (example, ()), configure=configure)
            prepared_at = time.perf_counter_ns()
            prepare_ms = (prepared_at - prepare_start) / 1e6
            preparation_ms = (prepared_at - setup_start) / 1e6

            layer_plans = []
            for plan in plans:
                stages = [{
                    "name": stage.name,
                    "mnk": list(stage.shape),
                    "instances": stage.instance_count,
                    "p_b": stage.p_b,
                    "batch_rounds": stage.batch_rounds,
                    "mnk_rounds": list(stage.mnk_rounds),
                    "execution_extent": list(stage.config.execution_extent),
                    "config": asdict(stage.config),
                    "expected_launches": stage.batch_rounds * math.prod(stage.mnk_rounds),
                    "program": stage.program,
                } for stage in plan.stages]
                layer_plans.append({
                    "layer": plan.index, "stages": stages,
                    "expected_launches": sum(stage["expected_launches"] for stage in stages),
                    "memory": plan.backend_plan,
                })
                print(f"layer={plan.index} planned_stages={len(stages)} "
                      f"expected_launches={layer_plans[-1]['expected_launches']}", flush=True)
            expected_launches = sum(layer["expected_launches"] for layer in layer_plans)
            assert expected_launches == 32 * (71 if args.gemm_config == "baseline" else 9)
            print(f"gemm_config={args.gemm_config} prepare_ms={prepare_ms:.3f} "
                  f"expected_launches={expected_launches}", flush=True)

            before = session.stats()
            print("Executing one request: tokenize -> 32 layers -> first token -> free",
                  flush=True)
            with torch._dynamo.config.patch(error_on_recompile=True):
                request_start = time.perf_counter_ns()
                input_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1024,
                )["input_ids"]
                if tuple(input_ids.shape) != (1, 1024):
                    raise ValueError("the prompt must contain at least 1024 tokens including special tokens")
                tokenized_at = time.perf_counter_ns()

                hidden, cache = session.run_model(model.model.embed_tokens(input_ids), ())
                for layer_id, (key, value) in enumerate(cache):
                    session.kv_table[layer_id] = {
                        "key": key, "value": value,
                        "capacity": key.shape[-2], "length": key.shape[-2],
                    }
                logits = model.lm_head(model.model.norm(hidden[:, -1, :]))
                token_id = logits.argmax(dim=-1).item()
                token_ready = time.perf_counter_ns()

            after = session.stats()
            request_transfers = {name: after[name] - before[name] for name in before}
            kv_bytes = sum(tensor.numel() * tensor.element_size()
                           for pair in cache for tensor in pair)

        assert request_transfers["launches"] == expected_launches
        assert request_transfers["weight_bytes"] == (
            sum(plan.weight_bytes_per_unit for plan in plans) * session.num_units
        )
        assert request_transfers["device_compiles"] == request_transfers["activation_hits"] == 0
        assert len(cache) == 32 and kv_bytes == 1024 ** 3
        assert session.stats()["resource_opens"] == session.stats()["resource_closes"] == 1

        report = {
            "status": "execution_complete", "runtime": "model", "gemm_config": args.gemm_config,
            "checkpoint": str(args.checkpoint.resolve()),
            "prompt_file": str(args.prompt_file.resolve()),
            "scope": "single request prefill to first greedy token ID; no decode steps",
            "measurement_scope": (
                "tokenize, embedding, per-layer weight upload and execution, KV registration, "
                "last-token norm/lm_head/argmax; excludes preparation and resource open/close"
            ),
            "batch_size": 1, "sequence_length": 1024, "layers": 32, "heads": 32,
            "gemm_dtype": "int8 x int8 -> int32", "cpu_dtype": "float32",
            "lm_head_dtype": "float32", "lm_head_scope": "last_token_only",
            "accuracy_checked": False,
            "quantization": {
                "weight": "per-tensor absmax/128, round, clip [-127, 127]",
                "activation_scale": ACTIVATION_SCALE,
                "probability": "per-row max/127", "calibration": "none",
            },
            "weight_policy": "one layer resident at a time; upload at each layer entry",
            "activation_transfer_policy": "always",
            "kv_cache_location": "cpu", "kv_cache_dtype": "float32", "kv_cache_bytes": kv_bytes,
            "cpu_threads": torch.get_num_threads(),
            "versions": {"torch": torch.__version__, "triton": triton.__version__},
            "target": asdict(session.spec),
            "resource_open_ms": session.open_ms, "resource_close_ms": session.close_ms,
            "prepare_ms": prepare_ms, "preparation_ms": preparation_ms,
            "tokenization_ms": (tokenized_at - request_start) / 1e6,
            "model_to_token_ms": (token_ready - tokenized_at) / 1e6,
            "ttft_ms": (token_ready - request_start) / 1e6,
            "setup_to_token_ms": (token_ready - setup_start) / 1e6,
            "token_id": token_id, "token_text": tokenizer.decode([token_id]),
            "expected_launches": expected_launches,
            "request_transfers": request_transfers,
            "preparation_transfers": before, "session_totals": session.stats(),
            "layer_plans": layer_plans,
        }
        (args.report_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for plan in plans:
            write_fx_graph(plan.exported_graph, args.report_dir / f"layer_{plan.index}_export", {})
            write_fx_graph(plan.graph, args.report_dir / f"layer_{plan.index}_partitions", {})
        print(f"TTFT_ms={report['ttft_ms']:.3f} first_token_id={token_id} "
              f"launches={request_transfers['launches']} "
              f"weight_bytes={request_transfers['weight_bytes']} resources=1/1", flush=True)
        print(f"Execution report saved to {args.report_dir.resolve() / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
