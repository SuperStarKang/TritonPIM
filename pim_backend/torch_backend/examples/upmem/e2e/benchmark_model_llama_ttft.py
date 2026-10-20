"""Checkpoint Llama-family (Llama-2-7b, Llama-3-8B) prefill through model compile.

Run 1024 tokens, every layer, 2048 DPUs and 16 tasklets. --kv share|repeat as
e2e/llama_workloads.py. compile_model tunes every
stage class of the first layer, compiles all layers and plans the MRAM slots
before the DPUs are opened. Two identical requests follow: cold TTFT includes
the first weight upload at each layer entry; warm TTFT uploads again only the
layers of the shared slot. Compile and resource open/close are excluded. CPU
FP32 KV tensors are retained through the first token. No calibration or
accuracy comparison is performed.
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
from triton.backends.pim.upmem_artifact import WEIGHT

from torch_backend import PIMPerformanceProfile, compile_model
from torch_backend.template import TORCH_GEMM_TEMPLATE_VERSION
from torch_backend._autotuner import DEFAULT_CACHE_PATH
from attention.int8_workloads import ACTIVATION_SCALE
from e2e.llama_workloads import KV_MODES, LlamaPrefillLayer
from fx_utils import write_fx_graph
from transformer.benchmark_model_transformer_block import PaddedFFNProjection


class ModelPrefillLayer(LlamaPrefillLayer):
    def __init__(self, layer, config, cosine, sine, causal_mask, kv):
        super().__init__(layer, config, cosine, sine, causal_mask, kv)
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
    parser.add_argument("--kv", choices=KV_MODES, default="share", help="how a query head reads its KV head")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--performance", help="single-unit saturation JSON")
    parser.add_argument("--max-benchmarks", type=int, default=20, help="per stage class")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--local-capacity-limit-bytes", type=int, required=True,
                        help="WRAM bytes per DPU for Block Group buffers; the rest holds tasklet stacks")
    args = parser.parse_args()
    performance = None
    if args.performance is not None:
        performance = PIMPerformanceProfile.load(args.performance, kernel_version=TORCH_GEMM_TEMPLATE_VERSION)
    logging.basicConfig(level=logging.INFO)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")

    with torch.no_grad():
        setup_start = time.perf_counter_ns()
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.checkpoint), local_files_only=True, dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)
        config = model.config
        layers = config.num_hidden_layers

        example = torch.zeros((1, 1024, config.hidden_size), dtype=torch.float32)
        cosine, sine = model.model.rotary_emb(
            example, position_ids=torch.arange(1024).unsqueeze(0),
        )
        causal_mask = torch.full(
            (1024, 1024), torch.finfo(torch.float32).min, dtype=torch.float32,
        ).triu(1)
        blocks = [
            ModelPrefillLayer(layer, config, cosine, sine, causal_mask, args.kv).eval()
            for layer in model.model.layers
        ]
        # 모든 layer의 FP32 key/value [1, kv_heads, 1024, head_dim]
        expected_kv_bytes = 2 * layers * config.num_key_value_heads * 1024 * config.head_dim * 4

        print(f"Compiling {layers} INT8 layers (kv={args.kv}): tuning -> compile -> memory plan ...",
              flush=True)
        compiled = compile_model(
            blocks, (example, ()), target="upmem", units=2048,
            performance=performance,
            runtime_options={"nr_tasklets": 16,
                             "local_capacity_limit_bytes": args.local_capacity_limit_bytes},
            cache_path=args.cache,
            max_benchmarks=args.max_benchmarks, max_generations=args.max_generations,
        )
        compile_report = compiled.report
        memory = compiled.session.memory
        print(f"slots={memory.slots} pinned={memory.pinned} "
              f"bytes_per_unit={memory.bytes_per_unit} capacity={memory.capacity_bytes} "
              f"tuning_ms={compile_report['tuning_ms']:.1f} "
              f"compile_ms={compile_report['compile_ms']:.1f}", flush=True)
        for stage_class in compile_report["stage_classes"]:
            selected = stage_class["selected"]
            print(f"  {stage_class['label']}: measured={stage_class['measured']} "
                  f"E_ms={selected['execution_ms']:.3f} U_ms={selected['first_upload_ms']:.3f} "
                  f"config={tuple(selected['config'].values())}", flush=True)

        with compiled.session as session:
            preparation_ms = (time.perf_counter_ns() - setup_start) / 1e6
            plans = session.layers
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
                    # upload 한 번에 weight를 받는 DPU 수(p_m*p_n*p_k)만큼 전송된다.
                    "upload_bytes": sum(
                        binding.size * stage.config.p_m * stage.config.p_n * stage.config.p_k
                        for stage in plan.stages for binding in stage.bindings
                        if binding.role == WEIGHT),
                    "memory": plan.backend_plan,
                })
            expected_launches = sum(layer["expected_launches"] for layer in layer_plans)
            print(f"slots={memory.slots} pinned={memory.pinned} open_ms={session.open_ms:.3f} "
                  f"expected_launches={expected_launches}", flush=True)

            preparation_transfers = session.stats()
            requests = []
            with torch._dynamo.config.patch(error_on_recompile=True):
                # 모든 slot은 layer 첫 진입에서 weight를 올리므로 cold 요청에 upload가 포함된다.
                for phase in ("cold", "warm"):
                    print(f"Executing {phase} request: tokenize -> {layers} layers -> first token",
                          flush=True)
                    before = session.stats()
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
                    kv_bytes = sum(tensor.numel() * tensor.element_size()
                                   for pair in cache for tensor in pair)
                    assert len(cache) == layers and kv_bytes == expected_kv_bytes
                    requests.append({
                        "phase": phase,
                        "tokenization_ms": (tokenized_at - request_start) / 1e6,
                        "model_to_token_ms": (token_ready - tokenized_at) / 1e6,
                        "ttft_ms": (token_ready - request_start) / 1e6,
                        "token_id": token_id, "token_text": tokenizer.decode([token_id]),
                        "transfers": {name: after[name] - before[name] for name in before},
                    })
                    print(f"{phase}: TTFT_ms={requests[-1]['ttft_ms']:.3f} "
                          f"first_token_id={token_id} "
                          f"launches={requests[-1]['transfers']['launches']} "
                          f"weight_bytes={requests[-1]['transfers']['weight_bytes']}", flush=True)

        weight_bytes = sum(layer["upload_bytes"] for layer in layer_plans)
        # 공유 slot이 있으면(pinned < slots) 그 slot의 layer들은 요청마다 서로를 덮어쓴다.
        shared_layers = [layer["layer"] for layer in layer_plans
                         if memory.pinned < memory.slots and layer["memory"]["slot"] == memory.pinned]
        warm_weight_bytes = sum(layer["upload_bytes"] for layer in layer_plans
                                if layer["layer"] in shared_layers)
        cold, warm = requests
        assert cold["transfers"]["weight_bytes"] == weight_bytes
        assert warm["transfers"]["weight_bytes"] == warm_weight_bytes
        for request in requests:
            assert request["transfers"]["launches"] == expected_launches
            assert request["transfers"]["device_compiles"] == request["transfers"]["activation_hits"] == 0
        assert session.stats()["resource_opens"] == session.stats()["resource_closes"] == 1

        report = {
            "status": "execution_complete", "runtime": "model_compile",
            "checkpoint": str(args.checkpoint.resolve()),
            "prompt_file": str(args.prompt_file.resolve()),
            "scope": "two identical prefill requests to the first greedy token ID; no decode steps",
            "measurement_scope": (
                "per request: tokenize, embedding, weight upload at layer entry when its slot "
                "holds another layer, execution, KV registration, last-token norm/lm_head/argmax; "
                "excludes model compile and resource open/close"
            ),
            "batch_size": 1, "sequence_length": 1024, "layers": layers,
            "heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads, "kv": args.kv,
            "gemm_dtype": "int8 x int8 -> int32", "cpu_dtype": "float32",
            "lm_head_dtype": "float32", "lm_head_scope": "last_token_only",
            "accuracy_checked": False,
            "quantization": {
                "weight": "per-tensor absmax/128, round, clip [-127, 127]",
                "activation_scale": ACTIVATION_SCALE,
                "probability": "per-row max/127", "calibration": "none",
            },
            "shared_slot_layers": shared_layers,
            "activation_transfer_policy": "always",
            "kv_cache_location": "cpu", "kv_cache_dtype": "float32", "kv_cache_bytes": expected_kv_bytes,
            "cpu_threads": torch.get_num_threads(),
            "versions": {"torch": torch.__version__, "triton": triton.__version__},
            "target": asdict(session.spec),
            "compile": compile_report,
            "resource_open_ms": session.open_ms, "resource_close_ms": session.close_ms,
            "preparation_ms": preparation_ms,
            "expected_launches": expected_launches,
            "expected_cold_weight_bytes": weight_bytes,
            "expected_warm_weight_bytes": warm_weight_bytes,
            "requests": requests,
            "preparation_transfers": preparation_transfers, "session_totals": session.stats(),
            "layer_plans": layer_plans,
        }
        (args.report_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for plan in plans:
            write_fx_graph(plan.exported_graph, args.report_dir / f"layer_{plan.index}_export", {})
            write_fx_graph(plan.graph, args.report_dir / f"layer_{plan.index}_partitions", {})
        print(f"slots={memory.slots} pinned={memory.pinned} cold_TTFT_ms={cold['ttft_ms']:.3f} "
              f"warm_TTFT_ms={warm['ttft_ms']:.3f} cold_weight_bytes={cold['transfers']['weight_bytes']} "
              f"warm_weight_bytes={warm['transfers']['weight_bytes']} resources=1/1", flush=True)
        print(f"Execution report saved to {args.report_dir.resolve() / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
