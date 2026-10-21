"""One Llama-family W8A8 decoder layer (Llama-2-7b, Llama-3-8B) on UPMEM against the CPU and Hugging Face.

The input is the prompt embedding passed through the HF FP32 layers before --layer.
The output of that layer is compared three ways:
  cpu_vs_hf      the same ModelPrefillLayer on the CPU vs HF FP32: W8A8 error alone (no calibration)
  upmem_vs_cpu   UPMEM vs the CPU run: INT32 GEMMs are exact, so only CPU float differences remain
  upmem runs     two UPMEM runs: run-to-run agreement
The UPMEM output is saved per --kv, so share and repeat can be compared afterwards.
UPMEM configs come from compile_model (the layer tuned with its weights resident).

  ../cmd.sh verify_llama_layer.py --checkpoint ~/models/Meta-Llama-3-8B --prompt-file prompt.txt \
      --kv share --local-capacity-limit-bytes <bytes>
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH
from e2e.benchmark_model_llama_ttft import ModelPrefillLayer
from e2e.llama_workloads import KV_MODES


def compare(output, reference):
    difference = (output.double() - reference.double()).flatten()
    reference = reference.double().flatten()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / reference.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(output.double().flatten(), reference, dim=0).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kv", choices=KV_MODES, default="share", help="how a query head reads its KV head")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--max-benchmarks", type=int, default=20, help="per stage class")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--local-capacity-limit-bytes", type=int, required=True,
                        help="WRAM bytes per DPU for Block Group buffers; the rest holds tasklet stacks")
    parser.add_argument("--report-dir", type=Path,
                        default=Path(__file__).resolve().with_name("llama_layer_report"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.checkpoint), local_files_only=True, dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)
        config = model.config
        input_ids = tokenizer(args.prompt_file.read_text(encoding="utf-8"), return_tensors="pt",
                              truncation=True, max_length=1024)["input_ids"]
        if tuple(input_ids.shape) != (1, 1024):
            raise ValueError("the prompt must contain at least 1024 tokens including special tokens")
        cosine, sine = model.model.rotary_emb(
            torch.zeros((1, 1024, config.hidden_size)), position_ids=torch.arange(1024).unsqueeze(0),
        )
        causal_mask = torch.full(
            (1024, 1024), torch.finfo(torch.float32).min, dtype=torch.float32,
        ).triu(1)

        hidden = model.model.embed_tokens(input_ids)
        for layer in model.model.layers[:args.layer]:
            hidden = layer(hidden, attention_mask=causal_mask[None, None], position_embeddings=(cosine, sine))
        reference = model.model.layers[args.layer](
            hidden, attention_mask=causal_mask[None, None], position_embeddings=(cosine, sine))

        block = ModelPrefillLayer(model.model.layers[args.layer], config, cosine, sine,
                                  causal_mask, args.kv).eval()
        start = time.perf_counter_ns()
        cpu, _ = block(hidden, ())
        cpu_ms = (time.perf_counter_ns() - start) / 1e6
        print(f"layer={args.layer} kv={args.kv} heads={config.num_attention_heads} "
              f"kv_heads={config.num_key_value_heads} cpu_eager_ms={cpu_ms:.1f}", flush=True)

        start = time.perf_counter_ns()
        compiled = compile_model(
            {"prefill": ([block], (torch.zeros_like(hidden), ()))}, target="upmem", units=2048,
            runtime_options={"nr_tasklets": 16,
                             "local_capacity_limit_bytes": args.local_capacity_limit_bytes},
            cache_path=args.cache,
            max_benchmarks=args.max_benchmarks, max_generations=args.max_generations,
        )
        compile_ms = (time.perf_counter_ns() - start) / 1e6
        session = compiled.session
        plan, = session.plans["prefill"]
        print(f"compile_ms={compile_ms:.1f} stages={len(plan.stages)} "
              f"slots={session.memory.phases['prefill'].slots} "
              f"bytes_per_unit={session.memory.bytes_per_unit}", flush=True)
        for stage in plan.stages:
            print(f"  stage={stage.name} mnk={stage.shape} instances={stage.instance_count} "
                  f"batch_rounds={stage.batch_rounds} dtype={stage.input_dtype}", flush=True)
        with session:
            outputs = []
            calls = []
            # 첫 호출은 layer 진입 시 weight upload를 포함한다.
            for phase in ("first", "second"):
                start = time.perf_counter_ns()
                output, _ = session.run_model("prefill", hidden, ())
                outputs.append(output)
                calls.append({"phase": phase, "wall_ms": (time.perf_counter_ns() - start) / 1e6})

    results = {
        "cpu_vs_hf": compare(cpu, reference),
        "upmem_vs_hf": compare(outputs[0], reference),
        "upmem_vs_cpu": compare(outputs[0], cpu),
        "upmem_runs_mismatches": (outputs[0] != outputs[1]).sum().item(),
    }
    for name, metrics in results.items():
        print(f"{name}: {metrics}", flush=True)
    for call in calls:
        print(f"upmem {call['phase']} wall_ms={call['wall_ms']:.1f}", flush=True)

    name = f"{args.checkpoint.name}_layer_{args.layer}_{args.kv}"
    torch.save(outputs[0], args.report_dir / f"{name}_upmem.pt")
    report = {
        "checkpoint": str(args.checkpoint), "layer": args.layer, "kv": args.kv, "sequence": 1024,
        "heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
        "cpu_eager_ms": cpu_ms, "compile_ms": compile_ms,
        "compile": compiled.report,
        "stages": [{"name": stage.name, "mnk": list(stage.shape), "instances": stage.instance_count,
                    "batch_rounds": stage.batch_rounds} for stage in plan.stages],
        "calls": calls, **results,
    }
    path = args.report_dir / f"{name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={path} output={args.report_dir / f'{name}_upmem.pt'}", flush=True)


if __name__ == "__main__":
    main()
