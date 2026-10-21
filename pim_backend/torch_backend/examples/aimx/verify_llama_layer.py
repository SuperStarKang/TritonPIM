"""One Llama-family decoder layer (Llama-2-7b, Llama-3-8B) on AiMX against the Hugging Face FP32 layer: a few minutes.

The input is the prompt embedding passed through the HF FP32 layers before --layer.
Three outputs of that layer are compared with the HF FP32 output:
  cpu   LlamaAimxLayer on the CPU (BF16 GEMMs): the error of BF16 GEMM operands alone
  aimx  LlamaAimxLayer on AiMX, run twice: the device error and run-to-run agreement
AiMX configs come from compile_model (the layer tuned with its weights resident).

  python verify_llama_layer.py --checkpoint ~/models/Llama-2-7b-hf \
      --prompt-file ../upmem/e2e/prompt.txt --fpgas 2
"""

import argparse
import json
from pathlib import Path
import time

from experiment_common import add_device_arguments, compile_session, memory_record
from llama_workloads import (
    KV_MODES, SEQUENCE, LlamaAimxLayer, load_llama, position_embeddings, prompt_ids, reference_layer,
)

import torch

from torch_backend._autotuner import DEFAULT_CACHE_PATH


def compare(output, reference):
    difference = (output.double() - reference.double()).flatten()
    reference = reference.double().flatten()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / reference.norm()).item(),
        "cosine": torch.nn.functional.cosine_similarity(output.double().flatten(), reference, dim=0).item(),
    }


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kv", choices=KV_MODES, default="share", help="how a query head reads its KV head")
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--report-dir", type=Path,
                        default=Path(__file__).resolve().with_name("llama_layer_report"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        tokenizer, model = load_llama(args.checkpoint)
        cosine, sine = position_embeddings(model)
        hidden = model.model.embed_tokens(prompt_ids(tokenizer, args.prompt_file.read_text(encoding="utf-8"))).float()
        for layer in model.model.layers[:args.layer]:
            hidden = reference_layer(layer, hidden, cosine, sine)
        start = time.perf_counter_ns()
        reference = reference_layer(model.model.layers[args.layer], hidden, cosine, sine)
        reference_ms = (time.perf_counter_ns() - start) / 1e6

        block = LlamaAimxLayer(model.model.layers[args.layer], cosine, sine, kv=args.kv).eval()
        start = time.perf_counter_ns()
        cpu = block(hidden)
        cpu_ms = (time.perf_counter_ns() - start) / 1e6
        print(f"layer={args.layer} hf_fp32_ms={reference_ms:.1f} cpu_bf16_gemm_ms={cpu_ms:.1f}", flush=True)

        start = time.perf_counter_ns()
        session, config_report = compile_session(
            [block], (torch.zeros(1, SEQUENCE, model.config.hidden_size),), args)
        compile_ms = (time.perf_counter_ns() - start) / 1e6
        plan, = session.plans["prefill"]
        print(f"compile_ms={compile_ms:.1f} stages={len(plan.stages)} "
              f"slots={session.memory.phases['prefill'].slots} bytes_per_fpga={session.memory.bytes_per_unit}",
              flush=True)
        for stage in plan.stages:
            print(f"  stage={stage.name} mnk={stage.shape} instances={stage.instance_count} "
                  f"batch_rounds={stage.batch_rounds} dtype={stage.input_dtype}", flush=True)
        with session:
            outputs = []
            calls = []
            # 첫 호출은 layer 진입 시 weight upload를 포함한다.
            for phase in ("first", "second"):
                start = time.perf_counter_ns()
                outputs.append(session.run_model("prefill", hidden))
                calls.append({"phase": phase, "wall_ms": (time.perf_counter_ns() - start) / 1e6})

    results = {
        "cpu_vs_hf": compare(cpu, reference),
        "aimx_vs_hf": compare(outputs[0], reference),
        "aimx_vs_cpu": compare(outputs[0], cpu),
        "aimx_runs_mismatches": (outputs[0] != outputs[1]).sum().item(),
    }
    for name, metrics in results.items():
        print(f"{name}: {metrics}", flush=True)
    for call in calls:
        print(f"aimx {call['phase']} wall_ms={call['wall_ms']:.1f}", flush=True)
    report = {
        "checkpoint": str(args.checkpoint), "layer": args.layer, "kv": args.kv, "fpgas": args.fpgas,
        "base_fpga": args.base_fpga, "sequence": SEQUENCE,
        "hf_fp32_ms": reference_ms, "cpu_bf16_gemm_ms": cpu_ms, "compile_ms": compile_ms,
        "config_report": config_report,
        "memory": memory_record(session.memory),
        "stages": [{"name": stage.name, "mnk": list(stage.shape), "instances": stage.instance_count,
                    "batch_rounds": stage.batch_rounds} for stage in plan.stages],
        "calls": calls, **results,
    }
    path = args.report_dir / f"{args.checkpoint.name}_layer_{args.layer}_{args.kv}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={path}", flush=True)


if __name__ == "__main__":
    main()
