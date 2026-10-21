"""Llama-family prefill (1024 tokens, 32 layers) on AiMX: TTFT and the first token.

One request = tokenize -> embed -> 32 LlamaAimxLayer on AiMX -> norm -> lm_head (CPU) -> argmax.
Configs come from compile_model (one layer tuned with its weights resident). The first request
includes the weight upload; with P-2 (all layers share one layer's slot) every request uploads
each layer's weights. Before compiling, the Hugging Face layers run in FP32 one at a time for
the reference logits; each HF layer is released once its BF16 operands are built.

  python benchmark_llama_ttft.py --checkpoint ~/models/Llama-2-7b-hf \
      --prompt-file ../upmem/e2e/prompt.txt --fpgas 2
"""

import argparse
import copy
import json
from pathlib import Path
import time

from experiment_common import add_device_arguments, compile_session, memory_record
from llama_workloads import (
    KV_MODES, SEQUENCE, LlamaAimxLayer, load_llama, position_embeddings, prompt_ids, reference_layer,
)

import torch

from torch_backend._autotuner import DEFAULT_CACHE_PATH
from triton.backends.pim.driver import _STAGE_NAMES


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--repeat", type=int, default=1, help="requests after the first")
    parser.add_argument("--kv", choices=KV_MODES, default="share", help="how a query head reads its KV head")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    args.report_dir = args.report_dir or Path(__file__).resolve().with_name(
        f"llama_ttft_report_{args.checkpoint.name}_{args.kv}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")

    with torch.no_grad():
        tokenizer, model = load_llama(args.checkpoint)
        cosine, sine = position_embeddings(model)
        # lm_head와 최종 norm은 CPU에서 FP32로 계산한다 (lm_head 32000 x 4096, 0.5 GB).
        norm = copy.deepcopy(model.model.norm).float()
        lm_head = copy.deepcopy(model.lm_head).float()

        start = time.perf_counter_ns()
        reference = model.model.embed_tokens(prompt_ids(tokenizer, prompt)).float()
        for layer in model.model.layers:
            reference = reference_layer(layer, reference, cosine, sine)
        reference_logits = lm_head(norm(reference[:, -1, :]))
        reference_ms = (time.perf_counter_ns() - start) / 1e6
        reference_token = reference_logits.argmax(dim=-1).item()
        print(f"hf_fp32_reference_ms={reference_ms:.1f} token={reference_token} "
              f"{tokenizer.decode([reference_token])!r}", flush=True)

        blocks = []
        for index in range(len(model.model.layers)):
            blocks.append(LlamaAimxLayer(model.model.layers[index], cosine, sine, kv=args.kv).eval())
            model.model.layers[index] = torch.nn.Identity()

        start = time.perf_counter_ns()
        session, config_report = compile_session(
            blocks, (torch.zeros(1, SEQUENCE, model.config.hidden_size),), args)
        compile_ms = (time.perf_counter_ns() - start) / 1e6
        plans = session.plans["prefill"]
        memory = session.memory
        prefill = memory.phases["prefill"]
        print(f"compile_ms={compile_ms:.1f} layers={len(plans)} stages/layer={len(plans[0].stages)} "
              f"slots={prefill.slots} groups={prefill.groups} bytes_per_fpga={memory.bytes_per_unit} "
              f"capacity={memory.capacity_bytes}", flush=True)

        with session:
            calls = []
            for phase, count in (("first", 1), ("sample", args.repeat)):
                for step in range(count):
                    before = session.stats()
                    start = time.perf_counter_ns()
                    embedded = model.model.embed_tokens(prompt_ids(tokenizer, prompt)).float()
                    tokenized_at = time.perf_counter_ns()
                    hidden = session.run_model("prefill", embedded)
                    layers_at = time.perf_counter_ns()
                    logits = lm_head(norm(hidden[:, -1, :]))
                    token = logits.argmax(dim=-1).item()
                    ttft_ms = (time.perf_counter_ns() - start) / 1e6
                    after = session.stats()
                    wall = {name: after[name] - before[name]
                            for name in ("upload_wall_ms", "run_wall_ms")}
                    # "upload:mm" 같은 key마다 이번 요청의 몫. native 칸은 FPGA thread 평균이다.
                    breakdown = {key: {field: value - before["breakdown"][key][field]
                                       for field, value in record.items() if field != "jobs"}
                                 for key, record in after["breakdown"].items()}
                    # stage를 합친 호출 종류별 합계. other = wall - native 합 (omp, lock, ctypes 등).
                    phases = {}
                    for kind in ("upload", "run"):
                        records = [record for key, record in breakdown.items()
                                   if key.startswith(kind + ":")]
                        phases[kind] = {field: sum(record[field] for record in records)
                                        for field in ("wall_ms", *_STAGE_NAMES["aimx"])}
                        phases[kind]["other"] = phases[kind]["wall_ms"] - sum(
                            phases[kind][name] for name in _STAGE_NAMES["aimx"])
                    call = {
                        "phase": phase, "ttft_ms": ttft_ms,
                        "tokenize_embed_ms": (tokenized_at - start) / 1e6,
                        "layers_ms": (layers_at - tokenized_at) / 1e6,
                        "token": token, "token_text": tokenizer.decode([token]),
                        "token_matches_hf": token == reference_token,
                        "top5_overlap": len(set(logits.topk(5).indices[0].tolist())
                                            & set(reference_logits.topk(5).indices[0].tolist())),
                        "hidden_max_abs": (hidden - reference).abs().max().item(),
                        "hidden_cosine": torch.nn.functional.cosine_similarity(
                            hidden.flatten().double(), reference.flatten().double(), dim=0).item(),
                        "logits_cosine": torch.nn.functional.cosine_similarity(
                            logits.flatten().double(), reference_logits.flatten().double(), dim=0).item(),
                        **{name: after[name] - before[name]
                           for name in ("weight_uploads", "weight_bytes", "invocations")},
                        **wall,
                        # layer 시간 중 PIM 호출 밖: CPU 연산(norm, RoPE, softmax 등)과 Python
                        "cpu_and_python_ms": (layers_at - tokenized_at) / 1e6 - sum(wall.values()),
                        "lm_head_ms": ttft_ms - (layers_at - start) / 1e6,
                        "phases": phases, "breakdown": breakdown,
                    }
                    calls.append(call)
                    print(f"{phase}={step} ttft_ms={ttft_ms:.1f} layers_ms={call['layers_ms']:.1f} "
                          f"token={token} {call['token_text']!r} matches_hf={call['token_matches_hf']} "
                          f"top5_overlap={call['top5_overlap']} logits_cosine={call['logits_cosine']:.6f} "
                          f"hidden_cosine={call['hidden_cosine']:.6f} "
                          f"weight_uploads={call['weight_uploads']} weight_bytes={call['weight_bytes']} "
                          f"invocations={call['invocations']}", flush=True)
                    for kind, totals in phases.items():
                        print(f"  {kind:<7}" + " ".join(f"{name}={value:.1f}"
                                                       for name, value in totals.items()), flush=True)
                    print(f"  cpu_and_python_ms={call['cpu_and_python_ms']:.1f} "
                          f"tokenize_embed_ms={call['tokenize_embed_ms']:.1f} "
                          f"lm_head_ms={call['lm_head_ms']:.1f}", flush=True)
            session_totals = session.stats()

    report = {
        "checkpoint": str(args.checkpoint), "kv": args.kv, "fpgas": args.fpgas, "base_fpga": args.base_fpga,
        "sequence": SEQUENCE, "layers": len(blocks),
        "gemm_dtype": "bfloat16 host and AiMX", "cpu_dtype": "float32",
        "reference": {"hf_fp32_ms": reference_ms, "token": reference_token,
                      "token_text": tokenizer.decode([reference_token])},
        "compile_ms": compile_ms, "config_report": config_report, "session_open_ms": session.open_ms, "session_close_ms": session.close_ms,
        "memory": memory_record(memory),
        "stages": [{"name": stage.name, "mnk": list(stage.shape), "instances": stage.instance_count,
                    "batch_rounds": stage.batch_rounds} for stage in plans[0].stages],
        "calls": calls, "session_totals": session_totals,
    }
    (args.report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={args.report_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
