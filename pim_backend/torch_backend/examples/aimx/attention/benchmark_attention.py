"""One Llama-family attention (QKV projections -> RoPE -> QK^T -> softmax (CPU) -> PV) on AiMX, weights resident.

The input is the RMSNorm output of --layer for the prompt, the output is the context before o_proj.
The first request uploads the QKV weights; later requests move only activations. The reference is
the same module on the CPU (BF16 GEMMs), so the cosine is the device error alone.
--qk-heads/--pv-heads/--kv as layer/benchmark_layer.py. Configs come from compile_model.

  python benchmark_attention.py --checkpoint ~/models/Meta-Llama-3-8B --prompt-file ../../upmem/e2e/prompt.txt \
      --fpgas 4
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_common import (
    add_device_arguments, compile_session, memory_record, print_request, request_breakdown, stage_records,
)
from llama_workloads import (
    KV_MODES, SEQUENCE, CheckpointRMSNorm, LlamaAimxAttention, load_llama, position_embeddings, prompt_ids,
    reference_layer,
)

import torch

from torch_backend._autotuner import DEFAULT_CACHE_PATH


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--qk-heads", type=int, default=1)
    parser.add_argument("--pv-heads", type=int, default=1)
    parser.add_argument("--kv", choices=KV_MODES, default="share")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or Path(__file__).resolve().with_name(
        f"report_{args.checkpoint.name}_{args.kv}_qk{args.qk_heads}_pv{args.pv_heads}_f{args.fpgas}")
    report_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        tokenizer, model = load_llama(args.checkpoint)
        cosine, sine = position_embeddings(model, SEQUENCE)
        hidden = model.model.embed_tokens(prompt_ids(tokenizer, args.prompt_file.read_text(encoding="utf-8"), SEQUENCE)).float()
        for layer in model.model.layers[:args.layer]:
            hidden = reference_layer(layer, hidden, cosine, sine)
        layer = model.model.layers[args.layer]
        normalized = CheckpointRMSNorm(layer.input_layernorm)(hidden)
        block = LlamaAimxAttention(layer, cosine, sine, args.qk_heads, args.pv_heads, args.kv).eval()
        reference = block(normalized)

        start = time.perf_counter_ns()
        session, config_report = compile_session([block], (torch.zeros(1, SEQUENCE, model.config.hidden_size),), args)
        compile_ms = (time.perf_counter_ns() - start) / 1e6
        plan, = session.plans["prefill"]
        stages = stage_records(plan)
        weight_stages = {stage["name"] for stage in stages if stage["kind"] == "weight"}
        print(f"compile_ms={compile_ms:.1f} slots={session.memory.phases['prefill'].slots} "
              f"groups={session.memory.phases['prefill'].groups} "
              f"bytes_per_fpga={session.memory.bytes_per_unit}", flush=True)
        for stage in stages:
            print(f"  stage={stage['name']} kind={stage['kind']} mnk={tuple(stage['mnk'])} "
                  f"instances={stage['instances']} p_b={stage['p_b']} batch_rounds={stage['batch_rounds']} "
                  f"config={tuple(stage['config'].values())}", flush=True)

        calls = []
        with session:
            for phase, count in (("first", 1), ("warmup", args.warmup), ("sample", args.repeat)):
                for step in range(count):
                    before = session.stats()
                    start = time.perf_counter_ns()
                    output = session.run_model("prefill", normalized)
                    attention_ms = (time.perf_counter_ns() - start) / 1e6
                    call = {"phase": phase,
                            **request_breakdown(before, session.stats(), weight_stages, attention_ms),
                            "cosine": torch.nn.functional.cosine_similarity(
                                output.flatten().double(), reference.flatten().double(), dim=0).item()}
                    calls.append(call)
                    print_request(f"{phase}={step} cosine={call['cosine']:.6f}", call)
            # slot buffer는 첫 실행에서 할당되므로 요청이 끝난 뒤 읽는다.
            memory_table = session.memory_table()

    samples = [call for call in calls if call["phase"] == "sample"]
    summary = {
        "attention_ms": statistics.median(call["layers_ms"] for call in samples),
        "kinds": {name: statistics.median(call["kinds"][name] for call in samples)
                  for name in samples[0]["kinds"]},
        "phases": {kind: {name: statistics.median(call["phases"][kind][name] for call in samples)
                          for name in samples[0]["phases"][kind]}
                   for kind in samples[0]["phases"]},
        "weight_operations": 2 * SEQUENCE * model.config.hidden_size * (block.heads + 2 * block.kv_heads) * block.head_dim,
        # QK^T와 PV의 유효 연산 수는 묶음과 무관하다. executed_operations와 비교하면 0 블록 계산량이 나온다.
        "useful_activation_operations": 2 * (2 * SEQUENCE * SEQUENCE * block.head_dim * block.heads),
        "executed_activation_operations": sum(stage["executed_operations"] for stage in stages
                                              if stage["kind"] == "activation"),
    }
    print(f"summary attention_ms={summary['attention_ms']:.1f} " + " ".join(
        f"{name}={value:.1f}" for name, value in summary["kinds"].items()), flush=True)

    report = {
        "checkpoint": str(args.checkpoint), "layer": args.layer, "fpgas": args.fpgas,
        "base_fpga": args.base_fpga, "sequence": SEQUENCE, "config_source": "autotune",
        "qk_heads": args.qk_heads, "pv_heads": args.pv_heads, "kv": args.kv,
        "compile_ms": compile_ms, "session_open_ms": session.open_ms, "session_close_ms": session.close_ms,
        "memory": memory_record(session.memory),
        "config_report": config_report, "memory_table": memory_table, "stages": stages,
        "calls": calls, "summary": summary,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
