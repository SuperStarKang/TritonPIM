"""Llama-2-7b-hf prefill through the first --layers decoder layers on AiMX: TTFT and per-layer time.

One request = tokenize -> embed -> layers on AiMX -> norm -> lm_head (CPU) -> argmax. The reference
runs the same layers in HF FP32, so the first token is compared for any --layers.
--config-source autotune compiles with compile_model:
  --strategy joint        best configs of every feasible policy; DP when the fastest configs do not fit
  --strategy layer_first  fastest configs of one layer; P-1 if every layer fits, else P-2
--config-source analytical uses seed configs; P-1 if every layer fits, else P-2 (--strategy unused).
--qk-heads/--pv-heads as layer/benchmark_layer.py.

  python benchmark_e2e.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../../upmem/e2e/prompt.txt \
      --config-source autotune --strategy joint --layers 4
"""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_common import add_config_arguments, compile_session, print_request, request_breakdown, stage_records
from llama_workloads import LlamaAimxLayer, load_llama, position_embeddings, prompt_ids, reference_layer

import torch

from fp32_workloads import HIDDEN, SEQUENCE
from torch_backend._autotuner import DEFAULT_CACHE_PATH
from torch_backend.model_compile import STRATEGIES


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--strategy", choices=STRATEGIES, default="joint")
    parser.add_argument("--qk-heads", type=int, default=2)
    parser.add_argument("--pv-heads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3, help="requests after the first")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or Path(__file__).resolve().with_name(
        f"report_{args.config_source}_{args.strategy}_l{args.layers}_qk{args.qk_heads}_pv{args.pv_heads}_f{args.fpgas}")
    report_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")

    with torch.no_grad():
        tokenizer, model = load_llama(args.checkpoint)
        cosine, sine = position_embeddings(model)
        # lm_head와 최종 norm은 CPU에서 FP32로 계산한다 (lm_head 32000 x 4096, 0.5 GB).
        norm = copy.deepcopy(model.model.norm).float()
        lm_head = copy.deepcopy(model.lm_head).float()
        layers = model.model.layers[:args.layers]

        reference = model.model.embed_tokens(prompt_ids(tokenizer, prompt)).float()
        for layer in layers:
            reference = reference_layer(layer, reference, cosine, sine)
        reference_logits = lm_head(norm(reference[:, -1, :]))
        reference_token = reference_logits.argmax(dim=-1).item()
        print(f"reference layers={args.layers} token={reference_token} "
              f"{tokenizer.decode([reference_token])!r}", flush=True)

        # HF layer는 BF16 operand를 만든 뒤 바로 놓아 host memory에 한 벌만 둔다.
        blocks = []
        for index in range(args.layers):
            blocks.append(LlamaAimxLayer(model.model.layers[index], cosine, sine,
                                         args.qk_heads, args.pv_heads).eval())
            model.model.layers[index] = torch.nn.Identity()

        start = time.perf_counter_ns()
        session, config_report = compile_session(blocks, (torch.zeros(1, SEQUENCE, HIDDEN),), args,
                                                 requests=1 + args.repeat, strategy=args.strategy)
        compile_ms = (time.perf_counter_ns() - start) / 1e6
        stages = stage_records(session.layers[0])
        weight_stages = {stage["name"] for stage in stages if stage["kind"] == "weight"}
        residency = session.residency
        print(f"compile_ms={compile_ms:.1f} layers={args.layers} policy={residency.policy} "
              f"selection={config_report['selection']} bytes_per_fpga={residency.bytes_per_unit} "
              f"capacity={residency.capacity_bytes}", flush=True)
        for stage_class in config_report["stage_classes"]:
            print(f"  {stage_class['label']}: config={tuple(stage_class['selected']['config'].values())}", flush=True)

        calls = []
        with session:
            for phase, count in (("first", 1), ("sample", args.repeat)):
                for step in range(count):
                    before = session.stats()
                    start = time.perf_counter_ns()
                    embedded = model.model.embed_tokens(prompt_ids(tokenizer, prompt)).float()
                    tokenized_at = time.perf_counter_ns()
                    hidden = session.run_model(embedded)
                    layers_at = time.perf_counter_ns()
                    logits = lm_head(norm(hidden[:, -1, :]))
                    token = logits.argmax(dim=-1).item()
                    end = time.perf_counter_ns()
                    ttft_ms = (end - start) / 1e6
                    call = {
                        "phase": phase, "ttft_ms": ttft_ms,
                        "tokenize_embed_ms": (tokenized_at - start) / 1e6,
                        "lm_head_ms": (end - layers_at) / 1e6,
                        **request_breakdown(before, session.stats(), weight_stages, (layers_at - tokenized_at) / 1e6),
                        "token": token, "token_matches_reference": token == reference_token,
                        "logits_cosine": torch.nn.functional.cosine_similarity(
                            logits.flatten().double(), reference_logits.flatten().double(), dim=0).item(),
                    }
                    call["per_layer_ms"] = call["layers_ms"] / args.layers
                    calls.append(call)
                    print_request(f"{phase}={step} ttft_ms={ttft_ms:.1f} per_layer_ms={call['per_layer_ms']:.1f} "
                                  f"token={token} matches={call['token_matches_reference']} "
                                  f"logits_cosine={call['logits_cosine']:.6f}", call)

    samples = [call for call in calls if call["phase"] == "sample"]
    summary = {name: statistics.median(call[name] for call in samples)
               for name in ("ttft_ms", "layers_ms", "per_layer_ms")}
    summary["kinds"] = {name: statistics.median(call["kinds"][name] for call in samples)
                        for name in samples[0]["kinds"]}
    print(f"summary ttft_ms={summary['ttft_ms']:.1f} per_layer_ms={summary['per_layer_ms']:.1f} "
          f"policy={residency.policy} selection={config_report['selection']}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint), "layers": args.layers, "fpgas": args.fpgas,
        "base_fpga": args.base_fpga, "sequence": SEQUENCE, "config_source": args.config_source,
        "strategy": args.strategy, "policy": residency.policy, "selection": config_report["selection"],
        "qk_heads": args.qk_heads, "pv_heads": args.pv_heads,
        "reference": {"token": reference_token, "token_text": tokenizer.decode([reference_token])},
        "compile_ms": compile_ms, "session_open_ms": session.open_ms, "session_close_ms": session.close_ms,
        "residency": {"policy": residency.policy, "bytes_per_fpga": residency.bytes_per_unit,
                      "capacity_bytes": residency.capacity_bytes},
        "config_report": config_report, "stages": stages, "calls": calls, "summary": summary,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
