"""One Llama-family decoder layer (Llama-2-7b, Llama-3-8B) on AiMX: upload every request vs weight-stationary.

--residency upload    every request uploads the weights into the kept slot again (as a shared slot)
--residency resident  the first request uploads; later requests move only activations (own slot)
--qk-heads/--pv-heads heads per GEMM written in the model; 1 and 1 leave packing to the compiler
--kv share|repeat     query heads of one group read the same KV tensor, or a copy each (repeat_kv)
Configs come from compile_model (every stage class tuned with its weights resident).

Each request reports the PIM wall per upload/run phase, the native slots (transfer, layout,
compute, setup), and the split into weight GEMMs, activation GEMMs and CPU. With upload,
the samples also give the weight-stationary prediction T(R) = E + U / R over R requests.

  python benchmark_layer.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../../upmem/e2e/prompt.txt \
      --residency upload
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
    KV_MODES, SEQUENCE, LlamaAimxLayer, load_llama, position_embeddings, prompt_ids, reference_layer,
)

import torch

from torch_backend._autotuner import DEFAULT_CACHE_PATH


REQUESTS_PREDICTED = (1, 2, 4, 8, 16, 32, 64)


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--residency", choices=("upload", "resident"), required=True)
    parser.add_argument("--qk-heads", type=int, default=1)
    parser.add_argument("--pv-heads", type=int, default=1)
    parser.add_argument("--kv", choices=KV_MODES, default="share")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or Path(__file__).resolve().with_name(
        f"report_{args.checkpoint.name}_{args.residency}_{args.kv}_qk{args.qk_heads}_pv{args.pv_heads}_f{args.fpgas}")
    report_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        tokenizer, model = load_llama(args.checkpoint)
        cosine, sine = position_embeddings(model)
        hidden = model.model.embed_tokens(prompt_ids(tokenizer, args.prompt_file.read_text(encoding="utf-8"))).float()
        for layer in model.model.layers[:args.layer]:
            hidden = reference_layer(layer, hidden, cosine, sine)
        reference = reference_layer(model.model.layers[args.layer], hidden, cosine, sine)
        block = LlamaAimxLayer(model.model.layers[args.layer], cosine, sine,
                               args.qk_heads, args.pv_heads, args.kv).eval()

        start = time.perf_counter_ns()
        session, config_report = compile_session(
            [block], (torch.zeros(1, SEQUENCE, model.config.hidden_size),), args)
        compile_ms = (time.perf_counter_ns() - start) / 1e6
        plan, = session.layers
        stages = stage_records(plan)
        weight_stages = {stage["name"] for stage in stages if stage["kind"] == "weight"}
        print(f"compile_ms={compile_ms:.1f} slots={session.memory.slots} groups={session.memory.groups} "
              f"bytes_per_fpga={session.memory.bytes_per_unit}", flush=True)
        for stage in stages:
            print(f"  stage={stage['name']} kind={stage['kind']} mnk={tuple(stage['mnk'])} "
                  f"instances={stage['instances']} p_b={stage['p_b']} batch_rounds={stage['batch_rounds']} "
                  f"config={tuple(stage['config'].values())}", flush=True)

        calls = []
        with session:
            for phase, count in (("first", 1), ("warmup", args.warmup), ("sample", args.repeat)):
                for step in range(count):
                    if args.residency == "upload":
                        # layer 하나는 planner가 P-1을 고른다. slot을 비워 P-2처럼 매 요청 weight를 다시 올린다.
                        session.backend.model.slots.owner.clear()
                    before = session.stats()
                    start = time.perf_counter_ns()
                    output = session.run_model(hidden)
                    layers_ms = (time.perf_counter_ns() - start) / 1e6
                    call = {"phase": phase, **request_breakdown(before, session.stats(), weight_stages, layers_ms),
                            "cosine": torch.nn.functional.cosine_similarity(
                                output.flatten().double(), reference.flatten().double(), dim=0).item()}
                    calls.append(call)
                    print_request(f"{phase}={step} cosine={call['cosine']:.6f}", call)
            # slot buffer는 첫 실행에서 할당되므로 요청이 끝난 뒤 읽는다.
            memory_table = session.memory_table()

    samples = [call for call in calls if call["phase"] == "sample"]
    summary = {
        "layers_ms": statistics.median(call["layers_ms"] for call in samples),
        "kinds": {name: statistics.median(call["kinds"][name] for call in samples)
                  for name in samples[0]["kinds"]},
        "phases": {kind: {name: statistics.median(call["phases"][kind][name] for call in samples)
                          for name in samples[0]["phases"][kind]}
                   for kind in samples[0]["phases"]},
        # QK^T와 PV의 유효 연산 수는 묶음과 무관하다. executed_operations와 비교하면 0 블록 계산량이 나온다.
        "useful_activation_operations": 2 * (2 * SEQUENCE * SEQUENCE * block.head_dim * block.heads),
        "executed_activation_operations": sum(stage["executed_operations"] for stage in stages
                                              if stage["kind"] == "activation"),
    }
    if args.residency == "upload":
        # U: 요청마다 내는 weight 업로드. E: 나머지. weight-stationary면 U를 R 요청이 나눠 낸다.
        upload_ms = statistics.median(call["upload_wall_ms"] for call in samples)
        execution_ms = summary["layers_ms"] - upload_ms
        summary["prediction"] = {"execution_ms": execution_ms, "upload_ms": upload_ms,
                                 "layers_ms": {requests: execution_ms + upload_ms / requests
                                               for requests in REQUESTS_PREDICTED}}
        print("prediction " + " ".join(f"R={requests}:{value:.1f}"
                                       for requests, value in summary["prediction"]["layers_ms"].items()), flush=True)
    print(f"summary layers_ms={summary['layers_ms']:.1f} " + " ".join(
        f"{name}={value:.1f}" for name, value in summary["kinds"].items()), flush=True)

    report = {
        "checkpoint": str(args.checkpoint), "layer": args.layer, "layers": 1, "fpgas": args.fpgas,
        "base_fpga": args.base_fpga, "sequence": SEQUENCE, "config_source": "autotune",
        "residency": args.residency,
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
