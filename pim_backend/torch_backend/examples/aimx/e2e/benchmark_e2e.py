"""Llama-family prefill through the first --layers decoder layers on AiMX: TTFT and per-layer time.

One request = tokenize -> embed -> layers on AiMX -> norm -> lm_head (CPU) -> argmax. The reference
runs the same layers in HF FP32, so the first token is compared for any --layers.
compile_model tunes one layer and compiles every layer with its configs: P-1 if every layer
fits, otherwise the first slots keep one layer each and the rest share the last slot.
--qk-heads/--pv-heads/--kv as
layer/benchmark_layer.py. --seq is the prompt length in tokens.

Besides report.json, <report-dir>/forward_breakdown.csv has one row per request (iter 0 is the
first request) and warm_mean, the mean of the later requests, in the columns of the UPMEM
run_upmem.py CSV: first_token_ms (embedding to the first token, without tokenization) in the
items h2d (weight pack + push, desc_h2d, h2d), pim (invoke), d2h and cpu (the rest), their
shares, then every breakdown item as detail_* (they add up to ttft_ms). gemm.csv (weight GEMMs)
and attention.csv (QK^T and PV) have one row per PIM stage, summed over the layers and averaged
over the later requests: calls, wall time and its native slots.

  python benchmark_e2e.py --checkpoint ~/models/Llama-2-7b-hf --prompt-file ../../upmem/e2e/prompt.txt \
      --layers 4 --seq 512
"""

import argparse
import copy
import csv
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_common import (
    SLOTS, add_device_arguments, compile_session, memory_record, print_request, request_breakdown, stage_records,
)
from llama_workloads import (
    KV_MODES, SEQUENCE, LlamaAimxLayer, load_llama, position_embeddings, prompt_ids, reference_layer,
)

import torch

from torch_backend._autotuner import DEFAULT_CACHE_PATH


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: f"{value:.3f}" if isinstance(value, float) else value
                             for name, value in row.items()})


def stage_rows(stages, kind, samples):
    """One row per stage of kind: the mean over samples of its run calls, summed over the layers."""
    rows = []
    for stage in stages:
        if stage["kind"] != kind:
            continue
        m, n, k = stage["mnk"]
        row = {"stage": stage["name"], "m": m, "n": n, "k": k, "instances": stage["instances"]}
        for field in ("calls", "wall_ms", *SLOTS):
            row[field] = statistics.mean(call["breakdown"]["run:" + stage["name"]][field] for call in samples)
        # native 칸 밖: ctypes 호출, lock, stub 안의 측정되지 않은 구간
        row["other_ms"] = row["wall_ms"] - sum(row[name] for name in SLOTS)
        rows.append(row)
    return rows


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--seq", type=int, default=SEQUENCE, help="prompt tokens")
    parser.add_argument("--qk-heads", type=int, default=1)
    parser.add_argument("--pv-heads", type=int, default=1)
    parser.add_argument("--kv", choices=KV_MODES, default="share")
    parser.add_argument("--repeat", type=int, default=3, help="requests after the first")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    report_dir = args.report_dir or Path(__file__).resolve().with_name(
        f"report_{args.checkpoint.name}_l{args.layers}_s{args.seq}_{args.kv}"
        f"_qk{args.qk_heads}_pv{args.pv_heads}_f{args.fpgas}")
    report_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")

    with torch.no_grad():
        tokenizer, model = load_llama(args.checkpoint)
        cosine, sine = position_embeddings(model, args.seq)
        # lm_head와 최종 norm은 CPU에서 FP32로 계산한다 (lm_head 32000 x 4096, 0.5 GB).
        norm = copy.deepcopy(model.model.norm).float()
        lm_head = copy.deepcopy(model.lm_head).float()
        layers = model.model.layers[:args.layers]

        reference = model.model.embed_tokens(prompt_ids(tokenizer, prompt, args.seq)).float()
        for layer in layers:
            reference = reference_layer(layer, reference, cosine, sine)
        # slice와 loop 변수가 HF layer를 붙잡지 않게 한다. 아래에서 Identity로 바꾸면 바로 해제된다.
        del layers, layer
        reference_logits = lm_head(norm(reference[:, -1, :]))
        reference_token = reference_logits.argmax(dim=-1).item()
        print(f"reference layers={args.layers} token={reference_token} "
              f"{tokenizer.decode([reference_token])!r}", flush=True)

        # HF layer는 BF16 operand를 만든 뒤 바로 놓아 host memory에 한 벌만 둔다.
        blocks = []
        for index in range(args.layers):
            blocks.append(LlamaAimxLayer(model.model.layers[index], cosine, sine,
                                         args.qk_heads, args.pv_heads, args.kv).eval())
            model.model.layers[index] = torch.nn.Identity()

        start = time.perf_counter_ns()
        session, config_report = compile_session(
            blocks, (torch.zeros(1, args.seq, model.config.hidden_size),), args)
        compile_ms = (time.perf_counter_ns() - start) / 1e6
        # plan이 필요한 weight를 직접 참조한다. packing으로 대체된 원본(q/k/v, gate/up)을 해제한다.
        del blocks
        stages = stage_records(session.plans["prefill"][0])
        weight_stages = {stage["name"] for stage in stages if stage["kind"] == "weight"}
        memory = session.memory
        prefill = memory.phases["prefill"]
        print(f"compile_ms={compile_ms:.1f} layers={args.layers} slots={prefill.slots} groups={prefill.groups} "
              f"bytes_per_fpga={memory.bytes_per_unit} capacity={memory.capacity_bytes}", flush=True)
        for stage_class in config_report["phases"]["prefill"]["stage_classes"]:
            print(f"  {stage_class['label']}: config={tuple(stage_class['selected']['config'].values())}", flush=True)

        calls = []
        with session:
            # session open이 slot마다 descriptor chain을 만들어 올린다 (요청 밖의 준비 비용).
            chain_build_ms = session.stats()["chain_build_ms"]
            print(f"session_open_ms={session.open_ms:.1f} chain_build_ms={chain_build_ms:.1f}", flush=True)
            for phase, count in (("first", 1), ("sample", args.repeat)):
                for step in range(count):
                    before = session.stats()
                    start = time.perf_counter_ns()
                    input_ids = prompt_ids(tokenizer, prompt, args.seq)
                    tokenized_at = time.perf_counter_ns()
                    embedded = model.model.embed_tokens(input_ids).float()
                    embedded_at = time.perf_counter_ns()
                    hidden = session.run_model("prefill", embedded)
                    layers_at = time.perf_counter_ns()
                    logits = lm_head(norm(hidden[:, -1, :]))
                    token = logits.argmax(dim=-1).item()
                    end = time.perf_counter_ns()
                    ttft_ms = (end - start) / 1e6
                    call = {
                        "phase": phase, "ttft_ms": ttft_ms,
                        "tokenization_ms": (tokenized_at - start) / 1e6,
                        "embedding_ms": (embedded_at - tokenized_at) / 1e6,
                        "lm_head_ms": (end - layers_at) / 1e6,
                        **request_breakdown(before, session.stats(), weight_stages, (layers_at - embedded_at) / 1e6),
                        "token": token, "token_matches_reference": token == reference_token,
                        "logits_cosine": torch.nn.functional.cosine_similarity(
                            logits.flatten().double(), reference_logits.flatten().double(), dim=0).item(),
                    }
                    call["per_layer_ms"] = call["layers_ms"] / args.layers
                    call["ttft_without_group0_load_ms"] = ttft_ms - call["group0_load_ms"]
                    calls.append(call)
                    print_request(f"{phase}={step} ttft_ms={ttft_ms:.1f} "
                                  f"ttft_without_group0_load_ms={call['ttft_without_group0_load_ms']:.1f} "
                                  f"per_layer_ms={call['per_layer_ms']:.1f} "
                                  f"token={token} matches={call['token_matches_reference']} "
                                  f"logits_cosine={call['logits_cosine']:.6f}", call)
            # slot buffer는 첫 실행에서 할당되므로 요청이 끝난 뒤 읽는다.
            memory_table = session.memory_table()

    summary = {}
    # --repeat 0이면 단일 요청만 잰다
    samples = [call for call in calls if call["phase"] == "sample"]
    if samples:
        summary = {name: statistics.median(call[name] for call in samples)
                   for name in ("ttft_ms", "layers_ms", "per_layer_ms")}
        summary["kinds"] = {name: statistics.median(call["kinds"][name] for call in samples)
                            for name in samples[0]["kinds"]}
        print(f"summary ttft_ms={summary['ttft_ms']:.1f} per_layer_ms={summary['per_layer_ms']:.1f} "
              f"slots={prefill.slots} groups={prefill.groups}", flush=True)
    # 단일 요청: 첫 요청이 group 0을 포함한 모든 weight 업로드를 한다
    first = next(call for call in calls if call["phase"] == "first")
    summary["first"] = {name: first[name] for name in
                        ("ttft_ms", "group0_load_ms", "ttft_without_group0_load_ms", "layers_ms",
                         "pack_wall_ms", "background_pack_ms", "upload_wall_ms", "weight_uploads")}
    print(f"first ttft_ms={first['ttft_ms']:.1f} group0_load_ms={first['group0_load_ms']:.1f} "
          f"ttft_without_group0_load_ms={first['ttft_without_group0_load_ms']:.1f} "
          f"pack_wall_ms={first['pack_wall_ms']:.1f} background_pack_ms={first['background_pack_ms']:.1f} "
          f"upload_wall_ms={first['upload_wall_ms']:.1f} weight_uploads={first['weight_uploads']} "
          f"slots={prefill.slots} groups={prefill.groups}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint), "layers": args.layers, "fpgas": args.fpgas,
        "base_fpga": args.base_fpga, "sequence": args.seq, "config_source": "autotune",
        "qk_heads": args.qk_heads, "pv_heads": args.pv_heads, "kv": args.kv,
        "reference": {"token": reference_token, "token_text": tokenizer.decode([reference_token])},
        "compile_ms": compile_ms, "session_open_ms": session.open_ms, "session_close_ms": session.close_ms,
        "chain_build_ms": chain_build_ms,
        "memory": memory_record(memory),
        "config_report": config_report, "memory_table": memory_table, "stages": stages,
        "calls": calls, "summary": summary,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rows = []
    for index, call in enumerate(calls):
        run = call["phases"]["run"]
        # 더하면 ttft_ms다. run_* 는 run 호출의 native 칸(FPGA thread 평균), cpu_layers는 layer 구간의 PIM 호출 밖.
        detail = {
            "tokenization_ms": call["tokenization_ms"],
            "embedding_ms": call["embedding_ms"],
            "weight_upload_ms": call["pack_wall_ms"] + call["upload_wall_ms"],
            **{f"run_{name}_ms": run[name] for name in SLOTS},
            "run_other_ms": run["other"],
            "cpu_layers_ms": call["kinds"]["cpu_ms"],
            "lm_head_ms": call["lm_head_ms"],
        }
        first_token_ms = call["ttft_ms"] - call["tokenization_ms"]
        items = {"h2d_ms": detail["weight_upload_ms"] + run["desc_h2d"] + run["h2d"],
                 "pim_ms": run["invoke"], "d2h_ms": run["d2h"]}
        items["cpu_ms"] = first_token_ms - sum(items.values())
        items.update({name.replace("_ms", "_pct"): 100 * value / first_token_ms for name, value in list(items.items())})
        rows.append({"iter": index, "phase": call["phase"], "seq": args.seq, "ttft_ms": call["ttft_ms"],
                     "first_token_ms": first_token_ms, "layers_ms": call["layers_ms"], **items,
                     **{"detail_" + name: value for name, value in detail.items()}})
    if samples:
        names = [name for name in rows[0] if name not in ("iter", "phase", "seq")]
        rows.append({"iter": "warm_mean", "phase": "sample", "seq": args.seq,
                     **{name: statistics.mean(row[name] for row in rows[1:]) for name in names}})
        # weight GEMM(q/k/v/o, gate/up/down)과 attention의 activation GEMM(QK^T, PV)을 나눠 둔다.
        write_csv(report_dir / "gemm.csv", stage_rows(stages, "weight", samples))
        write_csv(report_dir / "attention.csv", stage_rows(stages, "activation", samples))
    write_csv(report_dir / "forward_breakdown.csv", rows)
    print(f"report={report_dir / 'report.json'} (+ forward_breakdown.csv, gemm.csv, attention.csv)", flush=True)


if __name__ == "__main__":
    main()
