"""UPMEM runs of the Llama-family INT8 prefill, the PIM path of the CPU experiments.

The entry scripts in this directory choose the DPU runtime options and the Inductor
settings of the CPU partitions and call main(). llama runs every checkpoint layer of
e2e/llama_workloads.py on a prompt of --seq tokens up to the first greedy token: compile_model tunes and compiles
the layers before the DPUs are opened, then --repeat requests run; request 0 is cold
and includes the first weight push. Every request records its per-layer time and a
breakdown into CPU, weight upload, program load and the native stages of every PIM
call (h2d, kernel, d2h, ...), in <report-dir>/report.json. The same numbers also go to
two CSV files in <report-dir>, laid out like PIM-LLM's export_run_llm_breakdown.py:
  forward_breakdown.csv  one row per request (iter 0 is cold) and warm_mean: the time
                         from the embedding to the first token (first_token_ms, without
                         tokenization) in PIM-LLM's items h2d (with the weight upload),
                         d2h, pim (kernel) and cpu (the rest), their shares, then every
                         breakdown item as detail_* (they add up to ttft_ms)
  gemm.csv               one row per PIM partition, the mean of the warm requests
"""

import argparse
import csv
import ctypes
from dataclasses import asdict
from functools import partial
import json
import os
from pathlib import Path
import statistics
import sys
import time

UPMEM_EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(UPMEM_EXAMPLES.parents[2]))
sys.path.insert(0, str(UPMEM_EXAMPLES))

import torch
from torch._inductor import config as inductor_config
import triton
import triton.backends.pim.driver as pim_driver

from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH
from torch_backend.model import _LayerGemm
from e2e.benchmark_model_llama_ttft import ModelPrefillLayer
from e2e.llama_workloads import KV_MODES


NATIVE_STAGES = pim_driver._STAGE_NAMES["upmem"]
# 전송·launch·split-K 합산을 뺀 native 구간: host pool 준비와 layout 변환
HOST_LAYOUT_STAGES = tuple(name for name in NATIVE_STAGES
                           if name not in ("h2d", "kernel", "d2h", "reduction"))


def pimllm_items(total_ms, breakdown):
    """total_ms in PIM-LLM's items: h2d (with the weight upload), d2h, pim (kernel), cpu (the rest)."""
    h2d = breakdown["h2d_ms"] + breakdown["weight_upload_ms"]
    items = {"h2d_ms": h2d, "d2h_ms": breakdown["d2h_ms"], "pim_ms": breakdown["kernel_ms"],
             "cpu_ms": total_ms - h2d - breakdown["d2h_ms"] - breakdown["kernel_ms"]}
    items.update({name.replace("_ms", "_pct"): 100 * value / total_ms for name, value in list(items.items())})
    return items


def mean_row(label, rows):
    """The mean of every numeric column of rows; the first column holds label."""
    first, *names = rows[0]
    return {first: label, **{name: statistics.mean(row[name] for row in rows) for name in names}}


def gemm_rows(records):
    """One row per PIM partition: the mean over records, PartitionTimer.partitions of each run."""
    rows = []
    for name, record in records[0].items():
        if record["where"] != "PIM":
            continue
        m, n, k = record["mnk"]

        def mean(value):
            return statistics.mean(value(run[name]) for run in records)

        native = {stage: mean(lambda run: run["native_ms"][stage]) for stage in NATIVE_STAGES}
        rows.append({
            "partition": name, "stage": record["stage"], "m": m, "n": n, "k": k,
            "instances": record["instances"], "calls": mean(lambda run: run["calls"]),
            "wall_ms": mean(lambda run: run["wall_ms"]),
            "h2d_ms": native["h2d"], "kernel_ms": native["kernel"], "d2h_ms": native["d2h"],
            "reduction_ms": native["reduction"],
            "host_layout_ms": sum(native[stage] for stage in HOST_LAYOUT_STAGES),
        })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: f"{value:.3f}" if isinstance(value, float) else value
                             for name, value in row.items()})


class PartitionTimer:
    """Wall time of every top-level partition and the native stage times of every PIM call.

    A stage library resets pim_timing_ms when its host function is entered, so the
    forward hook reads it right after the call. Partitions keep their submodule name,
    which is the same in every layer, so the totals sum over the layers.
    """

    def __init__(self, plans):
        self.partitions = {}
        for plan in plans:
            for name, module in plan.graph.named_children():
                stages = [child for child in module.modules() if isinstance(child, _LayerGemm)]
                if len(stages) > 1:
                    raise ValueError(f"layer {plan.index} {name} holds {len(stages)} PIM stages")
                stage = stages[0] if stages else None
                timing = None if stage is None else (
                    (ctypes.c_double * len(NATIVE_STAGES)).in_dll(stage.library, "pim_timing_ms"))
                module.register_forward_pre_hook(self.start)
                module.register_forward_hook(partial(self.stop, name, stage, timing))

    def reset(self):
        self.partitions = {}

    def start(self, module, args):
        self.started = time.perf_counter_ns()

    def stop(self, name, stage, timing, module, args, output):
        elapsed = (time.perf_counter_ns() - self.started) / 1e6
        if name not in self.partitions:
            self.partitions[name] = {"where": "CPU", "calls": 0, "wall_ms": 0.0}
            if stage is not None:
                self.partitions[name].update(
                    where="PIM", stage=stage.name, mnk=list(stage.shape),
                    instances=stage.instance_count, native_ms=dict.fromkeys(NATIVE_STAGES, 0.0))
        record = self.partitions[name]
        record["calls"] += 1
        record["wall_ms"] += elapsed
        if stage is not None:
            for stage_name, value in zip(NATIVE_STAGES, timing, strict=True):
                record["native_ms"][stage_name] += value


def run_request(session, model, tokenizer, prompt, seq, timer):
    timer.reset()
    before = session.stats()
    start = time.perf_counter_ns()
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq)["input_ids"]
    if tuple(input_ids.shape) != (1, seq):
        raise ValueError(f"the prompt must contain at least {seq} tokens including special tokens")
    tokenized_at = time.perf_counter_ns()
    hidden = model.model.embed_tokens(input_ids)
    embedded_at = time.perf_counter_ns()

    # layer 시간에는 그 layer 진입 때의 weight push(enter_layer)도 들어간다
    layer_ms = []
    layer_start = embedded_at

    def after_layer(index, outputs):
        nonlocal layer_start
        now = time.perf_counter_ns()
        layer_ms.append((now - layer_start) / 1e6)
        layer_start = now

    hidden, cache = session.run_model("prefill", hidden, (), after_layer=after_layer)
    head_start = time.perf_counter_ns()
    token_id = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1).item()
    token_ready = time.perf_counter_ns()
    after = session.stats()
    transfers = {name: after[name] - before[name] for name in before}

    partitions = timer.partitions
    native = {name: sum(record["native_ms"][name] for record in partitions.values()
                        if record["where"] == "PIM")
              for name in NATIVE_STAGES}
    tokenization_ms = (tokenized_at - start) / 1e6
    embedding_ms = (embedded_at - tokenized_at) / 1e6
    head_ms = (token_ready - head_start) / 1e6
    ttft_ms = (token_ready - start) / 1e6
    cpu_partitions_ms = sum(record["wall_ms"] for record in partitions.values()
                            if record["where"] == "CPU")
    breakdown = {
        "cpu_ms": tokenization_ms + embedding_ms + head_ms + cpu_partitions_ms,
        # 요청 경로에서 group weight를 host에 pack(또는 background pack을 대기)하고 push한 시간
        "weight_upload_ms": transfers["pack_wall_ms"] + transfers["upload_wall_ms"],
        "program_load_ms": transfers["program_load_ms"],
        "kv_push_ms": transfers["kv_push_ms"],
        "h2d_ms": native["h2d"],
        "kernel_ms": native["kernel"],
        "d2h_ms": native["d2h"],
        "reduction_ms": native["reduction"],
        "host_layout_ms": sum(native[name] for name in HOST_LAYOUT_STAGES),
        "invoke_outside_native_ms": transfers["invoke_ms"] - sum(native.values()),
    }
    # Python, partition 사이 FX 처리, _LayerGemm의 입력 contiguous·출력 할당, exit_layer
    breakdown["other_ms"] = ttft_ms - sum(breakdown.values())

    return {
        "tokenization_ms": tokenization_ms,
        "embedding_ms": embedding_ms,
        "layer_ms": layer_ms,
        # decoder layer 구간 합 (embedding 뒤부터 마지막 layer까지, weight push 포함)
        "layers_ms": sum(layer_ms),
        "head_ms": head_ms,
        "model_to_token_ms": (token_ready - tokenized_at) / 1e6,
        "ttft_ms": ttft_ms,
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id]),
        "kv_cache_bytes": sum(tensor.numel() * tensor.element_size()
                              for pair in cache for tensor in pair),
        "cpu_partitions_ms": cpu_partitions_ms,
        "native_ms": native,
        "breakdown": breakdown,
        "transfers": transfers,
        "partitions": partitions,
    }


def run_llama(args, inductor_overrides, runtime_options):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    preparation_start = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint), local_files_only=True, dtype=torch.float32, attn_implementation="eager",
    ).cpu().eval().requires_grad_(False)
    config = model.config
    example = torch.zeros((1, args.seq, config.hidden_size), dtype=torch.float32)
    cosine, sine = model.model.rotary_emb(example, position_ids=torch.arange(args.seq).unsqueeze(0))
    causal_mask = torch.full(
        (args.seq, args.seq), torch.finfo(torch.float32).min, dtype=torch.float32,
    ).triu(1)
    blocks = [ModelPrefillLayer(layer, config, cosine, sine, causal_mask, args.kv).eval()
              for layer in model.model.layers]
    preparation_ms = (time.perf_counter_ns() - preparation_start) / 1e6

    options = {**runtime_options, "local_capacity_limit_bytes": args.local_capacity_limit_bytes}
    compile_start = time.perf_counter_ns()
    compiled = compile_model(
        {"prefill": (blocks, (example, ()))}, target="upmem", units=args.units,
        runtime_options=options, cache_path=args.cache,
        max_benchmarks=args.max_benchmarks, max_generations=args.max_generations,
    )
    compile_ms = (time.perf_counter_ns() - compile_start) / 1e6
    phase_names, phase_seconds = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    layout = compiled.session.memory.phases["prefill"]
    print(f"preparation_ms={preparation_ms:.3f} compile_ms={compile_ms:.3f} "
          f"slots={layout.slots} groups={layout.groups}", flush=True)

    prompt = args.prompt_file.read_text(encoding="utf-8")
    with compiled.session as session:
        plans = session.plans["prefill"]
        timer = PartitionTimer(plans)
        requests = []
        # 측정 중에 재컴파일이 일어나면 TTFT에 컴파일이 섞이므로 오류를 낸다
        with torch._dynamo.config.patch(error_on_recompile=True):
            for index in range(args.repeat):
                print(f"Executing TTFT request {index}: tokenize -> {len(plans)} layers -> first token",
                      flush=True)
                requests.append(run_request(session, model, tokenizer, prompt, args.seq, timer))
                request = requests[-1]
                print(f"request={index} TTFT_ms={request['ttft_ms']:.3f} "
                      f"first_token_id={request['token_id']} text={request['token_text']!r}", flush=True)
                print("  " + " ".join(f"{name}={value:.3f}" for name, value in request["breakdown"].items()),
                      flush=True)
        for name, record in requests[-1]["partitions"].items():
            if record["where"] == "PIM":
                print(f"  {name} stage={record['stage']} mnk={record['mnk']} calls={record['calls']} "
                      f"wall_ms={record['wall_ms']:.3f} h2d_ms={record['native_ms']['h2d']:.3f} "
                      f"kernel_ms={record['native_ms']['kernel']:.3f} "
                      f"d2h_ms={record['native_ms']['d2h']:.3f}", flush=True)
        target = asdict(session.spec)

    warm_ms = [request["ttft_ms"] for request in requests[1:]]
    report = {
        "scope": "prefill requests to the first greedy token ID; no decode steps",
        "measurement_scope": (
            "tokenize, embedding, every layer (weight push when the group is not resident, "
            "CPU partitions, PIM stages), KV kept on CPU, last-token norm/lm_head/argmax; "
            "excludes checkpoint load, quantization, model compile and resource open/close"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt_file": str(args.prompt_file.resolve()),
        "batch_size": 1, "sequence_length": args.seq, "layers": len(blocks), "kv": args.kv,
        "gemm_dtype": "int8 x int8 -> int32", "cpu_dtype": "float32",
        "mode": f"upmem+inductor-{inductor_overrides['cpu_backend']}",
        "inductor_overrides": inductor_overrides, "runtime_options": options, "units": args.units,
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", ""),
        "torch_threads": torch.get_num_threads(),
        "versions": {"torch": torch.__version__, "triton": triton.__version__},
        "target": target,
        "slots": layout.slots, "groups": layout.groups,
        "native_stages": list(NATIVE_STAGES),
        "preparation_ms": preparation_ms,
        "compile": {
            "compile_ms": compile_ms,
            "phases_s": {name: float(seconds) for name, seconds in zip(phase_names, phase_seconds)},
            "model_compile": compiled.report,
        },
        "resource_open_ms": session.open_ms, "resource_close_ms": session.close_ms,
        "cold_ttft_ms": requests[0]["ttft_ms"],
        "warm_ttft_ms": {"mean": sum(warm_ms) / len(warm_ms), "min": min(warm_ms), "max": max(warm_ms)},
        "requests": requests,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = [{"iter": index, "seq": args.seq, "ttft_ms": request["ttft_ms"],
             "first_token_ms": request["model_to_token_ms"], "layers_ms": request["layers_ms"],
             **pimllm_items(request["model_to_token_ms"], request["breakdown"]),
             **{"detail_" + name: value for name, value in request["breakdown"].items()}}
            for index, request in enumerate(requests)]
    write_csv(args.report_dir / "forward_breakdown.csv", rows + [mean_row("warm_mean", rows[1:])])
    write_csv(args.report_dir / "gemm.csv", gemm_rows([request["partitions"] for request in requests[1:]]))
    print(f"Execution report saved to {(args.report_dir / 'report.json').resolve()} "
          f"(+ forward_breakdown.csv, gemm.csv)", flush=True)


def main(description, inductor_overrides, runtime_options):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("workload", choices=("llama",))
    # 0번은 cold(weight 첫 push), 1번부터 warm
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=UPMEM_EXAMPLES / "e2e" / "prompt.txt")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--seq", type=int, default=1024, help="prompt tokens")
    parser.add_argument("--kv", choices=KV_MODES, default="share", help="how a query head reads its KV head")
    parser.add_argument("--units", type=int, default=2048)
    parser.add_argument("--local-capacity-limit-bytes", type=int, required=True,
                        help="WRAM bytes per DPU for Block Group buffers; the rest holds tasklet stacks")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--max-benchmarks", type=int, default=20, help="per stage class")
    parser.add_argument("--max-generations", type=int, default=10)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("--repeat must be at least 2: one cold and one warm request")

    torch.manual_seed(0)
    # 재컴파일 한도를 넘으면 eager로 넘어가지 않고 오류를 낸다
    with torch.no_grad(), inductor_config.patch(inductor_overrides), \
            torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        run_llama(args, inductor_overrides, runtime_options)
