"""CPU-only runs of the INT8 attention/transformer/Llama-2-7b prefill examples.

The entry scripts in this directory choose eager PyTorch (inductor_overrides=None) or an
Inductor CPU backend and call main(). attention-kernel/attention-layer/transformer use the
same synthetic inputs as the PIM benchmarks and print the difference from eager ATen.
llama runs the 32 checkpoint layers of upmem/e2e/llama_workloads.py up to the first greedy
token: --repeat requests for TTFT and its per-stage times, then one request under
torch.profiler for the per-stage kernel breakdown. Requests exclude compilation, which is
reported separately with the Dynamo/Inductor phase times, in <report-dir>/report.json.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# 연산 정의(attention/, transformer/, e2e/)는 PIM 예제와 같은 upmem/ 아래 모듈을 쓴다
UPMEM_EXAMPLES = Path(__file__).resolve().parents[1] / "upmem"
sys.path.insert(0, str(UPMEM_EXAMPLES))

import torch
from torch._inductor import config as inductor_config
from torch.profiler import ProfilerActivity, profile, record_function

from attention.int8_workloads import HEAD_DIM, HIDDEN, SEQUENCE, AttentionKernel, AttentionLayer
from transformer.int8_workloads import TransformerBlock


def compile_cpu(module, inductor_overrides):
    if inductor_overrides is None:
        return module
    return torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)


def run_block(module, inputs, repeat, inductor_overrides):
    compiled = compile_cpu(module, inductor_overrides)
    start = time.perf_counter()
    output = compiled(*inputs)
    print(f"compile_and_first_run_s={time.perf_counter() - start:.3f}", flush=True)
    for index in range(repeat):
        start = time.perf_counter()
        compiled(*inputs)
        print(f"sample={index} run_s={time.perf_counter() - start:.3f}", flush=True)
    start = time.perf_counter()
    reference = module(*inputs)
    print(f"eager_aten_s={time.perf_counter() - start:.3f} "
          f"max_abs_diff={(output - reference).abs().max().item():.6g}", flush=True)


def run_request(model, layers, tokenizer, prompt):
    start = time.perf_counter_ns()
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SEQUENCE)["input_ids"]
    if tuple(input_ids.shape) != (1, SEQUENCE):
        raise ValueError(f"the prompt must contain at least {SEQUENCE} tokens including special tokens")
    tokenized_at = time.perf_counter_ns()

    with record_function("embedding"):
        hidden = model.model.embed_tokens(input_ids)
    embedded_at = time.perf_counter_ns()

    layer_ms = []
    cache = []
    for index, layer in enumerate(layers):
        layer_start = time.perf_counter_ns()
        with record_function(f"layer_{index}"):
            hidden, key, value = layer(hidden)
        cache.append((key, value))
        layer_ms.append((time.perf_counter_ns() - layer_start) / 1e6)
        print(f"layer={index} layer_ms={layer_ms[-1]:.3f}", flush=True)

    head_start = time.perf_counter_ns()
    with record_function("head"):
        token_id = model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1).item()
    token_ready = time.perf_counter_ns()

    return {
        "tokenization_ms": (tokenized_at - start) / 1e6,
        "embedding_ms": (embedded_at - tokenized_at) / 1e6,
        "layer_ms": layer_ms,
        "head_ms": (token_ready - head_start) / 1e6,
        "model_to_token_ms": (token_ready - tokenized_at) / 1e6,
        "ttft_ms": (token_ready - start) / 1e6,
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id]),
        "kv_cache_bytes": sum(tensor.numel() * tensor.element_size()
                              for pair in cache for tensor in pair),
    }


def is_kernel(name):
    # eager/C++ backend의 ATen op, Triton 커널 launch, C++ 커널(cpp.enable_kernel_profile)
    return name.startswith(("aten::", "triton_")) or "cpp_fused_" in name


def is_gemm(name):
    return name == "aten::_int_mm" or name.startswith("triton_tem_")


def gemm_roles(heads):
    # LlamaPrefillLayer.forward의 _int_mm 호출 순서다. Inductor도 FX graph 순서대로 launch한다
    attention = [role for _ in range(heads) for role in ("attention_qk", "attention_pv")]
    return ["q_proj", "k_proj", "v_proj", *attention, "o_proj", "gate_proj", "up_proj", "down_proj"]


def breakdown_by_stage(profiler, heads):
    """Sums top-level kernel time per GEMM role and per non-GEMM stage between GEMMs.

    A non-GEMM kernel is charged to after_<previous GEMM role> in its layer, or
    before_q_proj; kernels outside the layers go to embedding/head/outside_regions.
    Kernels fused into a GEMM epilogue are part of that GEMM's time.
    """
    roles = gemm_roles(heads)
    stages = {}
    kernels = {}
    # 같은 layer 안에서 지금까지 만난 GEMM 수
    gemm_seen = {}
    for event in sorted(profiler.events(), key=lambda event: event.time_range.start):
        # 다른 커널 안에서 호출된 ATen op는 부모 커널 시간에 이미 들어 있다
        if not is_kernel(event.name) or (event.cpu_parent is not None and is_kernel(event.cpu_parent.name)):
            continue
        region = event.cpu_parent
        while region is not None and not region.name.startswith(("layer_", "embedding", "head")):
            region = region.cpu_parent
        if region is None:
            # tokenizer가 input_ids tensor를 만드는 ATen op 등
            stage = "outside_regions"
        elif not region.name.startswith("layer_"):
            stage = region.name
        elif is_gemm(event.name):
            stage = roles[gemm_seen.get(region.name, 0)]
            gemm_seen[region.name] = gemm_seen.get(region.name, 0) + 1
        elif region.name in gemm_seen:
            stage = f"after_{roles[gemm_seen[region.name] - 1]}"
        else:
            stage = "before_q_proj"

        milliseconds = event.cpu_time_total / 1e3
        stages.setdefault(stage, {"calls": 0, "total_ms": 0.0})
        stages[stage]["calls"] += 1
        stages[stage]["total_ms"] += milliseconds
        kernels.setdefault(event.name, {"calls": 0, "total_ms": 0.0})
        kernels[event.name]["calls"] += 1
        kernels[event.name]["total_ms"] += milliseconds

    for layer, count in gemm_seen.items():
        if count != len(roles):
            raise ValueError(f"{layer} launched {count} GEMMs, expected {len(roles)}")
    return stages, kernels


def run_llama(checkpoint, prompt_file, report_dir, inductor_overrides, repeat):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from e2e.llama_workloads import LlamaPrefillLayer

    preparation_start = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.float32, attn_implementation="eager",
    ).cpu().eval().requires_grad_(False)
    example = torch.zeros((1, SEQUENCE, HIDDEN), dtype=torch.float32)
    cosine, sine = model.model.rotary_emb(example, position_ids=torch.arange(SEQUENCE).unsqueeze(0))
    causal_mask = torch.full(
        (SEQUENCE, SEQUENCE), torch.finfo(torch.float32).min, dtype=torch.float32,
    ).triu(1)
    layers = [compile_cpu(LlamaPrefillLayer(layer, model.config, cosine, sine, causal_mask).eval(),
                          inductor_overrides)
              for layer in model.model.layers]
    preparation_ms = (time.perf_counter_ns() - preparation_start) / 1e6

    # 32개 layer는 같은 forward 코드를 공유하므로 layer 0 한 번으로 컴파일을 TTFT 밖으로 뺀다
    warmup_start = time.perf_counter_ns()
    layers[0](example)
    compile_and_warmup_ms = (time.perf_counter_ns() - warmup_start) / 1e6
    # dynamo_timed 구간별 누적 초. eager는 컴파일이 없어 비어 있다
    phase_names, phase_seconds = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    compile_phases_s = {name: float(seconds) for name, seconds in zip(phase_names, phase_seconds)}
    print(f"preparation_ms={preparation_ms:.3f} compile_and_warmup_ms={compile_and_warmup_ms:.3f}",
          flush=True)

    prompt = prompt_file.read_text(encoding="utf-8")
    # 측정 중에 재컴파일이 일어나면 TTFT에 컴파일이 섞이므로 오류를 낸다
    with torch._dynamo.config.patch(error_on_recompile=True):
        requests = []
        for index in range(repeat):
            print(f"Executing TTFT request {index}: tokenize -> 32 layers -> first token", flush=True)
            requests.append(run_request(model, layers, tokenizer, prompt))
            print(f"request={index} TTFT_ms={requests[-1]['ttft_ms']:.3f} "
                  f"first_token_id={requests[-1]['token_id']} text={requests[-1]['token_text']!r}",
                  flush=True)

        print("Executing profiled request for the kernel breakdown", flush=True)
        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            profiled = run_request(model, layers, tokenizer, prompt)

    stages, kernels = breakdown_by_stage(profiler, model.config.num_attention_heads)
    kernel_ms = sum(kernel["total_ms"] for kernel in kernels.values())
    gemm_ms = sum(kernel["total_ms"] for name, kernel in kernels.items() if is_gemm(name))
    breakdown = {
        "model_to_token_ms": profiled["model_to_token_ms"],
        "gemm_ms": gemm_ms,
        "other_kernel_ms": kernel_ms - gemm_ms,
        # Python, Inductor wrapper, Triton launcher의 Python 처리
        "outside_kernels_ms": profiled["model_to_token_ms"] - kernel_ms,
        "stages": stages,
        "kernels": dict(sorted(kernels.items(), key=lambda item: -item[1]["total_ms"])),
    }
    print(f"profiled model_to_token_ms={breakdown['model_to_token_ms']:.3f} "
          f"gemm_ms={gemm_ms:.3f} other_kernel_ms={breakdown['other_kernel_ms']:.3f} "
          f"outside_kernels_ms={breakdown['outside_kernels_ms']:.3f}", flush=True)
    for stage, entry in stages.items():
        print(f"  stage={stage} calls={entry['calls']} total_ms={entry['total_ms']:.3f}", flush=True)

    ttft_ms = [request["ttft_ms"] for request in requests]
    report = {
        "scope": "single prefill request to the first greedy token ID; no decode steps",
        "measurement_scope": (
            "tokenize, embedding, 32 layers, KV kept on CPU, last-token norm/lm_head/argmax; "
            "excludes checkpoint load, quantization and compilation"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "prompt_file": str(prompt_file.resolve()),
        "batch_size": 1, "sequence_length": SEQUENCE, "layers": len(layers),
        "gemm_dtype": "int8 x int8 -> int32", "cpu_dtype": "float32",
        "mode": "eager" if inductor_overrides is None else f"inductor-{inductor_overrides['cpu_backend']}",
        "inductor_overrides": inductor_overrides,
        "triton_shared_num_threads": os.getenv("TRITON_SHARED_NUM_THREADS", ""),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", ""),
        "torch_threads": torch.get_num_threads(),
        "versions": {"torch": torch.__version__},
        "preparation_ms": preparation_ms,
        "compile": {
            "compile_and_warmup_ms": compile_and_warmup_ms,
            "phases_s": compile_phases_s,
        },
        "ttft_ms": {"mean": sum(ttft_ms) / len(ttft_ms), "min": min(ttft_ms), "max": max(ttft_ms)},
        "requests": requests,
        "profiled_request": profiled,
        "breakdown": breakdown,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Execution report saved to {(report_dir / 'report.json').resolve()}", flush=True)


def main(description, inductor_overrides):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("workload", choices=("attention-kernel", "attention-layer", "transformer", "llama"))
    parser.add_argument("--heads", type=int, choices=(1, 32), default=1)
    # llama에서는 측정할 TTFT 요청 수다
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt-file", type=Path,
                        default=UPMEM_EXAMPLES / "e2e" / "prompt.txt")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.workload == "llama" and (args.checkpoint is None or args.report_dir is None):
        parser.error("llama requires --checkpoint and --report-dir")

    torch.manual_seed(0)
    # 재컴파일 한도를 넘으면 eager로 넘어가지 않고 오류를 낸다
    with torch.no_grad(), inductor_config.patch(inductor_overrides), \
            torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        if args.workload == "attention-kernel":
            inputs = tuple(torch.randint(-8, 8, (args.heads, SEQUENCE, HEAD_DIM), dtype=torch.int8)
                           for _ in range(3))
            run_block(AttentionKernel(args.heads).eval(), inputs, args.repeat, inductor_overrides)
        elif args.workload == "attention-layer":
            run_block(AttentionLayer().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args.repeat,
                      inductor_overrides)
        elif args.workload == "transformer":
            run_block(TransformerBlock().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args.repeat,
                      inductor_overrides)
        else:
            run_llama(args.checkpoint, args.prompt_file, args.report_dir, inductor_overrides,
                      args.repeat)
