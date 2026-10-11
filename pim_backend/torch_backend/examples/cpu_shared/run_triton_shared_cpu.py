"""Run the INT8 attention/transformer/Llama-2-7b prefill examples on the triton_shared CPU backend.

Every Inductor kernel, including each torch._int_mm, is generated as a Triton kernel and
compiled by triton_shared (TTIR -> linalg -> LLVM). No PIM driver or DPU is used.
attention-kernel/attention-layer/transformer use the same synthetic inputs as the PIM
benchmarks and print the difference from eager ATen. llama runs the 32 checkpoint
layers of e2e/torch_llama_ttft.py up to the first greedy token.
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
from torch._inductor import config as inductor_config
from triton.backends.triton_shared.driver import CPUDriver

from attention.int8_workloads import HEAD_DIM, HIDDEN, SEQUENCE, AttentionKernel, AttentionLayer
from transformer.int8_workloads import TransformerBlock


INDUCTOR_OVERRIDES = {
    # pointwise/reduction 커널을 C++ 대신 Triton으로 생성한다
    "cpu_backend": "triton",
    # _int_mm 후보에서 aten을 빼고 Triton mm template만 남긴다
    "max_autotune_gemm": True,
    "max_autotune_gemm_backends": "TRITON",
    # compile worker subprocess에는 CPUDriver가 active로 설정되지 않는다
    "compile_threads": 1,
    "cpp_wrapper": False,
}


def compile_cpu(module):
    return torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)


def run_block(module, inputs, repeat):
    compiled = compile_cpu(module)
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


def run_llama(checkpoint, prompt_file):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from e2e.torch_llama_ttft import LlamaPrefillLayer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.float32, attn_implementation="eager",
    ).cpu().eval().requires_grad_(False)
    example = torch.zeros((1, SEQUENCE, HIDDEN), dtype=torch.float32)
    cosine, sine = model.model.rotary_emb(example, position_ids=torch.arange(SEQUENCE).unsqueeze(0))
    causal_mask = torch.full(
        (SEQUENCE, SEQUENCE), torch.finfo(torch.float32).min, dtype=torch.float32,
    ).triu(1)
    layers = [compile_cpu(LlamaPrefillLayer(layer, model.config, cosine, sine, causal_mask).eval())
              for layer in model.model.layers]

    # 컴파일을 TTFT 측정 밖으로 뺀다
    start = time.perf_counter()
    layers[0](example)
    print(f"compile_and_warmup_s={time.perf_counter() - start:.3f}", flush=True)

    input_ids = tokenizer(prompt_file.read_text(encoding="utf-8"), return_tensors="pt",
                          truncation=True, max_length=SEQUENCE)["input_ids"]
    if tuple(input_ids.shape) != (1, SEQUENCE):
        raise ValueError(f"the prompt must contain at least {SEQUENCE} tokens including special tokens")

    start = time.perf_counter()
    hidden = model.model.embed_tokens(input_ids)
    for index, layer in enumerate(layers):
        hidden, _, _ = layer(hidden)
        print(f"layer={index} elapsed_s={time.perf_counter() - start:.3f}", flush=True)
    logits = model.lm_head(model.model.norm(hidden[:, -1, :]))
    token_id = logits.argmax(dim=-1).item()
    print(f"TTFT_s={time.perf_counter() - start:.3f} first_token_id={token_id} "
          f"text={tokenizer.decode([token_id])!r}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("attention-kernel", "attention-layer", "transformer", "llama"))
    parser.add_argument("--heads", type=int, choices=(1, 32), default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt-file", type=Path,
                        default=Path(__file__).resolve().parents[1] / "e2e" / "prompt.txt")
    args = parser.parse_args()
    if args.workload == "llama" and args.checkpoint is None:
        parser.error("llama requires --checkpoint")

    torch.manual_seed(0)
    triton.runtime.driver.set_active(CPUDriver())
    # 재컴파일 한도를 넘으면 eager로 넘어가지 않고 오류를 낸다
    with torch.no_grad(), inductor_config.patch(INDUCTOR_OVERRIDES), \
            torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        if args.workload == "attention-kernel":
            inputs = tuple(torch.randint(-8, 8, (args.heads, SEQUENCE, HEAD_DIM), dtype=torch.int8)
                           for _ in range(3))
            run_block(AttentionKernel(args.heads).eval(), inputs, args.repeat)
        elif args.workload == "attention-layer":
            run_block(AttentionLayer().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args.repeat)
        elif args.workload == "transformer":
            run_block(TransformerBlock().eval(), (torch.randn(1, SEQUENCE, HIDDEN),), args.repeat)
        else:
            run_llama(args.checkpoint, args.prompt_file)


if __name__ == "__main__":
    main()
