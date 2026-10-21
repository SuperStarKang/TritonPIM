"""Generate the same request with the CPU layers and compare the token IDs of a PIM report.

The prefill and decode layers of e2e/llama_generation.py run eagerly on the CPU
(torch._int_mm for every GEMM) through the same loop as
benchmark_model_llama_generate.py, with the report's checkpoint, prompt, kv mode and
token count. The INT32 GEMM results are the same on both sides; the FP32 parts on the
PIM side are compiled by Inductor, so a token whose top two logits are almost equal can
still differ. Prints the first position that differs, if any.

  python check_llama_generate_cpu.py --report <report-dir>/report.json
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2e.llama_generation import PROMPT_TOKENS, generate, generation_layers, run_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True, help="report.json of benchmark_model_llama_generate.py")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    checkpoint = report["checkpoint"]

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)
        prefill_layers, decode_layers = generation_layers(model, report["kv"])
        input_ids = tokenizer(Path(report["prompt_file"]).read_text(encoding="utf-8"), return_tensors="pt",
                              truncation=True, max_length=PROMPT_TOKENS)["input_ids"]
        result = generate(model, input_ids, report["max_new_tokens"],
                          lambda *inputs: run_layers(prefill_layers, *inputs),
                          lambda *inputs: run_layers(decode_layers, *inputs))

    pim = report["token_ids"]
    cpu = result["tokens"]
    print(f"cpu first_token_ms={result['first_token_ms']:.1f} "
          f"step_ms_median={statistics.median(result['step_ms']):.1f}", flush=True)
    print(f"cpu token_ids={cpu}", flush=True)
    print(f"cpu text={tokenizer.decode(cpu)!r}", flush=True)
    differing = [position for position, (left, right) in enumerate(zip(pim, cpu, strict=True)) if left != right]
    if differing:
        position = differing[0]
        print(f"first differing token {position}: pim={pim[position]} cpu={cpu[position]}; "
              f"{len(differing)} of {len(cpu)} differ", flush=True)
    assert not differing
    print(f"all {len(cpu)} tokens match the PIM report", flush=True)


if __name__ == "__main__":
    main()
