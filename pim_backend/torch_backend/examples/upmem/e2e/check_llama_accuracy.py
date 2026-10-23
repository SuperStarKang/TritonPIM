"""Where the W8A8 Llama layers of e2e/llama_generation.py lose the FP32 model, on the CPU only.

C2  decode against prefill: the prefill of the 1024 prompt tokens gives the key/value
    cache of positions 0..1022 (causal attention, so it equals a 1023-token prefill);
    token 1023 then goes through the decode layers at position 1023. Its last hidden
    must equal the last prefill row up to FP32 rounding, with the same next token.
C3b per layer against Hugging Face FP32: the hidden after each checkpoint layer of
    one FP32 forward, against the W8A8 layer fed the same FP32 input (isolated) and
    against the W8A8 layers chained from the embedding (chained).
C4  per layer, the FP32 inputs of the projections in that forward (q/k/v share one,
    gate/up share one): absmax, and the share of elements the earlier fixed scale
    (1/32, range +-127/32) clipped.
C3a greedy tokens of the Hugging Face FP32 model for the same prompt.

  python check_llama_accuracy.py --checkpoint <dir> --prompt-file prompt.txt
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2e.llama_generation import PROMPT_TOKENS, decode_inputs, generation_layers, run_layers
from e2e.llama_workloads import KV_MODES


def compare(value, reference):
    value = value.flatten().double()
    reference = reference.flatten().double()
    relative = ((value - reference).norm() / reference.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(value, reference, dim=0).item()
    return relative, cosine


# 이전 고정 scale 1/32의 표현 범위
FIXED_RANGE = 127 / 32
# (출력 이름, Hugging Face layer 안의 projection 이름)
PROJECTION_INPUTS = (("qkv_in", "self_attn.q_proj"), ("o_in", "self_attn.o_proj"),
                     ("gate_up_in", "mlp.gate_proj"), ("down_in", "mlp.down_proj"))


def next_token(model, hidden):
    return model.lm_head(model.model.norm(hidden[:, -1, :])).argmax(dim=-1).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--kv", choices=KV_MODES, default="share")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="C3a greedy tokens")
    args = parser.parse_args()

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(args.checkpoint), local_files_only=True, dtype=torch.float32, attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)
        input_ids = tokenizer(args.prompt_file.read_text(encoding="utf-8"), return_tensors="pt",
                              truncation=True, max_length=PROMPT_TOKENS)["input_ids"]
        if tuple(input_ids.shape) != (1, PROMPT_TOKENS):
            raise ValueError(f"the prompt must have {PROMPT_TOKENS} tokens, got {tuple(input_ids.shape)}")
        prefill_layers, decode_layers = generation_layers(model, args.kv)
        embedded = model.model.embed_tokens(input_ids)

        # Hugging Face FP32: checkpoint layer마다 출력 hidden을 hook으로 받는다.
        reference = []
        hooks = [layer.register_forward_hook(
                     lambda module, inputs, output: reference.append(
                         output[0] if isinstance(output, tuple) else output))
                 for layer in model.model.layers]
        # (layer, 출력 이름) -> (absmax, 고정 scale이 자른 비율)
        projection_inputs = {}
        for index, layer in enumerate(model.model.layers):
            for name, path in PROJECTION_INPUTS:
                hooks.append(layer.get_submodule(path).register_forward_pre_hook(
                    lambda module, inputs, key=(index, name): projection_inputs.__setitem__(
                        key, (inputs[0].abs().max().item(),
                              (inputs[0].abs() > FIXED_RANGE).float().mean().item()))))
        model(input_ids)
        for hook in hooks:
            hook.remove()
        reference_inputs = [embedded] + reference[:-1]

        print("C4 layer: projection input absmax (share clipped by the fixed scale)", flush=True)
        for index in range(len(model.model.layers)):
            print(f"  layer {index:2d}: " + " ".join(
                f"{name}={projection_inputs[index, name][0]:.1f} ({projection_inputs[index, name][1]:.6f})"
                for name, _ in PROJECTION_INPUTS), flush=True)

        print("C3b layer: isolated (relative error, cosine) | chained (relative error, cosine)", flush=True)
        chained = embedded
        cache = ()
        for index, layer in enumerate(prefill_layers):
            isolated, _ = layer(reference_inputs[index], ())
            chained, cache = layer(chained, cache)
            isolated_error, isolated_cosine = compare(isolated, reference[index])
            chained_error, chained_cosine = compare(chained, reference[index])
            print(f"  layer {index:2d}: isolated ({isolated_error:.4f}, {isolated_cosine:.4f}) | "
                  f"chained ({chained_error:.4f}, {chained_cosine:.4f})", flush=True)
        prefill_hidden = chained
        print(f"C3b next token: fp32={next_token(model, reference[-1])} w8a8={next_token(model, prefill_hidden)}",
              flush=True)

        # C2: 위치 0..1022만 유효한 prefill cache로 token 1023을 decode한다.
        position = PROMPT_TOKENS - 1
        decoded, *_ = run_layers(decode_layers, *decode_inputs(
            model, model.model.embed_tokens(input_ids[:, position:]), position, cache,
            torch.arange(cache[0][0].shape[1])))
        last_row = prefill_hidden[:, position:]
        decode_error, decode_cosine = compare(decoded, last_row)
        print(f"C2 decode vs prefill row {position}: max_abs={(decoded - last_row).abs().max().item():.3e} "
              f"relative={decode_error:.3e} cosine={decode_cosine:.6f} "
              f"next token prefill={next_token(model, last_row)} decode={next_token(model, decoded)}",
              flush=True)

        generated = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                   max_new_tokens=args.max_new_tokens, do_sample=False)[0, PROMPT_TOKENS:]
        print(f"C3a fp32 token_ids={generated.tolist()}", flush=True)
        print(f"C3a fp32 text={tokenizer.decode(generated)!r}", flush=True)


if __name__ == "__main__":
    main()
