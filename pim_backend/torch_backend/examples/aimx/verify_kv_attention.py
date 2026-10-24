"""Decode attention reading a device-resident key/value cache on AiMX: QK and PV against torch BF16.

The host cache is [positions, kv_heads * head_dim], one row per token. FPGA f owns the
consecutive KV heads of column window f, so every cache stage runs one instance per FPGA
in one round and its window stays at one address:
  QK  scores[f]  = q_rows[f] (rows x W) @ key_cache[:, window f].T   B = K window^T, positions on N
  PV  context[f] = p_rows[f] (rows x C) @ value_cache[:, window f]   B = V window, positions on K
rows = heads / fpgas query heads, W = kv_heads * head_dim / fpgas. The rows are random here:
the check is the GEMM over the resident cache, not the attention block structure.

Steps: write the whole cache and compare; overwrite the last --append positions on the host,
write only those and compare again (the device must see the new positions, the rest unchanged).
Each comparison is relative L2 against torch.mm on the same BF16 inputs. --p-scale e multiplies
the probabilities by 2**e (exact in BF16) to move the PV partial sums to another vACC exponent
group; the relative error of an exact power-of-two scale is comparable across e.

  python verify_kv_attention.py --fpgas 2 --positions 2048                  # Llama-2 MHA shape
  python verify_kv_attention.py --fpgas 2 --positions 2048 --kv-heads 8     # Llama-3 GQA shape
"""

import argparse

from experiment_common import add_device_arguments

import torch
from triton.backends.pim.residency import KEY, VALUE

from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH


class Attention(torch.nn.Module):
    def __init__(self, fpgas, width):
        super().__init__()
        self.fpgas = fpgas
        self.width = width

    def forward(self, q_rows, p_rows, key_cache, value_cache):
        scores = tuple(torch.mm(q_rows[f], key_cache.narrow(1, f * self.width, self.width).t())
                       for f in range(self.fpgas))
        context = tuple(torch.mm(p_rows[f], value_cache.narrow(1, f * self.width, self.width))
                        for f in range(self.fpgas))
        return scores, context


def relative_l2(outputs, references):
    return [((got.float() - reference.float()).norm() / reference.float().norm()).item()
            for got, reference in zip(outputs, references)]


def report(label, session, inputs, module, kv_bytes):
    scores, context = session.run_model("decode", *inputs)
    reference_scores, reference_context = module(*inputs)
    totals = session.backend.model.totals
    print(f"{label}: rel_l2 qk={[f'{value:.4%}' for value in relative_l2(scores, reference_scores)]} "
          f"pv={[f'{value:.4%}' for value in relative_l2(context, reference_context)]} "
          f"kv_bytes={totals['kv_bytes'] - kv_bytes} kv_push_ms={totals['kv_push_ms']:.2f} "
          f"kv_layout_check_ms={totals['kv_layout_check_ms']:.2f}", flush=True)
    return totals["kv_bytes"]


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_device_arguments(parser, DEFAULT_CACHE_PATH)
    parser.add_argument("--positions", type=int, default=2048, help="cache capacity C")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--append", type=int, default=16, help="positions rewritten in the second step")
    parser.add_argument("--p-scale", type=int, default=0, help="probabilities times 2**p_scale")
    args = parser.parse_args()
    if args.heads % args.fpgas or args.kv_heads % args.fpgas:
        parser.error("heads and kv-heads must split evenly over the FPGAs")
    rows = args.heads // args.fpgas
    width = args.kv_heads * args.head_dim // args.fpgas

    generator = torch.Generator().manual_seed(0)
    module = Attention(args.fpgas, width).eval()
    key_cache = torch.randn((args.positions, args.kv_heads * args.head_dim), generator=generator).bfloat16()
    value_cache = torch.randn((args.positions, args.kv_heads * args.head_dim), generator=generator).bfloat16()
    inputs = (torch.randn((args.fpgas, rows, width), generator=generator).bfloat16(),
              (torch.softmax(torch.randn((args.fpgas, rows, args.positions), generator=generator), dim=-1)
               * 2.0 ** args.p_scale).bfloat16(),
              key_cache, value_cache)
    print(f"fpgas={args.fpgas} positions={args.positions} heads={args.heads} kv_heads={args.kv_heads} "
          f"rows={rows} window={width}: qk ({rows},{args.positions},{width}) pv ({rows},{width},{args.positions}) "
          f"x{args.fpgas}", flush=True)

    compiled = compile_model(
        {"decode": ([module], inputs)}, target="aimx", units=args.fpgas,
        kv_inputs={"decode": lambda index, layer_inputs: {KEY: layer_inputs[2], VALUE: layer_inputs[3]}},
        runtime_options={"base_fpga": args.base_fpga},
        cache_path=args.cache, max_benchmarks=args.max_benchmarks,
    )
    memory = compiled.session.memory
    print(f"kv_bytes_per_fpga={memory.kv_bytes} kv_base={memory.kv_base} capacity={memory.capacity_bytes}",
          flush=True)
    for plan in compiled.session.plans["decode"]:
        for stage in plan.stages:
            print(f"  stage {stage.name} mnk={tuple(stage.shape)} x{stage.instance_count} kv={stage.kv} "
                  f"windows={stage.kv_windows} p_b={stage.p_b} rounds={stage.batch_rounds} "
                  f"config={stage.config.key()}", flush=True)

    with compiled.session as session:
        session.write_kv(0, KEY, key_cache, 0, args.positions)
        session.write_kv(0, VALUE, value_cache, 0, args.positions)
        kv_bytes = report("whole cache", session, inputs, module, 0)

        # 마지막 위치만 host에서 바꾸고 그 위치만 쓴다. 장치는 새 값과 나머지 옛 값을 함께 읽어야 한다.
        start = args.positions - args.append
        key_cache[start:] = torch.randn((args.append, key_cache.shape[1]), generator=generator).bfloat16()
        value_cache[start:] = torch.randn((args.append, value_cache.shape[1]), generator=generator).bfloat16()
        session.write_kv(0, KEY, key_cache, start, args.positions)
        session.write_kv(0, VALUE, value_cache, start, args.positions)
        report(f"append {args.append}", session, inputs, module, kv_bytes)


if __name__ == "__main__":
    main()
