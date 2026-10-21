"""Decode-shaped W8A8 GEMMs (one token) on UPMEM as GEMVs, against the CPU.

One token x (1 × hidden) INT8 goes through q/k/v projections that share it (packed into
one weight GEMM) and an output projection; one head of q then scores a key cache
(positions × head_dim) and its probabilities weight a value cache, per head. Every GEMM
has one output row (x·W, p·V) or one output column (K·q), so it runs as a GEMV
(execution.gemm_problem). The INT32 outputs must equal the CPU's _int_mm exactly.

  ../cmd.sh verify_decode_gemv.py --local-capacity-limit-bytes <bytes>
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from torch_backend import compile_model
from torch_backend._autotuner import DEFAULT_CACHE_PATH


class DecodeGemvs(torch.nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        # CheckpointInt8Linear처럼 weight를 (in, out)으로 둔다.
        for name in ("q", "k", "v", "o"):
            self.register_buffer(name, torch.randint(-8, 8, (hidden, hidden), dtype=torch.int8))

    def forward(self, token, key_cache, value_cache):
        q = torch._int_mm(token, self.q)
        k = torch._int_mm(token, self.k)
        v = torch._int_mm(token, self.v)
        heads = (q + k + v).clamp(-127, 127).to(torch.int8).reshape(self.heads, self.head_dim)
        # 한 열 GEMM: 위치마다 key · query
        scores = [torch._int_mm(key_cache[head], heads[head].reshape(self.head_dim, 1).contiguous())
                  for head in range(self.heads)]
        # 한 행 GEMM: 확률 · value
        outputs = [torch._int_mm(score.reshape(1, -1).clamp(-127, 127).to(torch.int8),
                                 value_cache[head])
                   for head, score in enumerate(scores)]
        context = torch.cat(outputs, dim=1).clamp(-127, 127).to(torch.int8)
        return torch._int_mm(context, self.o), torch.stack(scores), q


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--positions", type=int, default=1152, help="key/value cache token positions")
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--max-benchmarks", type=int, default=20, help="per stage class")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--local-capacity-limit-bytes", type=int, required=True,
                        help="WRAM bytes per DPU for Block Group buffers; the rest holds tasklet stacks")
    args = parser.parse_args()

    torch.manual_seed(0)
    head_dim = args.hidden // args.heads
    with torch.no_grad():
        module = DecodeGemvs(args.hidden, args.heads).eval()
        inputs = (torch.randint(-8, 8, (1, args.hidden), dtype=torch.int8),
                  torch.randint(-8, 8, (args.heads, args.positions, head_dim), dtype=torch.int8),
                  torch.randint(-8, 8, (args.heads, args.positions, head_dim), dtype=torch.int8))
        reference = module(*inputs)
        compiled = compile_model(
            {"decode": ([module], tuple(torch.zeros_like(tensor) for tensor in inputs))},
            target="upmem", units=2048,
            runtime_options={"nr_tasklets": 16,
                             "local_capacity_limit_bytes": args.local_capacity_limit_bytes},
            cache_path=args.cache,
            max_benchmarks=args.max_benchmarks, max_generations=args.max_generations,
        )
        session = compiled.session
        plan, = session.plans["decode"]
        for stage in plan.stages:
            config = stage.config
            print(f"stage={stage.name} mnk={tuple(stage.shape)} instances={stage.instance_count} "
                  f"p_b={stage.p_b} weight_mask={stage.weight_mask} "
                  f"P=({config.p_m},{config.p_n},{config.p_k})", flush=True)
        with session:
            outputs = session.run_model("decode", *inputs)
    for name, output, expected in zip(("output", "scores", "q"), outputs, reference):
        print(f"{name}: shape={tuple(output.shape)} exact={torch.equal(output, expected)}", flush=True)
    assert all(torch.equal(output, expected) for output, expected in zip(outputs, reference))
    print("decode GEMVs match the CPU exactly", flush=True)


if __name__ == "__main__":
    main()
