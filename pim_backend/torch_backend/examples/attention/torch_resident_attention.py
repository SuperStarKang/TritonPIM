"""Measure FP32 single-head attention with resident Q/K/V projection weights.

Shape: sequence/KV length 1024, input/head dimension 4096, one head.
Three warm-ups precede three uninstrumented samples and one partition profile.
Progress goes to stderr and the final JSON report goes to stdout.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import triton
import triton.backends.pim.driver as pim_driver

from autotuner import PIMConfig
from torch_backend import PIMTorchConfig, ResidentGemmConfig, prepare_block, target_spec
from torch_backend.fx_analysis import analyze_fx_gemms
from torch_backend.resident import _ResidentGemm


class SingleHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float32)
        self.k = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float32)
        self.v = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float32)

    def forward(self, activation):
        q = self.q(activation)
        k = self.k(activation)
        v = self.v(activation)
        scores = torch.mm(q, k.t()) / 64.0
        probability = torch.softmax(scores, dim=-1)
        return torch.mm(probability, v)


def check_run(block, output, reference, prepared_weight_bytes):
    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
    assert block.last_transfers["weight_bytes"] == 0
    assert block.last_transfers["activation_bytes"] > 0
    assert block.last_transfers["output_bytes"] > 0
    assert block.last_transfers["activation_hits"] == 0
    assert block.last_transfers["launches"] == 5
    assert block.stats()["weight_bytes"] == prepared_weight_bytes


def timed_partition(name, module, records):
    forward = module.forward
    is_pim = isinstance(module, _ResidentGemm)
    operators = ([module.name] if is_pim else [
        str(node.target)
        for node in module._orig_mod.graph.nodes
        if node.op == "call_function"
    ])

    def measured(*args, **kwargs):
        before = module.runtime.stats() if is_pim else None
        start = time.perf_counter_ns()
        output = forward(*args, **kwargs)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        record = {
            "partition": name,
            "where": "CPU+UPMEM" if is_pim else "CPU",
            "operators": operators,
            "wall_ms": elapsed_ms,
        }
        if is_pim:
            record["gemm"] = module.name
            record["native_ms"] = dict(zip(
                pim_driver.TIMING_STAGES, pim_driver.timing, strict=True))
            record["h2d_ms"] = dict(pim_driver.resident_h2d_timing)
            after = module.runtime.stats()
            record["transfers"] = {key: after[key] - before[key] for key in before}
            record["outside_native_stages_ms"] = elapsed_ms - sum(record["native_ms"].values())
        records.append(record)
        return output

    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation", choices=("normal", "dual"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    target = target_spec("upmem", num_units=2048, nr_tasklets=16)
    tiles_by_shape = {
        (1024, 4096, 4096): (2, 4, 128),
        (1024, 1024, 4096): (2, 1, 128),
        (1024, 4096, 1024): (2, 4, 32),
    }

    with torch.no_grad():
        model = SingleHeadAttention().eval()
        activations = (
            torch.randn(1024, 4096, dtype=torch.float32),
            torch.randn(1024, 4096, dtype=torch.float32),
        )
        print("Exporting attention and computing two CPU references ...",
              file=sys.stderr, flush=True)
        start = time.perf_counter_ns()
        exported = torch.export.export(model, (activations[0],))
        export_ms = (time.perf_counter_ns() - start) / 1e6
        gemms = analyze_fx_gemms(exported)
        assert [gemm.op for gemm in gemms] == ["aten.linear.default"] * 3 + ["aten.mm.default"] * 2
        assert [gemm.right.weight_target for gemm in gemms] == [
            "q.weight", "k.weight", "v.weight", None, None,
        ]
        assert [gemm.mnk for gemm in gemms] == [
            (1024, 4096, 4096), (1024, 4096, 4096), (1024, 4096, 4096),
            (1024, 1024, 4096), (1024, 4096, 1024),
        ]
        references = tuple(model(activation) for activation in activations)

        configs = {}
        for gemm in gemms:
            tile_m, tile_n, tile_k = tiles_by_shape[gemm.mnk]
            if args.orientation == "dual":
                tile_m, tile_n = tile_n, tile_m
            configs[gemm.node] = ResidentGemmConfig(
                torch_config=PIMTorchConfig(
                    spec=target,
                    config=PIMConfig(
                        block_m=16,
                        block_n=16,
                        block_k=32,
                        group_m=1,
                        group_n=1,
                        group_k=1,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        tile_k=tile_k,
                        p_m=32 if args.orientation == "normal" else 64,
                        p_n=64 if args.orientation == "normal" else 32,
                        p_k=1,
                    ),
                ),
                orientation=args.orientation,
                padded_shape=gemm.mnk,
            )
            print(f"{gemm.node}: {gemm.op} mnk={gemm.mnk} "
                  f"B_weight={gemm.right.weight_target}", file=sys.stderr, flush=True)

        print(f"Preparing orientation={args.orientation}, DPUs=2048, tasklets=16 ...",
              file=sys.stderr, flush=True)
        with prepare_block(
            exported, (activations[0],), configs=configs, dpu_count=2048,
        ) as block:
            prepared = block.stats()
            weight_regions = [region for region in block.layout if region["role"] == "weight"]
            assert len(weight_regions) == 3
            assert {region["kernel"] for region in weight_regions} == {
                gemm.node for gemm in gemms if gemm.right.weight_target is not None
            }
            assert prepared["weight_bytes"] == 2048 * sum(
                region["bytes"] for region in weight_regions)
            assert prepared["weight_bytes"] > 0
            assert prepared["launches"] == 0
            assert prepared["activation_bytes"] == 0
            assert prepared["output_bytes"] == 0
            print(f"prepare_ms={block.prepare_ms:.3f} "
                  f"prepare_weight_bytes={prepared['weight_bytes']}",
                  file=sys.stderr, flush=True)

            warmups = []
            for iteration in range(3):
                input_index = iteration % len(activations)
                start = time.perf_counter_ns()
                output = block.run(activations[input_index])
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                check_run(block, output, references[input_index], prepared["weight_bytes"])
                warmups.append({"input": input_index, "wall_ms": wall_ms})
                print(f"warmup={iteration} correctness=PASS wall_ms={wall_ms:.3f} (excluded)",
                      file=sys.stderr, flush=True)
                del output

            samples = []
            for iteration in range(3):
                input_index = iteration % len(activations)
                start = time.perf_counter_ns()
                output = block.run(activations[input_index])
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                check_run(block, output, references[input_index], prepared["weight_bytes"])
                samples.append({
                    "input": input_index,
                    "wall_ms": wall_ms,
                    "graph_ms": block.last_run_ms,
                    "transfers": dict(block.last_transfers),
                })
                print(f"sample={iteration} correctness=PASS wall_ms={wall_ms:.3f} "
                      f"weight_h2d={block.last_transfers['weight_bytes']} "
                      f"activation_h2d={block.last_transfers['activation_bytes']} "
                      f"activation_hits={block.last_transfers['activation_hits']} "
                      f"launches={block.last_transfers['launches']}",
                      file=sys.stderr, flush=True)
                del output

            # 구간별 계측 실행은 일반 실행의 median에 포함하지 않는다.
            records = []
            with ExitStack() as instrumentation:
                for name, module in block.graph.named_children():
                    instrumentation.enter_context(patch.object(
                        module, "forward", timed_partition(name, module, records)))
                start = time.perf_counter_ns()
                output = block.run(activations[0])
                profile_wall_ms = (time.perf_counter_ns() - start) / 1e6
            check_run(block, output, references[0], prepared["weight_bytes"])
            pim_records = [record for record in records if "gemm" in record]
            assert len(pim_records) == 5
            assert {record["gemm"] for record in pim_records} == {gemm.node for gemm in gemms}
            assert all(record["transfers"]["launches"] == 1 for record in pim_records)
            print("partition_profile correctness=PASS PIM_partitions=5 (excluded from median)",
                  file=sys.stderr, flush=True)

            report = {
                "scope": "single-head QKV projection + scaled dot-product attention",
                "shape": {"sequence_length": 1024, "kv_length": 1024,
                          "input_dimension": 4096, "head_dimension": 4096, "heads": 1},
                "dtype": "float32",
                "mask": "none",
                "bias": False,
                "dropout": 0.0,
                "qkv_fusion": False,
                "output_projection": False,
                "seed": 0,
                "orientation": args.orientation,
                "target": asdict(target),
                "versions": {"torch": torch.__version__, "triton": triton.__version__},
                "cpu_threads": torch.get_num_threads(),
                "environment": {
                    name: os.environ[name]
                    for name in ("UPMEM_PROFILE", "UPMEM_BACKEND", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                    if name in os.environ
                },
                "gemms": [
                    {"node": gemm.node, "op": gemm.op, "mnk": gemm.mnk,
                     "weight": gemm.right.weight_target,
                     "config": asdict(configs[gemm.node].torch_config.config)}
                    for gemm in gemms
                ],
                "preparation": {
                    "export_ms": export_ms,
                    "prepare_ms": block.prepare_ms,
                    "weight_prepare_ms": block.weight_prepare_ms,
                    "resident_capture_ms": block.runtime.capture_ms,
                    "resident_upload_ms": block.runtime.upload_ms,
                    "resident_upload_breakdown_ms": block.runtime.upload_breakdown_ms,
                    "transfers": prepared,
                    "layout": block.layout,
                },
                "warmups": warmups,
                "samples": samples,
                "median_wall_ms": statistics.median(sample["wall_ms"] for sample in samples),
                "median_graph_ms": statistics.median(sample["graph_ms"] for sample in samples),
                "profile": {
                    "wall_ms": profile_wall_ms,
                    "outside_partitions_ms": profile_wall_ms - sum(record["wall_ms"] for record in records),
                    "pim_timing_version": pim_driver.TIMING_VERSION,
                    "transfers": dict(block.last_transfers),
                    "partitions": records,
                },
                "correctness": {"status": "PASS", "rtol": 1e-4, "atol": 1e-4},
            }

    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
