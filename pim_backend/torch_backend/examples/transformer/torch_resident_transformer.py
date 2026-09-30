"""Measure 1, 2 or 4 FP32 single-head Transformer blocks in one resident graph.

Sequence length 1024, hidden dimension 4096, SwiGLU dimension 11008.
All GEMMs use 2048 DPUs and 16 tasklets in normal orientation.
Progress goes to stderr and the JSON report goes to stdout.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
import triton.backends.pim.driver as pim_driver

from autotuner import ConstraintEngine, PIMConfig, PIMProblem, pim_hierarchy
from torch_backend import PIMTorchConfig, ResidentGemmConfig, prepare_block, target_spec
from torch_backend.fx_analysis import analyze_fx_gemms
from torch_backend.resident import _ResidentGemm
from attention.torch_resident_attention import timed_partition

from torch_resident_transformer import checked_partition


class RMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4096, dtype=torch.float32))

    def forward(self, activation):
        return activation * torch.rsqrt(activation.square().mean(dim=-1, keepdim=True) + 1e-5) * self.weight


def apply_rope(activation, cosine, sine):
    even = activation[:, 0::2]
    odd = activation[:, 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(1)


class SingleHeadTransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = RMSNorm()
        self.q = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float32)
        self.k = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float32)
        self.v = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float32)
        self.o = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float32)
        self.ffn_norm = RMSNorm()
        self.gate = torch.nn.Linear(4096, 11008, bias=False, dtype=torch.float32)
        self.up = torch.nn.Linear(4096, 11008, bias=False, dtype=torch.float32)
        self.down = torch.nn.Linear(11008, 4096, bias=False, dtype=torch.float32)

    def forward(self, activation, cosine, sine, mask):
        normalized = self.attention_norm(activation)
        q = self.q(normalized)
        k = self.k(normalized)
        v = self.v(normalized)
        q = apply_rope(q, cosine, sine)
        k = apply_rope(k, cosine, sine)
        scores = torch.mm(q, k.t()) / 64.0
        probability = torch.softmax(scores + mask, dim=-1)
        hidden = activation + self.o(torch.mm(probability, v))
        normalized = self.ffn_norm(hidden)
        gate = self.gate(normalized)
        up = self.up(normalized)
        return hidden + self.down(torch.nn.functional.silu(gate) * up)


class TransformerStack(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(SingleHeadTransformerBlock() for _ in range(layers))
        frequencies = 10000.0 ** (-torch.arange(0, 4096, 2, dtype=torch.float32) / 4096)
        angles = torch.arange(1024, dtype=torch.float32)[:, None] * frequencies[None, :]
        self.register_buffer("rope_cosine", angles.cos())
        self.register_buffer("rope_sine", angles.sin())
        self.register_buffer("causal_mask", torch.full(
            (1024, 1024), float("-inf"), dtype=torch.float32).triu(1))

    def forward(self, activation):
        for layer in self.layers:
            activation = layer(activation, self.rope_cosine, self.rope_sine, self.causal_mask)
        return activation


def check_run(block, output, reference, prepared_weight_bytes, layers):
    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
    assert block.last_transfers["weight_bytes"] == 0
    assert block.last_transfers["activation_bytes"] > 0
    assert block.last_transfers["output_bytes"] > 0
    assert block.last_transfers["launches"] == 9 * layers
    assert block.last_transfers["activation_hits"] == 0
    assert block.stats()["weight_bytes"] == prepared_weight_bytes
    return (output - reference).abs().max().item()


def checked_partition(name, module, records):
    measured = timed_partition(name, module, records)
    is_pim = isinstance(module, _ResidentGemm)

    def checked(*inputs, **kwargs):
        print(f"profile_check partition={name} "
              f"source={module.name if is_pim else 'CPU'}", file=sys.stderr, flush=True)
        start = time.perf_counter_ns()
        if is_pim:
            left, right = (
                weight if weight is not None else inputs[index]
                for weight, index in zip(module.weights, module.input_indices, strict=True)
            )
            reference = torch.mm(left, right)
        else:
            reference = module._orig_mod(*inputs, **kwargs)
        reference_ms = (time.perf_counter_ns() - start) / 1e6

        output = measured(*inputs, **kwargs)
        start = time.perf_counter_ns()
        torch.testing.assert_close(
            output, reference, rtol=1e-4, atol=1e-4,
            msg=lambda message: f"partition={name}, operators={records[-1]['operators']}\n{message}",
        )
        records[-1]["validation_ms"] = reference_ms + (time.perf_counter_ns() - start) / 1e6
        records[-1]["correctness"] = "PASS"
        return output

    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, choices=(1, 2, 4), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    target = target_spec("upmem", num_units=2048, nr_tasklets=16)
    operations = (
        ("q", "attention", (1024, 4096, 4096), "q.weight"),
        ("k", "attention", (1024, 4096, 4096), "k.weight"),
        ("v", "attention", (1024, 4096, 4096), "v.weight"),
        ("qk", "attention", (1024, 1024, 4096), None),
        ("pv", "attention", (1024, 4096, 1024), None),
        ("o", "attention", (1024, 4096, 4096), "o.weight"),
        ("gate", "ffn", (1024, 11008, 4096), "gate.weight"),
        ("up", "ffn", (1024, 11008, 4096), "up.weight"),
        ("down", "ffn", (1024, 4096, 11008), "down.weight"),
    )

    with torch.no_grad():
        model = TransformerStack(args.layers).eval()
        activations = (
            torch.randn(1024, 4096, dtype=torch.float32),
            torch.randn(1024, 4096, dtype=torch.float32),
        )
        print(f"Exporting {args.layers} Transformer layers ...", file=sys.stderr, flush=True)
        start = time.perf_counter_ns()
        exported = torch.export.export(model, (activations[0],))
        export_ms = (time.perf_counter_ns() - start) / 1e6
        gemms = analyze_fx_gemms(exported)
        assert len(gemms) == 9 * args.layers

        configs = {}
        descriptions = {}
        weight_bytes_per_dpu = 0
        input_scratch_bytes = 0
        output_scratch_bytes = 0
        for layer in range(args.layers):
            for gemm, (operation, section, shape, weight) in zip(
                gemms[9 * layer:9 * (layer + 1)], operations, strict=True,
            ):
                assert gemm.mnk == shape
                assert gemm.left.weight_target is None
                assert f"layers.{layer}" in gemm.module_paths
                assert gemm.right.weight_target == (
                    None if weight is None else f"layers.{layer}.{weight}")
                assert gemm.op == ("aten.mm.default" if weight is None else "aten.linear.default")

                # Gate/up의 N을 64개 DPU 방향과 block_n=16에 맞춰 11264로 확장한다.
                m, n, k = gemm.mnk
                padded_n = ((n + 1023) // 1024) * 1024
                hierarchy = PIMConfig(
                    block_m=16, block_n=16, block_k=32,
                    group_m=1, group_n=1, group_k=1,
                    tile_m=2, tile_n=padded_n // 1024, tile_k=k // 32,
                    p_m=32, p_n=64, p_k=1,
                )
                problem = PIMProblem(m, padded_n, k, input_dtype="float32", output_dtype="float32")
                reason = ConstraintEngine(problem, target).check(hierarchy)
                if reason is not None:
                    raise ValueError(f"{gemm.node}: {reason}")
                assert pim_hierarchy(problem, hierarchy).rounds == (1, 1, 1)
                configs[gemm.node] = ResidentGemmConfig(
                    torch_config=PIMTorchConfig(spec=target, config=hierarchy),
                    orientation="normal", padded_shape=(m, padded_n, k),
                )
                extent_m, extent_n, extent_k = hierarchy.execution_extent
                resident_bytes = 0 if weight is None else extent_k * extent_n * 4
                weight_bytes_per_dpu += resident_bytes
                input_scratch_bytes = max(
                    input_scratch_bytes,
                    extent_m * extent_k * 4 + (extent_k * extent_n * 4 if weight is None else 0),
                )
                output_scratch_bytes = max(output_scratch_bytes, extent_m * extent_n * 4)
                descriptions[gemm.node] = {
                    "node": gemm.node, "layer": layer, "operation": operation, "section": section,
                    "op": gemm.op, "mnk": gemm.mnk, "weight": gemm.right.weight_target,
                    "padded_shape": configs[gemm.node].padded_shape,
                    "config": asdict(hierarchy), "weight_bytes_per_dpu": resident_bytes,
                }
                print(f"layer={layer} {operation}: {gemm.op} mnk={gemm.mnk} "
                      f"padded={configs[gemm.node].padded_shape} B_weight={gemm.right.weight_target}",
                      file=sys.stderr, flush=True)

        mram_bytes_per_dpu = weight_bytes_per_dpu + input_scratch_bytes + output_scratch_bytes
        if mram_bytes_per_dpu > target.common_profile.bank_capacity_limit_bytes:
            raise ValueError(f"stack requires {mram_bytes_per_dpu} bytes per DPU; "
                             f"budget is {target.common_profile.bank_capacity_limit_bytes}")
        print(f"preflight MRAM_bytes_per_DPU={mram_bytes_per_dpu} "
              f"captured_weight_host_bytes={2048 * weight_bytes_per_dpu} (other host buffers excluded)",
              file=sys.stderr, flush=True)
        print("Computing two CPU references ...", file=sys.stderr, flush=True)
        references = tuple(model(activation) for activation in activations)

        print(f"Preparing layers={args.layers} orientation=normal DPUs=2048 tasklets=16 ...",
              file=sys.stderr, flush=True)
        with prepare_block(exported, (activations[0],), configs=configs, dpu_count=2048) as block:
            prepared = block.stats()
            weight_regions = [region for region in block.layout if region["role"] == "weight"]
            assert len(weight_regions) == 7 * args.layers
            assert {region["kernel"]: region["bytes"] for region in weight_regions} == {
                node: description["weight_bytes_per_dpu"]
                for node, description in descriptions.items() if description["weight"] is not None
            }
            assert all(region["window"] == 0 for region in weight_regions)
            assert [region for region in block.layout if region["role"] != "weight"] == [
                {"role": "input_scratch", "offset": weight_bytes_per_dpu, "bytes": input_scratch_bytes},
                {"role": "output_scratch", "offset": weight_bytes_per_dpu + input_scratch_bytes,
                 "bytes": output_scratch_bytes},
            ]
            assert prepared["weight_bytes"] == 2048 * weight_bytes_per_dpu
            assert prepared["launches"] == 0
            assert prepared["activation_bytes"] == 0
            assert prepared["output_bytes"] == 0
            print(f"prepare_ms={block.prepare_ms:.3f} prepare_weight_bytes={prepared['weight_bytes']}",
                  file=sys.stderr, flush=True)

            warmups = []
            for iteration in range(3):
                input_index = iteration % len(activations)
                start = time.perf_counter_ns()
                output = block.run(activations[input_index])
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                max_abs_error = check_run(block, output, references[input_index],
                                          prepared["weight_bytes"], args.layers)
                warmups.append({"input": input_index, "wall_ms": wall_ms, "max_abs_error": max_abs_error})
                print(f"warmup={iteration} correctness=PASS wall_ms={wall_ms:.3f} (excluded)",
                      file=sys.stderr, flush=True)
                del output

            samples = []

            # ---------------- 삭제 --------------
            diagnostic_records = []

            with ExitStack() as instrumentation:
                for name, module in block.graph.named_children():
                    instrumentation.enter_context(
                        patch.object(
                            module,
                            "forward",
                            checked_partition(
                                name, module, diagnostic_records
                            ),
                        )
                    )

                for step in range(3):
                    print(f"diagnostic step={step}", flush=True)
                    outputs = block.run(activation)
                    check_outputs(
                        outputs,
                        references,
                        f"diagnostic={step}",
                    )

            return
            # ----------------------------------

            for iteration in range(3):
                input_index = iteration % len(activations)
                start = time.perf_counter_ns()
                output = block.run(activations[input_index])
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                max_abs_error = check_run(block, output, references[input_index],
                                          prepared["weight_bytes"], args.layers)
                samples.append({
                    "input": input_index, "wall_ms": wall_ms, "graph_ms": block.last_run_ms,
                    "transfers": dict(block.last_transfers), "max_abs_error": max_abs_error,
                })
                print(f"sample={iteration} correctness=PASS wall_ms={wall_ms:.3f} "
                      f"weight_h2d={block.last_transfers['weight_bytes']} "
                      f"activation_h2d={block.last_transfers['activation_bytes']} "
                      f"activation_hits={block.last_transfers['activation_hits']} "
                      f"launches={block.last_transfers['launches']}", file=sys.stderr, flush=True)
                del output

            records = []
            with ExitStack() as instrumentation:
                for name, module in block.graph.named_children():
                    instrumentation.enter_context(patch.object(
                        module, "forward", checked_partition(name, module, records)))
                start = time.perf_counter_ns()
                output = block.run(activations[0])
                profile_wall_ms = (time.perf_counter_ns() - start) / 1e6
            profile_error = check_run(block, output, references[0], prepared["weight_bytes"], args.layers)
            pim_records = [record for record in records if "gemm" in record]
            assert [record["gemm"] for record in pim_records] == [gemm.node for gemm in gemms]
            for record in pim_records:
                description = descriptions[record["gemm"]]
                record.update({key: description[key] for key in ("layer", "operation", "section")})
                assert record["transfers"]["launches"] == 1
                assert record["transfers"]["weight_bytes"] == 0
                assert record["transfers"]["activation_bytes"] > 0
                assert record["transfers"]["activation_hits"] == 0

            # CPU partition은 layer 경계를 넘을 수 있어 layer별 합계에는 GEMM 구간만 넣는다.
            layer_profiles = []
            for layer in range(args.layers):
                layer_records = [record for record in pim_records if record["layer"] == layer]
                layer_profiles.append({
                    "layer": layer,
                    "gemm_partition_ms": sum(record["wall_ms"] for record in layer_records),
                    "attention_gemm_ms": sum(record["wall_ms"] for record in layer_records
                                             if record["section"] == "attention"),
                    "ffn_gemm_ms": sum(record["wall_ms"] for record in layer_records
                                       if record["section"] == "ffn"),
                    "native_ms": {stage: sum(record["native_ms"][stage] for record in layer_records)
                                  for stage in pim_driver.TIMING_STAGES},
                    "transfers": {key: sum(record["transfers"][key] for record in layer_records)
                                  for key in prepared},
                })
            print(f"partition_profile correctness=PASS PIM_partitions={len(pim_records)} "
                  "(excluded from median)", file=sys.stderr, flush=True)

            report = {
                "scope": "single-head Transformer stack with causal attention, RoPE, RMSNorm and SwiGLU",
                "shape": {"layers": args.layers, "batch_size": 1, "sequence_length": 1024,
                          "kv_length": 1024, "input_dimension": 4096, "head_dimension": 4096,
                          "heads": 1, "intermediate_dimension": 11008},
                "dtype": "float32", "orientation": "normal", "mask": "causal",
                "normalization": {"kind": "RMSNorm", "eps": 1e-5},
                "rope": {"theta": 10000.0, "pairing": "interleaved", "start_position": 0},
                "ffn": "SwiGLU", "bias": False, "dropout": 0.0, "qkv_fusion": False,
                "output_projection": True, "residual": True, "kv_cache": False,
                "weights": "random", "seed": 0,
                "target": asdict(target),
                "versions": {"torch": torch.__version__, "triton": triton.__version__},
                "cpu_threads": torch.get_num_threads(),
                "environment": {
                    name: os.environ[name]
                    for name in ("UPMEM_PROFILE", "UPMEM_BACKEND", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                    if name in os.environ
                },
                "gemms": list(descriptions.values()),
                "preparation": {
                    "export_ms": export_ms, "prepare_ms": block.prepare_ms,
                    "weight_prepare_ms": block.weight_prepare_ms,
                    "resident_capture_ms": block.runtime.capture_ms,
                    "resident_upload_ms": block.runtime.upload_ms,
                    "resident_upload_breakdown_ms": block.runtime.upload_breakdown_ms,
                    "transfers": prepared, "layout": block.layout,
                    "mram_bytes_per_dpu": mram_bytes_per_dpu,
                    "mram_budget_bytes_per_dpu": target.common_profile.bank_capacity_limit_bytes,
                    "captured_weight_host_bytes": 2048 * weight_bytes_per_dpu,
                },
                "warmups": warmups, "samples": samples,
                "median_wall_ms": statistics.median(sample["wall_ms"] for sample in samples),
                "median_graph_ms": statistics.median(sample["graph_ms"] for sample in samples),
                "profile": {
                    "kind": "partition correctness diagnostic",
                    "wall_ms": profile_wall_ms,
                    "validation_ms": sum(record["validation_ms"] for record in records),
                    "outside_partitions_ms": profile_wall_ms - sum(
                        record["wall_ms"] + record["validation_ms"] for record in records),
                    "cpu_partition_ms": sum(record["wall_ms"] for record in records if "gemm" not in record),
                    "layer_timing_scope": "GEMM partitions only; CPU partitions reported separately",
                    "pim_timing_version": pim_driver.TIMING_VERSION,
                    "transfers": dict(block.last_transfers), "layers": layer_profiles,
                    "partitions": records, "max_abs_error": profile_error,
                },
                "correctness": {"status": "PASS", "rtol": 1e-4, "atol": 1e-4,
                                "scope": "final stack output for two distinct inputs"},
            }

    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
