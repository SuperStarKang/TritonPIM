"""Validate and profile connected Llama-2-7b-hf layers 0 and 1.

Batch size 1, sequence length 1024, 2048 DPUs and 16 tasklets.
Pass the local snapshot directory containing config.json as --checkpoint.
The checkpoint and tokenizer must already be downloaded.
Both layers' FP32 weights are resident before execution.
One warm-up precedes two uninstrumented samples and one partition profile.
FX graphs and timings are written to llama_two_layers_report beside this script.
Both layer outputs are returned for validation; timings are not model TTFT.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton.backends.pim.driver as pim_driver
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from autotuner import ConstraintEngine, PIMConfig, PIMProblem, pim_hierarchy
from torch_backend import (
    PIMTorchConfig,
    ResidentGemmConfig,
    prepare_block,
    target_spec,
)
from torch_backend.fx_analysis import analyze_fx_gemms
from torch_backend.resident import _ResidentGemm
from attention.torch_resident_attention import timed_partition
from fx_utils import write_fx_graph


class LlamaFirstLayer(torch.nn.Module):
    def __init__(self, layer, config, cosine, sine, causal_mask):
        super().__init__()
        self.self_attn = layer.self_attn
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

        self.hidden_size = config.hidden_size
        self.query_heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.group_size = self.query_heads // self.kv_heads

        self.register_buffer("cosine", cosine.contiguous())
        self.register_buffer("sine", sine.contiguous())
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, activation):
        return self.prefill(activation)[0]

    def prefill(self, activation):
        sequence = activation.shape[1]
        normalized = self.input_layernorm(activation)

        query = self.self_attn.q_proj(normalized).reshape(
            1, sequence, self.query_heads, self.head_dim
        ).transpose(1, 2)
        key = self.self_attn.k_proj(normalized).reshape(
            1, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.self_attn.v_proj(normalized).reshape(
            1, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)

        query, key = apply_rotary_pos_emb(
            query, key, self.cosine, self.sine
        )
        key_cache = key.contiguous()
        value_cache = value.contiguous()

        # 같은 KV head를 공유하는 query head만 GEMM의 M축으로 묶는다.
        query = query.reshape(
            self.kv_heads, self.group_size * sequence, self.head_dim
        ).contiguous()
        key = key_cache[0]
        value = value_cache[0]

        outputs = []
        for group in range(self.kv_heads):
            scores = torch.mm(query[group], key[group].t())
            scores = scores * self.self_attn.scaling
            scores = scores.reshape(self.group_size, sequence, sequence)
            probability = torch.softmax(
                scores + self.causal_mask,
                dim=-1,
                dtype=torch.float32,
            ).reshape(self.group_size * sequence, sequence)
            outputs.append(torch.mm(probability, value[group]))

        context = torch.cat(outputs, dim=0).reshape(
            self.query_heads, sequence, self.head_dim
        ).transpose(0, 1).reshape(
            1, sequence, self.hidden_size
        ).contiguous()

        hidden = activation + self.self_attn.o_proj(context)
        normalized = self.post_attention_layernorm(hidden)
        return hidden + self.mlp(normalized), key_cache, value_cache


class LlamaTwoLayers(torch.nn.Module):
    def __init__(self, layers, config, cosine, sine, causal_mask):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            LlamaFirstLayer(layer, config, cosine, sine, causal_mask)
            for layer in layers
        )

    def forward(self, activation):
        layer0_output = self.layers[0](activation)
        layer1_output = self.layers[1](layer0_output)
        return layer0_output, layer1_output


def check_outputs(outputs, references, phase):
    errors = []
    for layer, (output, reference) in enumerate(
        zip(outputs, references, strict=True)
    ):
        torch.testing.assert_close(
            output,
            reference,
            rtol=1e-4,
            atol=1e-4,
            msg=lambda message: f"{phase} layer={layer}\n{message}",
        )
        errors.append((output - reference).abs().max().item())
        print(
            f"{phase} layer={layer} correctness=PASS "
            f"max_abs_error={errors[-1]:.8g}",
            flush=True,
        )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.checkpoint),
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(args.checkpoint),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval().requires_grad_(False)

        if model.config.model_type != "llama" or (
            model.config.num_hidden_layers,
            model.config.hidden_size,
            model.config.intermediate_size,
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            model.config.head_dim,
        ) != (32, 4096, 11008, 32, 32, 128):
            raise ValueError("This example requires Llama-2-7b-hf.")

        input_ids = tokenizer(
            args.prompt_file.read_text(encoding="utf-8"),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )["input_ids"]
        if tuple(input_ids.shape) != (1, 1024):
            raise ValueError(
                "The prompt must contain at least 1024 tokens "
                "including tokenizer-added special tokens."
            )

        activation = model.model.embed_tokens(input_ids).contiguous()
        position_ids = torch.arange(1024).unsqueeze(0)
        cosine, sine = model.model.rotary_emb(
            activation, position_ids=position_ids
        )
        causal_mask = torch.full(
            (1024, 1024),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        ).triu(1)

        report_dir = Path(__file__).with_name("llama_two_layers_report")
        report_dir.mkdir(parents=True, exist_ok=True)

        adapter = LlamaTwoLayers(
            model.model.layers[:2],
            model.config,
            cosine,
            sine,
            causal_mask,
        ).eval()

        references = []
        hidden = activation
        for layer in model.model.layers[:2]:
            hidden = layer(
                hidden,
                attention_mask=causal_mask[None, None, :, :],
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=(cosine, sine),
            )
            references.append(hidden)

        adapter_errors = check_outputs(
            adapter(activation), references, "HF_vs_adapter"
        )

        exported = torch.export.export(adapter, (activation,))
        gemms = analyze_fx_gemms(exported)
        assert len(gemms) == 142

        plan = {}
        head_counts = Counter()
        for gemm in gemms:
            layer_index, = {
                int(path.split(".")[1])
                for path in gemm.module_paths
                if path.startswith("layers.")
            }
            head = None
            if gemm.right.weight_target is not None:
                operation = gemm.right.weight_target.split(".")[-2]
            else:
                assert gemm.mnk in (
                    (1024, 1024, 128),
                    (1024, 128, 1024),
                )
                operation = (
                    "qk" if gemm.mnk == (1024, 1024, 128) else "pv"
                )
                head = head_counts[layer_index, operation]
                head_counts[layer_index, operation] += 1

            plan[gemm.node] = {
                "layer": layer_index,
                "operation": operation,
                "head": head,
                "mnk": gemm.mnk,
                "weight": gemm.right.weight_target,
            }

        for layer_index in range(2):
            layer_gemms = [
                gemm for gemm in gemms
                if plan[gemm.node]["layer"] == layer_index
            ]
            assert Counter(gemm.mnk for gemm in layer_gemms) == Counter({
                (1024, 4096, 4096): 4,
                (1024, 1024, 128): 32,
                (1024, 128, 1024): 32,
                (1024, 11008, 4096): 2,
                (1024, 4096, 11008): 1,
            })
            assert {
                gemm.right.weight_target
                for gemm in layer_gemms
                if gemm.right.weight_target is not None
            } == {
                f"layers.{layer_index}.{name}.weight"
                for name in (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                )
            }

        write_fx_graph(
            exported.graph_module,
            report_dir / "exported_fx",
            {
                name: json.dumps(description)
                for name, description in plan.items()
            },
        )

        # 각 값은 (block_n, p_m, p_n)이며 p_k는 모두 1이다.
        placements = {
            (1024, 4096, 4096): (16, 32, 64),
            (1024, 1024, 128): (16, 32, 64),
            (1024, 128, 1024): (4, 64, 32),
            (1024, 11008, 4096): (16, 32, 64),
            (1024, 4096, 11008): (16, 32, 64),
        }
        target = target_spec("upmem", num_units=2048, nr_tasklets=16)
        configs = {}
        weight_sizes = {}
        input_scratch_bytes = 0
        output_scratch_bytes = 0

        for gemm in gemms:
            assert gemm.left.weight_target is None
            assert gemm.op == (
                "aten.mm.default"
                if gemm.right.weight_target is None
                else "aten.linear.default"
            )

            m, n, k = gemm.mnk
            block_n, p_m, p_n = placements[gemm.mnk]
            # Gate/up의 N=11008을 DPU별 N 타일에 맞춰 11264로 확장한다.
            padded_n = ((n + block_n * p_n - 1) // (block_n * p_n)) * (block_n * p_n)
            assert p_m * p_n == 2048
            assert m % (16 * p_m) == 0
            assert padded_n % (block_n * p_n) == 0
            assert k % 32 == 0

            hierarchy = PIMConfig(
                block_m=16, block_n=block_n, block_k=32,
                group_m=1, group_n=1, group_k=1,
                tile_m=m // (16 * p_m),
                tile_n=padded_n // (block_n * p_n),
                tile_k=k // 32,
                p_m=p_m, p_n=p_n, p_k=1,
            )
            problem = PIMProblem(
                m, padded_n, k, input_dtype="float32", output_dtype="float32"
            )
            reason = ConstraintEngine(problem, target).check(hierarchy)
            if reason is not None:
                raise ValueError(f"{gemm.node}: {reason}")
            assert pim_hierarchy(problem, hierarchy).rounds == (1, 1, 1)

            configs[gemm.node] = ResidentGemmConfig(
                torch_config=PIMTorchConfig(spec=target, config=hierarchy),
                orientation="normal",
                padded_shape=(m, padded_n, k),
            )
            extent_m, extent_n, extent_k = hierarchy.execution_extent
            if gemm.right.weight_target is not None:
                weight_sizes[gemm.node] = extent_k * extent_n * 4

            input_scratch_bytes = max(
                input_scratch_bytes,
                extent_m * extent_k * 4
                + (
                    extent_k * extent_n * 4
                    if gemm.right.weight_target is None else 0
                ),
            )
            output_scratch_bytes = max(
                output_scratch_bytes, extent_m * extent_n * 4
            )
            print(
                f"layer={plan[gemm.node]['layer']} "
                f"{gemm.node}: {gemm.op} mnk={gemm.mnk} "
                f"padded={(m, padded_n, k)} "
                f"B_weight={gemm.right.weight_target} "
                f"partitions=({p_m},{p_n},1)",
                flush=True,
            )

        assert len(weight_sizes) == 14
        weight_bytes_per_dpu = sum(weight_sizes.values())
        mram_bytes_per_dpu = (
            weight_bytes_per_dpu
            + input_scratch_bytes
            + output_scratch_bytes
        )
        if mram_bytes_per_dpu > target.common_profile.bank_capacity_limit_bytes:
            raise ValueError(
                f"MRAM capacity exceeded: {mram_bytes_per_dpu} bytes per DPU"
            )

        print(
            f"FX analysis=PASS gemms={len(gemms)} weights={len(weight_sizes)}\n"
            f"preflight MRAM_bytes_per_DPU={mram_bytes_per_dpu} "
            f"captured_weight_host_bytes={2048 * weight_bytes_per_dpu}",
            flush=True,
        )

        with prepare_block(
            exported,
            (activation,),
            configs=configs,
            dpu_count=2048,
        ) as block:
            prepared = block.stats()
            assert prepared["weight_bytes"] == 2048 * weight_bytes_per_dpu
            assert prepared["launches"] == 0
            assert {
                region["kernel"]: region["bytes"]
                for region in block.layout
                if region["role"] == "weight"
            } == weight_sizes
            assert all(
                region["window"] == 0
                for region in block.layout
                if region["role"] == "weight"
            )

            print(
                f"prepare_ms={block.prepare_ms:.3f} "
                f"prepare_weight_bytes={prepared['weight_bytes']}",
                flush=True,
            )

            execution_labels = {}
            for node in block.graph.graph.nodes:
                if node.op == "call_module":
                    module = block.graph.get_submodule(node.target)
                    if isinstance(module, _ResidentGemm):
                        execution_labels[node.name] = (
                            "CPU+UPMEM\n" + json.dumps(plan[module.name])
                        )
                    else:
                        execution_labels[node.name] = "CPU"

            write_fx_graph(
                block.graph,
                report_dir / "execution_fx",
                execution_labels,
            )
            with (report_dir / "execution_fx.txt").open(
                "a", encoding="utf-8"
            ) as graph_file:
                for name, module in block.graph.named_children():
                    if not isinstance(module, _ResidentGemm):
                        graph_file.write(
                            f"\n\n# CPU partition: {name}\n"
                            + module._orig_mod.code
                        )

            samples = []
            for step in range(3):
                phase = "warmup" if step == 0 else f"sample={step - 1}"
                start = time.perf_counter_ns()
                outputs = block.run(activation)
                wall_ms = (time.perf_counter_ns() - start) / 1e6

                errors = check_outputs(outputs, references, phase)
                assert block.last_transfers["weight_bytes"] == 0
                assert block.last_transfers["activation_bytes"] > 0
                assert block.last_transfers["output_bytes"] > 0
                assert block.last_transfers["launches"] == 142
                assert block.stats()["weight_bytes"] == prepared["weight_bytes"]

                measurement = {
                    "phase": phase,
                    "wall_ms": wall_ms,
                    "graph_ms": block.last_run_ms,
                    "max_abs_error_by_layer": errors,
                    "transfers": dict(block.last_transfers),
                }
                if step == 0:
                    warmup = measurement
                else:
                    samples.append(measurement)

                print(
                    f"{phase} wall_ms={wall_ms:.3f} "
                    f"weight_h2d={block.last_transfers['weight_bytes']} "
                    f"activation_h2d={block.last_transfers['activation_bytes']} "
                    f"activation_hits={block.last_transfers['activation_hits']} "
                    f"launches={block.last_transfers['launches']}",
                    flush=True,
                )
                del outputs

            report = {
                "status": "plain_pass_profile_pending",
                "scope": "Llama-2-7b-hf layers 0 -> 1; both outputs returned",
                "checkpoint": str(args.checkpoint.resolve()),
                "dtype": "float32",
                "sequence_length": 1024,
                "dpu_count": 2048,
                "tasklets": 16,
                "weight_policy": "both layers resident before run",
                "gemms": plan,
                "adapter_max_abs_error_by_layer": adapter_errors,
                "preparation": {
                    "prepare_ms": block.prepare_ms,
                    "weight_prepare_ms": block.weight_prepare_ms,
                    "capture_ms": block.runtime.capture_ms,
                    "upload_ms": block.runtime.upload_ms,
                    "upload_breakdown_ms": block.runtime.upload_breakdown_ms,
                    "transfers": prepared,
                    "layout": block.layout,
                    "mram_bytes_per_dpu": mram_bytes_per_dpu,
                    "captured_weight_host_bytes": 2048 * weight_bytes_per_dpu,
                },
                "warmup": warmup,
                "samples": samples,
                "median_wall_ms": statistics.median(
                    sample["wall_ms"] for sample in samples
                ),
            }
            (report_dir / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            records = []
            with ExitStack() as instrumentation:
                for name, module in block.graph.named_children():
                    instrumentation.enter_context(
                        patch.object(
                            module,
                            "forward",
                            timed_partition(name, module, records),
                        )
                    )
                start = time.perf_counter_ns()
                outputs = block.run(activation)
                profile_wall_ms = (time.perf_counter_ns() - start) / 1e6

            for record in records:
                if "gemm" in record:
                    record.update(plan[record["gemm"]])

            report["profile"] = {
                "correctness": "pending",
                "wall_ms": profile_wall_ms,
                "outside_partitions_ms": profile_wall_ms - sum(
                    record["wall_ms"] for record in records
                ),
                "cpu_partition_ms": sum(
                    record["wall_ms"]
                    for record in records
                    if "gemm" not in record
                ),
                "timing_version": pim_driver.TIMING_VERSION,
                "transfers": dict(block.last_transfers),
                "partitions": records,
            }
            # Profile 검사가 실패해도 일반 측정과 계측 원본을 남긴다.
            (report_dir / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            profile_errors = check_outputs(
                outputs, references, "partition_profile"
            )
            pim_records = [
                record for record in records if "gemm" in record
            ]
            assert [record["gemm"] for record in pim_records] == [
                gemm.node for gemm in gemms
            ]
            assert block.last_transfers["launches"] == 142
            assert block.last_transfers["weight_bytes"] == 0
            assert block.stats()["weight_bytes"] == prepared["weight_bytes"]

            for record in pim_records:
                assert record["transfers"]["launches"] == 1
                assert record["transfers"]["weight_bytes"] == 0

            operation_profiles = []
            for layer_index in range(2):
                for operation in (
                    "q_proj", "k_proj", "v_proj", "qk", "pv",
                    "o_proj", "gate_proj", "up_proj", "down_proj",
                ):
                    operation_records = [
                        record for record in pim_records
                        if record["layer"] == layer_index
                        and record["operation"] == operation
                    ]
                    operation_profiles.append({
                        "layer": layer_index,
                        "operation": operation,
                        "calls": len(operation_records),
                        "wall_ms": sum(
                            record["wall_ms"]
                            for record in operation_records
                        ),
                        "native_ms": {
                            stage: sum(
                                record["native_ms"][stage]
                                for record in operation_records
                            )
                            for stage in pim_driver.TIMING_STAGES
                        },
                        "transfers": {
                            key: sum(
                                record["transfers"][key]
                                for record in operation_records
                            )
                            for key in prepared
                        },
                    })

            report["profile"].update({
                "correctness": "PASS",
                "max_abs_error_by_layer": profile_errors,
                "operations": operation_profiles,
            })
            report["status"] = "PASS"
            (report_dir / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            print(
                "partition_profile correctness=PASS PIM_partitions=142",
                flush=True,
            )
            print(f"Reports saved to {report_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
