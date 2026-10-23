"""Compile time analysis of one layer: stage classes and the model-stage tuning constraint.

Nothing here touches the device. Every layer of the model has the structure of
the analyzed layer; backend/residency.py places the compiled layers in slots.
"""

from dataclasses import dataclass

import torch

from ._autotuner import build_config_space, pim_hierarchy
from .execution import gemm_problem
from .config import PIMTorchConfig
from .fx_analysis import analyze_fx_gemms
from .model import build_layer_plan


@dataclass(frozen=True)
class StageKey:
    """Stages with the same key share one measurement and one config."""

    mnk: tuple
    input_dtype: str
    output_dtype: str
    op: str
    weight_mask: tuple
    instances: int
    # operand마다 장치의 key/value cache 영역을 그대로 읽는지
    kv_mask: tuple

    @property
    def problem(self):
        return gemm_problem(*self.mnk, input_dtype=self.input_dtype, output_dtype=self.output_dtype)

    @property
    def has_weights(self):
        return any(self.weight_mask)

    @property
    def has_kv(self):
        return any(self.kv_mask)

    def label(self):
        kind = "weight" if self.has_weights else "kv" if self.has_kv else "activation"
        return f"{kind} {self.op} {self.mnk} x{self.instances} {self.input_dtype}"


@dataclass
class StageClass:
    key: StageKey
    # layer 하나에서 이 class에 속한 stage 수
    count: int
    # 측정에 쓸 weight operand (activation stage는 None)
    weights: tuple
    # packing으로 이어 붙인 class면 (축, member (m, n, k)들, member stage의 instance 수), 아니면 None
    origin: tuple | None
    # stage마다 그 stage가 계산하는 원래 GEMM 이름들
    stage_sources: list


@dataclass(frozen=True)
class Footprint:
    """Per-unit bytes in the fields of StageRequirement."""

    weight: int
    state: int
    input: int
    output: int
    # stage가 그대로 읽는 key/value cache 영역 (layer마다 하나씩 놓인다)
    kv: int


def stage_key(stage):
    return StageKey(
        mnk=tuple(stage.shape),
        input_dtype=str(stage.input_dtype).removeprefix("torch."),
        output_dtype=str(stage.output_dtype).removeprefix("torch."),
        op=stage.gemms[0].op,
        weight_mask=tuple(stage.weight_mask),
        instances=stage.instance_count,
        kv_mask=tuple(stage.kv is not None and stage.kv[0] == operand for operand in range(2)),
    )


def analyze_layer(layer, example_inputs, spec, kv):
    """Stage classes of one layer and its original GEMM name -> StageKey.

    Every GEMM gets the same placeholder config, so build_layer_plan packs and
    groups exactly as it will once each class has its own config. kv is the
    layer's {KEY/VALUE: cache input} (build_layer_plan) or None.
    """
    placeholder = PIMTorchConfig()
    classes = {}
    assignment = {}
    with torch.no_grad():
        exported = torch.export.export(layer, example_inputs)
        gemms = analyze_fx_gemms(exported, allow_bmm=True)
        plan = build_layer_plan(exported, example_inputs,
                                configs={gemm.node: placeholder for gemm in gemms}, index=0, spec=spec,
                                kv=kv)
        for stage in plan.stages:
            key = stage_key(stage)
            if key not in classes:
                classes[key] = StageClass(key, 0, tuple(stage.weights), stage.origin, [])
            classes[key].count += 1
            classes[key].stage_sources.append(tuple(stage.sources))
            for source in stage.sources:
                assignment[source] = key
    return list(classes.values()), assignment


def stage_constraint(key, max_rounds, units):
    """Static model-stage conditions.

    max_rounds bounds an activation stage's M/N/K scheduling rounds; without it
    the search would also measure one Block Group per round, thousands of rounds.
    A stage reading the key/value cache in place keeps every instance's part in
    memory, so all its instances run in one batch round on the units.
    p_k is left to the target (AiMX rejects p_k > 1 in its mapping_reason).
    """

    def check(problem, config):
        rounds = pim_hierarchy(problem, config).scheduling_rounds
        if key.has_weights and rounds != 1:
            return "weight stages require one M/N/K round"
        if key.has_kv and rounds != 1:
            return "key/value cache stages require one M/N/K round"
        if key.has_kv and key.instances * config.p_m * config.p_n * config.p_k > units:
            return (f"key/value cache stages run all {key.instances} instances in one batch round; "
                    f"{config.p_m * config.p_n * config.p_k} units per instance exceed {units} units")
        if rounds > max_rounds:
            return f"model stages allow at most {max_rounds} M/N/K rounds; got {rounds}"
        return None

    return check


def max_stage_rounds(problem, spec):
    # autotuner가 초기 후보에 허용하는 round 수와 같은 값이다.
    return build_config_space(problem, spec).max_initial_scheduling_rounds
