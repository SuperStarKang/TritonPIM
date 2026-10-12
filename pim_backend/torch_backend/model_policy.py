"""Compile time residency policy: stage classes, footprints and policy constraints.

Nothing here touches the device. Footprints come from the target model of
autotuner/model.py; the policy formulas are those of backend/residency.py.
"""

from dataclasses import dataclass
from itertools import product
import math

import torch

from triton.backends.pim.residency import (
    LAYER, MODEL, WEIGHTS, StageRequirement, UnitResources, plan_residency,
)

from ._autotuner import (
    ConstraintEngine, PIMConfig, PIMProblem, axis_values, build_config_space,
    legal_partitions, pim_hierarchy, target_model,
)
from .config import PIMTorchConfig
from .fx_analysis import analyze_fx_gemms
from .fx_normalization import normalize_fx_gemms
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

    @property
    def problem(self):
        return PIMProblem(*self.mnk, input_dtype=self.input_dtype, output_dtype=self.output_dtype)

    @property
    def has_weights(self):
        return any(self.weight_mask)

    def label(self):
        kind = "weight" if self.has_weights else "activation"
        return f"{kind} {self.op} {self.mnk} x{self.instances} {self.input_dtype}"


@dataclass
class StageClass:
    key: StageKey
    # layer마다 이 class에 속한 stage 수
    counts: list
    # 측정에 쓸 weight operand (activation stage는 None)
    weights: tuple

    @property
    def total(self):
        return sum(self.counts)

    @property
    def per_layer(self):
        return max(self.counts)


@dataclass(frozen=True)
class Footprint:
    """Per-unit bytes in the fields of StageRequirement."""

    weight: int
    state: int
    input: int
    output: int


def analyze_stages(layers, example_inputs):
    """Stage classes of every layer and, per layer, original GEMM name -> StageKey.

    Every GEMM gets the same placeholder config, so build_layer_plan groups
    exactly as it will once each class has its own config.
    """
    placeholder = PIMTorchConfig()
    classes = {}
    assignments = []
    with torch.no_grad():
        for index, layer in enumerate(layers):
            exported = torch.export.export(layer, example_inputs)
            gemms = analyze_fx_gemms(exported, allow_bmm=True)
            plan = build_layer_plan(exported, example_inputs,
                                    configs={gemm.node: placeholder for gemm in gemms}, index=index)
            _, mappings = normalize_fx_gemms(exported, allow_bmm=True)
            originals = {mapping.normalized.node: mapping.original.node for mapping in mappings}
            assignment = {}
            for stage in plan.stages:
                key = StageKey(
                    mnk=tuple(stage.shape),
                    input_dtype=str(stage.input_dtype).removeprefix("torch."),
                    output_dtype=str(stage.output_dtype).removeprefix("torch."),
                    op=stage.gemms[0].op,
                    weight_mask=tuple(stage.weight_mask),
                    instances=stage.instance_count,
                )
                if key not in classes:
                    classes[key] = StageClass(key, [0] * len(layers), tuple(stage.weights))
                classes[key].counts[index] += 1
                for gemm in stage.gemms:
                    assignment[originals[gemm.node]] = key
            assignments.append(assignment)
            outputs = layer(*example_inputs)
            example_inputs = outputs if isinstance(outputs, tuple) else (outputs,)
    return list(classes.values()), assignments


def footprint(key, spec, config):
    """Target model estimate of one stage's per-unit bytes."""
    model = target_model(spec)
    problem = key.problem
    accesses = model.access_allocations(problem, config)
    overhead = sum(model.overhead_allocations(problem, config).values())
    weight = 0
    inputs = 0
    outputs = 0
    # 기본 GEMM access 순서: A(m,k), B(k,n), C(m,n). weight_mask는 (A, B)이다.
    for access, size, is_weight in zip(problem.accesses, accesses, key.weight_mask + (False,)):
        if is_weight:
            weight += size
        elif access.direction == "out":
            outputs += size
        else:
            inputs += size
    if model.shared_activation_scratch:
        return Footprint(weight, overhead, inputs, outputs)
    return Footprint(weight, overhead + inputs + outputs, 0, 0)


def stage_constraint(key, max_rounds, budget=None):
    """Static model-stage conditions, plus the policy budget once it is known.

    max_rounds bounds an activation stage's M/N/K scheduling rounds; without it
    the smallest footprint is one Block Group per round, thousands of rounds.
    """

    def check(problem, config):
        if config.p_k != 1:
            return "model stages require p_k=1"
        rounds = pim_hierarchy(problem, config).scheduling_rounds
        if key.has_weights and rounds != 1:
            return "weight stages require one M/N/K round"
        if rounds > max_rounds:
            return f"model stages allow at most {max_rounds} M/N/K rounds; got {rounds}"
        if budget is not None:
            return budget.reason(key, config)
        return None

    return check


@dataclass(frozen=True)
class MinimumFootprint:
    config: PIMConfig
    footprint: Footprint


def max_stage_rounds(problem, spec):
    # autotuner가 초기 후보에 허용하는 round 수와 같은 값이다.
    return build_config_space(problem, spec).max_initial_scheduling_rounds


def minimum_footprint(stage_class, spec):
    """Smallest legal footprint of a stage class; computed, never measured.

    A weight stage runs in one round, so blocks, groups and partitions fix its
    tiles. An activation stage may use up to max_stage_rounds rounds; its tiles
    split each axis into at most that many rounds.
    """
    key = stage_class.key
    problem = key.problem
    max_rounds = max_stage_rounds(problem, spec)
    engine = ConstraintEngine(problem, spec, constraint=stage_constraint(key, max_rounds))
    space = build_config_space(problem, spec)
    if key.has_weights:
        partitions = [p for p in legal_partitions(problem, spec, space) if p[2] == 1]
    else:
        # activation stage의 unit은 instance(p_b)에 쓰고, footprint는 tile만으로 정해진다.
        partitions = [(1, 1, 1)]
    best = None
    for blocks in product(*(axis_values(problem, space, "block", axis) for axis in range(3))):
        for groups in product(*(axis_values(problem, space, "group", axis) for axis in range(3))):
            total_groups = tuple(
                math.ceil(problem.axis_extents[axis] / (blocks[axis] * groups[axis]))
                for axis in range(3))
            for p in partitions:
                if key.has_weights:
                    tile_choices = [(math.ceil(total_groups[axis] / p[axis]),) for axis in range(3)]
                else:
                    tile_choices = [sorted({math.ceil(total_groups[axis] / rounds)
                                            for rounds in range(1, max_rounds + 1)})
                                    for axis in range(3)]
                for tiles in product(*tile_choices):
                    config = PIMConfig(*blocks, *groups, *tiles, *p)
                    if not engine.valid(config):
                        continue
                    size = footprint(key, spec, config)
                    total = size.weight + size.state + size.input + size.output
                    if best is None or total < best[0]:
                        best = (total, MinimumFootprint(config, size))
    if best is None:
        raise ValueError(f"{key.label()}: no legal model-stage config on this target")
    return best[1]


def decide_policy(classes, minimum, spec):
    """P-1 -> P-2 -> P-3 with every stage at its minimum footprint."""
    profile = spec.common_profile
    resources = UnitResources(
        profile.num_units, profile.banks_per_unit * profile.bank_capacity_limit_bytes)
    requirements = []
    for stage_class in classes:
        size = minimum[stage_class.key].footprint
        for layer, count in enumerate(stage_class.counts):
            requirements.extend(
                StageRequirement(layer, f"{stage_class.key.label()}#{index}",
                                 size.weight, size.state, size.input, size.output)
                for index in range(count))
    return plan_residency(resources, requirements)


def coefficients(policy, stage_class):
    """How many times a class's weight and state bytes count in the policy formula."""
    if policy == MODEL:
        return stage_class.total, stage_class.total
    if policy == WEIGHTS:
        return stage_class.total, stage_class.per_layer
    if policy == LAYER:
        return stage_class.per_layer, stage_class.per_layer
    raise ValueError(f"unknown residency policy {policy}")


def policy_bytes(policy, stage_class, size):
    weight, state = coefficients(policy, stage_class)
    return weight * size.weight + state * size.state


class PolicyBudget:
    """A candidate fits if the model fits with every other class at its minimum."""

    def __init__(self, policy, classes, minimum, spec):
        profile = spec.common_profile
        self.policy = policy
        self.spec = spec
        self.capacity = profile.banks_per_unit * profile.bank_capacity_limit_bytes
        self.classes = {stage_class.key: stage_class for stage_class in classes}
        self.minimum = {key: minimum[key].footprint for key in self.classes}

    def needed(self, key, size):
        others = [other for other in self.classes if other != key]
        resident = policy_bytes(self.policy, self.classes[key], size) + sum(
            policy_bytes(self.policy, self.classes[other], self.minimum[other]) for other in others)
        scratch_in = max([size.input] + [self.minimum[other].input for other in others])
        scratch_out = max([size.output] + [self.minimum[other].output for other in others])
        return resident + scratch_in + scratch_out

    def reason(self, key, config):
        needed = self.needed(key, footprint(key, self.spec, config))
        if needed > self.capacity:
            return (f"policy {self.policy} needs {needed} bytes per unit with this config; "
                    f"capacity is {self.capacity}")
        return None
