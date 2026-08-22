# Stage 8, 10~15 — LFBO 탐색 loop (design/AutoTuner_Design.md §33, §37~§44, §52)
#
#   Initial Seeds → Benchmark → Dataset → Random Forest
#     → Tree-guided Candidate Generation → Constraint → Planner
#     → Plan Features → RF Scoring → Diversity Selection
#     → Compile + Benchmark → Retrain ↺                       (§43)

import math
import random
import time
from dataclasses import dataclass

import numpy as np

from .constraints import ConstraintEngine
from .features import config_plan_vector, extract_plan_features, feature_names
from .forest import RandomForestClassifier
from .planner import PIMPlanner
from .space import (build_config_space, heuristic_seeds, make_config,
                    random_valid_configs)


@dataclass
class SearchBudget:
    """Stage 15 — search termination 조건 (§44)."""

    max_generations: int = 10
    max_benchmarks: int = 100
    max_autotune_seconds: float = None
    improvement_epsilon: float = 0.01
    patience: int = 3


@dataclass
class TuningRecord:
    """benchmark 하나의 전체 기록: X_i = [Config, PlanFeature], y_i (§33)."""

    config: object
    plan: object
    features: object
    result: object


@dataclass
class TuningResult:
    best: TuningRecord
    records: list


@dataclass
class ScoredCandidate:
    """Stage 12 — 실행 전 RF로 점수만 매긴 candidate (§41)."""

    config: object
    plan: object
    features: object
    probability: float


# §39 — Feature → Config influence mapping.
# tree가 중요하게 본 결과 feature는 직접 tuning parameter가 아니므로,
# 그 feature에 영향을 주는 semantic fragment로 되돌린다.
PLAN_FEATURE_INFLUENCE = {
    "active_units": ("distribution", "execution_m", "execution_n"),
    "effective_utilization": ("distribution", "execution_m", "execution_n"),
    "execution_tile_count": ("execution_m", "execution_n", "execution_k"),
    "scheduling_rounds": ("distribution", "execution_m", "execution_n"),
    "h2d_bytes": ("distribution", "execution_m", "execution_n", "execution_k",
                  "residency", "streaming"),
    "d2h_bytes": ("ksplit", "execution_m", "execution_n"),
    "a_replication_bytes": ("distribution", "execution_m", "execution_n",
                            "residency", "replication"),
    "b_replication_bytes": ("distribution", "execution_m", "execution_n",
                            "residency", "replication"),
    "a_resident_bytes": ("residency", "replication"),
    "b_resident_bytes": ("residency", "replication"),
    "reduction_bytes": ("ksplit", "execution_k"),
    "reduction_rounds": ("ksplit",),
    "padding_ratio": ("execution_m", "execution_n", "execution_k"),
    "load_imbalance": ("distribution", "execution_m", "execution_n"),
    "local_memory_pressure": ("block_m", "block_n", "block_k",
                              "group_m", "group_n", "group_k"),
    "device_memory_pressure": ("execution_m", "execution_n", "execution_k",
                               "residency", "streaming"),
    "descriptor_build_count": ("persistence",),
    "descriptor_transfer_bytes": ("persistence",),
    "descriptor_reuse_factor": ("persistence",),
    "command_build_count": ("persistence",),
    "kernel_launch_count": ("persistence", "execution_m", "execution_n"),
    "device_alloc_count": ("persistence",),
}


def influenced_fragments(feature_name):
    """feature 이름 하나를 그 feature를 바꿀 수 있는 fragment로 mapping (§39)."""
    if feature_name in PLAN_FEATURE_INFLUENCE:
        return PLAN_FEATURE_INFLUENCE[feature_name]
    if feature_name.startswith("log2_block_"):
        return ("block_" + feature_name[-1],)
    if feature_name.startswith("log2_group_"):
        return ("group_" + feature_name[-1],)
    if feature_name.startswith("log2_tile_"):
        # tile 크기는 derived이므로 execution multiplier로 조절한다 (§5).
        return ("execution_" + feature_name[-1],)
    if feature_name in ("p_m", "p_n"):
        return ("distribution",)
    if feature_name == "p_k":
        return ("ksplit",)
    if feature_name.startswith("distribution="):
        return ("distribution",)
    if feature_name.startswith("reduction="):
        return ("ksplit",)
    if feature_name in ("a_residency", "b_residency"):
        return ("residency",)
    if feature_name.startswith("replication="):
        return ("replication",)
    if feature_name.startswith("streaming="):
        return ("streaming",)
    if feature_name in ("device_set_reuse", "descriptor_reuse",
                        "command_reuse", "persistent_execution"):
        return ("persistence",)
    raise ValueError(f"알 수 없는 feature 이름: {feature_name}")


_AXIS_INDEX = {"m": 0, "n": 1, "k": 2}


def _step_values(values, current):
    """value 목록에서 current의 위·아래 이웃 값."""
    index = values.index(current)
    neighbors = []
    if index > 0:
        neighbors.append(values[index - 1])
    if index + 1 < len(values):
        neighbors.append(values[index + 1])
    return neighbors


def semantic_mutate(config, fragment_kinds, problem, hardware, spec):
    """Stage 11 — 모델이 중요하다고 본 fragment만 바꾼 neighbor 생성 (§38, §40).

    모든 knob를 random하게 흔들지 않고, 지목된 fragment 하나씩만 바꾼
    configuration을 만든다. knob이 바뀌면 P_M/P_N은 다시 계산된다 (§6.1).
    """
    blocks = (config.block_m, config.block_n, config.block_k)
    groups = (config.group_m, config.group_n, config.group_k)
    executions = (config.execution_m, config.execution_n, config.execution_k)
    policy = config.distribution_policy
    strategy_index = spec.ksplit_strategies.index(
        (config.p_k, config.reduction_policy))
    persistence_index = spec.persistence_strategies.index(
        config.persistence_strategy())

    def rebuild(new_blocks=blocks, new_groups=groups,
                new_executions=executions, new_policy=policy,
                new_strategy_index=strategy_index,
                a_residency=config.a_residency,
                b_residency=config.b_residency,
                replication_policy=config.replication_policy,
                streaming_policy=config.streaming_policy,
                new_persistence_index=persistence_index):
        return make_config(problem, hardware, new_blocks, new_groups,
                           new_executions, new_policy,
                           spec.ksplit_strategies[new_strategy_index],
                           a_residency=a_residency, b_residency=b_residency,
                           replication_policy=replication_policy,
                           streaming_policy=streaming_policy,
                           persistence_strategy=spec.persistence_strategies[
                               new_persistence_index])

    neighbors = []
    for kind in fragment_kinds:
        if kind.startswith("block_"):
            axis = _AXIS_INDEX[kind[-1]]
            for value in _step_values(spec.block_values, blocks[axis]):
                new = list(blocks)
                new[axis] = value
                neighbors.append(rebuild(new_blocks=tuple(new)))
        elif kind.startswith("group_"):
            axis = _AXIS_INDEX[kind[-1]]
            for value in _step_values(spec.group_values, groups[axis]):
                new = list(groups)
                new[axis] = value
                neighbors.append(rebuild(new_groups=tuple(new)))
        elif kind.startswith("execution_"):
            axis = _AXIS_INDEX[kind[-1]]
            for value in _step_values(spec.execution_values, executions[axis]):
                new = list(executions)
                new[axis] = value
                neighbors.append(rebuild(new_executions=tuple(new)))
        elif kind == "distribution":
            for other in spec.distribution_policies:
                if other != policy:
                    neighbors.append(rebuild(new_policy=other))
        elif kind == "ksplit":
            for index in (strategy_index - 1, strategy_index + 1):
                if 0 <= index < len(spec.ksplit_strategies):
                    neighbors.append(rebuild(new_strategy_index=index))
        elif kind == "residency":
            for other in spec.residency_policies:
                if other != config.a_residency:
                    neighbors.append(rebuild(a_residency=other))
                if other != config.b_residency:
                    neighbors.append(rebuild(b_residency=other))
        elif kind == "replication":
            for other in spec.replication_policies:
                if other != config.replication_policy:
                    neighbors.append(rebuild(replication_policy=other))
        elif kind == "streaming":
            for other in spec.streaming_policies:
                if other != config.streaming_policy:
                    neighbors.append(rebuild(streaming_policy=other))
        elif kind == "persistence":
            for index in (persistence_index - 1, persistence_index + 1):
                if 0 <= index < len(spec.persistence_strategies):
                    neighbors.append(rebuild(new_persistence_index=index))
        else:
            raise ValueError(f"알 수 없는 fragment 종류: {kind}")
    return neighbors


def make_quantile_labels(records, good_quantile):
    """Stage 8 — 상위 quantile을 GOOD=1로 분류한다 (§33).

    compile/runtime 실패는 BAD sample로 기록한다 (§35).
    """
    ok = [r for r in records if r.result.ok]
    if not ok:
        return [0] * len(records)
    n_good = max(1, math.ceil(len(ok) * good_quantile))
    good = set(
        id(r) for r in sorted(ok, key=lambda r: r.result.total_ms)[:n_good])
    return [1 if id(r) in good else 0 for r in records]


def _similarity(a, b):
    """두 config가 공유하는 knob 값의 비율 (§42)."""
    knobs_a = a.knobs()
    knobs_b = b.knobs()
    same = sum(1 for x, y in zip(knobs_a, knobs_b) if x == y)
    return same / len(knobs_a)


def rank_with_diversity_penalty(scored, count, diversity_weight):
    """Stage 13 — high P(GOOD)이면서 이미 뽑힌 것과 다른 candidate 선호 (§42)."""
    selected = []
    remaining = list(scored)
    while remaining and len(selected) < count:
        def adjusted(candidate):
            if not selected:
                return candidate.probability
            penalty = max(_similarity(candidate.config, s.config)
                          for s in selected)
            return candidate.probability - diversity_weight * penalty
        best = max(remaining, key=adjusted)
        remaining.remove(best)
        selected.append(best)
    return selected


def tune(problem, hardware, benchmark_fn, *, space=None, cache=None,
         budget=None, seed=0, initial_random=10, search_copies=5,
         select_per_generation=8, good_quantile=0.1, diversity_weight=0.2,
         verbose=False):
    """§52 AutoTuner pseudocode의 구현.

    benchmark_fn(problem, hardware, config, plan) -> PIMBenchmarkResult 가
    Stage 6~7(candidate lowering, real hardware benchmark)을 수행한다.
    """
    spec = space if space is not None else build_config_space(problem, hardware)
    budget = budget if budget is not None else SearchBudget()
    engine = ConstraintEngine(problem, hardware)
    planner = PIMPlanner(problem, hardware)
    rng = random.Random(seed)
    names = feature_names(spec)
    start_time = time.monotonic()

    records = []
    seen = set()

    def out_of_budget():
        if len(records) >= budget.max_benchmarks:
            return True
        if (budget.max_autotune_seconds is not None
                and time.monotonic() - start_time
                >= budget.max_autotune_seconds):
            return True
        return False

    def measure(config, plan, features):
        # config 표기는 측정 전에 출력한다 — compile 단계에서 실패해도
        # 어떤 configuration이었는지 로그에 남는다.
        if verbose:
            grid = plan.active_device_grid
            print(f"[autotuner] "
                  f"B=({config.block_m},{config.block_n},{config.block_k}) "
                  f"G=({config.group_m},{config.group_n},{config.group_k}) "
                  f"tile={plan.execution_tile_shape} "
                  f"P={grid} {config.distribution_policy}/"
                  f"{config.reduction_policy}", end="", flush=True)
        result = benchmark_fn(problem, hardware, config, plan)
        result.config = config
        result.plan = plan
        result.plan_features = features
        record = TuningRecord(config, plan, features, result)
        records.append(record)
        if verbose:
            print(f" -> {result.total_ms:.2f} ms (ok={result.ok})")
        return record

    def best_ms():
        ok = [r for r in records if r.result.ok]
        if not ok:
            return None
        return min(r.result.total_ms for r in ok)

    # ── initial population (§7, §46, §52) ──────────────────────────────
    population = heuristic_seeds(problem, hardware, spec)
    if cache is not None:
        # warm start (§46): 과거 best의 knob을 가져오되, P_M/P_N은 현재
        # problem의 tile grid에 맞게 다시 계산한다 (§6.1).
        for cached in cache.lookup_similar(problem, hardware):
            population.append(make_config(
                problem, hardware,
                (cached.block_m, cached.block_n, cached.block_k),
                (cached.group_m, cached.group_n, cached.group_k),
                (cached.execution_m, cached.execution_n, cached.execution_k),
                cached.distribution_policy,
                (cached.p_k, cached.reduction_policy),
                a_residency=cached.a_residency,
                b_residency=cached.b_residency,
                replication_policy=cached.replication_policy,
                streaming_policy=cached.streaming_policy,
                persistence_strategy=cached.persistence_strategy()))
    population += random_valid_configs(
        problem, hardware, spec, initial_random, rng, engine)

    for config in population:
        if config.key() in seen:
            continue
        seen.add(config.key())
        if not engine.valid(config):
            continue
        if out_of_budget():
            break
        plan = planner.build(config)
        features = extract_plan_features(problem, hardware, config, plan)
        measure(config, plan, features)

    # ── iterative search (§43) ─────────────────────────────────────────
    generation = 0
    stale_generations = 0
    while generation < budget.max_generations and not out_of_budget():
        generation += 1

        # Stage 8~9 — dataset과 surrogate (§33, §36).
        labels = make_quantile_labels(records, good_quantile)
        if len(set(labels)) < 2:
            break
        X = np.asarray(
            [config_plan_vector(r.config, r.features, spec) for r in records],
            dtype=float)
        forest = RandomForestClassifier(seed=rng.randrange(2 ** 31))
        forest.fit(X, np.asarray(labels))

        # Stage 10 — search copies (§37): best configuration 여러 개 유지.
        ok_records = [r for r in records if r.result.ok]
        copies = sorted(
            ok_records, key=lambda r: r.result.total_ms)[:search_copies]

        # Stage 11 — tree-guided neighbor 생성 (§38~§40).
        candidates = []
        for record in copies:
            tree = forest.sample_tree(rng)
            vector = np.asarray(
                config_plan_vector(record.config, record.features, spec),
                dtype=float)
            path = tree.decision_path(vector)
            fragments = []
            for index in path:
                for fragment in influenced_fragments(names[index]):
                    if fragment not in fragments:
                        fragments.append(fragment)
            candidates += semantic_mutate(
                record.config, fragments, problem, hardware, spec)

        # Stage 12 — candidate scoring (§41): 실행 없이 RF로 P(GOOD) 계산.
        scored = []
        scored_keys = set()
        for config in candidates:
            key = config.key()
            if key in seen or key in scored_keys:
                continue
            scored_keys.add(key)
            if not engine.valid(config):
                continue
            plan = planner.build(config)
            features = extract_plan_features(problem, hardware, config, plan)
            probability = forest.predict_prob(
                config_plan_vector(config, features, spec))
            scored.append(ScoredCandidate(config, plan, features, probability))

        # Stage 13 — diversity 고려 batch 선택 (§42).
        selected = rank_with_diversity_penalty(
            scored, select_per_generation, diversity_weight)
        if not selected:
            break

        previous_best = best_ms()
        for candidate in selected:
            if out_of_budget():
                break
            seen.add(candidate.config.key())
            measure(candidate.config, candidate.plan, candidate.features)

        # Stage 15 — improvement threshold / patience (§44).
        new_best = best_ms()
        improvement = (previous_best - new_best) / previous_best
        if improvement < budget.improvement_epsilon:
            stale_generations += 1
        else:
            stale_generations = 0
        if verbose:
            print(f"[autotuner] generation {generation}: best {new_best:.2f} ms"
                  f" (improvement {improvement:.3f}, stale {stale_generations})")
        if stale_generations >= budget.patience:
            break

    ok_records = [r for r in records if r.result.ok]
    assert ok_records, "성공적으로 측정된 configuration이 없습니다"
    best = min(ok_records, key=lambda r: r.result.total_ms)
    if cache is not None:
        cache.store(problem, hardware, best)
    return TuningResult(best=best, records=records)
