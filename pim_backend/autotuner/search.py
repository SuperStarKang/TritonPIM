"""LFBO search over the values materialized in PIM Common IR."""

import math
import random
import time
from dataclasses import dataclass, replace

import numpy as np

from .constraints import ConstraintEngine
from .features import (
    config_feature_vector,
    extract_features,
    feature_names,
)
from .forest import RandomForestClassifier
from .model import target_model
from .space import axis_values, build_config_space, heuristic_seeds, make_config
from .space import ladder_neighbors, legal_partitions, regroup
from .space import structured_valid_configs
from .types import PIMBenchmarkResult, pim_hierarchy


@dataclass
class SearchBudget:
    max_generations: int = 10
    max_benchmarks: int = 100
    max_autotune_seconds: float = None
    # 측정 noise 수준. 이보다 크게 좋아진 generation은 stale을 0으로 되돌린다.
    improvement_epsilon: float = 0.002
    patience: int = 3


@dataclass
class TuningRecord:
    config: object
    features: object
    result: object


@dataclass
class TuningResult:
    best: TuningRecord
    records: list
    from_cache: bool = False


@dataclass
class ScoredCandidate:
    config: object
    features: object
    probability: float


FEATURE_INFLUENCE = {
    "block_group_compute_per_byte": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k"),
    "execution_tile_compute_per_byte": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k"),
    "active_units": ("p_m", "p_n", "p_k"),
    "useful_execution_tiles": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k"),
    "scheduled_execution_tiles": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k", "p_m", "p_n", "p_k"),
    "schedule_utilization": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k", "p_m", "p_n", "p_k"),
    "scheduling_rounds": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k", "p_m", "p_n", "p_k"),
    "stage_bytes": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k", "p_m", "p_n", "p_k"),
    "h2d_bytes": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k", "p_m", "p_n", "p_k"),
    "d2h_bytes": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k", "p_m", "p_n", "p_k"),
    "reduction_bytes": ("block_k", "group_k", "tile_k", "p_k"),
    "padding_ratio": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k", "p_m", "p_n", "p_k"),
    "local_memory_pressure": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k"),
    "unit_memory_pressure": (
        "block_m", "block_n", "block_k",
        "group_m", "group_n", "group_k",
        "tile_m", "tile_n", "tile_k"),
}


def influenced_fragments(feature_name):
    if feature_name in FEATURE_INFLUENCE:
        return FEATURE_INFLUENCE[feature_name]
    if feature_name.startswith("log2_"):
        return (feature_name[5:],)
    if feature_name in ("p_m", "p_n", "p_k"):
        return (feature_name,)
    raise ValueError(f"알 수 없는 feature 이름: {feature_name}")


_AXIS_INDEX = {"m": 0, "n": 1, "k": 2}


def _step_values(values, current):
    ordered = sorted(set(values) | {current})
    index = ordered.index(current)
    neighbors = []
    if index > 0:
        neighbors.append(ordered[index - 1])
    if index + 1 < len(ordered):
        neighbors.append(ordered[index + 1])
    return neighbors


def semantic_mutate(problem, target, config, fragment_kinds, spec, legal):
    """Neighbors of ``config`` for the fragments a decision path used.

    A Block or Group change moves to the new Block Group's ladder point with
    the closest Execution Tile count, so the frontier position is kept.
    A Tile or partition change also adds the adjacent frontier points.
    ``legal`` is ``legal_partitions(...)``.
    """
    blocks = (config.block_m, config.block_n, config.block_k)
    groups = (config.group_m, config.group_n, config.group_k)
    tiles = (config.tile_m, config.tile_n, config.tile_k)
    partitions = (config.p_m, config.p_n, config.p_k)

    def rebuild(new_blocks=blocks, new_groups=groups, new_tiles=tiles,
                new_partitions=partitions):
        return make_config(
            new_blocks, new_groups, new_tiles, new_partitions)

    def regrouped(new_blocks, new_groups):
        moved = regroup(
            problem, target, config, new_blocks, new_groups, spec, legal)
        if moved is not None:
            neighbors.append(moved)

    neighbors = []
    for kind in fragment_kinds:
        axis = _AXIS_INDEX[kind[-1]]
        if kind.startswith("block_"):
            for value in _step_values(
                    axis_values(problem, spec, "block", axis), blocks[axis]):
                new = list(blocks)
                new[axis] = value
                regrouped(tuple(new), groups)
        elif kind.startswith("group_"):
            for value in _step_values(
                    axis_values(problem, spec, "group", axis), groups[axis]):
                new = list(groups)
                new[axis] = value
                regrouped(blocks, tuple(new))
        elif kind.startswith("tile_"):
            for value in _step_values(
                    axis_values(problem, spec, "tile", axis), tiles[axis]):
                new = list(tiles)
                new[axis] = value
                neighbors.append(rebuild(new_tiles=tuple(new)))
        elif kind.startswith("p_"):
            for value in _step_values(
                    axis_values(problem, spec, "partition", axis),
                    partitions[axis]):
                new = list(partitions)
                new[axis] = value
                neighbors.append(rebuild(new_partitions=tuple(new)))
        else:
            raise ValueError(f"알 수 없는 fragment 종류: {kind}")
    if any(kind.startswith(("tile_", "p_")) for kind in fragment_kinds):
        neighbors.extend(
            ladder_neighbors(problem, target, config, spec, legal))
    return neighbors


def make_quantile_labels(records, good_quantile):
    ok = [record for record in records if record.result.ok]
    if not ok:
        return [0] * len(records)
    n_good = max(1, math.ceil(len(ok) * good_quantile))
    good = set(
        id(record)
        for record in sorted(
            ok, key=lambda record: record.result.total_ms)[:n_good]
    )
    return [1 if id(record) in good else 0 for record in records]


def _similarity(a, b):
    knobs_a = a.key()
    knobs_b = b.key()
    same = sum(x == y for x, y in zip(knobs_a, knobs_b))
    return same / len(knobs_a)


def rank_with_diversity_penalty(scored, count, diversity_weight):
    selected = []
    remaining = list(scored)
    while remaining and len(selected) < count:
        def adjusted(candidate):
            if not selected:
                return candidate.probability
            penalty = max(
                _similarity(candidate.config, item.config)
                for item in selected)
            return candidate.probability - diversity_weight * penalty

        best = max(remaining, key=adjusted)
        remaining.remove(best)
        selected.append(best)
    return selected


def tune(problem, target, benchmark_fn, *, space=None, cache=None,
         budget=None, seed=0, initial_random=6, search_copies=5,
         select_per_generation=8, good_quantile=0.1,
         diversity_weight=0.2, retune=False, verbose=False):
    spec = space if space is not None else build_config_space(problem, target)
    budget = budget if budget is not None else SearchBudget()
    engine = ConstraintEngine(problem, target)
    model = target_model(target)
    partitions = legal_partitions(problem, target, spec)
    rng = random.Random(seed)
    names = feature_names()
    start_time = time.monotonic()

    if cache is not None and not retune:
        hit = cache.lookup_result(problem, target)
        if hit is not None:
            config, result = hit
            features = extract_features(problem, target, config)
            if result.physical_allocations:
                features = replace(
                    features, unit_memory_pressure=model.physical_pressure(
                        result.physical_allocations))
            record = TuningRecord(config, features, result)
            if verbose:
                print(
                    f"[autotuner] cache hit: {result.total_ms:.2f} ms "
                    f"config={config.key()}")
            return TuningResult(
                best=record, records=[record], from_cache=True)

    measured = (cache.measured_results(problem, target)
                if cache is not None and not retune else {})
    if verbose:
        mode = "retune" if retune else "cache miss"
        print(
            f"[autotuner] {mode}: problem="
            f"({problem.M},{problem.N},{problem.K}) "
            f"target={target.common_profile.target}")
    records = []
    seen = set()
    benchmarks_run = 0

    def out_of_budget():
        if benchmarks_run >= budget.max_benchmarks:
            return True
        return (
            budget.max_autotune_seconds is not None
            and time.monotonic() - start_time
            >= budget.max_autotune_seconds
        )

    def measure(config, features):
        nonlocal benchmarks_run
        if verbose:
            extent = config.execution_extent
            hierarchy = pim_hierarchy(problem, config)
            print(
                f"[autotuner] "
                f"B=({config.block_m},{config.block_n},{config.block_k}) "
                f"G=({config.group_m},{config.group_n},{config.group_k}) "
                f"tile=({config.tile_m},{config.tile_n},{config.tile_k}) "
                f"extent={extent} "
                f"P=({config.p_m},{config.p_n},{config.p_k}) "
                f"groups={hierarchy.total_groups} "
                f"tiles={hierarchy.execution_tiles} "
                f"rounds={hierarchy.rounds} "
                f"compute/byte=({features.block_group_compute_per_byte:.3f},"
                f"{features.execution_tile_compute_per_byte:.3f}) "
                f"schedule_util={features.schedule_utilization:.3f} "
                f"bytes(stage/h2d/d2h/reduction)="
                f"({features.stage_bytes},{features.h2d_bytes},"
                f"{features.d2h_bytes},{features.reduction_bytes})",
                end="", flush=True)
        stored = measured.get(config.key())
        if stored is None:
            benchmarks_run += 1
            result = benchmark_fn(problem, target, config)
            source = ""
        else:
            result = PIMBenchmarkResult(**stored)
            source = " (cache)"
        if result.physical_allocations:
            features = replace(
                features, unit_memory_pressure=model.physical_pressure(
                    result.physical_allocations))
        record = TuningRecord(config, features, result)
        records.append(record)
        if verbose:
            stages = "".join(
                f" {name}={milliseconds:.2f}"
                for name, milliseconds in result.timings)
            print(
                f" -> {result.total_ms:.2f} ms (correctness={result.correctness})"
                f"{stages}{source}")
        return record

    def best_ms():
        ok = [record for record in records if record.result.ok]
        if not ok:
            return None
        return min(record.result.total_ms for record in ok)

    seeds = heuristic_seeds(problem, target, spec)
    structured_configs = structured_valid_configs(
        problem, target, spec, initial_random, rng, engine)
    warm = cache.lookup_similar(problem, target) if cache is not None else []
    population = seeds[:1] + warm + seeds[1:] + structured_configs

    for config in population:
        if config.key() in seen or not engine.valid(config):
            continue
        seen.add(config.key())
        if config.key() not in measured and out_of_budget():
            continue
        measure(config, extract_features(problem, target, config))

    generation = 0
    stale_generations = 0
    while generation < budget.max_generations and not out_of_budget():
        generation += 1
        labels = make_quantile_labels(records, good_quantile)
        if len(set(labels)) < 2:
            break
        X = np.asarray([
            config_feature_vector(record.config, record.features)
            for record in records
        ], dtype=float)
        forest = RandomForestClassifier(seed=rng.randrange(2 ** 31))
        forest.fit(X, np.asarray(labels))

        copies = sorted(
            (record for record in records if record.result.ok),
            key=lambda record: record.result.total_ms,
        )[:search_copies]
        candidates = []
        for record in copies:
            vector = np.asarray(
                config_feature_vector(record.config, record.features),
                dtype=float)
            path = forest.sample_tree(rng).decision_path(vector)
            fragments = []
            for index in path:
                for fragment in influenced_fragments(names[index]):
                    if fragment not in fragments:
                        fragments.append(fragment)
            candidates.extend(semantic_mutate(
                problem, target, record.config, fragments, spec, partitions))

        scored = []
        scored_keys = set()
        for config in candidates:
            key = config.key()
            if key in seen or key in scored_keys or not engine.valid(config):
                continue
            scored_keys.add(key)
            features = extract_features(problem, target, config)
            probability = forest.predict_prob(
                config_feature_vector(config, features))
            scored.append(ScoredCandidate(config, features, probability))

        selected = rank_with_diversity_penalty(
            scored, select_per_generation, diversity_weight)
        if not selected:
            break
        previous_best = best_ms()
        for candidate in selected:
            if candidate.config.key() not in measured and out_of_budget():
                continue
            seen.add(candidate.config.key())
            measure(candidate.config, candidate.features)

        new_best = best_ms()
        improvement = (previous_best - new_best) / previous_best
        if improvement < budget.improvement_epsilon:
            stale_generations += 1
        else:
            stale_generations = 0
        if verbose:
            print(
                f"[autotuner] generation {generation}: "
                f"best {new_best:.2f} ms "
                f"(improvement {improvement:.3f}, "
                f"stale {stale_generations})")
        if stale_generations >= budget.patience:
            break

    ok_records = [record for record in records if record.result.ok]
    assert ok_records, "성공적으로 측정된 configuration이 없습니다"
    best = min(ok_records, key=lambda record: record.result.total_ms)
    if cache is not None:
        cache.store(problem, target, best, records, replace=retune)
    if verbose:
        print(
            f"[autotuner] selected: {best.result.total_ms:.2f} ms "
            f"config={best.config.key()}")
    return TuningResult(best=best, records=records)
