"""Resolve one GEMM to a PIM Common IR configuration."""

import logging
from dataclasses import dataclass

from ._autotuner import (
    ConstraintEngine,
    PIMConfig,
    PIMConfigCache,
    PIMProblem,
    PIMTargetSpec,
    build_config_space,
    heuristic_seeds,
    pim_hierarchy,
    pim_launch_options,
    target_spec,
)
from .config import PIMTorchConfig, UnsupportedPIMGemm, validate_pim_options
from .template import TORCH_GEMM_TEMPLATE_VERSION


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PIMExecution:
    """Values needed to compile one Triton template through PIM Common IR."""

    problem: PIMProblem
    target: PIMTargetSpec
    config: PIMConfig
    torch_config: PIMTorchConfig

    def template_meta(self):
        return {
            "BLOCK_M": self.config.block_m,
            "BLOCK_N": self.config.block_n,
            "BLOCK_K": self.config.block_k,
            "ACC_TYPE": f"tl.{self.problem.output_dtype}",
        }

    def backend_options(self):
        return validate_pim_options(pim_launch_options(
            self.config, self.target))

    def describe(self):
        config = self.config
        profile = self.target.common_profile
        hierarchy = pim_hierarchy(self.problem, config)
        active = config.p_m * config.p_n * config.p_k
        return (
            f"block=({config.block_m},{config.block_n},{config.block_k}) "
            f"group=({config.group_m},{config.group_n},{config.group_k}) "
            f"tile=({config.tile_m},{config.tile_n},{config.tile_k}) "
            f"extent={config.execution_extent} "
            f"devices=({config.p_m},{config.p_n},{config.p_k})="
            f"{active}/{profile.num_units} "
            f"groups={hierarchy.total_groups} "
            f"tiles={hierarchy.execution_tiles} "
            f"rounds={hierarchy.rounds}")


def _seed_config(problem, target):
    space = build_config_space(problem, target)
    engine = ConstraintEngine(problem, target)
    for config in heuristic_seeds(problem, target, space):
        if engine.valid(config):
            return config
    raise UnsupportedPIMGemm("no valid Common IR seed")


def _cached_config(problem, target, torch_config):
    cache = PIMConfigCache(
        torch_config.autotune_cache_path,
        template_version=TORCH_GEMM_TEMPLATE_VERSION,
    )
    found = cache.lookup_result(problem, target)
    if found is None:
        raise UnsupportedPIMGemm(
            f"autotune cache miss for (M,N,K)=({problem.M},{problem.N},"
            f"{problem.K}) at {torch_config.autotune_cache_path}")
    config, result = found
    measured = len(cache.measured_results(problem, target))
    log.info(
        "PIM autotune cache hit (M,N,K)=(%d,%d,%d): %.2f ms, "
        "measured configs=%d",
        problem.M, problem.N, problem.K, result.total_ms, measured)
    return config


def resolve_execution(torch_config, problem):
    target = (torch_config.spec if torch_config.spec is not None
              else target_spec("upmem"))

    if torch_config.config is not None:
        config = torch_config.config
    elif torch_config.autotune_cache_path is not None:
        config = _cached_config(problem, target, torch_config)
    else:
        config = _seed_config(problem, target)

    reason = ConstraintEngine(problem, target).check(config)
    if reason is not None:
        raise UnsupportedPIMGemm(
            f"config rejected for (M,N,K)=({problem.M},{problem.N},"
            f"{problem.K}): {reason}")
    return PIMExecution(
        problem=problem, target=target, config=config,
        torch_config=torch_config)


def select_layer_config(problem, target, torch_config, instance_count, *, has_weights):
    """Choose an existing configuration after the session has acquired its units."""
    if torch_config.config is not None:
        candidates = ((torch_config.config, 0.0),)
    elif torch_config.autotune_cache_path is not None:
        cache = PIMConfigCache(torch_config.autotune_cache_path,
                               template_version=TORCH_GEMM_TEMPLATE_VERSION)
        candidates = tuple((PIMConfig(*key), record["total_ms"])
                           for key, record in cache.measured_results(problem, target).items()
                           if record["correctness"])
    else:
        raise ValueError("model layers require an explicit config or measured autotune records")
    engine = ConstraintEngine(problem, target)
    choices = []
    lanes = target.runtime_config.nr_tasklets
    if lanes == 0:
        lanes = target.common_profile.lanes_per_core
    for config, cost in candidates:
        if config.p_k != 1 or not engine.valid(config):
            continue
        if not 1 <= lanes <= target.common_profile.lanes_per_core:
            continue
        if config.block_m * config.group_m % lanes:
            continue
        hierarchy = pim_hierarchy(problem, config)
        if has_weights and (instance_count != 1 or hierarchy.scheduling_rounds != 1):
            continue
        units = config.p_m * config.p_n * config.p_k
        p_b = min(instance_count, target.common_profile.num_units // units)
        rounds = (instance_count + p_b - 1) // p_b
        choices.append((rounds * cost, config, p_b, rounds, hierarchy.rounds))
    if not choices:
        raise ValueError(
            "no model config fits the actual resources; resident weights require "
            "one instance and one M/N/K round"
        )
    _, config, p_b, rounds, mnk_rounds = min(choices, key=lambda choice: choice[0])
    return config, p_b, rounds, mnk_rounds
