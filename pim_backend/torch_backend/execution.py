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
from .config import PIMTorchConfig, UnsupportedPIMIntMM, validate_pim_options
from .template import TORCH_INT_MM_TEMPLATE_VERSION


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
    raise UnsupportedPIMIntMM("no valid Common IR seed")


def _cached_config(problem, target, torch_config):
    cache = PIMConfigCache(
        torch_config.autotune_cache_path,
        template_version=TORCH_INT_MM_TEMPLATE_VERSION,
    )
    found = cache.lookup_result(problem, target)
    if found is None:
        raise UnsupportedPIMIntMM(
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
        raise UnsupportedPIMIntMM(
            f"config rejected for (M,N,K)=({problem.M},{problem.N},"
            f"{problem.K}): {reason}")
    return PIMExecution(
        problem=problem, target=target, config=config,
        torch_config=torch_config)
