"""Resolve one GEMM to a PIM Common IR configuration."""

import logging
from dataclasses import dataclass

from ._autotuner import (
    ceil_div,
    ConstraintEngine,
    PIMConfig,
    PIMConfigCache,
    PIMProblem,
    PIMTargetSpec,
    build_config_space,
    heuristic_seeds,
    pim_launch_options,
    target_spec,
)
from .config import PIMTorchConfig, UnsupportedPIMIntMM, validate_pim_options


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
            "SPLIT_K": self.config.p_k,
            "EVEN_TILES": even_tiles(self.problem, self.config),
        }

    def backend_options(self):
        return validate_pim_options(pim_launch_options(
            self.config, self.target))

    def describe(self):
        config = self.config
        profile = self.target.common_profile
        t_m, t_n, _ = config.execution_extent
        e_m = ceil_div(self.problem.M, t_m)
        e_n = ceil_div(self.problem.N, t_n)
        rounds = ceil_div(e_m, config.p_m) * ceil_div(e_n, config.p_n)
        active = config.p_m * config.p_n * config.p_k
        return (
            f"block=({config.block_m},{config.block_n},{config.block_k}) "
            f"group=({config.group_m},{config.group_n},{config.group_k}) "
            f"tile=({config.tile_m},{config.tile_n},{config.tile_k}) "
            f"extent={config.execution_extent} "
            f"devices=({config.p_m},{config.p_n},{config.p_k})="
            f"{active}/{profile.num_units} rounds={rounds}")


def even_tiles(problem, config):
    return (
        problem.M % config.block_m == 0
        and problem.N % config.block_n == 0
    )


def check_template_support(problem, config, torch_config):
    divisor = config.p_k * config.block_k
    if problem.K % divisor:
        raise UnsupportedPIMIntMM(
            f"K={problem.K} must be divisible by "
            f"p_k*BLOCK_K={config.p_k}*{config.block_k}={divisor}")
    if not even_tiles(problem, config) and not torch_config.allow_masked_tiles:
        raise UnsupportedPIMIntMM(
            f"M={problem.M} must be divisible by BLOCK_M={config.block_m} "
            f"and N={problem.N} by BLOCK_N={config.block_n}")


def _seed_config(problem, target):
    space = build_config_space(problem, target)
    engine = ConstraintEngine(problem, target)
    for config in heuristic_seeds(problem, target, space):
        if engine.valid(config):
            return config
    raise UnsupportedPIMIntMM("no valid Common IR seed")


def _cached_config(problem, target, torch_config):
    cache = PIMConfigCache(torch_config.autotune_cache_path)
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
    check_template_support(problem, config, torch_config)
    return PIMExecution(
        problem=problem, target=target, config=config,
        torch_config=torch_config)
