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
            # BF16 출력도 FP32로 누산한다 (Triton은 BF16 누산을 받지 않는다). store가 출력 dtype으로 바꾼다.
            "ACC_TYPE": "tl.int32" if self.problem.output_dtype == "int32" else "tl.float32",
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
        # correctness None은 검사를 생략한 측정이다 (PIMBenchmarkResult.ok와 같은 의미).
        candidates = tuple((PIMConfig(*key), record["total_ms"])
                           for key, record in cache.measured_results(problem, target).items()
                           if record["correctness"] is not False)
    else:
        raise ValueError("model layers require an explicit config or measured autotune records")
    # Target 고유 조건(UPMEM tasklet 분할, AiMX 할당 등)은 ConstraintEngine이 검사한다.
    engine = ConstraintEngine(problem, target)
    choices = []
    # 거부한 후보마다 이유를 남긴다. 모두 거부되면 그대로 보고한다.
    rejected = []
    for config, cost in candidates:
        if config.p_k != 1:
            rejected.append((config.key(), "model layers require p_k=1"))
            continue
        reason = engine.check(config)
        if reason is not None:
            rejected.append((config.key(), reason))
            continue
        hierarchy = pim_hierarchy(problem, config)
        if has_weights and (instance_count != 1 or hierarchy.scheduling_rounds != 1):
            rejected.append((config.key(), (
                f"resident weights require one instance and one M/N/K round; got "
                f"{instance_count} instances and rounds {hierarchy.rounds}")))
            continue
        units = config.p_m * config.p_n * config.p_k
        p_b = min(instance_count, target.common_profile.num_units // units)
        rounds = (instance_count + p_b - 1) // p_b
        choices.append((rounds * cost, config, p_b, rounds, hierarchy.rounds))
    if not choices:
        raise ValueError(
            f"no model config fits the actual resources for (M,N,K)=({problem.M},"
            f"{problem.N},{problem.K}) on {target.common_profile.num_units} units: "
            + "; ".join(f"{key}: {reason}" for key, reason in rejected))
    _, config, p_b, rounds, mnk_rounds = min(choices, key=lambda choice: choice[0])
    return config, p_b, rounds, mnk_rounds
