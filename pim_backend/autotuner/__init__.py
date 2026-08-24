# PIM-aware Hierarchical AutoTuner (design/AutoTuner_Design.md)
#
# 사용 흐름 (§52):
#     problem  = PIMProblem(M=..., N=..., K=...)
#     hardware = PIMHardwareProfile(target="upmem", ...)
#     result   = tune(problem, hardware, benchmark_fn)
#     result.best.config / result.best.plan / result.best.result
#
# benchmark_fn(problem, hardware, config, plan) -> PIMBenchmarkResult 는
# Stage 6~7(candidate lowering + real hardware benchmark)을 수행한다.
# 예시는 pim_backend/examples/autotune_gemm.py 를 본다.

from .cache import PIMConfigCache
from .constraints import (ConstraintEngine, device_memory_footprint,
                          gemm_footprint_bytes, local_memory_footprint)
from .features import config_plan_vector, extract_plan_features, feature_names
from .forest import RandomForestClassifier
from .planner import PIMPlanner, resolve_distribution
from .search import (SearchBudget, TuningRecord, TuningResult,
                     make_quantile_labels, rank_with_diversity_penalty,
                     semantic_mutate, tune)
from .targets import TARGET_PROFILES, target_profile
from .space import (DISTRIBUTION_POLICIES, KSPLIT_PK_VALUES,
                    KSPLIT_STRATEGIES, PERSISTENCE_STRATEGIES,
                    REDUCTION_POLICIES, REPLICATION_POLICIES,
                    RESIDENCY_POLICIES, STREAMING_POLICIES,
                    TARGET_REDUCTION_POLICY, PIMConfigSpace,
                    build_config_space, heuristic_seeds, make_config,
                    random_valid_configs, strategy_for_pk)
from .types import (PIMBenchmarkResult, PIMConfig, PIMHardwareProfile,
                    PIMPlan, PIMPlanFeatures, PIMProblem)
