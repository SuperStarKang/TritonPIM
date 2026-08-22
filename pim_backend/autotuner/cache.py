# Stage 16 — Config Cache와 Warm Start (design/AutoTuner_Design.md §45~§46)
#
# 최종 best configuration을 (problem, hardware, version) key로 저장하고,
# 정확히 같은 shape가 아니어도 유사한 과거 configuration을 initial seed로
# 재사용한다 (§46).

import json
import math
import os
from dataclasses import asdict

from .types import PIMConfig


class PIMConfigCache:
    def __init__(self, path, template_version="", compiler_version=""):
        self.path = path
        self.template_version = template_version
        self.compiler_version = compiler_version
        self.entries = {}
        if os.path.exists(path):
            with open(path) as f:
                self.entries = json.load(f)

    def _context_key(self, problem, hardware):
        """shape를 제외한 key 부분 (§45). hardware나 template이 바뀌었는데
        이전 configuration을 잘못 재사용하는 것을 version이 막는다."""
        return "|".join(str(v) for v in (
            problem.input_dtype, problem.output_dtype,
            problem.a_layout, problem.b_layout,
            problem.a_resident, problem.b_resident,
            hardware.target, hardware.num_units,
            hardware.local_memory_bytes, hardware.device_memory_bytes,
            hardware.alignment_bytes,
            self.template_version, self.compiler_version,
        ))

    def _key(self, problem, hardware):
        return (f"{problem.M}x{problem.N}x{problem.K}|"
                + self._context_key(problem, hardware))

    def lookup(self, problem, hardware):
        """정확히 같은 tuning key의 best config (§45)."""
        entry = self.entries.get(self._key(problem, hardware))
        if entry is None:
            return None
        return PIMConfig(**entry["config"])

    def lookup_similar(self, problem, hardware, limit=3):
        """같은 context에서 shape가 가까운 과거 best config들 (§46)."""
        context = self._context_key(problem, hardware)
        matches = []
        for entry in self.entries.values():
            if entry["context"] != context:
                continue
            m, n, k = entry["problem"]["M"], entry["problem"]["N"], entry["problem"]["K"]
            distance = (abs(math.log2(problem.M / m))
                        + abs(math.log2(problem.N / n))
                        + abs(math.log2(problem.K / k)))
            matches.append((distance, entry))
        matches.sort(key=lambda pair: pair[0])
        return [PIMConfig(**entry["config"]) for _, entry in matches[:limit]]

    def store(self, problem, hardware, record):
        """best config와 plan metadata를 저장한다 (§45)."""
        plan = record.plan
        self.entries[self._key(problem, hardware)] = {
            "problem": {"M": problem.M, "N": problem.N, "K": problem.K},
            "context": self._context_key(problem, hardware),
            "config": asdict(record.config),
            "total_ms": record.result.total_ms,
            "plan": {
                "execution_tile_shape": list(plan.execution_tile_shape),
                "execution_grid": list(plan.execution_grid),
                "active_device_grid": list(plan.active_device_grid),
                "num_rounds": plan.num_rounds,
            },
        }
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)
