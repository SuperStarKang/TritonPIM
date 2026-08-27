# Stage 16 — Config Cache와 Warm Start (design/AutoTuner_Design.md §45~§46)
#
# 최종 best configuration을 (problem, hardware, version) key로 저장하고,
# 정확히 같은 shape가 아니어도 유사한 과거 configuration을 initial seed로
# 재사용한다 (§46). 같은 key의 측정 이력 전체도 함께 저장해서:
#   - 정확 일치 항목이 있으면 탐색 없이 best를 그대로 쓸 수 있고,
#   - 재탐색 시 이미 잰 configuration은 실측 없이 결과를 재사용한다.

import json
import math
import os
from dataclasses import asdict

from .types import PIMBenchmarkResult, PIMConfig

# 저장·복원하는 측정 결과 필드 (PIMBenchmarkResult의 스칼라 부분).
RESULT_FIELDS = ("total_ms", "stage_ms", "h2d_ms", "compute_ms",
                 "reduction_ms", "d2h_ms", "launch_ms",
                 "compile_status", "correctness")


def _decode_config(stored):
    """Read both the old execution_* spelling and the tile-count spelling."""
    values = dict(stored)
    for axis in "mnk":
        old = f"execution_{axis}"
        new = f"tile_{axis}"
        if old in values:
            values.setdefault(new, values[old])
            del values[old]
    return PIMConfig(**values)


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
            hardware.banks_per_device, hardware.cores_per_bank,
            hardware.lanes_per_core, hardware.usable_bank_capacity_bytes,
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
        return _decode_config(entry["config"])

    def lookup_result(self, problem, hardware):
        """정확히 같은 tuning key의 (best config, 저장된 측정 결과).

        탐색을 생략하고 cache를 그대로 쓸 때 사용한다. 없으면 None.
        """
        entry = self.entries.get(self._key(problem, hardware))
        if entry is None:
            return None
        config = _decode_config(entry["config"])
        stored = {"total_ms": entry["total_ms"]}
        for record in entry.get("records", ()):
            if _decode_config(record["config"]).key() == config.key():
                stored = {field: record[field] for field in RESULT_FIELDS}
                break
        return config, PIMBenchmarkResult(**stored)

    def measured_results(self, problem, hardware):
        """같은 tuning key로 저장된 모든 측정: config.key() → 결과 dict.

        재탐색 시 이미 잰 configuration의 실측을 생략하는 데 쓴다.
        """
        entry = self.entries.get(self._key(problem, hardware))
        if entry is None:
            return {}
        results = {}
        for record in entry.get("records", ()):
            config = _decode_config(record["config"])
            results[config.key()] = {
                field: record[field] for field in RESULT_FIELDS}
        return results

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
        return [_decode_config(entry["config"]) for _, entry in matches[:limit]]

    def store(self, problem, hardware, record, records=()):
        """best config·plan metadata와 측정 이력 전체를 저장한다 (§45).

        같은 key의 기존 이력과 config 단위로 합친다 (새 측정이 우선).
        """
        key = self._key(problem, hardware)
        plan = record.plan
        merged = {}
        previous = self.entries.get(key)
        if previous is not None:
            for old in previous.get("records", ()):
                merged[_decode_config(old["config"]).key()] = old
        for measured in records:
            merged[measured.config.key()] = {
                "config": asdict(measured.config),
                **{field: getattr(measured.result, field)
                   for field in RESULT_FIELDS},
            }
        self.entries[key] = {
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
            "records": list(merged.values()),
        }
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)
