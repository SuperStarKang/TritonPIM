"""Cache for measured PIM Common IR configurations."""

import json
import math
import os
from dataclasses import asdict

from .types import PIMBenchmarkResult, PIMConfig


CACHE_SCHEMA = "pim-common-ir-search-v3"
RESULT_FIELDS = (
    "total_ms", "correctness",
)


def _decode_config(stored):
    return PIMConfig(**stored)


class PIMConfigCache:
    def __init__(self, path, template_version="", compiler_version=""):
        self.path = path
        self.template_version = template_version
        self.compiler_version = compiler_version
        self.entries = {}
        if os.path.exists(path):
            with open(path) as handle:
                self.entries = json.load(handle)

    def _context_key(self, problem, target):
        profile = target.common_profile
        runtime = target.runtime_config
        return "|".join(str(value) for value in (
            CACHE_SCHEMA,
            problem.input_dtype, problem.output_dtype,
            problem.compute_kind,
            problem.access_key(),
            profile.target, profile.num_units,
            profile.banks_per_unit, profile.cores_per_bank,
            profile.lanes_per_core,
            profile.bank_capacity_limit_bytes,
            profile.local_capacity_limit_bytes,
            type(runtime).__name__,
            tuple(sorted(asdict(runtime).items())),
            self.template_version, self.compiler_version,
        ))

    def _key(self, problem, target):
        return (
            f"{problem.M}x{problem.N}x{problem.K}|"
            + self._context_key(problem, target))

    def lookup(self, problem, target):
        entry = self.entries.get(self._key(problem, target))
        if entry is None:
            return None
        return _decode_config(entry["config"])

    def lookup_result(self, problem, target):
        entry = self.entries.get(self._key(problem, target))
        if entry is None:
            return None
        config = _decode_config(entry["config"])
        stored = {"total_ms": entry["total_ms"]}
        for record in entry.get("records", ()):
            if _decode_config(record["config"]).key() == config.key():
                stored = {
                    field: record[field] for field in RESULT_FIELDS}
                break
        return config, PIMBenchmarkResult(**stored)

    def measured_results(self, problem, target):
        entry = self.entries.get(self._key(problem, target))
        if entry is None:
            return {}
        results = {}
        for record in entry.get("records", ()):
            config = _decode_config(record["config"])
            results[config.key()] = {
                field: record[field] for field in RESULT_FIELDS}
        return results

    def lookup_similar(self, problem, target, limit=3):
        context = self._context_key(problem, target)
        matches = []
        for entry in self.entries.values():
            if entry.get("context") != context:
                continue
            stored = entry["problem"]
            distance = (
                abs(math.log2(problem.M / stored["M"]))
                + abs(math.log2(problem.N / stored["N"]))
                + abs(math.log2(problem.K / stored["K"]))
            )
            matches.append((distance, entry))
        matches.sort(key=lambda pair: pair[0])
        return [
            _decode_config(entry["config"])
            for _, entry in matches[:limit]
        ]

    def store(self, problem, target, record, records=()):
        key = self._key(problem, target)
        merged = {}
        previous = self.entries.get(key)
        if previous is not None:
            for old in previous.get("records", ()):
                merged[_decode_config(old["config"]).key()] = old
        for measured in records:
            merged[measured.config.key()] = {
                "config": asdict(measured.config),
                **{
                    field: getattr(measured.result, field)
                    for field in RESULT_FIELDS
                },
            }
        self.entries[key] = {
            "problem": {
                "M": problem.M, "N": problem.N, "K": problem.K},
            "context": self._context_key(problem, target),
            "config": asdict(record.config),
            "total_ms": record.result.total_ms,
            "records": list(merged.values()),
        }
        with open(self.path, "w") as handle:
            json.dump(self.entries, handle, indent=2)
