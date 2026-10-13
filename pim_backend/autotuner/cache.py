"""Cache for measured PIM Common IR configurations."""

import json
import math
import os
import tempfile
from dataclasses import asdict

from .types import PIMBenchmarkResult, PIMConfig


CACHE_SCHEMA = "pim-common-ir-semantic-schedule-v7-aimx-vacc-reduction"
RESULT_FIELDS = (
    "total_ms", "correctness", "timings", "physical_allocations",
)


def _decode_config(stored):
    return PIMConfig(**stored)


class PIMConfigCache:
    def __init__(self, path, template_version="", compiler_version=""):
        self.path = path
        self.template_version = template_version
        self.compiler_version = compiler_version
        self.entries = {}
        if os.path.exists(path) and os.path.getsize(path) != 0:
            try:
                with open(path) as handle:
                    self.entries = json.load(handle)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"PIM autotuner cache is not valid JSON: {path}"
                ) from error

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

    def _complete_entry(self, problem, target):
        # store_partial만 거친 entry는 탐색이 끝나지 않았으므로 best가 없다.
        entry = self.entries.get(self._key(problem, target))
        if entry is None or not entry.get("complete", True):
            return None
        return entry

    def lookup(self, problem, target):
        entry = self._complete_entry(problem, target)
        if entry is None:
            return None
        return _decode_config(entry["config"])

    def lookup_result(self, problem, target):
        entry = self._complete_entry(problem, target)
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
            if entry.get("context") != context or not entry.get("complete", True):
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

    def _merged_records(self, key, records, replace):
        merged = {}
        previous = None if replace else self.entries.get(key)
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
        return list(merged.values())

    def store(self, problem, target, record, records=(), replace=False):
        """Finish a search: keep its best and every measured record."""
        key = self._key(problem, target)
        self.entries[key] = {
            "problem": {
                "M": problem.M, "N": problem.N, "K": problem.K},
            "context": self._context_key(problem, target),
            "complete": True,
            "config": asdict(record.config),
            "total_ms": record.result.total_ms,
            "records": self._merged_records(key, records, replace),
        }
        self._write()

    def store_partial(self, problem, target, records):
        """Keep the records of a search that has not finished.

        The entry has no best, so lookup treats it as a miss and the next
        search resumes with these records instead of measuring them again.
        """
        key = self._key(problem, target)
        previous = self.entries.get(key)
        if previous is not None and previous.get("complete", True):
            # 끝난 탐색을 retune하는 중이면 끝난 entry를 그대로 두고 마지막 store가 덮는다.
            return
        self.entries[key] = {
            "problem": {
                "M": problem.M, "N": problem.N, "K": problem.K},
            "context": self._context_key(problem, target),
            "complete": False,
            "records": self._merged_records(key, records, False),
        }
        self._write()

    def _write(self):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", dir=directory, prefix=".pim-autotune-",
                suffix=".json", delete=False) as handle:
            json.dump(self.entries, handle, indent=2)
            temporary = handle.name
        os.replace(temporary, self.path)
