"""Measured single-unit saturation, separate from compiler resource limits."""

import csv
import json
from dataclasses import asdict, dataclass, replace
from math import isfinite, prod
from statistics import median

from .constraints import ConstraintEngine
from .artifacts import prepare_run_directory, project_path
from .model import access_footprint_bytes, group_extent, target_model


PERFORMANCE_SCHEMA = "pim-single-unit-saturation-v2"
SATURATION_FRACTION = 0.95


def performance_context(problem, target):
    hardware = asdict(target.common_profile)
    del hardware["num_units"]
    return json.dumps({
        "hardware": hardware,
        "runtime_type": type(target.runtime_config).__name__,
        "runtime": asdict(target.runtime_config),
        "compute_kind": problem.compute_kind,
        "input_dtype": problem.input_dtype,
        "output_dtype": problem.output_dtype,
        "accesses": [asdict(access) for access in problem.accesses],
        "gemm_profile": (asdict(target_model(target).gemm_profile_spec)
                         if problem.compute_kind == "matmul" else None),
    }, sort_keys=True)


@dataclass(frozen=True)
class PIMUnitTiming:
    """One tile on one unit: synchronous launch time, excluding host transfers.

    device_ms is optional hardware-counter evidence, never a replacement for
    launch_ms. The latter includes the invocation overhead being amortized.
    """

    launch_ms: float
    device_ms: float | None = None

    def __post_init__(self):
        if not isfinite(self.launch_ms) or self.launch_ms <= 0:
            raise ValueError("single-unit launch_ms must be finite and positive")
        if self.device_ms is not None:
            if not isfinite(self.device_ms) or self.device_ms <= 0:
                raise ValueError("single-unit device_ms must be finite and positive")
            if self.device_ms > self.launch_ms:
                raise ValueError("device_ms cannot exceed synchronous launch_ms")


@dataclass(frozen=True)
class PIMSaturationSample:
    extent: tuple
    timing: PIMUnitTiming

    def __post_init__(self):
        if len(self.extent) != 3 or any(value <= 0 for value in self.extent):
            raise ValueError("a saturation sample needs three positive extents")

    @property
    def work(self):
        return prod(self.extent)

    @property
    def throughput(self):
        return self.work / self.timing.launch_ms


@dataclass(frozen=True)
class PIMGroupSample:
    groups: tuple
    sample: PIMSaturationSample


@dataclass(frozen=True)
class PIMGroupCurve:
    context: str
    blocks: tuple
    samples: tuple
    stop_reason: str

    def __post_init__(self):
        if not self.samples or not self.stop_reason:
            raise ValueError("a Group sweep needs measured samples and its stop reason")
        for point in self.samples:
            if len(point.groups) != 3 or min(point.groups) < 1:
                raise ValueError("a Group sample needs three positive group counts")
            if point.sample.extent != tuple(
                    block * group for block, group in zip(self.blocks, point.groups)):
                raise ValueError("a Group sample must execute exactly one Block Group")


@dataclass(frozen=True)
class PIMSaturationCurve:
    context: str
    blocks: tuple
    groups: tuple
    samples: tuple
    stop_reason: str = "saturation"

    def __post_init__(self):
        if len(self.blocks) != 3 or len(self.groups) != 3:
            raise ValueError("saturation curves require three block/group extents")
        if any(value <= 0 for value in (*self.blocks, *self.groups)):
            raise ValueError("block/group extents must be positive")
        if len(self.samples) < 3:
            raise ValueError(
                f"at least three single-unit measurements are required: "
                f"blocks={self.blocks}, groups={self.groups}")
        if any(extent % (block * group)
               for sample in self.samples
               for extent, block, group in zip(sample.extent, self.blocks, self.groups)):
            raise ValueError("single-unit samples must contain whole Block Groups")
        if any(left.work >= right.work
               for left, right in zip(self.samples, self.samples[1:])):
            raise ValueError("single-unit measurements must have increasing work")
        plateau = median(sample.throughput for sample in self.samples[-3:])
        if any(abs(sample.throughput / plateau - 1) > 0.05
               for sample in self.samples[-3:]):
            raise ValueError(
                f"single-unit throughput has not stabilized: blocks={self.blocks}, "
                f"groups={self.groups}; extend the sweep")
        if any(sample.throughput > plateau / SATURATION_FRACTION
               for sample in self.samples):
            raise ValueError(
                f"throughput declines before the sweep ends: blocks={self.blocks}, "
                f"groups={self.groups}; inspect the curve")

    @property
    def saturation_work(self):
        plateau = median(sample.throughput for sample in self.samples[-3:])
        for index, sample in enumerate(self.samples[:-2]):
            if all(point.throughput >= SATURATION_FRACTION * plateau
                   for point in self.samples[index:]):
                return sample.work
        raise ValueError("no sustained 95-percent saturation point in the measured curve")


@dataclass(frozen=True)
class PIMPerformanceProfile:
    kernel_version: str
    curves: tuple
    group_curves: tuple = ()

    def __post_init__(self):
        if not self.kernel_version:
            raise ValueError("performance measurements require a kernel_version")
        keys = [(curve.context, curve.blocks, curve.groups) for curve in self.curves]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate single-unit saturation curves")
        for curve in self.curves:
            curve.saturation_work
        group_keys = [(curve.context, curve.blocks) for curve in self.group_curves]
        if len(group_keys) != len(set(group_keys)):
            raise ValueError("duplicate Block Group sweeps")
        if self.group_curves and set(group_keys) != {
                (curve.context, curve.blocks) for curve in self.curves}:
            raise ValueError("Group sweeps and Tile saturation curves must cover the same Blocks")

    def validate_kernel(self, kernel_version):
        if self.kernel_version != kernel_version:
            raise ValueError(
                f"performance kernel mismatch: measured={self.kernel_version!r}, "
                f"requested={kernel_version!r}")

    def saturation_work(self, problem, target, blocks, groups):
        context = performance_context(problem, target)
        for curve in self.curves:
            if (curve.context, curve.blocks, curve.groups) == (context, blocks, groups):
                return curve.saturation_work
        raise ValueError(
            f"missing single-unit measurements: target={target.common_profile.target}, "
            f"operation={problem.compute_kind}, dtype={problem.input_dtype}/"
            f"{problem.output_dtype}, blocks={blocks}, groups={groups}; "
            "calibrate this kernel and device configuration first")

    def seed_families(self, problem, target, limit):
        """Use measured Groups and the Block's measured device-work reference.

        For a smaller problem, select its best legal measured Group. The
        reference work remains the measured Tile saturation of that Block;
        it is a seed heuristic, not an inferred timing for the smaller Group.
        """
        from .space import make_config

        context = performance_context(problem, target)
        matching = [curve for curve in self.curves if curve.context == context]
        if not matching:
            raise ValueError("missing single-unit measurements for this target, operation and dtype")
        engine = ConstraintEngine(problem, target)
        group_sweeps = {(curve.context, curve.blocks): curve for curve in self.group_curves}
        selected = []
        for curve in matching:
            if self.group_curves:
                legal = [point for point in group_sweeps[(context, curve.blocks)].samples
                         if engine.valid(make_config(
                             curve.blocks, point.groups, (1, 1, 1), (1, 1, 1)))]
                if not legal:
                    continue
                groups = max(legal, key=lambda point: point.sample.throughput).groups
            else:
                groups = curve.groups
                if not engine.valid(make_config(curve.blocks, groups, (1, 1, 1), (1, 1, 1))):
                    continue
            selected.append((curve, groups))
        selected.sort(key=lambda item: median(
            sample.throughput for sample in item[0].samples[-3:]), reverse=True)
        if not selected:
            raise ValueError("no measured Block/Group is legal for this problem")
        return [(curve.blocks, groups, curve.saturation_work)
                for curve, groups in selected[:limit]]

    def save(self, path):
        path = project_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"schema": PERFORMANCE_SCHEMA, **asdict(self)}, handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path, *, kernel_version):
        path = project_path(path)
        with open(path) as handle:
            stored = json.load(handle)
        if stored["schema"] != PERFORMANCE_SCHEMA:
            raise ValueError(f"unsupported performance schema: {stored['schema']}")
        profile = cls(stored["kernel_version"], tuple(
            PIMSaturationCurve(
                curve["context"], tuple(curve["blocks"]), tuple(curve["groups"]),
                tuple(PIMSaturationSample(tuple(sample["extent"]),
                                          PIMUnitTiming(**sample["timing"]))
                      for sample in curve["samples"]), curve["stop_reason"])
            for curve in stored["curves"]), tuple(
                PIMGroupCurve(
                    curve["context"], tuple(curve["blocks"]), tuple(
                        PIMGroupSample(tuple(point["groups"]), PIMSaturationSample(
                            tuple(point["sample"]["extent"]),
                            PIMUnitTiming(**point["sample"]["timing"])))
                        for point in curve["samples"]), curve["stop_reason"])
                for curve in stored["group_curves"]))
        profile.validate_kernel(kernel_version)
        return profile


def _throughput_stable(samples):
    if len(samples) < 4:
        return False
    previous_plateau = median(sample.throughput for sample in samples[-4:-1])
    plateau = median(sample.throughput for sample in samples[-3:])
    return (all(abs(sample.throughput / previous_plateau - 1) <= 0.05 for sample in samples[-4:])
            and all(abs(sample.throughput / plateau - 1) <= 0.05 for sample in samples[-3:])
            and all(sample.throughput <= plateau / SATURATION_FRACTION for sample in samples))


def _measure_sample(problem, target, config, measure_fn, record_fn, stage):
    local_problem = replace(problem, M=config.execution_extent[0],
                            N=config.execution_extent[1], K=config.execution_extent[2])
    reason = ConstraintEngine(local_problem, target).check(config)
    if reason is not None:
        raise ValueError(reason)
    print(f"[single-unit] stage={stage} units={target.common_profile.num_units} "
          f"blocks={config.key()[:3]} groups={config.key()[3:6]} "
          f"tile={config.key()[6:9]} extent={config.execution_extent}", end="", flush=True)
    timing = measure_fn(local_problem, target, config)
    if record_fn is not None:
        record_fn(stage, local_problem, target, config, timing)
    print(f" launch_ms={timing.launch_ms:.6f}", flush=True)
    return PIMSaturationSample(config.execution_extent, timing)


def _calibrate_gemm(problem, target, measure_fn, sweep_extents, record_fn):
    from .space import make_config

    curves = []
    group_curves = []
    context = performance_context(problem, target)
    spec = target_model(target).gemm_profile_spec
    for blocks in spec.block_candidates:
        points = []
        group_k = 1
        while True:
            groups = (spec.group_shape[0], spec.group_shape[1], spec.group_shape[2] * group_k)
            config = make_config(blocks, groups, (1, 1, 1), (1, 1, 1))
            local_problem = replace(problem, M=config.execution_extent[0],
                                    N=config.execution_extent[1], K=config.execution_extent[2])
            reason = ConstraintEngine(local_problem, target).check(config)
            if sweep_extents is not None and any(
                    extent > limit for extent, limit in zip(config.execution_extent, sweep_extents)):
                reason = "measurement extent limit"
            if reason is not None:
                group_stop = reason
                break
            sample = _measure_sample(problem, target, config, measure_fn, record_fn, "group")
            points.append(PIMGroupSample(groups, sample))
            if _throughput_stable([point.sample for point in points]):
                group_stop = "saturation"
                break
            group_k *= 2
        print(f"[single-unit] stage=group blocks={blocks} stop={group_stop}", flush=True)
        if not points:
            continue
        selected = max(points, key=lambda point: point.sample.throughput)
        print(f"[single-unit] selected blocks={blocks} groups={selected.groups} "
              f"work_per_ms={selected.sample.throughput:.3f}", flush=True)
        samples = [selected.sample]
        tile_k = 2
        while True:
            config = make_config(blocks, selected.groups, (1, 1, tile_k), (1, 1, 1))
            local_problem = replace(problem, M=config.execution_extent[0],
                                    N=config.execution_extent[1], K=config.execution_extent[2])
            reason = ConstraintEngine(local_problem, target).check(config)
            if sweep_extents is not None and any(
                    extent > limit for extent, limit in zip(config.execution_extent, sweep_extents)):
                reason = "measurement extent limit"
            if reason is not None:
                tile_stop = reason
                break
            samples.append(_measure_sample(problem, target, config, measure_fn, record_fn, "tile"))
            if _throughput_stable(samples):
                tile_stop = "saturation"
                break
            tile_k *= 2
        print(f"[single-unit] stage=tile blocks={blocks} groups={selected.groups} "
              f"stop={tile_stop}", flush=True)
        if not _throughput_stable(samples):
            raise ValueError(
                f"single-unit throughput has not stabilized: blocks={blocks}, "
                f"groups={selected.groups}; stopped at {tile_stop}")
        group_curves.append(PIMGroupCurve(context, blocks, tuple(points), group_stop))
        curves.append(PIMSaturationCurve(context, blocks, selected.groups, tuple(samples), tile_stop))
    if not curves:
        raise ValueError("no legal GEMM profiling Block for this target and measurement limits")
    return curves, group_curves


def calibrate_performance(cases, target, measure_fn, *, kernel_version, sweep_extents=None,
                          record_fn=None):
    """Collect measured seed references using one device and one Execution Tile.

    cases contains (problem, config_space) pairs whose seed families are needed.
    measure_fn(local_problem, single_unit_target, config) must execute the
    actual kernel with new inputs for local_problem and return PIMUnitTiming.
    GEMM sweeps the target's Blocks, selects the fastest measured Group for
    each Block, then grows only Tile K. Other operations retain their explicit
    sweep domain. sweep_extents optionally bounds GEMM's capacity-based sweep.
    No timing is inferred when a Tile sweep does not reach a stable plateau.
    """
    from .space import make_config, rank_block_groups, tile_growth

    if not kernel_version:
        raise ValueError("calibration requires a kernel_version")
    if sweep_extents is not None and (
            len(sweep_extents) != 3 or any(value <= 0 for value in sweep_extents)):
        raise ValueError("sweep_extents must contain three positive limits")
    single_target = replace(
        target, common_profile=replace(target.common_profile, num_units=1))
    curves = []
    group_curves = []
    measured = set()
    for problem, space in cases:
        if problem.compute_kind == "matmul":
            context = performance_context(problem, target)
            if context in measured:
                continue
            measured_curves, measured_groups = _calibrate_gemm(
                problem, single_target, measure_fn, sweep_extents, record_fn)
            curves.extend(measured_curves)
            group_curves.extend(measured_groups)
            measured.add(context)
            continue
        if sweep_extents is None:
            raise ValueError("non-GEMM calibration requires explicit sweep_extents")
        # 사용하지 않는 축은 1로 유지하고 실제 연산 축만 측정 상한까지 늘린다.
        extents = tuple(1 if extent == 1 else limit
                        for extent, limit in zip(problem.axis_extents, sweep_extents))
        sweep_problem = replace(problem, M=extents[0], N=extents[1], K=extents[2])
        families = rank_block_groups(problem, target, space)[:space.block_group_top_k]
        if not families:
            raise ValueError("no legal Block/Group family to calibrate")
        for blocks, groups in families:
            context = performance_context(problem, target)
            if (context, blocks, groups) in measured:
                continue
            samples = []
            for tiles in tile_growth(sweep_problem, single_target, blocks, groups, space):
                config = make_config(blocks, groups, tiles, (1, 1, 1))
                samples.append(_measure_sample(
                    problem, single_target, config, measure_fn, record_fn, "tile"))
            curves.append(PIMSaturationCurve(context, blocks, groups, tuple(samples)))
            measured.add((context, blocks, groups))
    if not curves:
        raise ValueError("at least one calibration case is required")
    return PIMPerformanceProfile(kernel_version, tuple(curves), tuple(group_curves))


def profile_single_unit(cases, target, measure_fn, *, kernel_version, output_dir, sweep_extents=None):
    """Persist measured points before validating saturation, then save a usable profile."""
    output_dir = prepare_run_directory(output_dir)
    print(f"[single-unit] raw measurements: {output_dir / 'single_unit.csv'}", flush=True)
    with open(output_dir / "single_unit.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "stage", "kernel_version", "context", "num_units", "p_m", "p_n", "p_k",
            "block_m", "block_n", "block_k", "group_m", "group_n", "group_k",
            "tile_m", "tile_n", "tile_k", "extent_m", "extent_n", "extent_k",
            "work", "launch_ms", "device_ms", "work_per_ms", "local_bytes", "resident_bytes"))
        handle.flush()

        def record(stage, problem, single_target, config, timing):
            sample = PIMSaturationSample(config.execution_extent, timing)
            model = target_model(single_target)
            writer.writerow((
                stage, kernel_version, performance_context(problem, single_target),
                single_target.common_profile.num_units, config.p_m, config.p_n, config.p_k,
                *config.key()[:9], *sample.extent, sample.work,
                timing.launch_ms, timing.device_ms, sample.throughput,
                access_footprint_bytes(problem, group_extent(config)),
                sum(model.access_allocations(problem, config))
                + sum(model.overhead_allocations(problem, config).values())))
            handle.flush()

        performance = calibrate_performance(
            cases, target, measure_fn, kernel_version=kernel_version,
            sweep_extents=sweep_extents, record_fn=record)
    performance.save(output_dir / "performance.json")
    for curve in performance.curves:
        print(f"[single-unit] blocks={curve.blocks} groups={curve.groups} "
              f"saturation_work={curve.saturation_work}", flush=True)
    print(f"[single-unit] saved: {output_dir / 'performance.json'}", flush=True)
    return performance
