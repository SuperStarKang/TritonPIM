"""Target-neutral memory management table of a compiled model.

Adapters report only numbers: their actual units, the capacity of one unit and
the per-unit bytes of every stage operand. The planner never sees vendor handles.

Every unit has the same layout, so one table of offsets from the unit's memory
base serves all units. Phase areas grow up from offset 0 and the key/value cache
grows down from the capacity: layer by layer, the key then the value region, each
below the previous one. A stage operand of kind KEY or VALUE reads that region in
place; its per-unit bytes, and which token positions a unit holds there, come
from the stage that reads it (its config and the adapter's operand layout), so
every stage reading one region must need the same bytes. Every phase of a
request (prefill, decode) lays its own area out from offset 0, and the phases
overlay one another: only the phase that runs holds the area, so its weights
overwrite those of the previous phase. The cache is written to the device only
right before the first phase that reads it, so the phases that run before that
one may use the whole unit, the cache's range included; the later phases end
below the cache.

One slot holds the weights and state of one layer's stages; layer i takes slot
i mod slots and belongs to group i // slots. If every layer fits there is one
group; otherwise the layers of one group fill every slot and are uploaded
together, overwriting only the weights of the previous group while buffers and
instructions stay in place. A phase area holds the weights of every slot, then
the state of every slot, then the input and output scratch. The weights of one
group are therefore one contiguous range, so a group upload is one transfer per
unit. Every stage uses the scratch from its start, so each scratch part is as
large as its largest stage.
"""

from dataclasses import dataclass


WEIGHT = "weight"
# weight가 아닌 장치 상주 memory (실행 사이에 남아야 하는 descriptor chain, PMM 등)
STATE = "state"
INPUT = "input"
OUTPUT = "output"
# key/value cache 항목, 그리고 그 영역을 그대로 읽는 stage operand
KEY = "key"
VALUE = "value"


@dataclass(frozen=True)
class UnitResources:
    units: int
    capacity_bytes: int


@dataclass(frozen=True)
class OperandRequirement:
    """Per-unit bytes of one stage operand; operand is the adapter's binding index."""

    operand: int
    kind: str
    bytes: int


@dataclass(frozen=True)
class StageRequirement:
    layer: int
    stage: str
    operands: tuple

    def _bytes(self, kind):
        return sum(operand.bytes for operand in self.operands if operand.kind == kind)

    @property
    def weight_bytes(self):
        return self._bytes(WEIGHT)

    @property
    def state_bytes(self):
        return self._bytes(STATE)

    @property
    def input_bytes(self):
        return self._bytes(INPUT)

    @property
    def output_bytes(self):
        return self._bytes(OUTPUT)

    @property
    def kv_bytes(self):
        return self._bytes(KEY) + self._bytes(VALUE)


@dataclass(frozen=True)
class MemoryEntry:
    """One region; offset is from the memory base of every unit.

    Stage operands belong to a phase. Key/value cache entries have no phase, stage
    or operand: they stay at the same offset through every phase. A KEY/VALUE stage
    operand entry has the offset of the cache entry it reads.
    """

    kind: str
    phase: str | None
    layer: int
    stage: str | None
    operand: int | None
    # weight/state는 slot과 group에 속한다. scratch(input/output)와 key/value cache는 None이다.
    slot: int | None
    group: int | None
    offset: int
    bytes: int


@dataclass(frozen=True)
class PhaseLayout:
    """The area of one phase, from offset 0."""

    # slot 수와 layer group 수. layer i는 slot i mod slots, group i // slots에 놓인다.
    slots: int
    groups: int
    # layer index -> slot
    slot_of_layer: dict
    # layer index -> group. 한 group의 layer들은 함께 올라간다.
    group_of_layer: dict
    # slot마다 weight + state bytes
    slot_bytes: tuple
    # slot마다 weight bytes와 state bytes. weight 영역은 phase 시작, state 영역은 그 뒤다.
    slot_weight_bytes: tuple
    slot_state_bytes: tuple
    scratch_bytes: int
    # phase 영역 전체 bytes
    bytes: int


@dataclass(frozen=True)
class MemoryTable:
    units: int
    capacity_bytes: int
    # key/value cache bytes와 그 시작 offset. cache는 [kv_base, capacity_bytes)를 차지한다.
    kv_bytes: int
    kv_base: int
    # 한 시점에 unit이 담는 최대 bytes: cache 전 phase 영역, 또는 cache + 그 뒤 phase 영역
    bytes_per_unit: int
    # phase -> PhaseLayout (실행 순서)
    phases: dict
    # (phase, layer, stage, operand) 또는 ("kv", layer, KEY/VALUE) -> MemoryEntry
    entries: dict


def _plan_phase(phase, requirements, base, capacity_bytes, entries):
    """Slots, groups and operand offsets of one phase area that starts at base."""
    layers = sorted({requirement.layer for requirement in requirements})
    stages = {layer: [requirement for requirement in requirements if requirement.layer == layer]
              for layer in layers}
    layer_weight = {layer: sum(requirement.weight_bytes for requirement in stages[layer])
                    for layer in layers}
    layer_state = {layer: sum(requirement.state_bytes for requirement in stages[layer])
                   for layer in layers}
    input_bytes = max(requirement.input_bytes for requirement in requirements)
    output_bytes = max(requirement.output_bytes for requirement in requirements)
    scratch = input_bytes + output_bytes
    if sum(layer_weight.values()) + sum(layer_state.values()) + scratch <= capacity_bytes:
        slot_weight_bytes = tuple(layer_weight[layer] for layer in layers)
        slot_state_bytes = tuple(layer_state[layer] for layer in layers)
    else:
        # 한 slot에 여러 layer가 차례로 들어가므로 operand별 크기가 모두 같아야 한다.
        layouts = {layer: [(operand.kind, operand.bytes) for requirement in stages[layer]
                           for operand in requirement.operands]
                   for layer in layers}
        for layer in layers[1:]:
            if layouts[layer] != layouts[layers[0]]:
                raise ValueError(
                    f"{phase}: layers share a slot; layer {layer} operands need {layouts[layer]} bytes, "
                    f"layer {layers[0]} operands {layouts[layers[0]]}")
        layer_bytes = layer_weight[layers[0]] + layer_state[layers[0]]
        slots = (capacity_bytes - scratch) // layer_bytes
        if slots < 1:
            raise ValueError(
                f"{phase}: one layer does not fit one unit: it needs {layer_bytes} bytes "
                f"and scratch {scratch}, the phase area has {capacity_bytes}")
        slot_weight_bytes = (layer_weight[layers[0]],) * slots
        slot_state_bytes = (layer_state[layers[0]],) * slots
    slot_bytes = tuple(weight + state for weight, state in zip(slot_weight_bytes, slot_state_bytes))
    slots = len(slot_bytes)
    slot_of_layer = {layer: position % slots for position, layer in enumerate(layers)}
    group_of_layer = {layer: position // slots for position, layer in enumerate(layers)}

    weight_end = base + sum(slot_weight_bytes)
    state_end = weight_end + sum(slot_state_bytes)
    for layer in layers:
        slot = slot_of_layer[layer]
        # 같은 slot의 layer는 같은 offset을 받는다. weight는 weight 영역, state는 state 영역의
        # 앞 slot들 바로 뒤에서 시작한다.
        offsets = {WEIGHT: base + sum(slot_weight_bytes[:slot]),
                   STATE: weight_end + sum(slot_state_bytes[:slot])}
        for requirement in stages[layer]:
            offsets[INPUT] = state_end
            offsets[OUTPUT] = state_end + input_bytes
            for operand in requirement.operands:
                if operand.kind in (KEY, VALUE):
                    # phase 영역을 쓰지 않고 key/value cache 영역을 그대로 읽는다.
                    entries[(phase, layer, requirement.stage, operand.operand)] = MemoryEntry(
                        operand.kind, phase, layer, requirement.stage, operand.operand, None, None,
                        entries[("kv", layer, operand.kind)].offset, operand.bytes)
                    continue
                resident = operand.kind in (WEIGHT, STATE)
                entries[(phase, layer, requirement.stage, operand.operand)] = MemoryEntry(
                    operand.kind, phase, layer, requirement.stage, operand.operand,
                    slot if resident else None, group_of_layer[layer] if resident else None,
                    offsets[operand.kind], operand.bytes)
                offsets[operand.kind] += operand.bytes
    return PhaseLayout(slots, max(group_of_layer.values()) + 1, slot_of_layer, group_of_layer,
                       slot_bytes, slot_weight_bytes, slot_state_bytes, scratch,
                       state_end + scratch - base)


def plan_memory(resources, requirements):
    """The memory table that fits one unit.

    requirements maps each phase, in execution order, to its stage requirements.
    The key/value cache regions are those the KEY/VALUE stage operands read; a
    model without them has no cache on the device. The phases before the first one
    that reads the cache may use the whole unit. Every stage is assumed to occupy
    the first unit, so stage sums bound the most loaded unit.
    """
    # (layer, KEY/VALUE) -> unit당 bytes. 같은 영역을 읽는 stage는 모두 같은 bytes여야 한다.
    regions = {}
    # cache를 처음 읽는 phase의 순서. 읽는 phase가 없으면 모든 phase가 그 앞이다.
    first_reader = len(requirements)
    for index, (phase, phase_requirements) in enumerate(requirements.items()):
        for requirement in phase_requirements:
            for operand in requirement.operands:
                if operand.kind not in (KEY, VALUE):
                    continue
                first_reader = min(first_reader, index)
                region = (requirement.layer, operand.kind)
                if regions.setdefault(region, operand.bytes) != operand.bytes:
                    raise ValueError(
                        f"{phase} layer {requirement.layer} {requirement.stage} reads the {operand.kind} "
                        f"cache with {operand.bytes} bytes per unit; another stage reads it with "
                        f"{regions[region]}")
    entries = {}
    kv_base = resources.capacity_bytes
    # capacity에서 아래로 layer마다 key, value 순서로 놓는다.
    for layer, kind in sorted(regions, key=lambda region: (region[0], region[1] != KEY)):
        kv_base -= regions[layer, kind]
        entries[("kv", layer, kind)] = MemoryEntry(
            kind, None, layer, None, None, None, None, kv_base, regions[layer, kind])
    kv_bytes = resources.capacity_bytes - kv_base
    if kv_base < 0:
        raise ValueError(f"the key/value cache needs {kv_bytes} bytes per unit; "
                         f"capacity is {resources.capacity_bytes}")
    # 모든 phase가 offset 0부터 같은 영역을 번갈아 쓴다. cache를 처음 읽는 phase부터는 cache 아래에서 끝난다.
    phases = {phase: _plan_phase(phase, phase_requirements, 0,
                                 kv_base if index >= first_reader else resources.capacity_bytes, entries)
              for index, (phase, phase_requirements) in enumerate(requirements.items())}
    # cache 전 phase는 자기 영역만, 그 뒤 phase는 cache와 함께 담긴다.
    held = [layout.bytes + (kv_bytes if index >= first_reader else 0)
            for index, layout in enumerate(phases.values())]
    return MemoryTable(resources.units, resources.capacity_bytes, kv_bytes, kv_base,
                       max(held), phases, entries)
