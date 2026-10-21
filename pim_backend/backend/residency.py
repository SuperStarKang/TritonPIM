"""Target-neutral memory management table of a compiled model.

Adapters report only numbers: their actual units, the capacity of one unit and
the per-unit bytes of every stage operand. The planner never sees vendor handles.

Every unit has the same layout, so one table of offsets from the unit's memory
base serves all units. The key/value cache comes first from offset 0: layer by
layer, the key then the value of the token positions this unit holds. Every
phase of a request (prefill, decode) then lays its own area out from the end of
the cache, and the phases overlay one another: only the phase that runs holds
the area, so its weights overwrite those of the previous phase.

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
import math


WEIGHT = "weight"
# weight가 아닌 장치 상주 memory (실행 사이에 남아야 하는 descriptor chain, PMM 등)
STATE = "state"
INPUT = "input"
OUTPUT = "output"
# key/value cache 항목
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


@dataclass(frozen=True)
class KVRequirement:
    """The key/value cache of a model, sized for the most token positions one request stores.

    Each layer has kv_heads heads; the token positions of one head are split into
    contiguous ranges over units / kv_heads units, so every unit holds the same bytes.
    """

    layers: int
    kv_heads: int
    head_dim: int
    max_tokens: int
    element_bytes: int


@dataclass(frozen=True)
class MemoryEntry:
    """One region; offset is from the memory base of every unit.

    Stage operands belong to a phase. Key/value cache entries have no phase, stage
    or operand: they stay at the same offset through every phase.
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
    """The area of one phase, from the end of the key/value cache."""

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
    # key/value head 하나의 token 위치를 나눠 담는 unit 수와 unit 하나가 담는 위치 수
    kv_split: int
    kv_tokens_per_unit: int
    # offset 0부터의 key/value cache bytes. phase 영역은 여기서 시작한다.
    kv_bytes: int
    # key/value cache + 가장 큰 phase 영역
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
                f"and scratch {scratch}, {capacity_bytes} are left after the key/value cache")
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
                resident = operand.kind in (WEIGHT, STATE)
                entries[(phase, layer, requirement.stage, operand.operand)] = MemoryEntry(
                    operand.kind, phase, layer, requirement.stage, operand.operand,
                    slot if resident else None, group_of_layer[layer] if resident else None,
                    offsets[operand.kind], operand.bytes)
                offsets[operand.kind] += operand.bytes
    return PhaseLayout(slots, max(group_of_layer.values()) + 1, slot_of_layer, group_of_layer,
                       slot_bytes, slot_weight_bytes, slot_state_bytes, scratch,
                       state_end + scratch - base)


def kv_parts(kv, units):
    """Units sharing one key/value head, token positions per unit, and the bytes of one
    layer's key (or value) per unit."""
    if units % kv.kv_heads:
        raise ValueError(f"{units} units cannot split {kv.kv_heads} key/value heads evenly")
    split = units // kv.kv_heads
    tokens = math.ceil(kv.max_tokens / split)
    return split, tokens, tokens * kv.head_dim * kv.element_bytes


def plan_memory(resources, requirements, kv=None):
    """The memory table that fits one unit.

    requirements maps each phase, in execution order, to its stage requirements;
    kv is the model's KVRequirement, or None when the key/value cache stays on the host.
    Every stage is assumed to occupy the first unit, so stage sums bound the most
    loaded unit.
    """
    entries = {}
    kv_split = kv_tokens_per_unit = kv_bytes = 0
    if kv is not None:
        kv_split, kv_tokens_per_unit, part = kv_parts(kv, resources.units)
        # layer마다 key, value 순서로, unit이 담는 token 위치들의 head_dim 원소
        for layer in range(kv.layers):
            entries[("kv", layer, KEY)] = MemoryEntry(
                KEY, None, layer, None, None, None, None, 2 * layer * part, part)
            entries[("kv", layer, VALUE)] = MemoryEntry(
                VALUE, None, layer, None, None, None, None, (2 * layer + 1) * part, part)
        kv_bytes = 2 * kv.layers * part
    if kv_bytes > resources.capacity_bytes:
        raise ValueError(f"the key/value cache needs {kv_bytes} bytes per unit; "
                         f"capacity is {resources.capacity_bytes}")
    # 모든 phase가 key/value cache 끝에서 시작하는 같은 영역을 번갈아 쓴다.
    phases = {phase: _plan_phase(phase, phase_requirements, kv_bytes,
                                 resources.capacity_bytes - kv_bytes, entries)
              for phase, phase_requirements in requirements.items()}
    return MemoryTable(resources.units, resources.capacity_bytes, kv_split, kv_tokens_per_unit,
                       kv_bytes, kv_bytes + max(layout.bytes for layout in phases.values()),
                       phases, entries)
