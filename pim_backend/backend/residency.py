"""Target-neutral memory management table of a compiled model.

Adapters report only numbers: their actual units, the capacity of one unit and
the per-unit bytes of every stage operand. The planner never sees vendor handles.

Every unit has the same layout, so one table of offsets from the unit's memory
base serves all units. One slot holds the weights and state of one layer's
stages; layer i takes slot i mod slots and belongs to group i // slots. If every
layer fits there is one group; otherwise the layers of one group fill every slot
and are uploaded together, overwriting only the weights of the previous group
while buffers and instructions stay in place.

The unit is laid out as three areas: the weights of every slot, then the state
of every slot, then the input and output scratch. The weights of one group are
therefore one contiguous range from the unit base, so a group upload is one
transfer per unit. Every stage uses the scratch from its start, so each scratch
part is as large as its largest stage.
"""

from dataclasses import dataclass


WEIGHT = "weight"
# weight가 아닌 장치 상주 memory (실행 사이에 남아야 하는 descriptor chain, PMM 등)
STATE = "state"
INPUT = "input"
OUTPUT = "output"


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
class MemoryEntry:
    """One operand region; offset is from the memory base of every unit."""

    kind: str
    layer: int
    stage: str
    operand: int
    # weight/state는 slot과 group에 속한다. scratch(input/output)는 None이다.
    slot: int | None
    group: int | None
    offset: int
    bytes: int


@dataclass(frozen=True)
class MemoryTable:
    units: int
    capacity_bytes: int
    # slot 수와 layer group 수. layer i는 slot i mod slots, group i // slots에 놓인다.
    slots: int
    groups: int
    # layer index -> slot
    slot_of_layer: dict
    # layer index -> group. 한 group의 layer들은 함께 올라간다.
    group_of_layer: dict
    # slot마다 weight + state bytes
    slot_bytes: tuple
    # slot마다 weight bytes와 state bytes. weight 영역은 unit 시작, state 영역은 그 뒤다.
    slot_weight_bytes: tuple
    slot_state_bytes: tuple
    scratch_bytes: int
    bytes_per_unit: int
    # (layer, stage, operand) -> MemoryEntry
    entries: dict


def plan_memory(resources, requirements):
    """The memory table that fits one unit.

    Every stage is assumed to occupy the first unit, so stage sums bound the
    most loaded unit.
    """
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
    if sum(layer_weight.values()) + sum(layer_state.values()) + scratch <= resources.capacity_bytes:
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
                    f"layers share a slot; layer {layer} operands need {layouts[layer]} bytes, "
                    f"layer {layers[0]} operands {layouts[layers[0]]}")
        layer_bytes = layer_weight[layers[0]] + layer_state[layers[0]]
        slots = (resources.capacity_bytes - scratch) // layer_bytes
        if slots < 1:
            raise ValueError(
                f"one layer does not fit one unit: it needs {layer_bytes} bytes "
                f"and scratch {scratch}, capacity is {resources.capacity_bytes}")
        slot_weight_bytes = (layer_weight[layers[0]],) * slots
        slot_state_bytes = (layer_state[layers[0]],) * slots
    slot_bytes = tuple(weight + state for weight, state in zip(slot_weight_bytes, slot_state_bytes))
    slots = len(slot_bytes)
    slot_of_layer = {layer: position % slots for position, layer in enumerate(layers)}
    group_of_layer = {layer: position // slots for position, layer in enumerate(layers)}

    weight_end = sum(slot_weight_bytes)
    state_end = weight_end + sum(slot_state_bytes)
    entries = {}
    for layer in layers:
        slot = slot_of_layer[layer]
        # 같은 slot의 layer는 같은 offset을 받는다. weight는 weight 영역, state는 state 영역의
        # 앞 slot들 바로 뒤에서 시작한다.
        offsets = {WEIGHT: sum(slot_weight_bytes[:slot]), STATE: weight_end + sum(slot_state_bytes[:slot])}
        for requirement in stages[layer]:
            offsets[INPUT] = state_end
            offsets[OUTPUT] = state_end + input_bytes
            for operand in requirement.operands:
                resident = operand.kind in (WEIGHT, STATE)
                entries[(layer, requirement.stage, operand.operand)] = MemoryEntry(
                    operand.kind, layer, requirement.stage, operand.operand,
                    slot if resident else None, group_of_layer[layer] if resident else None,
                    offsets[operand.kind], operand.bytes)
                offsets[operand.kind] += operand.bytes
    return MemoryTable(resources.units, resources.capacity_bytes, slots,
                       max(group_of_layer.values()) + 1, slot_of_layer, group_of_layer,
                       slot_bytes, slot_weight_bytes, slot_state_bytes, scratch,
                       state_end + scratch, entries)
