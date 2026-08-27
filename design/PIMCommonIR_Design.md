# PIM Common IR final design

This document defines the common IR contract and records the current
implementation checkpoint.  The contract is authoritative; target execution
details remain behind `TargetSpec` and target lowering.

## 1. Compiler boundary

The lowering pipeline has one common PIM stage and one target-specific stage.

```text
TTIR
  -> Triton Shared IR
  -> PIM Common IR
  -> Target IR (UPMEM IR, AiMX IR, ...)
  -> target code generation
```

Triton Shared IR recognition decides the logical operation, tensor accesses,
and the `m`, `n`, and `k` axes.  PIM Common passes decide the three data
levels, intra-device decomposition, inter-device scheduling, placement, and
reduction.  A target lowering translates those decisions and verifies only
constraints that are intrinsic to that target.

The common dialect does not contain target-specific field names, register
names, or execution-granularity fields.  The generic `target` selector only
chooses the profile and final lowering.

## 2. Three data levels

All shapes use the fixed axis order `[m, n, k]`.

1. **Block** is the tile recognized from Triton Shared IR.
2. **Block group** is an integer grid of adjacent blocks.  It is the unit that
   is decomposed across banks, cores, and lanes inside one device.
3. **Execution tile** is the actual data region placed on one PIM Device for
   one resident execution.  It contains an integer grid of block groups.

If `B`, `G`, and `E` are three-element vectors, their useful shapes are:

```text
block shape          = B
block-group shape    = B * G
execution-tile shape = B * G * E
```

An execution tile can contain several block groups.  In the initial
implementation the groups are visited sequentially.  Every visited block
group is decomposed using the same intra-device plan.

## 3. Axis identity

`#pim.axis<"m">`, `#pim.axis<"n">`, and `#pim.axis<"k">` are the only common
axis identities.  `m` and `n` are parallel output axes.  `k` is a reduction
axis.  An operation without a reduction simply omits `k`.

The IR does not use frontend names such as `output_row_tile`, generic names
such as `x/y/r`, or descriptive scope fields.

## 4. Target profile

`#pim.target_profile` contains only common topology and usable capacity:

```mlir
#pim.target_profile<
  target = "...",
  number_of_pim_devices = ...,
  number_of_banks_per_pim_device = ...,
  number_of_pim_cores_per_bank = ...,
  number_of_parallel_lanes_per_pim_core = ...,
  parallel_execution_kind = "...",
  usable_bank_capacity_bytes = ...,
  usable_local_capacity_bytes = ...
>
```

`usable_bank_capacity_bytes` is usable capacity **per bank**.  Homogeneous
devices derive aggregate usable capacity instead of recording another field:

```text
usable_device_capacity_bytes =
    number_of_banks_per_pim_device * usable_bank_capacity_bytes
```

Both per-bank footprint and aggregate device footprint must fit.  Replication,
padding, descriptors, and fixed target allocations count as physical
footprint.  The first implementation records usable rather than physical
capacity, so reserved runtime storage is not subtracted a second time.

Initial profiles are:

| field | UPMEM | AiMX/Silent-PIM |
|---|---:|---:|
| PIM Device | one DPU | one FPGA/card |
| banks per device | 1 | 256 (16 channels x 16 banks, flattened) |
| cores per bank | 1 | 1 |
| lanes per core | configured tasklets, initially 16 | 16 |
| usable bank capacity | usable MRAM, initially 60 MiB | 256 MiB |
| usable local capacity | usable WRAM, initially 48 KiB | 0 |

AiMX vecA and vACC are fixed execution state, not compiler-allocatable local
storage.  Their constraints remain in the AiMX lowering verifier.

## 5. Intra-device plan

`#pim.intra_device_plan` decomposes **one block group**, not a grid of separate
block groups:

```mlir
#pim.intra_device_plan<
  bank_partitions = [Pm, Pn, Pk],
  core_partitions = [Cm, Cn, Ck],
  lanes = L,
  lane_axis = "m" | "n" | "k"
>
```

For every block group:

```text
block group
  -> split into bank subtiles by bank_partitions
     -> split each bank subtile into core subtiles by core_partitions
        -> split each core subtile across lanes on lane_axis
```

The initial assignment is sequential and contiguous.  There is no assignment
policy field.  The common verifier checks:

```text
product(bank_partitions) <= banks_per_device
product(core_partitions) <= cores_per_bank
lanes <= lanes_per_core
```

The common verifier checks topology counts.  Each target lowering converts the
useful block-group shape to its physical execution shape and then checks
divisibility.  AiMX may pad to its batch/register grain; UPMEM requires the
useful `m` extent to divide its cooperative tasklet count.  Padded values are
zero-filled and only the useful output is returned.

Partitioning `k` creates partial outputs and therefore requires an explicit
reduction plan.  For matmul, partitioning `m` replicates B, partitioning `n`
replicates A, and partitioning `k` creates partial C.

The initial target plans are:

```text
UPMEM: bank=[1,1,1], core=[1,1,1], lanes=<tasklets>, lane_axis=m
AiMX:  bank=[batch, 256/batch, 1], core=[1,1,1], lanes=16, lane_axis=n
```

AiMX target lowering verifies `batch in {1,2,4,8}`, the exact physical bank
decomposition, and its descriptor/register granularity.  These values are not
added to the common target profile.

## 6. Inter-device plan

`#pim.inter_device_plan` schedules execution tiles over PIM Devices.  Its
partitions are device partitions, while `#pim.intra_device_plan` partitions a
block group inside each selected device.  These two partition spaces must not
be conflated.

`pim.device` remains the region containing the work and placement for one
abstract PIM Device.  `pim.place` is not used as the name of the device region;
it remains only the buffer-placement operation inside `pim.device`.

## 7. Autotuner contract

The autotuner supplies only plan parameters before PIM planning:

```text
block_m/n/k
group_m/n/k
execution_m/n/k
bank_partitions_m/n/k
core_partitions_m/n/k
lanes
lane_axis
inter_device_partitions_m/n/k
reduction policy
```

The backend serializes these values as pass options.  Common planning passes
validate them against `#pim.target_profile` and materialize the two plan
attributes.  Target lowering must not silently retune them; it either
translates the plan or rejects it with a target-specific diagnostic.

The first search space keeps core partitions fixed at one and materializes one
legal target seed (`UPMEM: m/tasklets`, `AiMX: batch=8`).  Block/group/execution
sizes and device partitions are searched now; mutation across legal bank/lane
plans is the next autotuner extension.

## 8. Dialect boundary

PIM Common IR deliberately uses a small dialect set:

| dialect | role |
|---|---|
| `pim` | access, transfer, placement, hierarchy plans, and reduction contract |
| `func` | kernel boundary |
| `scf` | block-group iteration, execution-tile rounds, and dynamic tails |
| `arith` | index and bound calculation |
| `tensor` + `linalg` | target-neutral compute, including matmul and elementwise maps |
| `memref` | host-buffer boundary until placement is lowered |

UPMEM and AiMX dialects appear only after the PIM Common IR boundary.  No
separate dialect is introduced for planning or autotuning; their decisions are
materialized as `pim` attributes.

## 9. Implementation order

1. Add topology fields, `#pim.axis`, `#pim.intra_device_plan`, and
   `#pim.inter_device_plan`.
2. Make common planning materialize and verify these attributes.
3. Change UPMEM and AiMX lowering to consume the new attributes.  Keep AiMX
   execution-granularity checks target-local.
4. Teach placement/footprint analysis to calculate per-bank physical
   footprint, replication, and `k` partials.
5. Add elementwise recognition and reuse the same three levels.  Elementwise
   operations normally use `m/n`, omit `k`, and therefore need no reduction
   plan.

## 10. Current implementation checkpoint

The current implementation completes items 1--4 for the matmul path: topology
and axis attributes, both plans, autotuner-to-pass option transport, common
validation, UPMEM/AiMX plan consumption, and AiMX per-bank physical-footprint
accounting.  AiMX uses 256 MiB of usable capacity per bank and performs its
256-bank/16-lane legality checks in target lowering.

Item 5 now has an initial end-to-end UPMEM implementation.  A reduction-free
`m/n` grid automatically forms an execution tile, and an identity-mapped
`linalg.generic` is lowered through the same block, block-group, and execution-
tile hierarchy.  All iterators must currently be parallel, every operand must
have the same static shape, and the scalar region may contain arithmetic such
as a fused add followed by multiply.  UPMEM row partitioning slices every
operand consistently and emits one flat device loop for the fused scalar
region.  The regression path starts from the Triton Shared memref/copy/tensor
idiom rather than hand-written PIM IR, so recognition and target lowering are
checked together.  AiMX elementwise target lowering is not implemented yet; it
remains a target-specific extension rather than a change to the common
hierarchy.

The pre-existing `execution_unit` and `lane_mode` profile fields have been
removed.  AiMX batch and physical padding now belong to AiMX target lowering;
the common block-group attribute records only Triton block grouping.  No new
`execution_granularity` field is introduced.

The Python hardware profile follows the same topology and derives device
capacity from banks-per-device times usable capacity per bank.  AiMX accounts
for A, rearranged A, B, C, descriptor, and PMM allocations separately.  Each
allocation is rounded to the next 8192-byte device-wide bank cycle and charged
one 32-byte burst per bank.  Summing those independent tails is a safe per-bank
upper bound when `pim_malloc` does not expose allocation start banks.  If the
runtime later exposes physical start addresses, diagnostics may additionally
report the exact bank-indexed distribution; legality does not depend on it.
