# PIM Common 변환 — 1차 구현 기록

> `--convert-ttshared-to-pim` pass를 "래퍼 합성"에서 "메모리 접근 변환"으로 바꾼 작업 기록.
> (옛 이름은 `--build-pim-offload`였다.)
> 설계 근거: `MemoryLowering_Design.md`, `ComputeLowering_design.md`.
> 입력은 `--triton-to-linalg-experimental` 출력(memref + linalg), 출력은 PIM Common IR.

---

## 1. 무엇을 만들었나 (한 줄)

메모리 접근 3중구조를 **주소 명세 + 전송** 두 op으로 끌어올려 staging 버퍼를 없앤다.
`scf.for`와 program_id 식은 그대로 두고(SPMD, symbolic), 계산(linalg)은 유지하며 종류 태그만 단다.

```
reinterpret_cast + alloc + copy + to_tensor   →   pim.access + pim.load
materialize_in_destination                     →   pim.access + pim.store
```

---

## 2. 확정한 방향

| 항목 | 결정 |
|---|---|
| 옛 래퍼 모델 | **제거**. memory_region / tile_plan / execution_group / compute / launch 삭제 |
| 새 모델 | func 본문 = `scf.for + pim.access/load/store + named linalg` (top-level, program_id symbolic) |
| 지원 범위 | ① strided / ② loop-carried / store. (③④⑤⑥⑦ 및 gather/scatter는 추후) |
| `!pim.access` 타입 | 지금은 파라미터 없는 핸들(shape/elemtype는 load 결과 tensor 타입이 보유). 파라미터형은 추후 |
| offset | reinterpret_cast와 1:1로 **선형 offset 하나** + 축별 strides |
| 효과 | pim.load = Read, pim.store = Write (추후 버퍼 제거 rewrite가 쓰도록 지금 달아둠) |

---

## 3. 새 PIM dialect (`include/.../Dialect/PIM/IR/PIMDialect.td`)

- 타입 `!pim.access` — 논리 주소 핸들(무비용).
- `pim.access %base offset %o sizes array<i64: ...> strides [%s...] : memref<*xT> -> !pim.access`
  - `Pure`. broadcast 축은 stride 0.
- `pim.load %d : !pim.access -> tensor<...>` — Read.
- `pim.store %v, %d : tensor<...>, !pim.access` — Write.

옛 타입(Region/TilePlan/ExecGroup)과 옛 op(memory_region/tile_plan/execution_group/compute/launch)은 삭제.

---

## 4. raising 규칙 (`lib/.../TTSharedToPIM/TTSharedToPIMPass.cpp`)

함수마다 아래 3개를 지역 rewrite로 적용한다. `scf.for`는 건드리지 않으므로 ②는 ①에서 저절로 따라온다.

1. **주소**: `memref.reinterpret_cast`의 offset/sizes/strides를 그대로 `pim.access`로.
   상수 offset/stride는 `arith.constant index`로 만든다.
2. **로드**: `copy(reinterpret_cast → alloc)` + `to_tensor(alloc)` → `pim.load`. 세 op 제거.
3. **저장**: `materialize_in_destination(value, reinterpret_cast)` → `pim.store`. 제거.

세부:
- 같은 reinterpret_cast를 load와 store가 함께 쓰면 **하나의 pim.access를 공유**한다(dedup).
- 지원 안 하는 패턴(예: wraparound의 subview 대상 copy)은 매칭 안 되면 그대로 둔다 — 조용히 바꾸지 않는다.
- 계산 분류: matmul→`gemm`, matvec→`gemv`, transpose→`transpose`, reduce→`reduce`,
  generic→(reduction 루프 있으면 `reduce`, 아니면 `elementwise`). named op은 유지하고 `pim.class` 태그만.
- 예외 처리/우회 경로 없음. 가정(sizes static 등)이 깨지면 그 자리에서 그대로 실패한다.

---

## 5. 제거한 것 (옛 target 다리)

옛 래퍼 op을 소비하던 `legalize-pim-to-upmem`은 새 모델과 안 맞아 제거한다.

- 빌드에서 제외: `lib/Conversion/CMakeLists.txt`와 `include/triton-shared/Conversion/CMakeLists.txt`
  둘 다에서 `add_subdirectory(PIMToUPMEM)` 삭제.
- 등록 해제: `tools/triton-shared-opt/RegisterTritonSharedDialects.h`에서
  `registerLegalizePIMToUPMEMPass()` 호출과 `PIMToUPMEM/Passes.h` include 삭제.

**직접 삭제 필요(파일/디렉토리)** — 아래는 빌드/등록에서 빠졌으니 지워도 된다:
- `lib/Conversion/PIMToUPMEM/`
- `include/triton-shared/Conversion/PIMToUPMEM/`
- `test/Conversion/PIMToUPMEM/`

> UPMEM **dialect**와 **Target/UPMEM**(device.c 생성)은 그대로 둔다 — `upmem` dialect만 쓰므로
> 이번 변경과 무관하고, 나중에 PIM Common(pim.access/load/store)을 소비하는 새 target 다리에서 재사용한다.

---

## 6. 손댄 파일 목록

수정:
- `include/triton-shared/Dialect/PIM/IR/PIMDialect.td` (op/type 교체)
- `lib/Conversion/TTSharedToPIM/TTSharedToPIMPass.cpp` (raising으로 재작성)
- `include/triton-shared/Conversion/TTSharedToPIM/Passes.td` (설명·옵션·의존 dialect 정리)
- `tools/triton-shared-opt/RegisterTritonSharedDialects.h` (PIMToUPMEM 등록 해제)
- `lib/Conversion/CMakeLists.txt`, `include/triton-shared/Conversion/CMakeLists.txt` (PIMToUPMEM 빌드 제외)
- `test/Dialect/PIM/ops.mlir` (새 op roundtrip)
- `test/Conversion/TTSharedToPIM/elementwise.mlir`, `matmul.mlir` (새 기대 출력)

추가:
- `test/Conversion/TTSharedToPIM/loop_carried.mlir` (② 검증)

파이프라인:
- `pim_backend/backend/compiler.py` — `pim_common` 단계를 `--convert-ttshared-to-pim`으로 바꾸고
  분배 옵션(split/local_lanes) 전달을 뺐다(분배는 target 관심사).

## 6.1 지금 끊긴 것 (의도된 상태)

`pim_backend`의 `vendor_ir`/`pimobj` 단계는 옛 다리(`--legalize-pim-to-upmem` + upmem translate)를
쓰므로, `vendor="upmem"`으로 끝까지 돌리면 **legalize 단계에서 그대로 실패한다**(no-fallback).
`pim_common` 단계까지(=이번 변환)는 정상 동작한다. UPMEM 끝단은 새 target 다리(§8)가 나오면 이어진다.

---

## 7. 검증 방법 (직접 실행)

빌드 후:
```
triton-shared-opt --convert-ttshared-to-pim test/Conversion/TTSharedToPIM/elementwise.mlir
triton-shared-opt --convert-ttshared-to-pim test/Conversion/TTSharedToPIM/loop_carried.mlir
```
또는 lit:
```
lit test/Conversion/TTSharedToPIM test/Dialect/PIM -v
```
기대: `elementwise`/`loop_carried`는 `memref.*`·`bufferization.*`가 사라지고 `pim.access/load/store`가 나온다.
`matmul`은 `linalg.matmul`이 유지되고 `pim.class = "gemm"` 태그가 붙는다.

---

## 8. 아직 안 한 것 (다음 작업)

- `pim.access` clause: ③ order · ④ wrap · ⑤ valid+pad · ⑥ gather, 그리고 ⑦ `pim.gather`/`pim.scatter`.
- Stage B 버퍼 제거 rewrite(passthrough elision / load CSE / LICM / store→load forwarding / dead-transfer DCE)
  — 효과(Read/Write)는 달아뒀으니 여기에 얹으면 된다.
- `!pim.access` 파라미터형 타입 + verifier(설계 §3 R1~R7).
- 새 target 다리: PIM Common(pim.access/load/store) → UPMEM. (옛 legalize-pim-to-upmem 대체)
