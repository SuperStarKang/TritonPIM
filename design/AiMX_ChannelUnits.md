# AiMX 채널 unit — 실측 사실과 보류한 설계 메모

상태: **보류**. 실측은 끝났고 구현은 하지 않았다. 지금 decode attention은 규칙 ④/⑤
(0으로 채워 KV head를 FPGA 수만큼 묶음, [TorchIntegration](TorchIntegration.md) "key/value cache를 읽는 GEMM",
`torch_backend/fx_packing.py`의 `_pack_cache_windows`)로 실행하고, cache stage는 instance 하나가
FPGA 하나를 쓴다([Silent-PIM 설계](Silent-PIM_Design.md) §8.4).

## 1. FPGA 실측 (`~/kangmlee/Experiment/ChannelVecA`)

| 실험 | 결과 |
| --- | --- |
| `channel_veca` (b=1): per_channel, rotated, sweep(lane×k 1,048,576쌍), dense K=256/512 | 모두 PASS. dense own rel_l2 1.1~1.9% (공유 A 기준선과 같음), 다른 채널 A 기준 1.2~1.4 |
| `batch_veca` (b=1,2,4,8): layout, decode(2,097,152 검사), dense | 모두 PASS. `stripe_match=4096/4096` |

확정된 것:

- Arel의 32 B stripe g(16개)는 채널 g의 vecA 몫이다. host가 Arel을 직접 쓰면 채널마다 다른 A를
  줄 수 있다(`bc|tp` 재배치는 16개를 같은 값으로 복제할 뿐).
- 출력 (r, j)의 채널은 C 주소의 stripe `(j/16)%16`이다. B 열도 같은 규칙이다.
- batch b에서 채널 하나 = b행 × 256/b열, b행이 그 채널의 B 열을 공유한다.
- Arel 순열 (모든 b):

```text
Arel[s·4096 + u·256 + g·16 + t] = A_g[r][k]
  s = MAC step (8 KiB), u = 512 B 구간, g = 채널, t = 0..15
  r = u / (16/b),  k = s·(256/b) + (u mod (16/b))·16 + t
```

모든 채널이 같은 A이면(현재 lowering) b=1, m=1에서 `Arel[c·256 + g·16 + t] = A[16c + t]`가 된다.
물리 "채널"이라는 이름은 주소 사상 `(PA>>5)&15` 가정이다. 컴파일러에는 논리 대응만 필요하다.

## 2. decode attention 구조 (보류한 설계)

- instance = KV head, query 행 G개(GQA group)를 쌓는다. unit = 채널(FPGA × 16, lockstep).
- QK `(G, C, d)`: B = K windowᵀ, 위치를 p_n으로 채널에 분할. batch b = G (Llama-3 b=4: k step 64,
  head_dim 128 = 2 step, 0 없음, key 복제 없음).
- PV `(G, d, C)`: B = V window, 위치(K)를 p_k로 분할, host가 device-k 순서로 합산(지금의
  `aimx.device_reduction` 경로와 같은 순서 규칙).
- 추정(FPGA당, 1057 위치): Llama-3 QK 98 → 28 µs, PV 69 → 28 µs; Llama-2 QK 394 → 189 µs,
  PV 275 → 189 µs (b=1, MHA).

저장: host cache `[kv_heads, positions, head_dim]`와 위치 API(`write_kv`)는 그대로 두고, 채널 g의
데이터를 512 B마다 32 B stripe g에 둔다. K는 행 = head 차원·열 = 위치, V는 행 = 위치·열 = head
차원. AiMX는 lockstep으로 capacity 전체를 계산하므로 위치는 block 분할(`p // S`)이 맞다(decode
예제의 flush 단위 `TAIL = 16` 위치 = stripe 하나, 32 B 정렬 쓰기).

## 3. 바꿔야 할 곳

| 계층 | 지금 코드 | 수정 |
| --- | --- | --- |
| Common IR, 3계층 | device 단위 SPMD | 없음 (채널 = lane 256개 device) |
| target profile | `aimx_target_spec`: `banks_per_unit = 256`, unit = FPGA | 채널 모드 unit 수 ×16 |
| `AiMXModel` | `_compute_unit`, `stage_target`(GEMM batch), `resident_operand_config/reason`(cache B padding) | cache stage만 연산 단위 `(b, 256/b, 256/b)`, 채널 단위 상주 B 크기 |
| PIMToAiMX | `ConvertPIMToAiMXPass`가 `getBanksPerUnit() == kNumBanks × kNumChannels`를 요구, `DeviceLowering::run`이 device(FPGA)마다 `foreach_device` 몸체에서 chain·할당, `lowerMACReduction`이 `buildMatmulRelayout`(`bc\|tp`, Arel `× kNumChannels`)을 낸다 | FPGA당 unit 16개를 chain 하나로, stripe 배치, `bc\|tp` 대신 host Arel stage |
| runtime stub | `emitStage`/`emitUnstage`의 `group_layout` panel 배치, `pim_model_slot_layout` | stripe 단위 stage/unstage와 그 layout 표 |
| `aimx_model.py` | `_create_stage`: cache stage는 `devices = 1`, instance i가 FPGA `base + i`. `_kv_offsets`·`_push_kv`·`_check_kv_layout`이 panel 배치로 위치를 쓴다 | instance를 채널에 배정, stripe offset과 block 위치 분할, 대조식도 같이 |
| fx_packing | cache reader는 규칙 (1)–(3)을 받지 않고(`_instance_groups(reading=True)`), GEMM 수가 unit 수보다 많을 때만 `_pack_cache_windows`가 (4)/(5)로 묶는다 | 같은 window를 읽는 query 행 쌓기((1))를 cache reader에 허용. UPMEM cache stage에도 적용되므로 그쪽 영향 확인 필요 |

먼저 할 것: QK b=4 채널 묶음 하나를 descriptor로 직접 실행하는 micro-benchmark로 28 µs 추정 확인.
