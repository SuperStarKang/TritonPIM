# 3계층 tiling 계산기 — 사용자가 execution tile 크기(원소 단위)로 생각하고,
# launch kwargs(group·분할·SPLIT_K)는 여기서 유도한다.
#
#   BLOCK_M/N/K   Triton tile 눈금            (커널 constexpr)
#   group_m/n/k   block group = WRAM 상주 단위 (tile 개수 단위)
#   tile_m/n/k    execution tile = DPU 하나가 맡는 영역 (원소 단위)
#
# 왜 backend 옵션이 아니라 launch 시점 helper 인가: device 분할 수는 IR 의 컴파일
# 시점 상수인데, tile 크기 → 분할 수 변환에는 문제 크기(M, N, K)가 필요하고 그것은
# 런타임 kernel 인자다. 문제 크기를 아는 곳은 launch 지점뿐이다. 값을 직접 계산해
# 넣고 싶으면 이 helper 를 거치지 않고 device_axis_partitions/SPLIT_K 를 그대로
# 주면 된다 — 반환값이 정확히 그 kwargs 다.


def execution_tile(M, N, K, *,
                   BLOCK_M, BLOCK_N, BLOCK_K,
                   group_m=1, group_n=1, group_k=1,
                   tile_m=None, tile_n=None, tile_k=None,
                   split_k=None, num_units=2048):
    """execution tile(원소 단위)로부터 kernel launch kwargs 를 만든다.

    tile_* 을 주지 않은 축은 나누지 않는다(그 축 전체가 각 DPU 의 몫).
    tile_k 를 주면 SPLIT_K 가 함께 유도되므로 split-K kernel(K 를 연속 구간으로
    가르는 blocked 형태)이어야 하고, 반환 dict 의 SPLIT_K 를 커널 constexpr 로
    받는다. split_k 를 따로 주면 한 DPU 가 여러 k 조각(plane)을 맡는다.
    """
    gm, gn, gk = group_m * BLOCK_M, group_n * BLOCK_N, group_k * BLOCK_K
    parts = {}

    def output_axis(tile, unit, name, axis_name):
        if tile is None:
            return
        # execution tile 경계는 group 경계 위에 있어야 balanced block 배분이 성립한다.
        assert tile % unit == 0, (
            f"{name}({tile})은 block group의 그 방향 크기({unit})의 배수여야 합니다")
        extent = M if axis_name == "output_row_tile" else N
        parts[axis_name] = -(-extent // tile)  # ceildiv — 마지막 DPU 는 덜 맡는다

    output_axis(tile_m, gm, "tile_m", "output_row_tile")
    output_axis(tile_n, gn, "tile_n", "output_column_tile")

    opts = dict(group_m=group_m, group_n=group_n, group_k=group_k)
    if tile_k is not None:
        assert tile_k % gk == 0, (
            f"tile_k({tile_k})는 block group의 k 방향 크기({gk})의 배수여야 합니다")
        # k 조각(plane)은 균등해야 host 합산이 성립한다 — 출력 축과 달리 꼬리가 없다.
        assert K % tile_k == 0, f"K({K})는 tile_k({tile_k})의 배수여야 합니다"
        pk = K // tile_k
        sk = split_k if split_k is not None else pk
        assert sk % pk == 0, (
            f"SPLIT_K({sk})는 k 분할 수({pk})의 배수여야 합니다 — plane 이 DPU 에 "
            f"통째로 배분됩니다")
        assert (K // sk) % BLOCK_K == 0, (
            f"각 k 조각(K/SPLIT_K = {K // sk})이 BLOCK_K({BLOCK_K}) 격자에 맞아야 "
            f"MRAM tile 주소가 성립합니다")
        parts["explicit_k_partition"] = pk
        opts["SPLIT_K"] = sk
    elif split_k is not None:
        assert (K // split_k) % BLOCK_K == 0, (
            f"각 k 조각(K/SPLIT_K = {K // split_k})이 BLOCK_K({BLOCK_K}) 격자에 "
            f"맞아야 합니다")
        opts["SPLIT_K"] = split_k

    total = 1
    for p in parts.values():
        total *= p
    assert total <= num_units, (
        f"분할 곱({total})이 PIM Device 수({num_units})를 넘습니다 — tile_* 을 "
        f"키우거나 축 하나를 나누지 마십시오")

    if parts:
        opts["device_axis_partitions"] = ",".join(
            f"{a}:{p}" for a, p in parts.items())
    return opts
