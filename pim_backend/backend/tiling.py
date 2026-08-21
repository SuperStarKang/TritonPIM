# Common three-level PIM planner used at a kernel launch site.
#
#   BLOCK_M/N/K   Triton block grain
#   group_m/n/k   block group (local-memory or hardware execution unit)
#   tile_m/n/k    execution tile resident in one abstract PIM Device
#
# Execution tiles may outnumber ``num_units`` on every spatial axis. The common
# lowering reuses the same PIM Device set over multiple scheduling rounds.
# ``split_k`` is the number of independent spatial partial-output planes;
# ``tile_k`` is the resident temporal K window inside each plane.


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _active_device_grid(M, N, K, execution_m, execution_n,
                        spatial_planes, num_units):
    """Choose the active M/N/K PIM Device grid of one scheduling round."""
    execution_tiles_m = _ceil_div(M, execution_m)
    execution_tiles_n = _ceil_div(N, execution_n)

    best_grid = None
    best_score = None
    max_k = min(spatial_planes, num_units)
    work = execution_tiles_m * execution_tiles_n * spatial_planes
    for devices_k in range(1, max_k + 1):
        output_budget = num_units // devices_k
        for devices_m in range(
                1, min(execution_tiles_m, output_budget) + 1):
            max_n = min(execution_tiles_n, output_budget // devices_m)
            for devices_n in range(1, max_n + 1):
                rounds_m = _ceil_div(execution_tiles_m, devices_m)
                rounds_n = _ceil_div(execution_tiles_n, devices_n)
                rounds_k = _ceil_div(spatial_planes, devices_k)
                rounds = rounds_m * rounds_n * rounds_k
                active = devices_m * devices_n * devices_k
                idle = rounds * active - work

                # M 분할은 B를, N 분할은 A를 복제한다. 값은 정확한 시간 예측이
                # 아니라 round와 idle이 같은 후보 사이의 전송량 tie-breaker다.
                transfer_proxy = (M * K * devices_n +
                                  K * N * devices_m +
                                  4 * M * N * devices_k)
                score = (rounds, idle, transfer_proxy, -active,
                         devices_k, devices_m, devices_n)
                if best_score is None or score < best_score:
                    best_score = score
                    best_grid = (devices_m, devices_n, devices_k)

    assert best_grid is not None, "PIM Device grid 후보가 없습니다"
    return best_grid


def device_plan(M, N, K, *,
                BLOCK_M, BLOCK_N, BLOCK_K,
                group_m=1, group_n=1, group_k=1,
                tile_m=None, tile_n=None, tile_k=None,
                split_k=None, num_units=2048):
    """Build launch options for an execution-tile PIM Device schedule.

    Each scheduling round assigns at most one M/N execution tile to an abstract
    PIM Device. More M/N tiles create more rounds over the same Device set. A
    ``split_k`` controls independent spatial output planes, while ``tile_k``
    controls the resident temporal window accumulated inside each plane.
    """
    assert tile_m is not None and tile_n is not None, (
        "device_plan은 구체적인 tile_m과 tile_n을 필요로 합니다")
    assert num_units > 0, "num_units는 양수여야 합니다"
    assert min(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
               group_m, group_n, group_k, tile_m, tile_n) > 0, (
        "문제 크기, BLOCK, group, tile_m, tile_n은 모두 양수여야 합니다")
    if tile_k is not None:
        assert tile_k > 0, "tile_k는 양수여야 합니다"
    if split_k is not None:
        assert split_k > 0, "split_k는 양수여야 합니다"

    group_size_m = group_m * BLOCK_M
    group_size_n = group_n * BLOCK_N
    group_size_k = group_k * BLOCK_K
    assert tile_m % group_size_m == 0, (
        f"tile_m({tile_m})은 M block group 크기({group_size_m})의 배수여야 "
        "합니다")
    assert tile_n % group_size_n == 0, (
        f"tile_n({tile_n})은 N block group 크기({group_size_n})의 배수여야 "
        "합니다")

    options = {
        "group_m": group_m,
        "group_n": group_n,
        "group_k": group_k,
    }

    spatial_planes = 1
    if tile_k is not None:
        assert tile_k % group_size_k == 0, (
            f"tile_k({tile_k})는 K block group 크기({group_size_k})의 배수여야 "
            "합니다")
        # split_k를 생략한 기존 사용법은 tile_k 하나를 spatial plane 하나로 본다.
        # 더 큰 plane 안에서 temporal window를 사용하려면 split_k를 명시한다.
        if split_k is None:
            assert K % tile_k == 0, (
                f"split_k를 생략하면 K({K})는 tile_k({tile_k})의 배수여야 합니다")
            spatial_planes = K // tile_k
        else:
            spatial_planes = split_k
        assert K % spatial_planes == 0, (
            f"K({K})는 SPLIT_K({spatial_planes})로 나누어떨어져야 합니다")
        local_k = K // spatial_planes
        assert local_k % BLOCK_K == 0, (
            f"각 K plane({local_k})이 BLOCK_K({BLOCK_K}) 격자에 "
            "맞아야 합니다")
        assert tile_k <= local_k, (
            f"tile_k({tile_k})는 각 spatial K plane({local_k})보다 클 수 없습니다")
        options["SPLIT_K"] = spatial_planes
    else:
        assert split_k is None, (
            "split_k를 사용하면 spatial execution tile을 나타내는 "
            "tile_k도 지정해야 합니다")

    devices_m, devices_n, devices_k = _active_device_grid(
        M, N, K, tile_m, tile_n, spatial_planes, num_units)
    parts = {
        "output_row_tile": devices_m,
        "output_column_tile": devices_n,
    }
    if tile_k is not None:
        parts["explicit_k_partition"] = devices_k

    options["device_axis_partitions"] = ",".join(
        f"{axis}:{count}" for axis, count in parts.items())
    options["num_units"] = num_units
    options["tile_m"] = tile_m
    options["tile_n"] = tile_n
    if tile_k is not None:
        options["tile_k"] = tile_k
    return options


def execution_tile(*args, **kwargs):
    """Compatibility name for :func:`device_plan`."""
    return device_plan(*args, **kwargs)
