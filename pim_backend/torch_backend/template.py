# PIM split-K TritonTemplate.
#
# kernel body는 examples/split_k.py의 matmul_split_k_kernel과 같은 연산이다.
# 같아야 하는 이유: 그 커널은 PIM 파이프라인(TTIR → TTShared IR → PIM IR →
# artifact)을 통과하는 것이 확인된 유일한 GEMM이다. Inductor의 stock mm template를
# 가져다 쓰면 tl.assume / tensor descriptor / GPU 전용 heuristic이 섞여 들어와,
# 실패했을 때 그것이 PIM 연동 문제인지 IR 변환 문제인지 갈라내지 못한다.
#
# [M/N/K와 stride를 런타임 인자로 넘기는 이유]
# Inductor의 {{size}}/{{stride}} hook은 static shape에서 정수 리터럴을 낸다. 그러면
# PIMRecoverAccessOriginPass가 split-K 출력면을 별도 축으로 승격하지 못한다.
# 그 pass의 canonicalStride는 stride 식을 (기준 SSA 값, 상수 배수)로 가르는데,
#
#   stride_ck 가 상수 M*N  → base = null, coeff = M*N
#   tile 축 0 stride 도 상수 N → base = null, coeff = N
#   → base 가 둘 다 null 로 일치하고 (M*N) % N == 0 이라 pid_k 항이 행 축에 흡수된다
#
# 그 결과 출력 access가 rank-2가 되고, 버퍼 축 0을 pid_k와 pid_m이 함께 몰게 되어
# PIMPlaceDataPass의 checkBlockedComposite가 진단한다.
#
# stride가 런타임 인자면 base가 SSA 값이라 어느 tile 축의 base(null)와도 일치하지
# 않는다. 그때 pass는 promote()로 크기 1짜리 축을 앞에 새로 만든다 — 주석 그대로
# "split-K의 출력면처럼 원본 버퍼에는 있는데 tile에는 없는 축"이다. 축마다 loop가
# 하나씩만 남으므로 composite 판정 자체가 일어나지 않는다.
# examples/split_k.py가 통과하는 이유가 이것이고, 여기서도 같은 시그니처를 만든다.
#
# stock template와 다른 점은 넷이고, 전부 Inductor 규약이나 위 구조를 맞추기 위한 것이다.
#   - a_ptr/b_ptr/c_ptr 대신 def_kernel이 만들어 준 이름(A, B, PARTIAL)을 쓴다.
#   - M/N/K와 stride를 {{rt_size}}/{{rt_stride}}로 런타임 인자에 싣는다.
#   - store_output을 쓰지 않는다. 출력은 mutated input인 PARTIAL이다 (아래 참고).
#   - 나누어떨어지는 shape에서는 mask를 만들지 않는다 (EVEN_TILES, 아래 참고).
#
# [PARTIAL을 mutated input으로 두는 이유]
# examples/split_k.py는 partial을 torch.zeros로 만든다. Inductor의 template output
# buffer는 초기화되지 않은 버퍼라 그 의미가 유지되지 않는다. 그래서 zero로 채운
# 버퍼를 lowering이 따로 만들어 template의 mutated input으로 넘긴다. template의
# 선언상 출력은 쓰이지 않는 [1] int32 buffer다 (lowering.py 참고).
# PIM generated host code가 partial의 모든 원소를 반드시 덮어쓴다는 것이 확인되면
# store_output을 쓰는 보통의 template output으로 바꿀 수 있다.
#
# [mask를 지우는 이유 — EVEN_TILES]
# mask의 extent가 상수가 되면 PIMPlaceDataPass의 isExtentCandidate가 상수를 extent
# 후보에서 배제하므로 extent 복원이 실패한다. tile이 나누어떨어지는 shape에서는
# mask가 항상 참이므로 아예 만들지 않는다. 나누어떨어지지 않는 shape는 lowering이
# 막는다 (config.allow_masked_tiles 참고).

import sympy

from torch._inductor.select_algorithm import (
    SymbolicGridFn,
    TritonTemplate,
    TritonTemplateKernel,
)
from torch._inductor.utils import sympy_str


@SymbolicGridFn
def pim_split_k_grid(m, n, meta, *, cdiv):
    """split-K grid: (C tile 수, K 분할 수, 1).

    examples/split_k.py의 grid와 같다. 두 번째 축이 K 분할이라 program_id(1)이
    자기 몫의 K 구간을 가리킨다."""
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"], 1)


class PIMTemplateKernel(TritonTemplateKernel):
    """shape/stride를 상수 대신 런타임 스칼라 인자로 싣는 template kernel.

    Inductor의 통로를 그대로 쓴다 — args.sizevars에 넣으면 python_argdefs가
    SizeArg로 인자를 만들고, call_kernel이 그 값을 호출부에 실어 준다. 값이 1인
    인자는 Triton이 constexpr로 특수화하는데, 이는 직접 경로에서도 같다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_extra_template_env_fns(self.rt_size, self.rt_stride)

    def _runtime_scalar(self, expr) -> str:
        expr = sympy.expand(expr)
        if expr.free_symbols:
            # dynamic shape면 기존 통로가 이미 sizevar 인자를 만든다.
            return sympy_str(self.rename_indexing(expr))
        # static shape면 정수라 그 통로를 타지 않는다. 같은 값은 한 인자로 모인다 —
        # 값이 같으므로 의미가 달라지지 않는다.
        return self.args._lookup("ks", self.args.sizevars, expr)

    def rt_size(self, name: str, index: int) -> str:
        """{{rt_size("A", 0)}} — 크기를 런타임 인자로."""
        return self._runtime_scalar(self.named_input_nodes[name].get_size()[index])

    def rt_stride(self, name: str, index: int) -> str:
        """{{rt_stride("A", 0)}} — stride를 런타임 인자로."""
        strides = self.get_stride_and_maybe_freeze_layout(
            self.named_input_nodes[name])
        return self._runtime_scalar(strides[index])


class PIMTritonTemplate(TritonTemplate):
    kernel_type = PIMTemplateKernel


PIM_SPLIT_K_SOURCE = r"""
{{def_kernel("A", "B", "PARTIAL")}}
    M = {{rt_size("A", 0)}}
    N = {{rt_size("B", 1)}}
    K = {{rt_size("A", 1)}}

    stride_am = {{rt_stride("A", 0)}}
    stride_ak = {{rt_stride("A", 1)}}
    stride_bk = {{rt_stride("B", 0)}}
    stride_bn = {{rt_stride("B", 1)}}
    stride_ck = {{rt_stride("PARTIAL", 0)}}
    stride_cm = {{rt_stride("PARTIAL", 1)}}
    stride_cn = {{rt_stride("PARTIAL", 2)}}

    # (1) grid가 2차원이고, 두 번째 축이 K를 가리킨다
    pid   = tl.program_id(0)      # 어느 C 조각인가
    pid_k = tl.program_id(1)      # K의 몇 번째 몫인가

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # (2) 각 k 분할이 K의 연속 구간(blocked)을 맡는다. cyclic으로 가르면 k를
    #     device 축으로 나눌 때 한 device의 K 영역이 띄엄띄엄이 되어
    #     pim-place-data가 진단한다. host가 K % (SPLIT_K*BLOCK_K) == 0을 보장한다.
    K_PER = K // SPLIT_K
    offs_k = pid_k * K_PER + tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # (3) 자기 몫의 연속 구간을 BLOCK_K씩 전진한다
    for k in range(0, tl.cdiv(K_PER, BLOCK_K)):
        {% if EVEN_TILES %}
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        {% else %}
        k_now = offs_k + k * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (k_now[None, :] < K), other=0.0)
        b = tl.load(b_ptrs,
                    mask=(k_now[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        {% endif %}
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = PARTIAL + pid_k * stride_ck + (offs_m[:, None] * stride_cm
                                            + offs_n[None, :] * stride_cn)
    {% if EVEN_TILES %}
    tl.store(c_ptrs, acc)
    {% else %}
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
    {% endif %}
"""


# TritonTemplate은 이름을 전역 등록표에 넣고 중복을 막으므로 모듈 최상위에서
# 한 번만 만든다.
pim_split_k_template = PIMTritonTemplate(
    name="pim_split_k",
    grid=pim_split_k_grid,
    source=PIM_SPLIT_K_SOURCE,
)
