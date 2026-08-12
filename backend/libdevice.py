# CPU 백엔드용 libdevice.
#
# triton/language/extra/libdevice.py 는 이름만 있는 빈 스텁이고(본문이 `...`), 백엔드가
# get_module_map() 으로 실제 구현을 끼워 넣게 되어 있다. nvidia 는 CUDA libdevice 를 준다
# (third_party/nvidia/backend/compiler.py:219-221). CPU 는 그 자리가 비어 있어서 커널이
# libdevice.exp 를 부르면 None 이 돌아오고, 그 다음 연산에서
#   AttributeError: 'NoneType' object has no attribute 'type'
# 로 죽는다. torch.compile 이 만든 CPU Triton 커널은 libdevice 를 무조건 쓰므로
# (torch/_inductor/codegen/triton.py, 끄는 config 없음) 이 파일이 없으면 그 경로가 안 선다.
#
# 여기 있는 것은 @triton.jit 함수라 호출부에 인라인된다. tl.math 로 갈 수 있는 것은 그대로
# 넘기고, 없는 것만 식으로 푼다. 전부를 덮지는 않는다 — 없는 이름을 부르면 AttributeError 가
# 나서 무엇이 빠졌는지 바로 보인다(None 이 조용히 흘러가는 것보다 낫다).

import triton
import triton.language as tl


# ── tl.math 로 바로 가는 것 ──────────────────────────────────────────────────
@triton.jit
def exp(x):
    return tl.math.exp(x)


@triton.jit
def exp2(x):
    return tl.math.exp2(x)


@triton.jit
def log(x):
    return tl.math.log(x)


@triton.jit
def log2(x):
    return tl.math.log2(x)


@triton.jit
def sqrt(x):
    return tl.math.sqrt(x)


@triton.jit
def rsqrt(x):
    return tl.math.rsqrt(x)


@triton.jit
def sin(x):
    return tl.math.sin(x)


@triton.jit
def cos(x):
    return tl.math.cos(x)


@triton.jit
def erf(x):
    return tl.math.erf(x)


@triton.jit
def floor(x):
    return tl.math.floor(x)


@triton.jit
def ceil(x):
    return tl.math.ceil(x)


@triton.jit
def fabs(x):
    return tl.math.abs(x)


@triton.jit
def abs(x):
    return tl.math.abs(x)


@triton.jit
def fma(x, y, z):
    return tl.math.fma(x, y, z)


# ── 식으로 푸는 것 ───────────────────────────────────────────────────────────
@triton.jit
def trunc(x):
    # 0 쪽으로 버린다. floor 는 -inf 쪽이라 음수에서 다르다.
    return tl.where(x < 0.0, tl.math.ceil(x), tl.math.floor(x))


@triton.jit
def nearbyint(x):
    # round-half-to-even. inductor 의 aten.round 가 이것을 부른다.
    # 0.5 딱 걸리는 경우만 짝수 쪽으로 보내고 나머지는 가까운 쪽이다.
    f = tl.math.floor(x)
    d = x - f
    even = (f * 0.5 - tl.math.floor(f * 0.5)) == 0.0
    half = tl.where(even, f, f + 1.0)
    return tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, half))


@triton.jit
def rint(x):
    return nearbyint(x)


@triton.jit
def round(x):
    # C 의 round 는 half-away-from-zero 다(nearbyint 와 다르다).
    return tl.where(x < 0.0, -tl.math.floor(-x + 0.5), tl.math.floor(x + 0.5))


@triton.jit
def tan(x):
    return tl.math.sin(x) / tl.math.cos(x)


@triton.jit
def tanh(x):
    # 큰 |x| 에서 exp 가 넘치지 않도록 부호를 빼고 계산한다.
    a = tl.math.abs(x)
    e = tl.math.exp(-2.0 * a)
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x < 0.0, -t, t)


@triton.jit
def sigmoid(x):
    return 1.0 / (1.0 + tl.math.exp(-x))


@triton.jit
def log1p(x):
    return tl.math.log(1.0 + x)


@triton.jit
def expm1(x):
    return tl.math.exp(x) - 1.0


@triton.jit
def pow(x, y):
    # x > 0 전제. 정수 지수의 음수 밑은 이 식으로 안 된다.
    return tl.math.exp2(y * tl.math.log2(x))


@triton.jit
def fmod(x, y):
    # 나머지의 부호가 x 를 따른다(트런케이션). floor 기반 mod 와 다르다.
    return x - trunc(x / y) * y


@triton.jit
def isnan(x):
    return x != x


@triton.jit
def isinf(x):
    return (x == float("inf")) | (x == float("-inf"))


@triton.jit
def signbit(x):
    return (x < 0.0) | ((x == 0.0) & (1.0 / x < 0.0))


@triton.jit
def copysign(x, y):
    a = tl.math.abs(x)
    return tl.where(signbit(y), -a, a)


@triton.jit
def maxnum(x, y):
    return tl.where(x > y, x, y)


@triton.jit
def minnum(x, y):
    return tl.where(x < y, x, y)
