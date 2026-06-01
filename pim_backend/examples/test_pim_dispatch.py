# device="pim" 라우팅 검증용 smoke test.
#
# 핵심: Triton에서 backend는 kernel 호출의 device= kwarg가 아니라 활성 driver가 결정한다.
# (jit.py가 device=/device_type=/stream= kwarg를 deprecated로 assert 처리한다.)
# 따라서 PIM을 쓰려면 triton.runtime.driver.set_active(PIMDriver())로 활성화한다.
# vendor는 PIMOptions의 필드이므로 kernel 호출에 vendor="upmem" kwarg로 넘기면
# parse_options를 거쳐 PIMOptions.vendor로 들어간다.
#
# 이 시점의 실제 lowering은 CPU 경로를 재사용하므로 결과는 CPU 실행과 동일하다.
# 여기서 검증하는 것은 "PIM으로 라우팅되고, vendor가 강제되며, 실행이 된다"는 것뿐이다.

import torch
import triton
import triton.language as tl

from triton.backends.pim.driver import PIMDriver
from triton.backends.pim.compiler import PIMBackend, PIMOptions
from triton.compiler.compiler import make_backend

# PIM driver를 활성화한다. 이것이 backend를 PIMBackend로 라우팅하는 실제 트리거다.
triton.runtime.driver.set_active(PIMDriver())


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def test_routes_to_pim_backend():
    # 활성 driver의 target이 "pim"이고, 그 target으로 선택되는 backend가 PIMBackend여야 한다.
    target = triton.runtime.driver.active.get_current_target()
    assert target.backend == "pim"
    backend = make_backend(target)
    assert isinstance(backend, PIMBackend)


def test_vendor_is_required():
    # vendor 누락은 fallback 없이 즉시 에러여야 한다.
    try:
        PIMOptions()
    except ValueError:
        return
    raise AssertionError("PIMOptions가 vendor 누락 시 ValueError를 내야 한다")


def test_add_runs_under_pim():
    n = 128
    BLOCK = 64
    x = torch.rand(n, dtype=torch.float32)
    y = torch.rand(n, dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, BLOCK), )
    # vendor는 PIMOptions 필드라 kwarg로 전달된다. device= 는 deprecated이므로 쓰지 않는다.
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK, vendor="upmem")

    torch.testing.assert_close(out, x + y)


if __name__ == "__main__":
    test_routes_to_pim_backend()
    test_vendor_is_required()
    test_add_runs_under_pim()
    print("OK: device=pim routing + vendor 강제 + 실행 확인")
