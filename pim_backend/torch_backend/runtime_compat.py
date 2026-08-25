# Torch runtime의 최소 수정.
#
# 고친 것은 두 군데다.
#
#   1) triton_helpers.set_driver_to_backend(name)
#      backend를 이름으로 명시 선택한다. 기존 set_driver_to_gpu()는 backends를
#      훑어 is_active()가 True인 것을 스스로 고른다. 그 방식이면 PIM은 "자동
#      탐지되는 backend"가 되어야 하고, PIM을 안 쓰는 torch.compile까지 끌어간다.
#      PIMDriver.is_active()는 이제 항상 False다 — 자동으로 선택되지 않는다.
#
#   2) CachingAutotuner._precompile_config
#      stock 코드는 tensor의 device type만 보고 driver를 고른다
#      (cpu면 CPU driver, 아니면 GPU driver). PIM에서는 tensor device가 CPU인데
#      compile target은 PIM이라 그 기준이 성립하지 않는다. 그래서 판단 기준을
#      "이 template이 무엇으로 컴파일되기를 요구하는가"(triton_meta의
#      compile_backend)로 바꾼다.
#
# 두 수정 모두 monkeypatch다. site-packages의 파일을 고치지 않는 이유:
#   - torch 업데이트마다 patch 충돌을 관리하지 않아도 된다.
#   - PIM backend를 import하지 않은 프로세스는 stock torch 그대로다.
#   - 되돌리기가 import 유무로 끝난다.
#
# 대신 조건이 하나 붙는다. async compile worker(subprocess)에는 이 patch가 없다.
# 그래서 PIMInductorBackend가 compile_threads=1로 묶는다 (config.py 참고).

import triton

from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.triton_compat import ASTSource

from .config import validate_pim_options

_installed = False
_orig_precompile_config = None
_orig_make_launchers = None


def set_driver_to_backend(name: str):
    """이름으로 Triton backend driver를 명시 선택한다.

    자동 탐지(is_active 훑기)를 쓰지 않는 이유는 모듈 머리말 참고."""
    backend = triton.backends.backends.get(name)
    if backend is None:
        raise RuntimeError(
            f"Triton backend {name!r} is not registered. "
            f"Registered: {sorted(triton.backends.backends)}")

    driver = triton.runtime.driver
    # property가 아니라 슬롯(_active)을 본다. driver.active는 아직 아무도 고르지
    # 않았으면 _create_driver()로 자동 탐지를 시도하는데, is_active()가 True인
    # backend가 하나도 없는 환경(GPU 없음 + CPU/PIM 모두 명시 선택 전용)에서는
    # "0 active drivers"로 터진다. 지금 고르려는 참이므로 그 경로에 들어가면 안 된다.
    active = getattr(driver, "_active", None)
    # 이미 골라 둔 것이 있으면 그대로 쓴다 (LazyProxy면 _obj로 감싸여 있다).
    if active is not None and (
            isinstance(active, backend.driver)
            or (hasattr(active, "_obj") and isinstance(active._obj, backend.driver))):
        return active

    new_active = backend.driver()
    driver.set_active(new_active)
    return new_active


def _pim_precompile_config(self, cfg):
    """compile_backend == "pim"인 template만 PIM으로 컴파일한다."""
    if self.triton_meta.get("compile_backend") != "pim":
        return _orig_precompile_config(self, cfg)

    compile_meta = self._create_compile_meta(cfg)

    driver = set_driver_to_backend("pim")
    # tensor device는 CPU지만 compile target은 PIM이다. arch/warp_size를
    # device_props(cpu)에서 끌어오면 안 된다.
    target = driver.get_current_target()

    ast_source = ASTSource(
        self.fn,
        compile_meta["signature"],
        compile_meta["constants"],
        compile_meta["configs"][0],
    )

    options = self._create_compile_options(cfg, compile_meta)
    # PIM 고유 옵션. lowering이 template metadata에 실어 보낸 값이다.
    options.update(validate_pim_options(compile_meta["backend_options"]))

    binary = triton.compile(ast_source, target=target, options=options)
    return triton_heuristics.TritonCompileResult(
        binary, cfg, compile_meta, self.inductor_meta)


def _pim_make_launchers(self):
    """launcher를 만들 때도 PIM driver가 active여야 한다.

    CompiledKernel._init_handles가 driver.active.launcher_cls로 launcher를
    만들기 때문이다. 보통은 바로 앞의 _precompile_config가 이미 바꿔 두지만,
    compile 결과를 캐시에서 되살린 경우에는 그 단계가 없다."""
    if self.triton_meta.get("compile_backend") == "pim":
        set_driver_to_backend("pim")
    return _orig_make_launchers(self)


def install():
    """monkeypatch를 건다. 여러 번 불러도 한 번만 걸린다."""
    global _installed, _orig_precompile_config, _orig_make_launchers
    if _installed:
        return

    from torch._inductor.runtime import triton_helpers

    # 생성된 커널 모듈이나 다른 torch 코드가 이름으로 찾을 수 있게 심어 둔다.
    triton_helpers.set_driver_to_backend = set_driver_to_backend

    _orig_precompile_config = triton_heuristics.CachingAutotuner._precompile_config
    _orig_make_launchers = triton_heuristics.CachingAutotuner._make_launchers
    triton_heuristics.CachingAutotuner._precompile_config = _pim_precompile_config
    triton_heuristics.CachingAutotuner._make_launchers = _pim_make_launchers
    _installed = True
