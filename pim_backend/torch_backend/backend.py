# torch.compile backend entry.
#
#     compiled = torch.compile(
#         fn,
#         backend=PIMInductorBackend(PIMTorchConfig(...)),
#         fullgraph=True,
#         dynamic=False,
#     )
#
# callable backend로 두는 이유: 이름 등록("pim_inductor")은 전역 등록표를 건드리고
# import 순서에 얽힌다. 경로가 안정된 다음에 붙여도 늦지 않다.
#
# [범위]
# lowering 교체와 scheduling 교체는 compile_fx 호출 하나로 묶고 나가면서 되돌린다.
# 전역 dict를 영구히 바꾸면 같은 프로세스의 다른 torch.compile까지 PIM으로 간다.
# 반대로 runtime patch(runtime_compat.install)는 되돌리지 않는다 — 커널의
# precompile은 compile_fx가 끝난 뒤(첫 실행 시점)에 일어날 수 있고, 그때 필요한
# 정보는 생성된 코드의 triton_meta에 이미 실려 있어서 PIM이 아닌 커널에는
# 아무 영향이 없다.

from contextlib import ExitStack

from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import compile_fx

from .config import PIMTorchConfig
from .lowering import PIMGemmLowering
from .runtime_compat import install as install_runtime_compat
from .runtime_compat import set_driver_to_backend
from .scheduling import PIMCombinedSchedulingInstalled


def _inductor_overrides(config: PIMTorchConfig) -> dict:
    overrides = {
        # 비-template 노드는 stock C++ 경로로 간다.
        "cpu_backend": "cpp",
        # 컴파일 시점에 임의 입력으로 커널을 돌려 보는 기능. 켜져 있으면
        # compile 중에 PIM launch가 일어난다.
        "triton.autotune_at_compile_time": False,
    }
    if config.force_single_compile_thread:
        # worker subprocess에는 runtime patch가 없다 (runtime_compat 참고).
        overrides["compile_threads"] = 1
    if config.disable_fx_graph_cache:
        # 캐시 key에 "lowering이 교체되었다"가 들어가지 않는다. 히트가 나면
        # PIM이 아닌 예전 CPU 코드가 그대로 돌아온다.
        overrides["fx_graph_cache"] = False
        overrides["fx_graph_remote_cache"] = False
    return overrides


class PIMInductorBackend:
    """Send FP32 mm and INT8 _int_mm to the PIM GEMM template."""

    def __init__(self, config: PIMTorchConfig = None):
        self.config = config if config is not None else PIMTorchConfig()

    def __call__(self, graph_module, example_inputs):
        install_runtime_compat()

        # compile_fx에 들어가기 전에 PIM driver를 명시 선택한다.
        #
        # triton.runtime.driver.active는 아직 아무도 고르지 않았으면
        # _create_driver()로 자동 탐지를 시도하는데, is_active()가 True인 backend가
        # 하나도 없으면(GPU 없음 + triton_shared CPUDriver False + PIM False)
        # "0 active drivers"로 터진다.
        #
        # 이 시점이어야 하는 이유: driver를 처음 요구하는 곳은 compile 단계가 아니라
        # lowering 중 template 렌더다. jit_lines() → inductor_meta_common() →
        # triton_hash_with_backend()가 Inductor 캐시 키를 만들려고
        # driver.active.get_current_target()을 본다. 게다가 그 함수는
        # functools.cache라 첫 호출의 backend가 프로세스 전체의 hash를 결정하므로,
        # PIM이 active인 상태에서 불려야 캐시 키가 PIM backend를 반영한다.
        #
        # 나가면서 되돌리지 않는다. _active를 None으로 되돌리면 다음에 driver를
        # 건드리는 코드가 다시 같은 자리에서 터진다. 다른 backend를 쓰는 코드는
        # 각자 set_driver_to_* 로 바꾼다(그것이 torch의 기존 방식이다).
        set_driver_to_backend("pim")

        with ExitStack() as stack:
            stack.enter_context(inductor_config.patch(_inductor_overrides(self.config)))
            stack.enter_context(PIMGemmLowering(self.config))
            stack.enter_context(PIMCombinedSchedulingInstalled())
            return compile_fx(graph_module, example_inputs)

    def __repr__(self):
        return f"PIMInductorBackend({self.config})"
