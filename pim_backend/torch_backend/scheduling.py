# PIMCombinedScheduling — 한 CPU 그래프 안에서 두 codegen backend를 섞는다.
#
#     PIM split-K template  → TritonScheduling  (Triton 소스 → PIMBackend)
#     그 밖의 모든 노드      → CppScheduling     (C++/OpenMP, stock 그대로)
#
# CUDACombinedScheduling(torch/_inductor/codegen/cuda_combined_scheduling.py)이
# CUDA 위에서 Triton/CUTLASS를 섞는 것과 같은 구조다. 다른 점은 "기본"이 반대라는
# 것이다 — 여기서는 Cpp가 기본이고 PIM template만 Triton으로 빠진다.
#
# [device를 늘리지 않고 cpu backend를 갈아 끼우는 이유]
# tensor는 계속 CPU에 있다. PIM은 tensor device가 아니라 template의 compile /
# execution backend다. device를 새로 만들면 입력 텐서를 옮겨야 하고, aten 연산
# 전부에 대해 dispatch를 새로 채워야 한다.
#
# [비-template 노드가 전부 Cpp로 가야 하는 이유]
# 이 단계의 목표는 "_int_mm 하나만 PIM에서 돈다"를 확인하는 것이다. 나머지
# 노드까지 Triton으로 보내면 CPU Triton 경로의 문제와 PIM 경로의 문제가 섞인다.

import torch

from torch._inductor.codegen.common import (
    DeviceCodegen,
    device_codegens,
    init_backend_registration,
)
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V

from .lowering import is_pim_template_buffer


class PIMCombinedScheduling(BaseScheduling):
    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)
        self._cpp_scheduling = CppScheduling(scheduler)
        self._triton_scheduling = TritonScheduling(scheduler)

    # ── PIM template 판별 ──────────────────────────────────────────────────
    @staticmethod
    def is_pim_template(node) -> bool:
        template = node.get_template_node()
        return template is not None and is_pim_template_buffer(template)

    def choose_node_backend(self, node) -> BaseScheduling:
        return (self._triton_scheduling if self.is_pim_template(node)
                else self._cpp_scheduling)

    # ── 그래프 전체 성질은 Cpp 쪽을 따른다 ────────────────────────────────
    # 노드의 절대다수가 Cpp로 가므로 feature/group 규약이 Cpp와 어긋나면 안 된다.
    # template 노드는 이 둘을 쓰지 않는다 (group은 layout에서 나오고, feature는
    # template 소스에 영향을 주지 않는다).
    def get_backend_features(self, device: torch.device):
        return self._cpp_scheduling.get_backend_features(device)

    def group_fn(self, sizes):
        return self._cpp_scheduling.group_fn(sizes)

    # ── fusion: PIM template 주변은 전부 막는다 ───────────────────────────
    # epilogue/prologue가 붙으면 생성되는 Triton 소스가 split-K kernel과 달라져
    # PIM 파이프라인이 실패했을 때 원인이 갈라지지 않는다. lowering에서
    # allow_epilogue_fusion / allow_prologue_fusion을 False로 두었고, 여기서
    # 한 번 더 막는다 (수평 fusion까지 포함).
    def can_fuse_vertical(self, node1, node2) -> bool:
        if self.is_pim_template(node1) or self.is_pim_template(node2):
            return False
        return self._cpp_scheduling.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1, node2) -> bool:
        if self.is_pim_template(node1) or self.is_pim_template(node2):
            return False
        return self._cpp_scheduling.can_fuse_horizontal(node1, node2)

    def can_fuse_multi_outputs_template(self, node1, node2) -> bool:
        if self.is_pim_template(node1) or self.is_pim_template(node2):
            return False
        return self._cpp_scheduling.can_fuse_multi_outputs_template(node1, node2)

    def get_fusion_pair_priority(self, node1, node2) -> int:
        return self._cpp_scheduling.get_fusion_pair_priority(node1, node2)

    def fuse(self, node1, node2):
        return self._cpp_scheduling.fuse(node1, node2)

    # ── codegen ──────────────────────────────────────────────────────────
    def codegen_template(self, template_node, epilogue_nodes, prologue_nodes):
        if self.is_pim_template(template_node):
            assert not epilogue_nodes, "PIM template does not take an epilogue yet"
            assert not prologue_nodes, "PIM template does not take a prologue yet"
            return self._codegen_pim_template(template_node)
        return self._cpp_scheduling.codegen_template(
            template_node, epilogue_nodes, prologue_nodes)

    def _codegen_pim_template(self, template_node):
        """SIMDScheduling.codegen_template의 단일 template 분기와 같은 일을 한다.

        그대로 부르지 못하는 이유는 그 함수의 첫 두 줄이다.

            _, (_numel, rnumel) = template_node.group
            assert rnumel == 1

        node.group은 device backend의 group_fn이 만든다. Triton의 group_fn은
        축마다 크기를 곱해 정수를 내고(rnumel == 1), Cpp의 group_fn은 크기
        tuple을 그대로 남긴다(rnumel == ()). 이 backend는 노드의 절대다수가
        Cpp로 가므로 group_fn을 Cpp 것으로 두어야 하고, 그러면 저 assert가
        성립하지 않는다. group은 template codegen 자체에는 쓰이지 않으므로
        (TritonTemplateKernel은 tiling을 output layout에서 직접 만든다)
        assert만 지나치면 된다."""
        scheduling = self._triton_scheduling
        node = template_node.node
        kernel, render = node.make_kernel_render(node)
        kernel = scheduling._codegen_single_template(
            kernel, render, template_node, [], [], only_gen_src_code=False)

        scheduling.codegen_comment([template_node], kernel.kernel_name)
        kernel.call_kernel(kernel.kernel_name, node)

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        scheduling.free_buffers_in_scheduler()
        return None

    def codegen_node(self, node) -> None:
        return self._cpp_scheduling.codegen_node(node)

    def codegen_sync(self) -> None:
        return self._cpp_scheduling.codegen_sync()

    def codegen_comment(self, node_schedule, kernel_name=None) -> None:
        return self._cpp_scheduling.codegen_comment(node_schedule, kernel_name)

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False,
                                        hint_override=None):
        return self._cpp_scheduling.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel, hint_override=hint_override)

    def benchmark_fused_nodes(self, nodes):
        return self._cpp_scheduling.benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        return self._cpp_scheduling.benchmark_codegened_module(module)

    # ── flush 순서 ───────────────────────────────────────────────────────
    # CppScheduling은 노드를 kernel_group에 모아 두었다가 flush 때 커널 정의와
    # 호출을 한꺼번에 낸다. Scheduler는 template 노드를 만나면 codegen 직전에
    # flush를 부르므로(scheduler.py의 `node.is_template()` 분기), 여기서 Cpp를
    # 먼저 비워야 wrapper에 찍히는 호출 순서가 프로그램 순서와 같아진다.
    # TritonScheduling.flush()는 아무것도 모으지 않으므로(no-op) 뒤에 둔다.
    def flush(self) -> None:
        self._cpp_scheduling.flush()
        self._triton_scheduling.flush()

    def ready_to_flush(self) -> bool:
        return (self._cpp_scheduling.ready_to_flush()
                or self._triton_scheduling.ready_to_flush())

    def free_buffers_in_scheduler(self) -> None:
        self._cpp_scheduling.free_buffers_in_scheduler()


class PIMCombinedSchedulingInstalled:
    """cpu device의 scheduling을 PIMCombinedScheduling으로 바꾸는 context manager.

    register_backend_for_device는 프로세스 전역이라, 범위를 compile_fx 호출
    하나로 묶어 두고 나가면서 되돌린다. wrapper codegen은 건드리지 않는다 —
    생성되는 것은 여전히 보통의 CPU Python wrapper다."""

    def __init__(self):
        self._saved = None

    def __enter__(self):
        init_backend_registration()
        previous = device_codegens["cpu"]
        self._saved = previous
        device_codegens["cpu"] = DeviceCodegen(
            PIMCombinedScheduling,
            previous.wrapper_codegen,
            previous.cpp_wrapper_codegen,
            previous.fx_wrapper_codegen,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        device_codegens["cpu"] = self._saved
        self._saved = None
        return False
