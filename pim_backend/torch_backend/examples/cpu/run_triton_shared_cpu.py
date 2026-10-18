"""Run the CPU workloads with every Inductor kernel, including torch._int_mm, on triton_shared.

Kernels are generated as Triton and compiled by triton_shared (TTIR -> linalg -> LLVM).
No PIM driver or DPU is used.
"""

import triton
from triton.backends.triton_shared.driver import CPUDriver

from cpu_workloads import main


INDUCTOR_OVERRIDES = {
    # pointwise/reduction 커널을 C++ 대신 Triton으로 생성한다
    "cpu_backend": "triton",
    # _int_mm 후보에서 aten을 빼고 Triton mm template만 남긴다
    "max_autotune_gemm": True,
    "max_autotune_gemm_backends": "TRITON",
    # epilogue fusion 벤치마크용 모듈 생성이 CPU device에서 깨진 Python을 만든다(torch 2.12)
    "benchmark_epilogue_fusion": False,
    # compile worker subprocess에는 CPUDriver가 active로 설정되지 않는다
    "compile_threads": 1,
    "cpp_wrapper": False,
}


if __name__ == "__main__":
    # torch의 set_driver_to_cpu()는 backends["cpu"]만 찾으므로 triton_shared를 직접 고른다
    triton.runtime.driver.set_active(CPUDriver())
    main(__doc__, INDUCTOR_OVERRIDES)
