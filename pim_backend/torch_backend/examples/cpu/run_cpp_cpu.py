"""Run the CPU workloads on Inductor's C++ backend, the CPU path of the PIM experiments.

Non-GEMM kernels are generated as C++ (at::vec SIMD + OpenMP). torch._int_mm stays an
extern ATen call, which uses oneDNN on CPUs with AVX512-VNNI.
"""

from cpu_workloads import main


INDUCTOR_OVERRIDES = {
    "cpu_backend": "cpp",
    # max-autotune을 끄면 _int_mm 후보는 aten 하나다
    "max_autotune": False,
    "max_autotune_gemm": False,
    "cpp_wrapper": False,
    # 생성된 C++ 커널마다 profiler 구간(graph_N_cpp_fused_*)을 남긴다
    "cpp.enable_kernel_profile": True,
}


if __name__ == "__main__":
    main(__doc__, INDUCTOR_OVERRIDES)
