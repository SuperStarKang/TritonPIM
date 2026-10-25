"""Run the UPMEM workloads: INT8 GEMMs on the DPUs, the other kernels on Inductor's C++ backend.

The CPU partitions use the Inductor settings of the CPU baseline's run_cpp_cpu.py, so
only the GEMMs differ from the CPU run.
"""

from upmem_workloads import main


INDUCTOR_OVERRIDES = {
    "cpu_backend": "cpp",
    # max-autotune을 끄면 _int_mm 후보는 aten 하나다
    "max_autotune": False,
    "max_autotune_gemm": False,
    "cpp_wrapper": False,
    # 생성된 C++ 커널마다 profiler 구간(graph_N_cpp_fused_*)을 남긴다
    "cpp.enable_kernel_profile": True,
}
RUNTIME_OPTIONS = {"nr_tasklets": 16}


if __name__ == "__main__":
    main(__doc__, INDUCTOR_OVERRIDES, RUNTIME_OPTIONS)
