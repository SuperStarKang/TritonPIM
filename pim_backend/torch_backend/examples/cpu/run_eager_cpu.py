"""Run the CPU workloads in eager PyTorch without torch.compile.

Every op, including torch._int_mm, runs as an ATen kernel (oneDNN for _int_mm on CPUs with
AVX512-VNNI). The ATen intra-op thread count follows OMP_NUM_THREADS.
"""

from cpu_workloads import main


if __name__ == "__main__":
    main(__doc__, None)
