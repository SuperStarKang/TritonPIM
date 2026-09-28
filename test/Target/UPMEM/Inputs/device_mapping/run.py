from pathlib import Path
import subprocess
import sys

inputs = Path(__file__).resolve().parent
backend = Path(sys.argv[1]).resolve()
subprocess.run([
    "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
    "-I", str(inputs), "-I", str(backend),
    str(backend / "upmem_runtime.c"), str(inputs / "runtime.c"),
    "-o", sys.argv[2],
], check=True)
subprocess.run([sys.argv[2]], check=True)
