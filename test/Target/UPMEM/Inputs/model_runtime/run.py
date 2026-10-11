from pathlib import Path
import subprocess
import sys

inputs = Path(__file__).resolve().parent
backend = Path(sys.argv[1]).resolve()
objects = []
for source in (backend / "upmem_runtime.c", backend / "upmem_model.c", inputs / "runtime.c"):
    output = sys.argv[3] + "." + source.stem + ".o"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                    "-I", str(inputs), "-I", str(backend), "-c", str(source), "-o", output],
                   check=True)
    objects.append(output)
subprocess.run(["g++", "-std=c++17", "-fopenmp", "-I", str(inputs), "-I", str(backend),
                sys.argv[2], *objects, "-o", sys.argv[3]], check=True)
subprocess.run([sys.argv[3]], check=True)
