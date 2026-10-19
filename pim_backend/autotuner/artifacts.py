"""Project-local artifacts for profiling and autotuning commands."""

import os
import tempfile
from pathlib import Path

from triton import knobs


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "workspace" / "Experiment" / "Autotuner"
DEFAULT_CACHE_PATH = str(RESULTS_ROOT / "autotune_cache.json")


def project_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"profiling/autotuning artifacts must be inside {PROJECT_ROOT}: {path}")
    return path


def prepare_run_directory(path):
    """Set only this command's process and compiler-child artifact locations."""
    directory = project_path(path)
    temporary = directory / "temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    # Python과 외부 compiler가 같은 프로젝트 내부 임시 경로를 사용한다.
    tempfile.tempdir = str(temporary)
    os.environ["TMPDIR"] = str(temporary)
    knobs.cache.dir = str(directory / "triton_cache")
    knobs.cache.dump_dir = str(directory / "triton_dump")
    knobs.cache.override_dir = str(directory / "triton_override")
    return directory
