"""UPMEM resources of the PIMSession model path.

Compile time (begin_model, requirements, prepare_model) builds stage hosts, lays
out MRAM and links DPU programs without the device. open() allocates the DPU set
and bind_model() creates the native stage handles on it.
"""

from contextlib import ExitStack
import ctypes

from .residency import UnitResources
from .runtime import resident_lock
from .upmem_artifact import TransferStats


class UPMEMAdapter:
    def __init__(self, spec):
        from .driver import _build_upmem_runtime

        self.spec = spec
        self.path = _build_upmem_runtime()
        self.library = ctypes.CDLL(self.path)
        self.library.upmem_session_open.argtypes = [ctypes.c_uint32]
        self.library.upmem_session_open.restype = None
        self.library.upmem_session_close.argtypes = []
        self.library.upmem_session_close.restype = None
        self.library.upmem_session_num_units.argtypes = []
        self.library.upmem_session_num_units.restype = ctypes.c_uint32
        self.model = None
        self.totals = {name: 0 for name, _ in TransferStats._fields_}
        self.totals.update(resource_opens=0, resource_closes=0)
        self.totals["device_compiles"] = 0
        self.totals["weight_pushes"] = 0
        self.cleanup = None

    def open(self):
        if self.cleanup is not None or self.totals["resource_opens"]:
            raise RuntimeError("UPMEM adapter cannot be opened twice")
        with ExitStack() as cleanup:
            cleanup.enter_context(resident_lock)
            self.library.upmem_session_open(self.spec.common_profile.num_units)
            self.num_units = self.library.upmem_session_num_units()
            self.totals["resource_opens"] += 1
            cleanup.callback(self._release_resources)
            cleanup.callback(self._close_model)
            self.cleanup = cleanup.pop_all()

    def _release_resources(self):
        self.library.upmem_session_close()
        self.totals["resource_closes"] += 1

    def _close_model(self):
        if self.model is not None:
            for name, value in self.model.stats().items():
                self.totals[name] += value
            self.model.close()
            self.model = None

    def stats(self):
        totals = dict(self.totals)
        if self.model is not None:
            for name, value in self.model.stats().items():
                totals[name] += value
        return totals

    def begin_model(self):
        from .upmem_model import UPMEMModelRuntime

        if self.model is not None:
            raise RuntimeError("prepare one model per session")
        self.model = UPMEMModelRuntime(self)

    def resources(self):
        # compile time에도 부른다. unit 수는 spec이 정한다. session-first 경로는 open
        # 뒤 spec을 실제 unit 수로 바꾸므로 두 경로 모두 실제 수와 같다.
        profile = self.spec.common_profile
        return UnitResources(
            profile.num_units, profile.banks_per_unit * profile.bank_capacity_limit_bytes)

    def requirements(self, plans):
        return self.model.requirements(plans)

    def prepare_model(self, plans, memory):
        self.model.place(plans, memory)

    def bind_model(self, plans):
        self.model.bind(plans, self.num_units)

    def weight_table(self, plans):
        return self.model.weight_table(plans)

    def memory_table(self, plans):
        return self.model.memory_table(plans)

    def close(self):
        if self.cleanup is not None:
            self.cleanup.close()
            self.cleanup = None
