"""AiMX resources for the PIMSession model path.

The adapter reports real FPGA resources and per-stage needs to the common
memory plan; its slots decide what stays on the FPGAs.
"""

import ctypes

from .residency import UnitResources


class AiMXAdapter:
    def __init__(self, spec):
        from .driver import _aimx_resident_runtime_path

        self.spec = spec
        self.path = _aimx_resident_runtime_path()
        self.library = ctypes.CDLL(self.path)
        signatures = {
            "aimx_model_open": ([], ctypes.c_int),
            "aimx_model_region_create": ([ctypes.c_int, ctypes.c_int64, ctypes.c_int64], ctypes.c_void_p),
            "aimx_model_region_destroy": ([ctypes.c_void_p], None),
            "aimx_model_push": ([ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64)], None),
            "aimx_model_create": ([ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
                                   ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_void_p)],
                                  ctypes.c_void_p),
            "aimx_host_image_create": ([ctypes.c_int64], ctypes.c_void_p),
            "aimx_host_image_destroy": ([ctypes.c_void_p, ctypes.c_int64], None),
            "aimx_model_destroy": ([ctypes.c_void_p], None),
            "aimx_model_live_bytes": ([ctypes.c_void_p], ctypes.c_int64),
            "aimx_host_cache_clear": ([], None),
            "aimx_host_cache_clear_thread": ([], None),
        }
        for name, (arguments, returns) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = returns
        self.model = None
        self.visible_fpgas = 0
        self.num_units = 0

    def open(self):
        if self.visible_fpgas:
            raise RuntimeError("AiMX adapter cannot be opened twice")
        self.visible_fpgas = self.library.aimx_model_open()
        base = self.spec.runtime_config.base_fpga
        requested = self.spec.common_profile.num_units
        if base + requested > self.visible_fpgas:
            raise ValueError(
                f"FPGAs {base}..{base + requested - 1} requested, "
                f"{self.visible_fpgas} visible")
        self.num_units = requested

    def resources(self):
        # compile time에도 부른다. unit 수는 spec이 정하고 open()이 같은 수를 확보한다.
        profile = self.spec.common_profile
        return UnitResources(
            profile.num_units, profile.banks_per_unit * profile.bank_capacity_limit_bytes)

    def begin_model(self):
        from .aimx_model import AiMXModelRuntime

        if self.model is not None:
            raise RuntimeError("prepare one model per session")
        self.model = AiMXModelRuntime(self)

    def end_model(self):
        # FPGA는 열어 둔 채 stage context만 해제한다.
        self.model.close()
        self.model = None

    def requirements(self, plans):
        return self.model.requirements(plans)

    def prepare_model(self, plans, memory):
        self.model.place(plans, memory)

    def bind_model(self, plans):
        self.model.bind(plans)

    def weight_table(self, plans):
        return self.model.weight_table(plans)

    def memory_table(self, plans):
        return self.model.memory_table(plans)

    def stats(self):
        totals = {"visible_fpgas": self.visible_fpgas, "units": self.num_units}
        if self.model is not None:
            totals.update(self.model.stats())
        return totals

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
