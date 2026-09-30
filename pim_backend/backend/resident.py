"""Host transfer capture and link-time MRAM placement for a resident block."""

import ctypes
import hashlib
import json
import re
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from triton.runtime.cache import get_cache_manager

from .runtime import current_invocation, resident_lock


MRAM_BASE = 0x08000000
WEIGHT = 1
INPUT = 2
OUTPUT = 3


@dataclass(frozen=True)
class Binding:
    symbol: str
    size: int
    role: int


class TransferStats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "weight_bytes", "activation_bytes", "output_bytes", "metadata_bytes",
        "launches", "program_loads", "activation_hits")]


class TransferTimings(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "weight_h2d_ns", "activation_h2d_ns", "metadata_h2d_ns", "program_load_ns")]

    def h2d_ms(self):
        return {role: getattr(self, role + "_h2d_ns") / 1e6
                for role in ("weight", "activation", "metadata")}


def _artifact_bindings(target_ir, arguments, physical_inputs, weight_mask, units):
    acquire = re.findall(r"upmem\.acquire\s+(\d+)\s*,", target_ir)
    if acquire != [str(units)]:
        raise ValueError(f"resident artifact must acquire exactly {units} DPUs: {acquire}")
    host = re.search(r"func\.func\s+@[\w.$]+\((.*?)\)\s*(?:attributes|\{)",
                     target_ir, re.DOTALL)
    if host is None:
        raise ValueError("UPMEM host function is missing from target_ir")
    host_args = re.findall(r"(%[\w.$]+)\s*:", host.group(1))
    stages = dict(re.findall(r"(%[\w.$]+)\s*=\s*upmem\.stage\s+(%[\w.$]+)\s+origin",
                             target_ir))
    transfers = re.findall(r"upmem\.to_mram\s+%[\w.$]+\s*,\s*(%[\w.$]+)\s*,\s*@([\w.$]+)",
                           target_ir)
    sources = {symbol: stages[stage] for stage, symbol in transfers}
    outputs = re.findall(r"upmem\.gather\s+%[\w.$]+\s+assignment\s+%[\w.$]+\s*,\s*@([\w.$]+)\s+capacity",
                          target_ir)
    declarations = re.findall(
        r"upmem\.mram\s+@([\w.$]+)\s*:\s*memref<(\d+)x(i8|i32|f32)\s*,", target_ir)
    if len(declarations) != 3 or len(transfers) != 2 or len(outputs) != 1:
        raise ValueError("resident Torch GEMM requires two staged inputs and one gathered output")
    if {name for name, _, _ in declarations} != set(sources) | set(outputs):
        raise ValueError("UPMEM MRAM declarations and transfer bindings disagree")
    writes = set(re.findall(r"upmem\.mram_write\s+%[\w.$]+\s*,\s*@([\w.$]+)", target_ir))
    if writes != set(outputs):
        raise ValueError("resident device code may write only its output MRAM binding")
    bindings = []
    for symbol, count, dtype in declarations:
        if symbol in outputs:
            role = OUTPUT
        else:
            argument = arguments[host_args.index(sources[symbol])]
            roles = {weight_mask[index] for index, tensor in enumerate(physical_inputs)
                     if tensor.data_ptr() == argument.data_ptr()}
            if len(roles) != 1:
                raise ValueError(f"cannot associate {symbol} with the FX operand provenance")
            role = WEIGHT if roles.pop() else INPUT
        size = int(count) * {"i8": 1, "i32": 4, "f32": 4}[dtype]
        if size % 8:
            raise ValueError(f"{symbol}: MRAM transfer size {size} is not aligned to 8 bytes")
        bindings.append(Binding(symbol, size, role))
    return tuple(bindings)


def _build_host(bundle, runtime_path):
    from .driver import _upmem_host_flags

    directory = Path(__file__).resolve().parent
    key = hashlib.sha256((bundle["host"] + "|resident-host-v1|" + runtime_path
                          + (directory / "upmem_runtime.h").read_text()
                          + (directory / "upmem_resident.h").read_text()).encode()).hexdigest()
    cache = get_cache_manager(key)
    path = cache.get_file("resident_host.so")
    if path is not None:
        return path
    with tempfile.TemporaryDirectory() as build_dir:
        source = Path(build_dir, "host.cpp")
        library = Path(build_dir, "resident_host.so")
        source.write_text(bundle["host"])
        subprocess.check_call([
            "g++", "-O2", "-std=c++17", "-fPIC", "-shared", "-fopenmp",
            "-DUPMEM_RESIDENT_HOST", f"-I{directory}", str(source),
            *_upmem_host_flags(), runtime_path,
            "-Wl,-rpath," + str(Path(runtime_path).parent), "-o", str(library),
        ])
        return cache.put(library.read_bytes(), "resident_host.so", binary=True)


def _check_device_elf(data, addresses):
    if data[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("resident UPMEM linking requires a little-endian ELF32 file")
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    program_offset, section_offset = header[5], header[6]
    program_size, program_count = header[9], header[10]
    section_size, section_count = header[11], header[12]
    for index in range(program_count):
        program = struct.unpack_from("<IIIIIIII", data, program_offset + index * program_size)
        kind, _, virtual, physical, file_bytes, memory_bytes, _, _ = program
        if kind == 1 and (file_bytes or memory_bytes):
            for start in (virtual, physical):
                if start < MRAM_BASE + 64 * 1024 * 1024 and start + memory_bytes > MRAM_BASE:
                    raise ValueError("device ELF has a loadable MRAM segment that can overwrite resident data")
    sections = [struct.unpack_from("<IIIIIIIIII", data, section_offset + i * section_size)
                for i in range(section_count)]
    for section in sections:
        if (section[2] & 2 and section[5]
                and section[3] < MRAM_BASE + 64 * 1024 * 1024
                and section[3] + section[5] > MRAM_BASE):
            raise ValueError("device ELF retains an MRAM allocation outside the resident address plan")
    symbols = {}
    for section in sections:
        if section[1] != 2:
            continue
        strings = sections[section[6]]
        names = data[strings[4]:strings[4] + strings[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, address, _, _, _, _ = struct.unpack_from("<IIIBBH", data, offset)
            symbols[names[name:names.index(b"\0", name)].decode()] = address
    for symbol, address in addresses.items():
        if symbols[symbol] != address:
            raise ValueError(f"linker did not place {symbol} at {address:#x}")


def _build_program(bundle, addresses):
    key = hashlib.sha256((bundle["device"] + "|resident-device-v1|"
                          + str(bundle["nr_tasklets"])
                          + json.dumps(addresses, sort_keys=True)).encode()).hexdigest()
    cache = get_cache_manager(key)
    path = cache.get_file("resident.elf")
    if path is not None:
        _check_device_elf(Path(path).read_bytes(), addresses)
        return path
    with tempfile.TemporaryDirectory() as build_dir:
        source = Path(build_dir, "device.c")
        binary = Path(build_dir, "resident.elf")
        source.write_text(bundle["device"])
        subprocess.check_call([
            "dpu-upmem-dpurte-clang", "-O2", f'-DNR_TASKLETS={bundle["nr_tasklets"]}',
            str(source), *[f"-Wl,--defsym={name}={address:#x}"
                          for name, address in addresses.items()], "-o", str(binary),
        ])
        data = binary.read_bytes()
        _check_device_elf(data, addresses)
        return cache.put(data, "resident.elf", binary=True)


class ResidentRuntime:
    @staticmethod
    def validate_config(config):
        target = config.torch_config.spec
        hierarchy = config.torch_config.config
        if target.common_profile.target != "upmem":
            raise ValueError("this resident runtime requires UPMEM")
        lanes = target.runtime_config.nr_tasklets
        if (lanes < 1 or lanes > target.common_profile.lanes_per_core
                or hierarchy.block_m * hierarchy.group_m % lanes):
            raise ValueError("nr_tasklets must divide the physical Block Group M extent")

    def __init__(self, units, capacity):
        from .driver import _build_upmem_runtime

        if units < 1 or capacity < 1 or capacity > 64 * 1024 * 1024:
            raise ValueError("invalid resident DPU count or MRAM capacity")
        self.units = units
        self.capacity = capacity
        self.path = _build_upmem_runtime()
        self.library = ctypes.CDLL(self.path)
        pointer = ctypes.c_void_p
        uint = ctypes.c_uint32
        signatures = {
            "create": ([uint], pointer),
            "add_kernel": ([pointer, ctypes.POINTER(ctypes.c_char_p),
                            ctypes.POINTER(uint), ctypes.POINTER(uint)], pointer),
            "begin": ([pointer, ctypes.c_int], None),
            "end": ([pointer], None),
            "cancel": ([], None),
            "windows": ([pointer, uint], uint),
            "rounds": ([pointer], uint),
            "window": ([pointer, uint, uint], uint),
            "address": ([pointer, uint, uint, uint], None),
            "program": ([pointer, uint, ctypes.c_char_p], None),
            "upload": ([pointer], None),
            "reset_inputs": ([pointer], None),
            "get_stats": ([pointer, ctypes.POINTER(TransferStats)], None),
            "get_timings": ([pointer, ctypes.POINTER(TransferTimings)], None),
            "close": ([pointer], None),
        }
        for name, (arguments, returns) in signatures.items():
            function = getattr(self.library, "upmem_resident_" + name)
            function.argtypes = arguments
            function.restype = returns
        self.handle = self.library.upmem_resident_create(units)
        self.kernels = []
        self.layout = []
        self.ready = False
        self.capture_ms = 0.0
        self.upload_ms = 0.0
        self.upload_breakdown_ms = None

    def finalize(self):
        if self.ready or self.handle is None or not self.kernels:
            raise RuntimeError("resident runtime cannot be finalized in its current state")
        offset = 0
        for kernel in self.kernels:
            kernel.addresses = []
            for index, binding in enumerate(kernel.bindings):
                addresses = []
                if binding.role == WEIGHT:
                    for window in range(self.library.upmem_resident_windows(kernel.handle, index)):
                        addresses.append(MRAM_BASE + offset)
                        self.layout.append({"kernel": kernel.name, "symbol": binding.symbol,
                                            "window": window, "offset": offset,
                                            "bytes": binding.size, "role": "weight"})
                        offset += binding.size
                kernel.addresses.append(addresses)
        input_base = offset
        input_bytes = max(sum(b.size for b in k.bindings if b.role == INPUT) for k in self.kernels)
        output_base = input_base + input_bytes
        output_bytes = max(b.size for k in self.kernels for b in k.bindings if b.role == OUTPUT)
        if output_base + output_bytes > self.capacity:
            raise ValueError(f"resident block needs {output_base + output_bytes} bytes per DPU; "
                             f"MRAM budget is {self.capacity}")
        self.layout.extend((
            {"role": "input_scratch", "offset": input_base, "bytes": input_bytes},
            {"role": "output_scratch", "offset": output_base, "bytes": output_bytes},
        ))
        for kernel in self.kernels:
            cursor = input_base
            for index, binding in enumerate(kernel.bindings):
                if binding.role == INPUT:
                    kernel.addresses[index] = [MRAM_BASE + cursor]
                    cursor += binding.size
                elif binding.role == OUTPUT:
                    kernel.addresses[index] = [MRAM_BASE + output_base]
                for window, address in enumerate(kernel.addresses[index]):
                    self.library.upmem_resident_address(kernel.handle, index, window, address)
            for round_index in range(self.library.upmem_resident_rounds(kernel.handle)):
                addresses = {}
                for index, binding in enumerate(kernel.bindings):
                    window = (self.library.upmem_resident_window(kernel.handle, round_index, index)
                              if binding.role == WEIGHT else 0)
                    addresses[binding.symbol] = kernel.addresses[index][window]
                path = self.program_for(kernel, addresses)
                self.library.upmem_resident_program(kernel.handle, round_index, path.encode())
        start = time.perf_counter_ns()
        self.library.upmem_resident_upload(self.handle)
        self.upload_ms = (time.perf_counter_ns() - start) / 1e6
        transfer_timings = self.transfer_timings()
        h2d_ms = sum(transfer_timings.h2d_ms().values())
        program_load_ms = transfer_timings.program_load_ns / 1e6
        resident_host_ms = self.upload_ms - h2d_ms - program_load_ms
        if resident_host_ms < 0:
            raise RuntimeError("resident upload timing intervals overlap or exceed upload wall time")
        self.upload_breakdown_ms = {
            "h2d": h2d_ms, "program_load": program_load_ms, "resident_host": resident_host_ms,
        }
        self.ready = True

    def library_for(self, name, bundle):
        return ctypes.CDLL(_build_host(bundle, self.path))

    def program_for(self, kernel, addresses):
        return _build_program(kernel.bundle, addresses)

    def invocation(self, kernel, physical_inputs, *, capture):
        return ResidentInvocation(kernel, physical_inputs, capture)

    def reset_inputs(self):
        self.library.upmem_resident_reset_inputs(self.handle)

    def stats(self):
        if self.handle is None:
            raise RuntimeError("ResidentBlock is closed")
        stats = TransferStats()
        self.library.upmem_resident_get_stats(self.handle, ctypes.byref(stats))
        return {name: getattr(stats, name) for name, _ in stats._fields_}

    def transfer_timings(self):
        if self.handle is None:
            raise RuntimeError("ResidentBlock is closed")
        timings = TransferTimings()
        self.library.upmem_resident_get_timings(self.handle, ctypes.byref(timings))
        return timings

    def close(self):
        if self.handle is not None:
            self.library.upmem_resident_close(self.handle)
            self.handle = None
            self.ready = False


class ResidentInvocation:
    def __init__(self, owner, physical_inputs, capture):
        self.owner = owner
        self.physical_inputs = physical_inputs
        self.capture = capture
        self.calls = 0

    def __enter__(self):
        if current_invocation.get() is not None:
            raise RuntimeError("nested resident invocation")
        self.token = current_invocation.set(self)
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.owner.runtime.library.upmem_resident_cancel()
        current_invocation.reset(self.token)
        if exception_type is None and self.calls != 1:
            raise RuntimeError(f"expected one PIM kernel for {self.owner.name}, got {self.calls}")
        return False

    def launch(self, launcher, bundle, entry, grid, arguments):
        from . import driver

        if launcher.pim_target != "upmem":
            raise ValueError("UPMEM invocation received a different target")
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("a resident GEMM must launch exactly one generated host function")
        runtime = self.owner.runtime
        host_arguments = [arguments[index] for index in launcher.arg_indices]
        if self.capture:
            if self.owner.artifact is not None:
                raise RuntimeError("resident GEMM has already been captured")
            bindings = _artifact_bindings(bundle["target_ir"], host_arguments,
                                          self.physical_inputs, self.owner.weight_mask, runtime.units)
            self.owner.artifact = ResidentArtifact(self.owner.name, bundle, entry, bindings, runtime)
            runtime.kernels.append(self.owner.artifact)
        artifact = self.owner.artifact
        if artifact.bundle != bundle or artifact.entry != entry:
            raise RuntimeError("compiled resident artifact changed; prepare a new block")
        function = getattr(artifact.library, entry)
        function.argtypes = [ctypes.c_void_p if pointer else ctypes.c_int32
                             for pointer in launcher.arg_is_pointer] + [ctypes.c_int32] * 3
        function.restype = None
        call = [ctypes.c_void_p(value.data_ptr()) if pointer else ctypes.c_int32(int(value))
                for value, pointer in zip(host_arguments, launcher.arg_is_pointer)]
        start = time.perf_counter_ns()
        runtime.library.upmem_resident_begin(artifact.handle, int(self.capture))
        function(*call, *(ctypes.c_int32(size) for size in grid))
        runtime.library.upmem_resident_end(artifact.handle)
        if self.capture:
            runtime.capture_ms += (time.perf_counter_ns() - start) / 1e6
        host_stages = driver._STAGE_NAMES["upmem"]
        host_timing = list((ctypes.c_double * len(host_stages)).in_dll(
            artifact.library, "pim_timing_ms"))
        transfer_timings = runtime.transfer_timings()
        h2d_by_role = transfer_timings.h2d_ms()
        h2d_ms = sum(h2d_by_role.values())
        program_load_ms = transfer_timings.program_load_ns / 1e6
        h2d_index = host_stages.index("h2d")
        # The generated v3 interval encloses SDK transfers and resident host work.
        resident_host_ms = host_timing[h2d_index] - h2d_ms - program_load_ms
        if resident_host_ms < 0:
            raise RuntimeError("resident transfer timing intervals exceed the generated H2D interval")
        host_timing[h2d_index] = h2d_ms
        driver.TIMING_STAGES = driver.UPMEM_RESIDENT_TIMING_STAGES
        driver.TIMING_VERSION = driver.UPMEM_RESIDENT_TIMING_VERSION
        driver.timing[:] = host_timing + [program_load_ms, resident_host_ms]
        driver.resident_h2d_timing = h2d_by_role
        driver.invokes = 0


class ResidentArtifact:
    def __init__(self, name, bundle, entry, bindings, runtime):
        from .driver import UPMEM_TIMING_VERSION

        self.name = name
        self.bundle = bundle
        self.entry = entry
        self.bindings = bindings
        self.library = runtime.library_for(name, bundle)
        version = ctypes.c_uint32.in_dll(self.library, "pim_timing_version").value
        if version != UPMEM_TIMING_VERSION:
            raise RuntimeError(f"UPMEM timing ABI mismatch: {version} != {UPMEM_TIMING_VERSION}")
        self.handle = runtime.library.upmem_resident_add_kernel(
            runtime.handle,
            (ctypes.c_char_p * 3)(*(binding.symbol.encode() for binding in bindings)),
            (ctypes.c_uint32 * 3)(*(binding.size for binding in bindings)),
            (ctypes.c_uint32 * 3)(*(binding.role for binding in bindings)),
        )
