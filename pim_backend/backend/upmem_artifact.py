"""UPMEM stage artifacts: MRAM bindings of a compiled GEMM and its linked DPU program."""

import ctypes
import hashlib
import json
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from triton.runtime.cache import get_cache_manager


MRAM_BASE = 0x08000000
WEIGHT = 1
INPUT = 2
OUTPUT = 3


@dataclass(frozen=True)
class Binding:
    symbol: str
    size: int
    role: int
    operand: int | None = None


class TransferStats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "weight_bytes", "activation_bytes", "output_bytes", "metadata_bytes",
        "launches", "program_loads", "activation_hits")]


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
            operand = None
        else:
            argument = arguments[host_args.index(sources[symbol])]
            operands = [index for index, tensor in enumerate(physical_inputs)
                        if tensor.data_ptr() == argument.data_ptr()]
            roles = {weight_mask[index] for index in operands}
            if len(roles) != 1:
                raise ValueError(f"cannot associate {symbol} with the FX operand provenance")
            role = WEIGHT if roles.pop() else INPUT
            operand = operands[0]
        size = int(count) * {"i8": 1, "i32": 4, "f32": 4}[dtype]
        if size % 8:
            raise ValueError(f"{symbol}: MRAM transfer size {size} is not aligned to 8 bytes")
        bindings.append(Binding(symbol, size, role, operand))
    return tuple(bindings)


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


def _build_program(bundle, addresses, *, stats=None):
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
        path = cache.put(data, "resident.elf", binary=True)
        if stats is not None:
            stats["device_compiles"] += 1
        return path
