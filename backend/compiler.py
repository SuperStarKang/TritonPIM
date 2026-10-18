from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
import triton
from pathlib import Path

def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))

def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")
    
    return sanitizer_type

def _ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        _dump_ir_if_needed([src_path])
        triton_shared_opt_path = _get_triton_shared_opt_path()

        subprocess_args = [triton_shared_opt_path, src_path, "--triton-to-linalg-experimental", "--mlir-print-debuginfo", "-o", dst_path]

        if _get_sanitizer_type() != "":
            print("Building with sanitizer support...")

            # has to run before the other passes as operates on the tt dialect
            subprocess_args.insert(2, "--add-llvm-debug-info")

        subprocess.check_call(subprocess_args)
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        # mlir-opt는 tptr dialect를 등록하지 않아 TritonToPtr가 남긴 tptr op를 읽지 못한다.
        # triton-shared-opt는 mlir-opt의 pass를 모두 등록하고 tptr도 읽을 수 있다.
        triton_shared_opt_path = _get_triton_shared_opt_path()
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call([triton_shared_opt_path, ttshared_path,
            "--convert-linalg-to-affine-loops",
            # Note: eliminate-empty-tensors fails when there are multiple func.return ops
            # in a single kernel which are the results of early returns.
            # See python/examples/test_early_return.py for examples.
            # We disable this pass for now since performance on CPU isn't the main
            # focus at the moment.
            # "--eliminate-empty-tensors",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=allow-return-allocs-from-loops=true",
            "--lower-affine",
            "--convert-linalg-to-loops",
            "--expand-strided-metadata",
            "--convert-scf-to-cf",
            "--memref-expand",
            # arith/math/complex/vector/index/finalize-memref/func/cf-to-llvm 패턴을
            # 한 번의 변환으로 실행한다. memref<Nx!ptr.ptr> 원소 타입을 memref 변환과
            # 같은 타입 변환기가 !llvm.ptr로 바꿔야 해서 pass를 나눌 수 없다.
            "--tptr-to-llvm",
            # Lowering memrefs creates more affine.apply ops.
            # Lowering these affine ops again creates further arith ops,
            # so we have to run these two passes again here.
            "--lower-affine",
            "--convert-arith-to-llvm",
            # Remove all unrealized casts created
            "--reconcile-unrealized-casts",
            "--mlir-print-debuginfo",
            "-o",
            llmlir_path])

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = _get_llvm_bin_path("mlir-translate")
        subprocess.check_call([mlir_translate_path, llmlir_path,
            "--mlir-to-llvmir",
            "-o",
            llir_path])
        _dump_ir_if_needed([ttshared_path, llmlir_path, llir_path])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_bin(llir: str, metadata):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)

        sanitizer_type = _get_sanitizer_type()

        if sanitizer_type != "":
            # using a sanitizer
            # invoke pass to append sanitizer attributes
            instrumented_src_path = os.path.join(tmpdir, "kernel-instrumented.ll")
        
            opt_path = _get_llvm_bin_path("opt")
            top_level_triton_path = os.path.dirname(triton.__file__)
            sanitizer_attributes_pass_path = str(next(Path(top_level_triton_path).rglob("libSanitizerAttributes.so"), None))

            if not sanitizer_attributes_pass_path:
                raise Exception(f"libSanitizerAttributes.so does not exist.")

            subprocess.check_call([opt_path, "-load-pass-plugin", sanitizer_attributes_pass_path, 
                "-passes=sanitizer-attributes", f"-sanitizer-type={sanitizer_type}", "-S", src_path, 
                "-o", instrumented_src_path])

            # compile to object file
            clang_path = _get_llvm_bin_path("clang++")

            subprocess_args = [clang_path, "-c", instrumented_src_path, "-o", dst_path]

            if sanitizer_type == "asan":
                subprocess_args.extend(["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"])
            elif sanitizer_type == "tsan":
                subprocess_args.extend(["-g", "-fsanitize=thread"])
                
            subprocess.check_call(subprocess_args)
        else:
            llc_path = _get_llvm_bin_path("llc")
            subprocess.check_call([llc_path, src_path, "-filetype=obj", "-relocation-model=pic", "-o", dst_path])
        
        return Path(dst_path).read_bytes()



@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = 'obj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(_ttir_to_ttsharedir(src))
        stages["llir"] = lambda src, metadata: _optimize_llir(_ttsharedir_to_llir(src))
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata)


    @functools.lru_cache()
    def hash(self):
        return self.target

    # triton/language/extra/libdevice.py 는 이름만 있는 빈 스텁이고, 백엔드가 여기서 실제
    # 구현을 끼워 넣는다(nvidia 는 CUDA libdevice 를 준다). 비워 두면 커널이 libdevice.exp 를
    # 부를 때 None 이 돌아오고 그 다음 연산에서 죽는다. torch.compile 이 만든 CPU Triton
    # 커널은 libdevice 를 무조건 쓰므로 이 자리가 채워져야 그 경로가 선다.
    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.backends.triton_shared import libdevice
        return {"triton.language.extra.libdevice": libdevice}
