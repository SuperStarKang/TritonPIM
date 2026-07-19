set(LLVM_TARGET_DEFINITIONS UPMEMDialect.td)
mlir_tablegen(UPMEMDialect.h.inc -gen-dialect-decls -dialect=upmem)
mlir_tablegen(UPMEMDialect.cpp.inc -gen-dialect-defs -dialect=upmem)
mlir_tablegen(UPMEMTypes.h.inc -gen-typedef-decls -typedefs-dialect=upmem)
mlir_tablegen(UPMEMTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=upmem)
mlir_tablegen(UPMEMOps.h.inc -gen-op-decls)
mlir_tablegen(UPMEMOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(UPMEMTableGen)
