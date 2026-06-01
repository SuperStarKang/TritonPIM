set(LLVM_TARGET_DEFINITIONS PIMDialect.td)
mlir_tablegen(PIMDialect.h.inc -gen-dialect-decls -dialect=pim)
mlir_tablegen(PIMDialect.cpp.inc -gen-dialect-defs -dialect=pim)
mlir_tablegen(PIMTypes.h.inc -gen-typedef-decls -typedefs-dialect=pim)
mlir_tablegen(PIMTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=pim)
mlir_tablegen(PIMOps.h.inc -gen-op-decls)
mlir_tablegen(PIMOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(PIMTableGen)
