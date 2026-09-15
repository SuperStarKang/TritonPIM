#ifndef TRITON_SHARED_TARGET_UPMEM_METADATA_LAYOUT_H_
#define TRITON_SHARED_TARGET_UPMEM_METADATA_LAYOUT_H_

#include "CEmitter.h"
#include "triton-shared/Dialect/PIMTarget/UPMEM/IR/UPMEMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace upmem_codegen {

// One wire ABI for both the host C++ and DPU C translations. In particular,
// index travels as int32_t, as it did in the individual scalar transfers.
// Explicit padding and packed/aligned declarations avoid depending on either
// compiler's native struct layout. Every field still has natural alignment.
struct MetadataLayout {
  struct Field {
    StringRef symbol;
    Type type;
    std::string name;
    std::string ctype;
    int64_t offset;
    int64_t bytes;
  };
  SmallVector<Field> fields;
  std::string symbol = "_pim_metadata";
  int64_t bytes = 0;

  LogicalResult init(ModuleOp module) {
    unsigned suffix = 0;
    while (SymbolTable::lookupSymbolIn(module, symbol))
      symbol = "_pim_metadata_" + std::to_string(++suffix);
    for (auto var : module.getOps<upmem::HostVarOp>()) {
      Type type = var.getType();
      std::string ct = CEmitter::scalarType(type, /*hostSide=*/false);
      if (ct.empty())
        return var.emitError("unsupported metadata field type");
      int64_t size = type.isIndex() ? 4 : (type.getIntOrFloatBitWidth() + 7) / 8;
      bytes = (bytes + size - 1) / size * size;
      fields.push_back({var.getSymName(), type,
                        "f" + std::to_string(fields.size()), ct, bytes, size});
      bytes += size;
    }
    bytes = (bytes + 7) / 8 * 8;
    return success();
  }

  const Field *find(StringRef name) const {
    for (const Field &field : fields)
      if (field.symbol == name)
        return &field;
    return nullptr;
  }

  void emit(raw_ostream &os) const {
    if (fields.empty())
      return;
    os << "#include <stddef.h>\n"
       << "struct __attribute__((packed, aligned(8))) _pim_metadata_t {\n";
    int64_t cursor = 0;
    unsigned padding = 0;
    auto padTo = [&](int64_t end) {
      if (end != cursor)
        os << "  uint8_t _pad" << padding++ << "[" << end - cursor << "];\n";
      cursor = end;
    };
    for (const Field &field : fields) {
      padTo(field.offset);
      os << "  " << field.ctype << " " << field.name << "; // "
         << field.symbol << "\n";
      cursor += field.bytes;
    }
    padTo(bytes);
    os << "};\n";
    // These assertions work in both C and C++ and are checked when users build
    // the generated sources, including with different host/DPU compilers.
    // The UPMEM stddef.h offsetof macro casts a pointer to size_t, which is not
    // an integer constant expression in C. Use the GCC/Clang builtin directly.
    os << "typedef char _pim_metadata_size_check["
       << "sizeof(struct _pim_metadata_t) == " << bytes << " ? 1 : -1];\n";
    for (const Field &field : fields)
      os << "typedef char _pim_metadata_" << field.name << "_check["
         << "__builtin_offsetof(struct _pim_metadata_t, " << field.name << ") == "
         << field.offset << " && sizeof(((struct _pim_metadata_t*)0)->"
         << field.name << ") == " << field.bytes << " ? 1 : -1];\n";
    os << "\n";
  }
};

} // namespace upmem_codegen
} // namespace mlir

#endif
