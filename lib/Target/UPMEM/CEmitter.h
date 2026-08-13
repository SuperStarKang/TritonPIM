#ifndef TRITON_SHARED_TARGET_UPMEM_CEMITTER_H_
#define TRITON_SHARED_TARGET_UPMEM_CEMITTER_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace upmem_codegen {

// host C++와 device C가 함께 쓰는 부분. 두 쪽 다 MLIR 값 하나가 C 변수 하나가 되고,
// op 하나가 대입문 하나가 된다. 그래야 생성된 코드의 모든 줄을 IR의 op으로 되짚을 수 있다.
class CEmitter {
public:
  CEmitter(raw_ostream &os) : os(os) {}

  // 값의 C 이름. 아직 없으면 새로 짓는다.
  std::string name(Value v) {
    auto it = names.find(v);
    if (it != names.end())
      return it->second;
    std::string n = "v" + std::to_string(counter++);
    names[v] = n;
    return n;
  }
  void bind(Value v, StringRef n) { names[v] = n.str(); }

  void indent(unsigned n) { os.indent(n * 2); }

  // 스칼라 타입 → C 타입. index는 정수 인덱스 계산에 쓰이므로 부호 있는 정수로 둔다
  // (subi/ceildivsi가 음수를 거친다).
  static std::string scalarType(Type t, bool hostSide) {
    if (t.isF32())
      return "float";
    if (t.isF64())
      return "double";
    if (isa<IndexType>(t))
      return hostSide ? "int64_t" : "int32_t";
    if (auto i = dyn_cast<IntegerType>(t)) {
      switch (i.getWidth()) {
      case 1:
        return "uint8_t";
      case 8:
        return "int8_t";
      case 16:
        return "int16_t";
      case 32:
        return "int32_t";
      case 64:
        return "int64_t";
      }
    }
    return "";
  }

  // arith 스칼라 연산 하나를 대입문 하나로 옮긴다. 다루지 못하면 실패.
  LogicalResult emitArith(Operation *op, unsigned depth, bool hostSide) {
    auto binary = [&](StringRef fmt) {
      indent(depth);
      os << scalarType(op->getResult(0).getType(), hostSide) << " "
         << name(op->getResult(0)) << " = ";
      std::string a = name(op->getOperand(0)), b = name(op->getOperand(1));
      for (size_t i = 0; i < fmt.size(); i++) {
        if (fmt[i] == 'A')
          os << a;
        else if (fmt[i] == 'B')
          os << b;
        else
          os << fmt[i];
      }
      os << ";\n";
    };
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      indent(depth);
      os << scalarType(c.getType(), hostSide) << " " << name(c.getResult())
         << " = ";
      if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
        os << ia.getValue().getSExtValue();
      else if (auto fa = dyn_cast<FloatAttr>(c.getValue()))
        os << fa.getValueAsDouble();
      else
        return op->emitError("unsupported constant");
      os << ";\n";
      return success();
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      const char *sym = "==";
      switch (cmp.getPredicate()) {
      case arith::CmpIPredicate::eq: sym = "=="; break;
      case arith::CmpIPredicate::ne: sym = "!="; break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult: sym = "<"; break;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule: sym = "<="; break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt: sym = ">"; break;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge: sym = ">="; break;
      }
      binary(std::string("A ") + sym + " B");
      return success();
    }
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      indent(depth);
      os << scalarType(sel.getResult().getType(), hostSide) << " "
         << name(sel.getResult()) << " = " << name(sel.getCondition()) << " ? "
         << name(sel.getTrueValue()) << " : " << name(sel.getFalseValue())
         << ";\n";
      return success();
    }
    if (isa<arith::AddIOp>(op))
      binary("A + B");
    else if (isa<arith::SubIOp>(op))
      binary("A - B");
    else if (isa<arith::MulIOp>(op))
      binary("A * B");
    else if (isa<arith::DivSIOp>(op))
      binary("A / B");
    else if (isa<arith::RemSIOp>(op))
      binary("A % B");
    else if (isa<arith::CeilDivSIOp>(op))
      binary("(A + B - 1) / B");
    else if (isa<arith::MinSIOp>(op))
      binary("(A < B ? A : B)");
    else if (isa<arith::MaxSIOp>(op))
      binary("(A > B ? A : B)");
    else if (isa<arith::AddFOp>(op))
      binary("A + B");
    else if (isa<arith::MulFOp>(op))
      binary("A * B");
    else if (isa<arith::IndexCastOp>(op) || isa<arith::ExtSIOp>(op) ||
             isa<arith::TruncIOp>(op)) {
      indent(depth);
      os << scalarType(op->getResult(0).getType(), hostSide) << " "
         << name(op->getResult(0)) << " = ("
         << scalarType(op->getResult(0).getType(), hostSide) << ")"
         << name(op->getOperand(0)) << ";\n";
    } else
      return op->emitError("no C form for this op");
    return success();
  }

  raw_ostream &os;

private:
  llvm::DenseMap<Value, std::string> names;
  unsigned counter = 0;
};

// memref의 원소 개수. 정적 shape만.
inline int64_t elementCount(MemRefType t) {
  int64_t n = 1;
  for (int64_t d : t.getShape())
    n *= d;
  return n;
}

// 원소 하나의 바이트.
inline int64_t elementBytes(Type elem) {
  return (elem.getIntOrFloatBitWidth() + 7) / 8;
}

} // namespace upmem_codegen
} // namespace mlir

#endif // TRITON_SHARED_TARGET_UPMEM_CEMITTER_H_
