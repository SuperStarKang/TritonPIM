#ifndef TRITON_SHARED_TARGET_AIMX_CEMITTER_H_
#define TRITON_SHARED_TARGET_AIMX_CEMITTER_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace mlir {
namespace aimx_codegen {

// descriptor program 과 runtime stub 이 함께 쓰는 부분. 두 쪽 다 MLIR 값 하나가 C 변수
// 하나가 되고 op 하나가 대입문 하나가 된다. 그래야 생성된 코드의 모든 줄을 IR 의 op 으로
// 되짚을 수 있다.
class CEmitter {
public:
  CEmitter(raw_ostream &os) : os(os) {}

  raw_ostream &os;

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

  // 이름이 겹치지 않는 임시 변수.
  std::string temp(StringRef prefix) {
    return (prefix + Twine(counter++)).str();
  }

  void indent(unsigned n) { os.indent(n * 2); }

  // 스칼라 타입 → C 타입. index 는 정수 인덱스 계산에 쓰이므로 부호 있는 정수로 둔다
  // (subi / ceildivsi 가 음수를 거친다). bf16 과 f16 은 host 쪽에서 할 연산이 없어
  // raw 16 bit 로 옮긴다.
  static std::string scalarType(Type t) {
    if (t.isF32())
      return "float";
    if (t.isF64())
      return "double";
    if (t.isBF16() || t.isF16())
      return "uint16_t";
    if (isa<IndexType>(t))
      return "int64_t";
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

  static int64_t elementBytes(Type t) {
    return std::max<int64_t>(1, t.getIntOrFloatBitWidth() / 8);
  }

  static std::string unsignedType(Type type) {
    unsigned width = isa<IndexType>(type) ? 64 : cast<IntegerType>(type).getWidth();
    return "uint" + std::to_string(std::max(width, 8u)) + "_t";
  }

  // 정수 스칼라 연산 하나를 대입문 하나로 옮긴다. 다루지 못하면 실패한다 —
  // 모르는 것을 짐작해 채우지 않는다.
  LogicalResult emitArith(Operation *op, unsigned depth) {
    auto decl = [&](Value r) {
      indent(depth);
      os << scalarType(r.getType()) << " " << name(r) << " = ";
    };
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      auto ia = dyn_cast<IntegerAttr>(c.getValue());
      if (!ia)
        return op->emitError("aimx codegen: only integer constants are used here");
      decl(c.getResult());
      os << ia.getValue().getSExtValue() << ";\n";
      return success();
    }
    // 폭만 바꾸는 것들은 C 캐스트 하나다.
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp>(op)) {
      Value r = op->getResult(0);
      decl(r);
      os << "(" << scalarType(r.getType()) << ")" << name(op->getOperand(0))
         << ";\n";
      return success();
    }
    if (isa<arith::IndexCastUIOp, arith::ExtUIOp>(op)) {
      Value result = op->getResult(0), source = op->getOperand(0);
      decl(result);
      os << "(" << scalarType(result.getType()) << ")(" << unsignedType(source.getType())
         << ")" << name(source) << ";\n";
      return success();
    }
    // 비교와 선택. min/max 를 정규화하면 이 둘로 갈라져 나온다.
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      const char *sym = nullptr;
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
      decl(cmp.getResult());
      bool unsignedComparison = cmp.getPredicate() == arith::CmpIPredicate::ult ||
                                cmp.getPredicate() == arith::CmpIPredicate::ule ||
                                cmp.getPredicate() == arith::CmpIPredicate::ugt ||
                                cmp.getPredicate() == arith::CmpIPredicate::uge;
      std::string cast = unsignedComparison ? "(" + unsignedType(cmp.getLhs().getType()) + ")" : "";
      os << cast << name(cmp.getLhs()) << " " << sym << " " << cast << name(cmp.getRhs())
         << ";\n";
      return success();
    }
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      decl(sel.getResult());
      os << name(sel.getCondition()) << " ? " << name(sel.getTrueValue())
         << " : " << name(sel.getFalseValue()) << ";\n";
      return success();
    }
    const char *fmt = nullptr;
    if (isa<arith::AddIOp>(op))
      fmt = "A + B";
    else if (isa<arith::SubIOp>(op))
      fmt = "A - B";
    else if (isa<arith::MulIOp>(op))
      fmt = "A * B";
    else if (isa<arith::DivSIOp>(op))
      fmt = "A / B";
    else if (isa<arith::RemSIOp>(op))
      fmt = "A % B";
    else if (isa<arith::MinSIOp>(op))
      fmt = "(A) < (B) ? (A) : (B)";
    else if (isa<arith::MaxSIOp>(op))
      fmt = "(A) > (B) ? (A) : (B)";
    else if (isa<arith::FloorDivSIOp>(op))
      fmt = "A / B - ((A % B != 0) && ((A < 0) != (B < 0)))";
    else if (isa<arith::CeilDivSIOp>(op))
      fmt = "A / B + ((A % B != 0) && ((A < 0) == (B < 0)))";
    if (!fmt)
      return op->emitError("aimx codegen: unsupported scalar operation");
    decl(op->getResult(0));
    std::string a = name(op->getOperand(0)), b = name(op->getOperand(1));
    for (char ch : StringRef(fmt)) {
      if (ch == 'A')
        os << a;
      else if (ch == 'B')
        os << b;
      else
        os << ch;
    }
    os << ";\n";
    return success();
  }

  // Keep integer/index loop state as descriptor-address/control arithmetic.
  // Tensor data is already lowered to PIM buffers; no floating-point source
  // computation is executed here. Results retain their init value on zero trips.
  LogicalResult emitFor(scf::ForOp loop, unsigned depth,
                        function_ref<LogicalResult(Block &, unsigned)> body) {
    if (scalarType(loop.getInductionVar().getType()).empty())
      return loop.emitError("aimx codegen: unsupported induction variable type");
    for (auto [argument, init, result] :
         llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs(), loop.getResults())) {
      if (!argument.getType().isIntOrIndex() || scalarType(argument.getType()).empty())
        return loop.emitError("aimx codegen: loop-carried control values require supported integer/index types");
      indent(depth);
      os << scalarType(argument.getType()) << " " << name(argument) << " = "
         << name(init) << ";\n";
      bind(result, name(argument));
    }
    std::string iv = name(loop.getInductionVar());
    std::string comparisonCast = loop.getUnsignedCmp()
                                     ? "(" + unsignedType(loop.getInductionVar().getType()) + ")"
                                     : "";
    indent(depth);
    os << "for (" << scalarType(loop.getInductionVar().getType()) << " " << iv
       << " = " << name(loop.getLowerBound()) << "; "
       << comparisonCast << iv << " < " << comparisonCast << name(loop.getUpperBound()) << "; " << iv;
    if (loop.getUnsignedCmp()) {
      // An unsigned traversal may cross the signed representation's sign bit.
      // Perform the increment in the unsigned domain, too.
      os << " = (" << scalarType(loop.getInductionVar().getType()) << ")(" << comparisonCast
         << iv << " + " << comparisonCast << name(loop.getStep()) << ")";
    } else {
      os << " += " << name(loop.getStep());
    }
    os << ") {\n";
    if (failed(body(*loop.getBody(), depth + 1)))
      return failure();
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    SmallVector<std::string> yielded;
    for (Value value : yield.getOperands()) {
      if (yield.getNumOperands() == 1) {
        yielded.push_back(name(value));
      } else {
        // Read all incoming values before assigning any carried variable.
        // Sequential assignments alone would corrupt a swap such as x,y=y,x.
        std::string snapshot = temp("_yield");
        indent(depth + 1);
        os << scalarType(value.getType()) << " " << snapshot << " = " << name(value) << ";\n";
        yielded.push_back(snapshot);
      }
    }
    for (auto [argument, value] : llvm::zip(loop.getRegionIterArgs(), yielded)) {
      indent(depth + 1);
      os << name(argument) << " = " << value << ";\n";
    }
    indent(depth);
    os << "}\n";
    return success();
  }

private:
  llvm::DenseMap<Value, std::string> names;
  unsigned counter = 0;
};

} // namespace aimx_codegen
} // namespace mlir

#endif // TRITON_SHARED_TARGET_AIMX_CEMITTER_H_
