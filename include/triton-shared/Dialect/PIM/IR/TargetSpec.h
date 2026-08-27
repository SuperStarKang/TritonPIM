#ifndef TRITON_SHARED_DIALECT_PIM_IR_TARGET_SPEC_H_
#define TRITON_SHARED_DIALECT_PIM_IR_TARGET_SPEC_H_

#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <utility>

namespace mlir {
namespace pim {

// execution tile 산출에 쓰는 buffer 하나의 정적 모양. 상한 축(m/n/k)을 따라가는
// 차원은 상한이 크기가 되고, 따라가지 않는 차원은 정적 tile 크기 그대로다.
struct BufferDim {
  enum Axis { M, N, K, Static } axis = Static;
  int64_t staticSize = 1; // axis == Static 일 때만 쓰인다
};

struct BufferShape {
  int64_t elemBytes = 1;
  llvm::SmallVector<BufferDim, 3> dims;
};

// target 이 공통 pass 에 제공하는 지식. 결정(탐색·확정·attribute 부착)은 공통 pass 가
// 하고, 여기는 "execution tile (m, n, k)가 이 기계에서 몇 바이트를 차지하는가" 같은
// 계산식만 담는다. IR 에는 결과 수치만 남으므로 공통 IR 의 target 무관 계약
// (PIMDialect.td 의 "target 확장 attribute 는 두지 않는다")은 유지된다.
class TargetSpec {
public:
  virtual ~TargetSpec() = default;

  // execution tile이 device의 큰 메모리 예산에서 차지하는 바이트. mT/nT/kT는
  // physicalGroupExtent가 target 실행 눈금에 맞춘 물리 원소수다. per-bank tail이나
  // replication이 있으면 target은 이를 device 전체에 환산한 보수적 값을 반환한다.
  virtual int64_t deviceFootprintBytes(TargetProfileAttr profile,
                                       IntraDevicePlanAttr plan,
                                       llvm::ArrayRef<BufferShape> buffers,
                                       int64_t mT, int64_t nT,
                                       int64_t kT) const = 0;

  // n 상한의 기본 block group 개수. 0 이면 공통 기본(group 1개)이다.
  virtual int64_t preferredNGroups(TargetProfileAttr profile,
                                   IntraDevicePlanAttr plan,
                                   int64_t nGroupUseful,
                                   int64_t nGroupPhys) const {
    return 0;
  }

  // k 탐색 눈금 {useful, phys}. 기본은 k block group 그대로다.
  virtual std::pair<int64_t, int64_t> kGrain(TargetProfileAttr profile,
                                             IntraDevicePlanAttr plan,
                                             int64_t kGroupUseful,
                                             int64_t kGroupPhys) const {
    return {kGroupUseful, kGroupPhys};
  }

  // Common block-group useful extent를 target이 실제로 실행하는 물리 extent로
  // 바꾼다. axis는 0=m, 1=n, 2=k다. 기본 target은 padding이 없다.
  virtual int64_t physicalGroupExtent(TargetProfileAttr profile,
                                      IntraDevicePlanAttr plan, unsigned axis,
                                      int64_t useful) const {
    return useful;
  }
};

// target 이름(profile 의 target 문자열)으로 등록·조회한다. 등록은 도구 초기화에서
// 명시 호출로 한다 — 정적 초기화 순서에 기대지 않는다.
void registerTargetSpec(llvm::StringRef target, std::unique_ptr<TargetSpec> spec);
const TargetSpec *lookupTargetSpec(llvm::StringRef target);

} // namespace pim
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_PIM_IR_TARGET_SPEC_H_
