#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMVERIFY
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// 배치 pass가 만드는 clamp 식인가. 런타임 값이면 부등식을 증명할 수 없으므로 clamp가
// 있었는지를 구조로 확인한다.
static bool isClamped(Value v) {
  Operation *def = v.getDefiningOp();
  return getConstantIntValue(v).has_value() ||
         isa_and_nonnull<arith::SelectOp, arith::MinSIOp, arith::MaxSIOp>(def);
}

struct PIMVerifyPass
    : public mlir::triton::impl::PIMVerifyBase<PIMVerifyPass> {
  using PIMVerifyBase::PIMVerifyBase;

  void runOnOperation() override {
    pim::TargetProfileAttr profile = pim::targetProfile(getOperation());
    if (!profile) {
      getOperation()->emitError("pim-verify: the module has no #pim.target_profile");
      return signalPassFailure();
    }
    if (failed(verifyCores(profile)))
      return signalPassFailure();
    WalkResult res = getOperation().walk([&](pim::DeviceOp d) {
      return succeeded(verifyDevice(d, profile)) ? WalkResult::advance()
                                                 : WalkResult::interrupt();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }

  LogicalResult verifyCores(pim::TargetProfileAttr profile) {
    ArrayRef<int64_t> coreShape = profile.getPimCoreShape().asArrayRef();
    IntegerAttr local = profile.getUsableLocalCapacityBytes();
    bool simd = profile.getParallelExecutionKind() == "simd";
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      auto cores = func->getAttrOfType<pim::CoreMappingAttr>("pim.core_mapping");
      if (!cores)
        return WalkResult::advance();
      ArrayRef<int64_t> footprint = cores.getTileFootprint().asArrayRef();
      if (footprint.size() != coreShape.size()) {
        func->emitError("pim-verify: tile_footprint rank does not match "
                        "pim_core_shape rank");
        return WalkResult::interrupt();
      }
      for (auto [f, c] : llvm::zip(footprint, coreShape))
        if (f <= 0 || c % f != 0) {
          func->emitError("pim-verify: tile_footprint does not divide "
                          "pim_core_shape");
          return WalkResult::interrupt();
        }
      // lane 예산. simd 와 협력 모드(lane_axis_role = output_row)는 버퍼가 lane 마다
      // 복제되지 않으므로 lane 수를 곱하지 않는다.
      if (!local) {
        func->emitError("pim-verify: usable_local_capacity_bytes is not "
                        "recorded, so the lane budget cannot be checked");
        return WalkResult::interrupt();
      }
      if (local.getInt() == 0)
        return WalkResult::advance();
      bool sharedBuffers =
          simd || cores.getLaneAxisRole().getValue() == "output_row";
      int64_t bytes = pim::localTileBytes(func);
      int64_t need = sharedBuffers ? bytes : bytes * cores.getLanes();
      if (need > local.getInt()) {
        func->emitError("pim-verify: tile buffers need ")
            << need << " byte of local storage but a PIM Core has "
            << local.getInt();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  LogicalResult verifyDevice(pim::DeviceOp device,
                             pim::TargetProfileAttr profile) {
    int64_t used = 1;
    for (int64_t p : device.getMapping().getPartitions().asArrayRef())
      used *= p;
    if (used > profile.getNumberOfPimDevices())
      return device->emitError("pim-verify: the partition counts need ")
             << used << " PIM Devices but the machine has "
             << profile.getNumberOfPimDevices();

    llvm::DenseMap<int64_t, pim::AccessOp> accessIds;
    llvm::DenseSet<int64_t> placeIds;
    WalkResult res = device.walk([&](pim::AccessOp a) {
      if (a.getOrigin().size() != a.getSizes().size()) {
        a->emitError("pim-verify: origin rank does not match sizes rank");
        return WalkResult::interrupt();
      }
      // 단계 1은 pad = 0만 지원한다. 0이 아닌 pad는 calloc으로 나오지 않아 별도 채우기가
      // 필요하다.
      if (Value pad = a.getPad())
        if (!matchPattern(pad, m_Zero()) && !matchPattern(pad, m_AnyZeroFloat())) {
          a->emitError("pim-verify: only a zero pad value is supported");
          return WalkResult::interrupt();
        }
      if (!accessIds.insert({(int64_t)a.getId(), a}).second) {
        a->emitError("pim-verify: duplicate pim.access id");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      return failure();

    ArrayAttr axes = device.getMapping().getAxes();
    for (Operation &op : device.getBody().front()) {
      auto place = dyn_cast<pim::PlaceOp>(&op);
      if (!place)
        continue;
      if (!placeIds.insert(place.getAccess()).second)
        return place->emitError("pim-verify: two placements for one pim.access");
      auto it = accessIds.find((int64_t)place.getAccess());
      if (it == accessIds.end())
        return place->emitError("pim-verify: no pim.access with id ")
               << place.getAccess() << " in this device";
      pim::AccessOp access = it->second;
      for (Attribute a : place.getMapping().getAxisSources()) {
        StringRef source = cast<StringAttr>(a).getValue();
        if (!source.empty() && !llvm::is_contained(axes, a))
          return place->emitError("pim-verify: axis source ")
                 << source << " is not a device mapping axis";
      }
      for (auto [d, s] : llvm::enumerate(place.getSize())) {
        // mask 가 있는 축만 잘라야 한다. 없으면 도는 범위가 그대로 영역이다.
        bool masked = !access.getValid().empty() &&
                      !getConstantIntValue(access.getValid()[d]);
        if (masked && !isClamped(s))
          return place->emitError("pim-verify: the size of buffer axis ")
                 << d << " is not clamped to the buffer extent";
        std::optional<int64_t> size = getConstantIntValue(s);
        std::optional<int64_t> capacity =
            getConstantIntValue(place.getCapacity()[d]);
        if (size && *size < 0)
          return place->emitError("pim-verify: negative size on buffer axis ")
                 << d;
        if (size && capacity && *size > *capacity)
          return place->emitError("pim-verify: buffer axis ")
                 << d << " needs " << *size << " elements but only " << *capacity
                 << " are allocated per device";
      }
    }
    if (accessIds.size() != placeIds.size())
      return device->emitError("pim-verify: ")
             << accessIds.size() << " accesses but " << placeIds.size()
             << " placements";
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMVerifyPass() {
  return std::make_unique<PIMVerifyPass>();
}
