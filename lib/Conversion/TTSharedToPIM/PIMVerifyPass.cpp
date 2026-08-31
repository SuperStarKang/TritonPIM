#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIM/IR/PIMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PIMVERIFY
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static int64_t canonicalAxis(StringRef axis) {
  return axis == "m" ? 0 : axis == "n" ? 1 : axis == "k" ? 2 : -1;
}

// 배치 pass가 만드는 clamp 식인가. 런타임 값이면 부등식을 증명할 수 없으므로 clamp가
// 있었는지를 구조로 확인한다.
static bool isClamped(Value v) {
  Operation *def = v.getDefiningOp();
  return getConstantIntValue(v).has_value() ||
         isa_and_nonnull<arith::SelectOp, arith::MinSIOp, arith::MaxSIOp>(def);
}

// pim-map-devices가 만드는 local window의 upper bound는
// min(execution_tile.groups[k], remaining)이다. operand 순서가 바뀌어도 같은
// 계약으로 보고, 다른 임의의 clamp는 temporal K window로 인정하지 않는다.
// scheduling 산술은 index지만 원래 Triton K loop가 i32이면 scf.for
// 경계에서만 index_cast가 생긴다. 이 cast는 window 의미를 바꾸지 않으므로
// 벗긴 뒤 map pass가 만든 min 형태를 검증한다.
static bool isClampedTo(Value value, int64_t limit) {
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    value = cast.getIn();
  auto min = value.getDefiningOp<arith::MinSIOp>();
  if (!min)
    return false;
  return getConstantIntValue(min.getLhs()) == limit ||
         getConstantIntValue(min.getRhs()) == limit;
}

static std::optional<int64_t> constantIndexExpr(Value value) {
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return constant;
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return constantIndexExpr(cast.getIn());
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    std::optional<int64_t> lhs = constantIndexExpr(mul.getLhs());
    std::optional<int64_t> rhs = constantIndexExpr(mul.getRhs());
    if (!lhs || !rhs)
      return std::nullopt;
    if (*lhs < 0 || *rhs < 0 ||
        (*lhs != 0 &&
         *rhs > std::numeric_limits<int64_t>::max() / *lhs))
      return std::nullopt;
    return *lhs * *rhs;
  }
  return std::nullopt;
}

static bool dependsOn(Value value, ValueRange roots) {
  SmallVector<Value> work{value};
  llvm::DenseSet<Value> seen;
  while (!work.empty()) {
    Value current = work.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    if (llvm::is_contained(roots, current))
      return true;
    if (Operation *def = current.getDefiningOp())
      work.append(def->getOperands().begin(), def->getOperands().end());
  }
  return false;
}

static bool dependsOn(Value value, Value root) {
  SmallVector<Value, 1> roots{root};
  return dependsOn(value, roots);
}

// Scheduling arithmetic is built with index values and may be cast only at the
// original Triton loop boundary. Casts do not change which Execution Tile is
// selected, so structural scheduling checks look through them.
static Value stripIndexCasts(Value value) {
  while (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    value = cast.getIn();
  return value;
}

struct LinearForm {
  int64_t coefficient;
  int64_t constant;
};

struct SpatialLinearForm {
  int64_t roundCoefficient;
  int64_t deviceCoefficient;
  int64_t constant;
};

static std::optional<int64_t> checkedLinearValue(__int128 value) {
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(value);
}

// Read the small affine expression produced for an Execution Tile start. The
// accepted grammar is intentionally limited to casts, add/sub, and a multiply
// by a constant; arbitrary arithmetic that merely depends on the round IV is
// not sufficient to establish the Common scheduling contract.
static std::optional<LinearForm> linearForm(Value value, Value variable) {
  value = stripIndexCasts(value);
  variable = stripIndexCasts(variable);
  if (value == variable)
    return LinearForm{1, 0};
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return LinearForm{0, *constant};
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    std::optional<LinearForm> lhs = linearForm(add.getLhs(), variable);
    std::optional<LinearForm> rhs = linearForm(add.getRhs(), variable);
    if (!lhs || !rhs)
      return std::nullopt;
    std::optional<int64_t> coefficient = checkedLinearValue(
        static_cast<__int128>(lhs->coefficient) + rhs->coefficient);
    std::optional<int64_t> constant = checkedLinearValue(
        static_cast<__int128>(lhs->constant) + rhs->constant);
    if (!coefficient || !constant)
      return std::nullopt;
    return LinearForm{*coefficient, *constant};
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    std::optional<LinearForm> lhs = linearForm(sub.getLhs(), variable);
    std::optional<LinearForm> rhs = linearForm(sub.getRhs(), variable);
    if (!lhs || !rhs)
      return std::nullopt;
    std::optional<int64_t> coefficient = checkedLinearValue(
        static_cast<__int128>(lhs->coefficient) - rhs->coefficient);
    std::optional<int64_t> constant = checkedLinearValue(
        static_cast<__int128>(lhs->constant) - rhs->constant);
    if (!coefficient || !constant)
      return std::nullopt;
    return LinearForm{*coefficient, *constant};
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    Value expression;
    std::optional<int64_t> factor = getConstantIntValue(mul.getLhs());
    if (factor)
      expression = mul.getRhs();
    else {
      factor = getConstantIntValue(mul.getRhs());
      expression = mul.getLhs();
    }
    if (!factor)
      return std::nullopt;
    std::optional<LinearForm> form = linearForm(expression, variable);
    if (!form)
      return std::nullopt;
    std::optional<int64_t> coefficient = checkedLinearValue(
        static_cast<__int128>(form->coefficient) * *factor);
    std::optional<int64_t> constant = checkedLinearValue(
        static_cast<__int128>(form->constant) * *factor);
    if (!coefficient || !constant)
      return std::nullopt;
    return LinearForm{*coefficient, *constant};
  }
  return std::nullopt;
}

// Read the exact first-group expression emitted for a spatial axis.  Only the
// scheduling round and the corresponding pim.device coordinate may vary:
//   first = (round * partitions + device) * groupsPerTile
static std::optional<SpatialLinearForm>
spatialLinearForm(Value value, Value round, Value device) {
  value = stripIndexCasts(value);
  round = stripIndexCasts(round);
  device = stripIndexCasts(device);
  if (value == round)
    return SpatialLinearForm{1, 0, 0};
  if (value == device)
    return SpatialLinearForm{0, 1, 0};
  if (std::optional<int64_t> constant = getConstantIntValue(value))
    return SpatialLinearForm{0, 0, *constant};
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    std::optional<SpatialLinearForm> lhs =
        spatialLinearForm(add.getLhs(), round, device);
    std::optional<SpatialLinearForm> rhs =
        spatialLinearForm(add.getRhs(), round, device);
    if (!lhs || !rhs)
      return std::nullopt;
    std::optional<int64_t> roundCoefficient = checkedLinearValue(
        static_cast<__int128>(lhs->roundCoefficient) +
        rhs->roundCoefficient);
    std::optional<int64_t> deviceCoefficient = checkedLinearValue(
        static_cast<__int128>(lhs->deviceCoefficient) +
        rhs->deviceCoefficient);
    std::optional<int64_t> constant = checkedLinearValue(
        static_cast<__int128>(lhs->constant) + rhs->constant);
    if (!roundCoefficient || !deviceCoefficient || !constant)
      return std::nullopt;
    return SpatialLinearForm{*roundCoefficient, *deviceCoefficient,
                             *constant};
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    std::optional<SpatialLinearForm> lhs =
        spatialLinearForm(sub.getLhs(), round, device);
    std::optional<SpatialLinearForm> rhs =
        spatialLinearForm(sub.getRhs(), round, device);
    if (!lhs || !rhs)
      return std::nullopt;
    std::optional<int64_t> roundCoefficient = checkedLinearValue(
        static_cast<__int128>(lhs->roundCoefficient) -
        rhs->roundCoefficient);
    std::optional<int64_t> deviceCoefficient = checkedLinearValue(
        static_cast<__int128>(lhs->deviceCoefficient) -
        rhs->deviceCoefficient);
    std::optional<int64_t> constant = checkedLinearValue(
        static_cast<__int128>(lhs->constant) - rhs->constant);
    if (!roundCoefficient || !deviceCoefficient || !constant)
      return std::nullopt;
    return SpatialLinearForm{*roundCoefficient, *deviceCoefficient,
                             *constant};
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    Value expression;
    std::optional<int64_t> factor = getConstantIntValue(mul.getLhs());
    if (factor)
      expression = mul.getRhs();
    else {
      factor = getConstantIntValue(mul.getRhs());
      expression = mul.getLhs();
    }
    if (!factor)
      return std::nullopt;
    std::optional<SpatialLinearForm> form =
        spatialLinearForm(expression, round, device);
    if (!form)
      return std::nullopt;
    std::optional<int64_t> roundCoefficient = checkedLinearValue(
        static_cast<__int128>(form->roundCoefficient) * *factor);
    std::optional<int64_t> deviceCoefficient = checkedLinearValue(
        static_cast<__int128>(form->deviceCoefficient) * *factor);
    std::optional<int64_t> constant = checkedLinearValue(
        static_cast<__int128>(form->constant) * *factor);
    if (!roundCoefficient || !deviceCoefficient || !constant)
      return std::nullopt;
    return SpatialLinearForm{*roundCoefficient, *deviceCoefficient,
                             *constant};
  }
  return std::nullopt;
}

static bool matchScaledCoordinate(Value value, Value coordinate,
                                  Value &scale) {
  value = stripIndexCasts(value);
  coordinate = stripIndexCasts(coordinate);
  if (value == coordinate) {
    scale = Value();
    return true;
  }
  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  Value lhs = stripIndexCasts(mul.getLhs());
  Value rhs = stripIndexCasts(mul.getRhs());
  if (lhs == coordinate) {
    scale = rhs;
    return true;
  }
  if (rhs == coordinate) {
    scale = lhs;
    return true;
  }
  return false;
}

// The local logical coordinate must be the original loop coordinate at the
// current Execution Tile start:
//   start  = lower + first * step
//   global = start + local * step
static bool hasExactSpatialOrigin(scf::ForOp local, Value first) {
  Operation *origin = nullptr;
  for (Operation &op : *local.getBody()) {
    if (!op.hasAttr("pim.device_origin"))
      continue;
    if (origin)
      return false;
    origin = &op;
  }
  if (!origin || origin->getNumResults() != 1)
    return false;

  auto global = stripIndexCasts(origin->getResult(0))
                    .getDefiningOp<arith::AddIOp>();
  if (!global)
    return false;
  Value globalOperands[2] = {global.getLhs(), global.getRhs()};
  for (int localOperand = 0; localOperand < 2; ++localOperand) {
    Value localScale;
    if (!matchScaledCoordinate(globalOperands[localOperand],
                               local.getInductionVar(), localScale))
      continue;
    Value start = stripIndexCasts(globalOperands[1 - localOperand]);
    auto startAdd = start.getDefiningOp<arith::AddIOp>();
    if (!startAdd)
      continue;
    Value startOperands[2] = {startAdd.getLhs(), startAdd.getRhs()};
    for (int firstOperand = 0; firstOperand < 2; ++firstOperand) {
      Value firstScale;
      if (!matchScaledCoordinate(startOperands[firstOperand], first,
                                 firstScale))
        continue;
      Value lower = startOperands[1 - firstOperand];
      bool sameScale = (!localScale && !firstScale) ||
                       (localScale && firstScale &&
                        stripIndexCasts(localScale) ==
                            stripIndexCasts(firstScale));
      if (sameScale && !dependsOn(lower, first) &&
          !dependsOn(lower, local.getInductionVar()))
        return true;
    }
  }
  return false;
}

// Verify the exact spatial scheduling contract emitted by PIMMapDevices:
//   executionTiles = ceildiv(totalGroups, groupsPerTile)
//   rounds          = ceildiv(executionTiles, partitions)
//   first           = (round * partitions + device) * groupsPerTile
//   localTrips      = min(groupsPerTile, max(totalGroups - first, 0))
static LogicalResult verifySpatialSchedule(scf::ForOp round,
                                           scf::ForOp local, Value device,
                                           StringRef axis,
                                           int64_t groupsPerTile,
                                           int64_t partitions, Value &first) {
  if (!matchPattern(round.getLowerBound(), m_Zero()) ||
      getConstantIntValue(round.getStep()) != 1)
    return round->emitError("pim-verify: the spatial ")
           << axis << " scheduling round must be zero-based with unit step";

  auto roundCount = stripIndexCasts(round.getUpperBound())
                        .getDefiningOp<arith::CeilDivSIOp>();
  if (!roundCount ||
      getConstantIntValue(roundCount.getRhs()) != partitions)
    return round->emitError("pim-verify: the spatial ")
           << axis
           << " scheduling round upper bound must divide the Execution Tile "
              "count by p_"
           << axis;
  auto executionTiles = stripIndexCasts(roundCount.getLhs())
                            .getDefiningOp<arith::CeilDivSIOp>();
  if (!executionTiles ||
      getConstantIntValue(executionTiles.getRhs()) != groupsPerTile)
    return round->emitError("pim-verify: the spatial ")
           << axis
           << " Execution Tile count must be ceildiv(total groups, groups per "
              "Execution Tile)";
  Value totalGroups = stripIndexCasts(executionTiles.getLhs());

  if (!matchPattern(local.getLowerBound(), m_Zero()) ||
      getConstantIntValue(local.getStep()) != 1)
    return local->emitError("pim-verify: the device-local spatial ")
           << axis << " loop must be zero-based with unit step";
  auto window = stripIndexCasts(local.getUpperBound())
                    .getDefiningOp<arith::MinSIOp>();
  if (!window)
    return local->emitError("pim-verify: the device-local spatial ")
           << axis << " loop is not clamped to the current Execution Tile";
  Value remaining;
  if (getConstantIntValue(window.getLhs()) == groupsPerTile)
    remaining = window.getRhs();
  else if (getConstantIntValue(window.getRhs()) == groupsPerTile)
    remaining = window.getLhs();
  else
    return local->emitError("pim-verify: the device-local spatial ")
           << axis << " loop does not use its Execution Tile group count";

  auto nonnegative = stripIndexCasts(remaining).getDefiningOp<arith::MaxSIOp>();
  if (!nonnegative)
    return local->emitError("pim-verify: the remaining spatial ")
           << axis << " group count is not clamped at zero";
  Value difference;
  if (getConstantIntValue(nonnegative.getLhs()) == 0)
    difference = nonnegative.getRhs();
  else if (getConstantIntValue(nonnegative.getRhs()) == 0)
    difference = nonnegative.getLhs();
  else
    return local->emitError("pim-verify: the remaining spatial ")
           << axis << " group count is not clamped at zero";

  auto left = stripIndexCasts(difference).getDefiningOp<arith::SubIOp>();
  if (!left || stripIndexCasts(left.getLhs()) != totalGroups)
    return local->emitError("pim-verify: the spatial ")
           << axis
           << " host round and local loop do not use the same total group "
              "count";
  first = stripIndexCasts(left.getRhs());
  std::optional<SpatialLinearForm> form = spatialLinearForm(
      first, round.getInductionVar(), device);
  std::optional<int64_t> roundCoefficient = checkedLinearValue(
      static_cast<__int128>(partitions) * groupsPerTile);
  if (!form || !roundCoefficient ||
      form->roundCoefficient != *roundCoefficient ||
      form->deviceCoefficient != groupsPerTile || form->constant != 0)
    return local->emitError("pim-verify: the spatial ")
           << axis
           << " Execution Tile start must be "
              "(round * partitions + device) * groups per tile";
  if (!hasExactSpatialOrigin(local, first))
    return local->emitError("pim-verify: the spatial ")
           << axis
           << " global origin must be lower + (first + local) * original "
              "step";
  return success();
}

// Verify that the host round and device-local temporal K loop are two views of
// the same Execution Tile window:
//   round       = 0 .. ceildiv(totalGroups, tileK)
//   first       = round * tileK
//   local trips = min(tileK, max(totalGroups - first, 0))
// The marked global origin must use both first and the local induction value.
static LogicalResult verifyTemporalSchedule(scf::ForOp round,
                                            scf::ForOp reduction,
                                            int64_t tileK,
                                            Value &first) {
  if (!matchPattern(round.getLowerBound(), m_Zero()) ||
      getConstantIntValue(round.getStep()) != 1)
    return round->emitError(
        "pim-verify: the temporal K scheduling round must be zero-based "
        "with unit step");

  Value roundUpper = stripIndexCasts(round.getUpperBound());
  auto roundCount = roundUpper.getDefiningOp<arith::CeilDivSIOp>();
  if (!roundCount || getConstantIntValue(roundCount.getRhs()) != tileK)
    return round->emitError(
        "pim-verify: the temporal K scheduling round upper bound must be "
        "ceildiv(total reduction groups, execution_tile.groups[k])");
  Value totalGroups = stripIndexCasts(roundCount.getLhs());

  Value localUpper = stripIndexCasts(reduction.getUpperBound());
  auto window = localUpper.getDefiningOp<arith::MinSIOp>();
  if (!window)
    return reduction->emitError(
        "pim-verify: the temporal K local window is not clamped to the "
        "remaining reduction groups");
  Value remaining;
  if (getConstantIntValue(window.getLhs()) == tileK)
    remaining = window.getRhs();
  else if (getConstantIntValue(window.getRhs()) == tileK)
    remaining = window.getLhs();
  else
    return reduction->emitError(
        "pim-verify: the temporal K local window does not use "
        "execution_tile.groups[k]");

  auto nonnegative = stripIndexCasts(remaining).getDefiningOp<arith::MaxSIOp>();
  if (!nonnegative)
    return reduction->emitError(
        "pim-verify: the temporal K remaining group count is not clamped "
        "at zero");
  Value difference;
  if (getConstantIntValue(nonnegative.getLhs()) == 0)
    difference = nonnegative.getRhs();
  else if (getConstantIntValue(nonnegative.getRhs()) == 0)
    difference = nonnegative.getLhs();
  else
    return reduction->emitError(
        "pim-verify: the temporal K remaining group count is not clamped "
        "at zero");

  auto left = stripIndexCasts(difference).getDefiningOp<arith::SubIOp>();
  if (!left || stripIndexCasts(left.getLhs()) != totalGroups)
    return reduction->emitError(
        "pim-verify: the temporal K host round and local window do not use "
        "the same total reduction-group count");
  first = stripIndexCasts(left.getRhs());
  std::optional<LinearForm> form =
      linearForm(first, round.getInductionVar());
  if (!form || form->coefficient != tileK || form->constant != 0)
    return reduction->emitError(
        "pim-verify: the temporal K window start must be round * "
        "execution_tile.groups[k]");

  Operation *origin = nullptr;
  for (Operation &op : *reduction.getBody()) {
    if (!op.hasAttr("pim.device_origin"))
      continue;
    if (origin)
      return reduction->emitError(
          "pim-verify: the temporal K loop has more than one device origin");
    origin = &op;
  }
  if (!origin || origin->getNumResults() != 1)
    return reduction->emitError(
        "pim-verify: the temporal K loop needs one marked device origin");
  Value global = origin->getResult(0);
  if (!dependsOn(global, first))
    return origin->emitError(
        "pim-verify: the temporal K global origin does not use the current "
        "Execution Tile window start");
  if (!dependsOn(global, reduction.getInductionVar()))
    return origin->emitError(
        "pim-verify: the temporal K global origin does not use the local "
        "reduction-group coordinate");
  return success();
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
    int64_t local = profile.getLocalCapacityLimitBytes();
    WalkResult res = getOperation().walk([&](func::FuncOp func) {
      bool hasAccess = false;
      func.walk([&](pim::AccessOp) { hasAccess = true; });
      if (!hasAccess)
        return WalkResult::advance();
      auto block = func->getAttrOfType<pim::BlockAttr>("pim.block");
      auto group =
          func->getAttrOfType<pim::BlockGroupAttr>("pim.block_group");
      if (!block || !group) {
        func->emitError(
            "pim-verify: every PIM function needs #pim.block and "
            "#pim.block_group");
        return WalkResult::interrupt();
      }

      // hierarchy metadata가 실제 local compute tile과 같은 크기를 말하는지 확인한다.
      // m/n은 이름 붙은 logical axis, k는 linalg/accumulator 의미에서 명시된
      // semantic reduction loop다. access 주소를 모는 loop로 reduction을 추측하지
      // 않는다.
      SmallVector<scf::ForOp> loops;
      func.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
        // pim.device 바깥의 scheduling round도 tag 없는 scf.for다. 그것을
        // reduction loop로 오인하지 않고 device-local compute loop만 본다.
        if (loop->getParentOfType<pim::DeviceOp>())
          loops.push_back(loop);
      });
      scf::ForOp hierarchyAxes[3];
      for (scf::ForOp loop : loops) {
        auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis");
        if (!axis)
          continue;
        if (axis.getName().getValue() == "m")
          hierarchyAxes[0] = loop;
        else if (axis.getName().getValue() == "n")
          hierarchyAxes[1] = loop;
      }
      for (scf::ForOp loop : loops)
        if (auto reduction =
                loop->getAttrOfType<StringAttr>("pim.reduction")) {
          if (reduction.getValue() != "add") {
            loop->emitError("pim-verify: unsupported reduction '")
                << reduction.getValue() << "'";
            return WalkResult::interrupt();
          }
          if (hierarchyAxes[2]) {
            func->emitError(
                "pim-verify: more than one semantic reduction loop is "
                "present");
            return WalkResult::interrupt();
          }
          hierarchyAxes[2] = loop;
        }

      SmallVector<int64_t, 3> groupShape = pim::groupShape(func);
      pim::DeviceOp device;
      func.walk([&](pim::DeviceOp candidate) {
        if (!device)
          device = candidate;
      });
      auto execution =
          device ? device->getAttrOfType<pim::ExecutionTileAttr>(
                       "pim.execution_tile")
                 : pim::ExecutionTileAttr();
      if (!execution) {
        func->emitError(
            "pim-verify: every PIM function needs a pim.device with "
            "#pim.execution_tile");
        return WalkResult::interrupt();
      }
      ArrayRef<int64_t> tileGroups = execution.getGroups().asArrayRef();
      static const char *axisNames[3] = {"m", "n", "k"};
      for (int i = 0; i < 3; ++i) {
        bool hasComputeAxis = static_cast<bool>(hierarchyAxes[i]);
        if (!hasComputeAxis &&
            (groupShape.size() != 3 || groupShape[i] != 1 ||
             tileGroups[i] != 1)) {
          func->emitError("pim-verify: inactive ")
              << axisNames[i]
              << " axis must have identity Block, Block Group, and "
                 "Execution Tile extents of 1";
          return WalkResult::interrupt();
        }
        if (!hasComputeAxis)
          continue;
        if (i == 2 &&
            !hierarchyAxes[i]->hasAttr("pim.temporal_reduction")) {
          func->emitError(
              "pim-verify: the reduction loop is not materialized as a "
              "temporal K Execution Tile");
          return WalkResult::interrupt();
        }
        int64_t actual = pim::tileElementsAlong(hierarchyAxes[i], loops);
        if (actual != groupShape[i]) {
          func->emitError("pim-verify: #pim.block × #pim.block_group gives ")
              << groupShape[i] << " elements on " << axisNames[i]
              << " but the local compute tile uses " << actual;
          return WalkResult::interrupt();
        }
      }
      // bank/core/lane 분해와 물리 padding은 target lowering의 책임이다. Common IR은
      // block group의 working set이 공통 local capacity 상한을 넘는지만 검사한다.
      if (local == 0)
        return WalkResult::advance();
      int64_t bytes = pim::localTileBytes(func);
      if (bytes > local) {
        func->emitError("pim-verify: tile buffers need ")
            << bytes << " byte of local storage but a PIM Core has " << local;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(res.wasInterrupted());
  }

  LogicalResult verifyDevice(pim::DeviceOp device,
                             pim::TargetProfileAttr profile) {
    auto execution =
        device->getAttrOfType<pim::ExecutionTileAttr>("pim.execution_tile");
    if (!execution)
      return device->emitError(
          "pim-verify: every pim.device needs #pim.execution_tile");
    SmallVector<int64_t, 3> executionShape =
        pim::executionTileShape(device);
    if (executionShape.size() != 3)
      return device->emitError(
          "pim-verify: cannot derive the execution tile shape from the "
          "three hierarchy attributes");
    if (llvm::any_of(executionShape,
                     [](int64_t extent) { return extent < 1; }))
      return device->emitError(
          "pim-verify: execution tile extents must be positive");

    SmallVector<scf::ForOp> deviceLoops;
    device.getBody().walk<WalkOrder::PreOrder>(
        [&](scf::ForOp loop) { deviceLoops.push_back(loop); });
    SmallVector<scf::ForOp> temporalReductions;
    SmallVector<scf::ForOp> semanticReductions;
    SmallVector<scf::ForOp> spatialAxes[3];
    for (scf::ForOp loop : deviceLoops) {
      if (loop->hasAttr("pim.temporal_reduction"))
        temporalReductions.push_back(loop);
      if (loop->getAttrOfType<StringAttr>("pim.reduction"))
        semanticReductions.push_back(loop);
      if (auto axis = loop->getAttrOfType<pim::AxisAttr>("pim.axis")) {
        int64_t number = canonicalAxis(axis.getName().getValue());
        if (number >= 0)
          spatialAxes[number].push_back(loop);
      }
    }
    if (temporalReductions.size() > 1)
      return device->emitError(
          "pim-verify: one Execution Tile may have only one temporal K "
          "reduction loop");
    if (semanticReductions.size() > 1)
      return device->emitError(
          "pim-verify: one Execution Tile may have only one semantic "
          "reduction loop");
    static const char *axisNames[3] = {"m", "n", "k"};
    for (int64_t axis = 0; axis < 3; ++axis)
      if (spatialAxes[axis].size() > 1)
        return device->emitError("pim-verify: one Execution Tile may have "
                                 "only one spatial ")
               << axisNames[axis] << " axis";

    ArrayRef<int64_t> partitions =
        device.getMapping().getPartitions().asArrayRef();
    for (int64_t axis = 0; axis < 3; ++axis)
      if (partitions[axis] > 1 && spatialAxes[axis].empty())
        return device->emitError("pim-verify: p_")
               << axisNames[axis]
               << " greater than one requires an explicit spatial pim.axis=\""
               << axisNames[axis] << "\"";

    bool hasStore = false;
    device.walk([&](pim::StoreOp) { hasStore = true; });
    if (!spatialAxes[2].empty() && hasStore && semanticReductions.empty())
      return device->emitError(
          "pim-verify: a spatial K output needs one semantic pim.reduction");

    scf::ForOp temporalReduction;
    Value temporalFirst;
    if (!temporalReductions.empty()) {
      temporalReduction = temporalReductions.front();
      scf::ForOp reduction = temporalReduction;
      auto combiner = reduction->getAttrOfType<StringAttr>("pim.reduction");
      if (!combiner)
        return reduction->emitError(
            "pim-verify: a temporal K loop must also carry the semantic "
            "pim.reduction attribute");
      if (combiner.getValue() != "add")
        return reduction->emitError("pim-verify: unsupported reduction '")
               << combiner.getValue() << "'";
      int64_t tileK = execution.getGroups().asArrayRef()[2];
      if (!matchPattern(reduction.getLowerBound(), m_Zero()) ||
          getConstantIntValue(reduction.getStep()) != 1 ||
          !isClampedTo(reduction.getUpperBound(), tileK))
        return reduction->emitError(
            "pim-verify: the temporal K loop must be a zero-based unit-step "
            "window clamped to execution_tile.groups[k] = ")
               << tileK;
    }

    SmallVector<scf::ForOp> temporalRounds;
    SmallVector<scf::ForOp> spatialRounds[3];
    for (Operation *parent = device->getParentOp(); parent;
         parent = parent->getParentOp())
      if (auto loop = dyn_cast<scf::ForOp>(parent))
        if (auto axis = loop->getAttrOfType<StringAttr>("pim.scheduling_axis")) {
          if (axis.getValue() == "reduction_k")
            temporalRounds.push_back(loop);
          else {
            int64_t number = canonicalAxis(axis.getValue());
            if (number >= 0)
              spatialRounds[number].push_back(loop);
          }
        }
    if (temporalReductions.empty() && !temporalRounds.empty())
      return device->emitError(
          "pim-verify: a temporal K scheduling round has no device-local "
          "reduction window");
    if (!temporalReductions.empty() && temporalRounds.size() != 1)
      return device->emitError(
          "pim-verify: one temporal K reduction requires exactly one "
          "reduction_k scheduling round");
    if (temporalReduction &&
        failed(verifyTemporalSchedule(temporalRounds.front(), temporalReduction,
                                      execution.getGroups().asArrayRef()[2],
                                      temporalFirst)))
      return failure();

    Value spatialFirst[3];
    for (int64_t axis = 0; axis < 3; ++axis) {
      if (spatialAxes[axis].size() != 1 || spatialRounds[axis].size() != 1)
        continue;
      int64_t groupsPerTile =
          axis == 2 ? 1 : execution.getGroups().asArrayRef()[axis];
      if (failed(verifySpatialSchedule(
              spatialRounds[axis].front(), spatialAxes[axis].front(),
              device.getBody().front().getArgument(axis), axisNames[axis],
              groupsPerTile, partitions[axis], spatialFirst[axis])))
        return failure();
    }

    int64_t used = 1;
    for (int64_t p : device.getMapping().getPartitions().asArrayRef())
      used *= p;
    if (used > profile.getNumUnits())
      return device->emitError("pim-verify: the partition counts need ")
             << used << " PIM units but the profile has "
             << profile.getNumUnits();

    llvm::DenseMap<int64_t, pim::AccessOp> accessIds;
    llvm::DenseSet<int64_t> placeIds;
    WalkResult res = device.walk([&](pim::AccessOp a) {
      if (a.getOrigin().size() != a.getSizes().size()) {
        a->emitError("pim-verify: origin rank does not match sizes rank");
        return WalkResult::interrupt();
      }
      for (auto [d, valid] : llvm::enumerate(a.getValid())) {
        int64_t size = a.getSizes()[d];
        if (std::optional<int64_t> constant = getConstantIntValue(valid)) {
          if (*constant < 0 || *constant > size) {
            a->emitError("pim-verify: constant valid extent on buffer axis ")
                << d << " is outside [0, " << size << "]";
            return WalkResult::interrupt();
          }
          continue;
        }
        if (!isClampedTo(valid, size)) {
          a->emitError("pim-verify: dynamic valid extent on buffer axis ")
              << d << " is not clamped to pim.access size " << size;
          return WalkResult::interrupt();
        }
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

    int64_t staticAllocationBytes = 0;
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
      int64_t elements = 1;
      for (auto [d, s] : llvm::enumerate(place.getSize())) {
        // mask 가 있는 축만 잘라야 한다. 없으면 도는 범위가 그대로 영역이다.
        bool masked = !access.getValid().empty() &&
                      !getConstantIntValue(access.getValid()[d]);
        if (masked && !isClamped(s))
          return place->emitError("pim-verify: the size of buffer axis ")
                 << d << " is not clamped to the buffer extent";
        std::optional<int64_t> size = getConstantIntValue(s);

        // placement가 실제 Execution Tile 하나의 resident buffer인지 hierarchy와
        // 독립적으로 맞춰 본다. access dimension을 모는 m/n loop는 해당 Tile의
        // Group 수만큼, spatial split-K는 plane 하나만큼, temporal reduction은
        // tile_k Group 수만큼 상주해야 한다.
        llvm::SmallPtrSet<Operation *, 4> drivers;
        pim::collectDrivers(access.getOrigin()[d], deviceLoops, drivers);
        // The resident K slice for this round must begin at the same window
        // start that bounds the device-local temporal reduction. Merely using
        // the round IV somewhere else in the address expression is not enough.
        if (temporalReduction &&
            drivers.contains(temporalReduction.getOperation()) &&
            !dependsOn(place.getStart()[d], temporalFirst))
          return place->emitError("pim-verify: resident buffer axis ")
                 << d
                 << " is driven by temporal K but start does not use the "
                    "current Execution Tile window start";
        if (temporalReduction &&
            drivers.contains(temporalReduction.getOperation()) &&
            !dependsOn(place.getSize()[d],
                       temporalReduction.getUpperBound()))
          return place->emitError("pim-verify: resident buffer axis ")
                 << d
                 << " is driven by temporal K but size does not use the "
                    "current local window length";
        int64_t expected = access.getSizes()[d];
        for (Operation *driver : drivers) {
          if (auto axis = driver->getAttrOfType<pim::AxisAttr>("pim.axis")) {
            StringRef name = axis.getName().getValue();
            int64_t axisNumber = canonicalAxis(name);
            if (axisNumber < 0)
              return place->emitError("pim-verify: unsupported spatial axis ")
                     << name;
            if (spatialFirst[axisNumber] &&
                !dependsOn(place.getStart()[d], spatialFirst[axisNumber]))
              return place->emitError("pim-verify: resident buffer axis ")
                     << d << " is driven by spatial " << name
                     << " but start does not use the current Execution Tile "
                        "start";
            if (spatialFirst[axisNumber] &&
                !dependsOn(place.getSize()[d],
                           spatialAxes[axisNumber].front().getUpperBound()))
              return place->emitError("pim-verify: resident buffer axis ")
                     << d << " is driven by spatial " << name
                     << " but size does not use the current local Tile "
                        "length";
            ArrayRef<int64_t> groups = execution.getGroups().asArrayRef();
            int64_t factor = name == "m"   ? groups[0]
                             : name == "n" ? groups[1]
                             : name == "k" ? 1
                                           : 0;
            if (factor < 1 ||
                expected > std::numeric_limits<int64_t>::max() / factor)
              return place->emitError(
                  "pim-verify: resident allocation extent overflows i64");
            expected *= factor;
          } else if (driver->hasAttr("pim.temporal_reduction")) {
            int64_t factor = execution.getGroups().asArrayRef()[2];
            if (expected > std::numeric_limits<int64_t>::max() / factor)
              return place->emitError(
                  "pim-verify: resident allocation extent overflows i64");
            expected *= factor;
          } else {
            return place->emitError("pim-verify: resident buffer axis ")
                   << d
                   << " is driven by a loop outside the "
                      "Block-Group-Execution Tile hierarchy";
          }
        }
        std::optional<int64_t> allocation =
            constantIndexExpr(place.getAllocationShape()[d]);
        if (!allocation)
          return place->emitError("pim-verify: resident allocation_shape[")
                 << d << "] must be a static Execution Tile extent";
        if (*allocation != expected)
          return place->emitError("pim-verify: allocation_shape[")
                 << d << "] is " << *allocation
                 << " but the Execution Tile requires " << expected;
        if (size && *size < 0)
          return place->emitError("pim-verify: negative size on buffer axis ")
                 << d;
        if (allocation && *allocation < 0)
          return place->emitError(
                     "pim-verify: negative allocation_shape on buffer axis ")
                 << d;
        if (size && allocation && *size > *allocation)
          return place->emitError("pim-verify: buffer axis ")
                 << d << " needs " << *size << " elements but only " << *allocation
                 << " are allocated per device";
        if (*allocation != 0 &&
            elements > std::numeric_limits<int64_t>::max() / *allocation) {
          return place->emitError(
              "pim-verify: allocation_shape element count overflows i64");
        } else {
          elements *= *allocation;
        }
      }
      Type elementType =
          cast<BaseMemRefType>(place.getBuffer().getType()).getElementType();
      int64_t elementBytes =
          std::max<int64_t>(1, elementType.getIntOrFloatBitWidth() / 8);
      if (elements != 0 &&
          elementBytes > std::numeric_limits<int64_t>::max() / elements)
        return place->emitError(
            "pim-verify: allocation byte count overflows i64");
      int64_t bytes = elements * elementBytes;
      if (staticAllocationBytes >
          std::numeric_limits<int64_t>::max() - bytes)
        return place->emitError(
            "pim-verify: total allocation byte count overflows i64");
      staticAllocationBytes += bytes;
    }
    if (accessIds.size() != placeIds.size())
      return device->emitError("pim-verify: ")
             << accessIds.size() << " accesses but " << placeIds.size()
             << " placements";

    for (int64_t axis = 0; axis < 3; ++axis) {
      if (spatialAxes[axis].empty() && !spatialRounds[axis].empty())
        return spatialRounds[axis].front()->emitError(
                   "pim-verify: spatial scheduling round has no device-local ")
               << axisNames[axis] << " axis";
      if (!spatialAxes[axis].empty() && spatialRounds[axis].size() != 1)
        return spatialAxes[axis].front()->emitError(
                   "pim-verify: one device-local spatial ")
               << axisNames[axis]
               << " axis requires exactly one matching scheduling round";
    }

    // pim.combine is the original semantic combiner for every partial source.
    // The sources are explicit in Common IR: reduction_k ancestors are temporal
    // windows and pim.axis="k" plus mapping.partitions[k] are spatial planes and
    // devices.  A target may choose where to perform the combine, but it must
    // not rediscover either the operation or its scope from access arithmetic.
    if (!semanticReductions.empty() &&
        semanticReductions.front().getNumResults() != 0) {
      bool sawPartialStore = false;
      bool missingCombine = false;
      bool mismatchedCombine = false;
      bool hasPreCombineEpilogue = false;
      StringAttr reductionCombiner = semanticReductions.front()
                                         ->getAttrOfType<StringAttr>(
                                             "pim.reduction");
      device.walk([&](pim::StoreOp store) {
        if (!dependsOn(store.getValue(), semanticReductions.front().getResults()))
          return;
        sawPartialStore = true;
        hasPreCombineEpilogue |= !llvm::is_contained(
            semanticReductions.front().getResults(), store.getValue());
        StringAttr combine =
            store->getAttrOfType<StringAttr>("pim.combine");
        missingCombine |= !combine;
        mismatchedCombine |=
            combine && combine.getValue() != reductionCombiner.getValue();
      });
      if (!sawPartialStore)
        return device->emitError(
            "pim-verify: semantic K reduction has no partial output store");
      if (hasPreCombineEpilogue)
        return device->emitError(
            "pim-verify: an epilogue consumes a K partial result "
            "before it is combined");
      if (missingCombine)
        return device->emitError(
            "pim-verify: every K partial store needs an explicit "
            "pim.combine");
      if (mismatchedCombine)
        return device->emitError(
            "pim-verify: K partial store combiner does not match the "
            "semantic pim.reduction combiner");
    }

    // A combine marker without a semantic reduction would force the target to
    // guess what value is partial.  Reject it in Common IR.
    if (semanticReductions.empty()) {
      pim::StoreOp unexplainedCombine;
      device.walk([&](pim::StoreOp store) {
        if (!unexplainedCombine && store->hasAttr("pim.combine"))
          unexplainedCombine = store;
      });
      if (unexplainedCombine)
        return unexplainedCombine->emitError(
            "pim-verify: pim.combine requires one semantic pim.reduction");
    }
    std::optional<int64_t> unitLimit = pim::unitCapacityLimitBytes(profile);
    if (unitLimit && *unitLimit > 0 && staticAllocationBytes > *unitLimit)
      return device->emitError("pim-verify: the execution tile needs ")
             << staticAllocationBytes << " byte but one PIM unit is limited to "
             << *unitLimit << " byte";
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPIMVerifyPass() {
  return std::make_unique<PIMVerifyPass>();
}
