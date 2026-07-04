#include "triton-shared/Conversion/TTSharedToPIM/TTSharedToPIM.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

// GEN_PASS_DEF가 생성하는 impl::...Base는 mlir::triton 네임스페이스 안에 있어야 한다.
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_BUILDPIMOFFLOAD
#include "triton-shared/Conversion/TTSharedToPIM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// tt.dot은 triton-to-linalg가 linalg.matmul로 낮춘다(별도 marker attribute는 붙지 않는다).
// MVP는 커널당 dot 1개(단일 tile)를 가정하므로 첫 linalg.matmul을 그대로 집는다.
static linalg::MatmulOp findMatmul(func::FuncOp func) {
  linalg::MatmulOp found;
  func.walk([&](linalg::MatmulOp op) {
    found = op;
    return WalkResult::interrupt();
  });
  return found;
}

// dynamic memref(src)에서 static 크기 [size0,size1]의 타일을 dynamic offset [off0,off1]로 떠낸다.
// strides는 1(연속 타일). 결과 타입은 빌더가 src 레이아웃으로부터 추론한다(strided<[?,1],offset:?>).
static Value makeTileSubview(OpBuilder &b, Location loc, Value src, Value off0,
                             Value off1, int64_t size0, int64_t size1) {
  SmallVector<OpFoldResult, 2> offsets{off0, off1};
  SmallVector<OpFoldResult, 2> sizes{b.getIndexAttr(size0), b.getIndexAttr(size1)};
  SmallVector<OpFoldResult, 2> strides{b.getIndexAttr(1), b.getIndexAttr(1)};
  return b.create<memref::SubViewOp>(loc, src, offsets, sizes, strides);
}

// pim.compute의 region 본문을 채운다 (faithful-body, 한 유닛의 로컬 슬라이스 계산).
//
// block arg %a=[Mu,K], %b=[K,Nu], %c=[Mu,Nu]. Mu/Nu/K는 런타임(host scatter가 결정)이라
// block arg의 dim에서 읽는다. BLOCK_M/N/K는 Triton constexpr이므로 루프 step·누산기 크기로
// 상수로 박힌다.
//
// 구조 = device.c가 만드는 루프 nest와 1:1: tm/tn(복원된 grid) → BLOCK_M×BLOCK_N 누산기 →
// K-loop(BLOCK_K 청크) in-place 누적 → 누산기를 C 타일로 copy. program_id 전역 오프셋은
// 본문에서 사라지고 tm/tn 루프로 흡수된다(분배층으로 이동, DESIGN 4.1 ②).
// accElemTy = 누산기/출력(C) element type. 입력(A/B) type은 block arg(%a/%b)에서 추론되므로
// (subview가 source memref 타입을 따름) 별도 인자 없이 mixed(int8 입력 + int32 누적)도 표현된다.
static void buildGemmBody(OpBuilder &b, Location loc, Block *body, int64_t blockM,
                          int64_t blockN, int64_t blockK, Type accElemTy) {
  b.setInsertionPointToStart(body);
  Value a = body->getArgument(0);
  Value bMat = body->getArgument(1);
  Value c = body->getArgument(2);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value stepM = b.create<arith::ConstantIndexOp>(loc, blockM);
  Value stepN = b.create<arith::ConstantIndexOp>(loc, blockN);
  Value stepK = b.create<arith::ConstantIndexOp>(loc, blockK);
  Value zero = b.create<arith::ConstantOp>(loc, cast<TypedAttr>(b.getZeroAttr(accElemTy)));

  // 유닛이 맡은 로컬 슬라이스 크기(런타임). %a=[Mu,K], %c=[Mu,Nu].
  Value mu = b.create<memref::DimOp>(loc, a, c0);
  Value kDim = b.create<memref::DimOp>(loc, a, c1);
  Value nu = b.create<memref::DimOp>(loc, c, c1);

  auto accTy = MemRefType::get({blockM, blockN}, accElemTy);

  // grid(M) 복원: pid_m*BLOCK_M 전역 오프셋이 이 루프로 들어옴.
  b.create<scf::ForOp>(
      loc, c0, mu, stepM, ValueRange{},
      [&](OpBuilder &nb, Location lm, Value tm, ValueRange) {
        // grid(N) 복원: pid_n*BLOCK_N 전역 오프셋이 이 루프로.
        nb.create<scf::ForOp>(
            lm, c0, nu, stepN, ValueRange{},
            [&](OpBuilder &nb2, Location ln, Value tn, ValueRange) {
              // BLOCK_M×BLOCK_N 누산기(device_local). K-loop 동안 in-place 누적.
              Value acc = nb2.create<memref::AllocOp>(ln, accTy);
              nb2.create<linalg::FillOp>(ln, ValueRange{zero}, ValueRange{acc});
              // K-loop 보존: BLOCK_K 청크를 스트리밍하며 누적.
              nb2.create<scf::ForOp>(
                  ln, c0, kDim, stepK, ValueRange{},
                  [&](OpBuilder &nb3, Location lk, Value kc, ValueRange) {
                    Value at = makeTileSubview(nb3, lk, a, tm, kc, blockM, blockK);
                    Value bt = makeTileSubview(nb3, lk, bMat, kc, tn, blockK, blockN);
                    // linalg.matmul(memref)은 outs에 in-place 누적(C += A*B).
                    auto mm = nb3.create<linalg::MatmulOp>(lk, ValueRange{at, bt},
                                                           ValueRange{acc});
                    // 본문 op별 class 태그(DESIGN 10.6). 재생성한 gemm nest도 동일 계약을 따른다 —
                    // generic 경로가 classifyComputeOp로 다는 태그와 같은 의미다.
                    mm->setAttr("pim.class", nb3.getStringAttr("gemm"));
                    nb3.create<scf::YieldOp>(lk);
                  });
              // 누산기를 이 유닛의 C 타일로 기록.
              Value cv = makeTileSubview(nb2, ln, c, tm, tn, blockM, blockN);
              nb2.create<memref::CopyOp>(ln, acc, cv);
              nb2.create<scf::YieldOp>(ln);
            });
        nb.create<scf::YieldOp>(lm);
      });
}

// split factor로부터 vendor-neutral mapping_policy를 정한다(DESIGN 11.4). split_k>1이면
// reduction(reduce 존재)이 특징이라 reduction_k_partitioned로 표기한다(출력 분배와 병행 가능).
static std::string mappingPolicy(unsigned splitM, unsigned splitN, unsigned splitK) {
  if (splitK > 1)
    return "reduction_k_partitioned";
  if (splitM > 1 && splitN > 1)
    return "output_2d_partitioned";
  if (splitM > 1)
    return "output_m_partitioned";
  if (splitN > 1)
    return "output_n_partitioned";
  return "none";
}

// 계산 op을 basic-op vocabulary class로 분류한다(DESIGN 10.6). 치환이 아니라 pim.class 태그용.
// control/glue(fill/for/subview/copy/to_tensor 등)는 "" 반환 → 태그 없음.
static StringRef classifyComputeOp(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return "gemm";
  if (isa<linalg::MatvecOp>(op))
    return "gemv";
  if (isa<linalg::TransposeOp>(op))
    return "transpose";
  if (isa<linalg::ReduceOp>(op))
    return "reduce";
  if (isa<linalg::GenericOp>(op)) {
    // reduction 루프가 있으면 reduce, 전부 parallel이면 elementwise.
    if (cast<linalg::LinalgOp>(op).getNumReductionLoops() > 0)
      return "reduce";
    return "elementwise";
  }
  return "";
}

// buffer(함수 인자)가 저장(output) 대상인지 판정한다(경계 인식). store idiom =
// reinterpret_cast(buffer)를 memref.copy의 target 또는 materialize_in_destination의 dest로 쓰는 것.
// 그 외(로드만)는 input으로 본다.
static bool isStoredBuffer(Value buf) {
  for (Operation *user : buf.getUsers()) {
    auto rc = dyn_cast<memref::ReinterpretCastOp>(user);
    if (!rc)
      continue;
    for (Operation *u2 : rc->getUsers()) {
      if (auto cp = dyn_cast<memref::CopyOp>(u2))
        if (cp.getTarget() == rc.getResult())
          return true;
      // bufferization dialect에 링크 의존을 만들지 않으려 op 이름으로 확인(dest=operand 1).
      if (u2->getName().getStringRef() ==
              "bufferization.materialize_in_destination" &&
          u2->getNumOperands() >= 2 && u2->getOperand(1) == rc.getResult())
        return true;
    }
  }
  return false;
}

struct BuildPIMOffloadPass
    : public mlir::triton::impl::BuildPIMOffloadBase<BuildPIMOffloadPass> {
  using BuildPIMOffloadBase::BuildPIMOffloadBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) {
      if (func.getBody().empty())
        return; // 선언만 있는 함수는 대상 아님.
      // gemm 패턴이면 로컬 슬라이스 경로(기존, 최적화 유지), 아니면 본문을 그대로 읽는
      // generic 경로 — 둘 다 정식 경로다(DESIGN 11.1, no-fallback).
      if (linalg::MatmulOp matmul = findMatmul(func))
        rewriteToFaithfulGemm(func, matmul);
      else
        rewriteGenericFaithful(func);
    });
  }

  // faithful-body 변환 (DESIGN 4.1/10.2):
  // - 본문을 erase하지 않고, memory_region/transfer/tile_plan/launch를 합성한 뒤
  //   계산 본문은 pim.compute region에 (로컬 슬라이스 기준으로) 보존한다.
  // - BLOCK_M/N/K는 matmul operand/result shape에서 읽는다 (Triton constexpr 타일 크기).
  //   (operand0 = A tile [BLOCK_M, BLOCK_K], result = C tile [BLOCK_M, BLOCK_N])
  // - 전체 M/N/K는 IR에 박지 않는다 (본문 block arg의 런타임 dim, host는 launcher에서 받음).
  // - A/B/C는 함수 인자 앞 3개로 가정.
  // - split_m/n은 pass 옵션(=compile option). vendor가 M/N축을 유닛에 분배할 때 쓴다.
  //
  // faithful-body: legalize가 이 본문을 device_func로 옮기고, UPMEMToDevice가 본문의 gemm nest에서
  // BLOCK을 읽어 device.c를 생성한다(O1.4a). device.c emission은 아직 gemm 템플릿이라 비-gemm 본문
  // 충실 번역은 O1.4b. gemm 결과는 동일(M6 유지).
  void rewriteToFaithfulGemm(func::FuncOp func, linalg::MatmulOp matmul) {
    auto lhsTy = cast<RankedTensorType>(matmul->getOperand(0).getType());
    auto resTy = cast<RankedTensorType>(matmul->getResult(0).getType());
    int64_t blockM = resTy.getDimSize(0);
    int64_t blockN = resTy.getDimSize(1);
    int64_t blockK = lhsTy.getDimSize(1);
    // 입력(A/B)과 누적/출력(C) element type을 따로 — fp32면 둘 다 f32(기존), int8 gemm이면
    // 입력 i8 + 누적 i32(mixed). vendor codegen이 이 타입들을 IR에서 읽어 C 타입을 정한다.
    Type inElemTy = lhsTy.getElementType();
    Type accElemTy = resTy.getElementType();

    Value A = func.getArgument(0);
    Value B = func.getArgument(1);
    Value C = func.getArgument(2);

    // 기존 본문을 뒤에서부터 모두 지운다 (사용처를 먼저 지워 use-def 위반 방지).
    // block 인자(A,B,C,...)는 유지된다. 보존할 "본문"은 단일-tile·program_id 주소라 그대로
    // 옮길 수 없고, 아래에서 유닛-로컬 nest로 재구성한다(program_id를 분배층으로 들어올림).
    Block &entry = func.getBody().front();
    while (!entry.empty())
      entry.back().erase();

    OpBuilder b = OpBuilder::atBlockBegin(&entry);
    Location loc = func.getLoc();
    auto regionTy = pim::RegionType::get(b.getContext());
    auto tpTy = pim::TilePlanType::get(b.getContext());
    auto egTy = pim::ExecGroupType::get(b.getContext());

    // 세 buffer 모두 host에 있다가 transfer로 device_global로 올라간다(DESIGN 10.5).
    Value aReg = b.create<pim::MemoryRegionOp>(loc, regionTy, A, "host", "row_major",
                                               "input_activation");
    Value bReg = b.create<pim::MemoryRegionOp>(loc, regionTy, B, "host", "row_major",
                                               "weight");
    Value cReg = b.create<pim::MemoryRegionOp>(loc, regionTy, C, "host", "row_major",
                                               "output");

    Value tp = b.create<pim::TilePlanOp>(loc, tpTy, blockM, blockN, blockK,
                                         /*split_m=*/(uint64_t)splitM,
                                         /*split_n=*/(uint64_t)splitN,
                                         /*split_k=*/(uint64_t)splitK);
    // 유닛 분배 정책(DESIGN 10.2): split로 mapping_policy 결정, num_units=split_m*split_n*split_k,
    // local_lanes는 option(DESIGN 16). gemm은 weight(B)가 상주 입력.
    Value eg = b.create<pim::ExecutionGroupOp>(
        loc, egTy, mappingPolicy(splitM, splitN, splitK),
        /*num_units=*/(uint64_t)splitM * (uint64_t)splitN * (uint64_t)splitK,
        /*local_lanes=*/(uint64_t)localLanes, /*stationary_data=*/"B");

    b.create<pim::TransferOp>(loc, aReg, "host", "device_global");
    b.create<pim::TransferOp>(loc, bReg, "host", "device_global");

    // pim.compute를 먼저 빈 region으로 만들고, 그 다음 entry의 나머지(transfer/launch/return)를
    // 채운다. 본문 region은 별도 builder로 채워 entry의 insertion을 흐트러뜨리지 않는다.
    auto compute = b.create<pim::ComputeOp>(loc, ValueRange{aReg, bReg},
                                            ValueRange{cReg}, eg, tp, "gemm");

    b.create<pim::TransferOp>(loc, cReg, "device_global", "host");
    b.create<pim::LaunchOp>(loc, func.getName(), vendor, /*device_id=*/0);
    b.create<func::ReturnOp>(loc);

    // region block arg는 유닛-로컬 슬라이스(dynamic memref). operand의 region 핸들과 별개.
    // %a/%b = 입력타입(inElemTy), %c = 누적/출력타입(accElemTy) — mixed dtype을 그대로 표현.
    auto kDyn = ShapedType::kDynamic;
    auto inTy = MemRefType::get({kDyn, kDyn}, inElemTy);
    auto accTy = MemRefType::get({kDyn, kDyn}, accElemTy);
    SmallVector<Location, 3> argLocs(3, loc);
    OpBuilder bodyBuilder(b.getContext());
    Block *body = bodyBuilder.createBlock(&compute.getBody(), {},
                                          {inTy, inTy, accTy}, argLocs);
    buildGemmBody(bodyBuilder, loc, body, blockM, blockN, blockK, accElemTy);
  }

  // 비-gemm 본문의 faithful 변환 (DESIGN 4.1/10.6, 1단계 simple faithful):
  // 재생성하지 않고 원본 계산 본문을 pim.compute region으로 그대로 옮긴다. host buffer 인자를
  // memory_region/transfer로 감싸(경계 인식) 본문 각 op에 pim.class 태그를 단다(분류). program_id
  // 주소는 이 단계에선 본문에 보존한다(로컬 슬라이스 승격=2단계, 인식 패턴에서만).
  void rewriteGenericFaithful(func::FuncOp func) {
    Block &entry = func.getBody().front();

    // host buffer = memref 타입 함수 인자. 스칼라(M/N/K: i32)는 제외. load/store 판정으로 역할 분류.
    // TTShared 인자는 memref<*xT>(UnrankedMemRefType)라 ranked/unranked 공통 base로 판정한다
    // (isa<MemRefType>은 ranked만 잡아 언랭크드 인자를 놓친다).
    SmallVector<Value> inBufs, outBufs;
    for (Value arg : func.getArguments()) {
      if (!isa<BaseMemRefType>(arg.getType()))
        continue;
      if (isStoredBuffer(arg))
        outBufs.push_back(arg);
      else
        inBufs.push_back(arg);
    }
    // 버퍼가 없으면 PIM 오프로드 대상이 아니다(그대로 둔다).
    if (inBufs.empty() && outBufs.empty())
      return;

    // 원본 계산 본문(터미네이터 제외)을 prologue 삽입 전에 수집한다 — 이후 region으로 옮긴다.
    SmallVector<Operation *> bodyOps;
    for (Operation &op : entry.without_terminator())
      bodyOps.push_back(&op);

    OpBuilder b = OpBuilder::atBlockBegin(&entry);
    Location loc = func.getLoc();
    auto regionTy = pim::RegionType::get(b.getContext());
    auto tpTy = pim::TilePlanType::get(b.getContext());
    auto egTy = pim::ExecGroupType::get(b.getContext());

    // 경계 인식: 각 host buffer를 memory_region으로 감싼다(입력/출력 역할 기록).
    SmallVector<Value> inRegions, outRegions;
    for (Value buf : inBufs)
      inRegions.push_back(b.create<pim::MemoryRegionOp>(
          loc, regionTy, buf, "host", "row_major", "input_activation"));
    for (Value buf : outBufs)
      outRegions.push_back(b.create<pim::MemoryRegionOp>(
          loc, regionTy, buf, "host", "row_major", "output"));

    // 분배 구조: 1단계 generic은 로컬 슬라이스를 안 하므로 BLOCK은 0(파생 안 함), SPLIT/local_lanes만
    // 기록한다. 인식 패턴(gemm 등)이 BLOCK을 채우는 건 2단계.
    Value tp = b.create<pim::TilePlanOp>(loc, tpTy, /*block_m=*/0, /*block_n=*/0,
                                         /*block_k=*/0, (uint64_t)splitM,
                                         (uint64_t)splitN, (uint64_t)splitK);
    Value eg = b.create<pim::ExecutionGroupOp>(
        loc, egTy, mappingPolicy(splitM, splitN, splitK),
        (uint64_t)splitM * (uint64_t)splitN * (uint64_t)splitK,
        (uint64_t)localLanes, /*stationary_data=*/"none");

    for (Value r : inRegions)
      b.create<pim::TransferOp>(loc, r, "host", "device_global");

    // 본문을 담을 pim.compute (variadic ins/outs). primitive는 generic으로 표기(본문 op별 세부
    // class는 아래에서 pim.class로 단다).
    auto compute = b.create<pim::ComputeOp>(loc, inRegions, outRegions, eg, tp,
                                            "generic");

    for (Value r : outRegions)
      b.create<pim::TransferOp>(loc, r, "device_global", "host");
    b.create<pim::LaunchOp>(loc, func.getName(), vendor, /*device_id=*/0);
    // 기존 return은 entry 끝에 그대로 둔다.

    // region block arg = 본문이 참조하는 buffer(전체). operand 순서(inRegions 다음 outRegions)와 맞춘다.
    SmallVector<Value> orderedBufs(inBufs.begin(), inBufs.end());
    orderedBufs.append(outBufs.begin(), outBufs.end());
    SmallVector<Type> argTys;
    SmallVector<Location> argLocs;
    for (Value buf : orderedBufs) {
      argTys.push_back(buf.getType());
      argLocs.push_back(loc);
    }
    OpBuilder rb(b.getContext());
    Block *body = rb.createBlock(&compute.getBody(), {}, argTys, argLocs);

    // 원본 계산 본문을 region으로 그대로 이동(치환 없음, 순서 보존).
    for (Operation *op : bodyOps)
      op->moveBefore(body, body->end());

    // region 안에서만 buffer 인자 사용을 block arg로 remap(밖의 memory_region은 안 건드린다).
    Region *region = &compute.getBody();
    for (unsigned i = 0; i < orderedBufs.size(); ++i) {
      Value buf = orderedBufs[i];
      Value barg = body->getArgument(i);
      buf.replaceUsesWithIf(barg, [&](OpOperand &use) {
        return region->isAncestor(use.getOwner()->getParentRegion());
      });
    }

    // 분류: 본문 각 op에 pim.class 태그(치환 아님, DESIGN 10.6).
    body->walk([&](Operation *op) {
      StringRef cls = classifyComputeOp(op);
      if (!cls.empty())
        op->setAttr("pim.class", StringAttr::get(op->getContext(), cls));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createBuildPIMOffloadPass() {
  return std::make_unique<BuildPIMOffloadPass>();
}
