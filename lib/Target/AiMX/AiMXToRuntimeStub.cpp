#include "triton-shared/Target/AiMX/AiMXTranslations.h"
#include "triton-shared/Dialect/PIM/IR/PIMDialect.h"
#include "triton-shared/Dialect/PIMTarget/AiMX/IR/AiMXDialect.h"

#include "CEmitter.h"
#include "DescriptorReuse.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>

using namespace mlir;
using namespace mlir::aimx_codegen;

namespace {

// PIM DRAM 할당 하나에 대해 stub 이 쥐고 있는 것들.
struct Buffer {
  std::string va, pa, host;
  int64_t bytes = 0;
  aimx::AllocKind kind = aimx::AllocKind::Data;
  // kind = descriptor 일 때만 쓴다. chain 을 채워 나가는 상태다.
  std::string idx, next;
  // persistent plan 의 resident slot 번호. 아니면 -1.
  int slot = -1;
};

// host C++ 진입점을 낸다. 참조 구현의 함수(_matmul_silent 등)는 부르지 않는다 —
// descriptor 는 aimx-to-descriptors 가 낸 program 이 직접 만들고, 여기서는 할당과
// 자료 이동과 실행만 한다.
struct StubEmitter {
  CEmitter e;
  // model: PIMSession 이 소유한 context 와 instance 표를 인자로 받는 model stage stub.
  bool model;
  StubEmitter(raw_ostream &os, bool model) : e(os), model(model) {}
  raw_ostream &os() { return e.os; }

  func::FuncOp hostFunc;
  // model stage 가 Python 에 내보내는 표. slot 마다 크기와 그 slot 을 채우거나 비우는 host 인자.
  SmallVector<int64_t> slotBytes;
  SmallVector<int64_t> slotInput, slotOutput;
  // slot 마다 aimx::AllocKind (0 data, 1 descriptor, 2 PMM). adapter 가 memory table 종류를 정한다.
  SmallVector<int64_t> slotKinds;
  // slot 을 채우는 stage 의 group_layout 6칸과 transpose. 없으면 0. adapter 가 key/value cache 를
  // 이 배치로 직접 써 넣는다.
  SmallVector<std::array<int64_t, 7>> slotLayouts;
  int64_t modelDevices = 0;
  unsigned modelPlans = 0;

  llvm::DenseMap<Value, Buffer> buffers;
  // 놓는 순서를 잡은 순서의 역순으로 두려면 순서가 필요하다. DenseMap 은 순서가 없다.
  SmallVector<Value> allocated;
  Buffer pmm;
  bool havePMM = false;
  bool persistentBody = false;
  unsigned residentSite = 0, residentSlot = 0;
  // shared_over가 있는 stage의 host pool과 slot별 once flag. foreach_device 앞에서 잡는다.
  struct SharedPool {
    std::string pool, once;
    int64_t slots, bytes;
  };
  llvm::DenseMap<Operation *, SharedPool> sharedPools;

  // ── PIM_TIMING (임시) ──────────────────────────────────────────────────────
  // 어느 단계가 오래 걸리는지 보려고 넣은 계측이다. 성능 판단이 끝나면 지운다.
  // 지우려면 PIM_TIMING 표시가 붙은 곳(이 블록, run 의 memset 한 줄, emitOp 의 mark
  // 호출, emitForeachDevice 의 시계 시작과 mark 호출, emitConstruction / emitStage /
  // emitUnstage 안의 mark 호출)과 driver.py 의 같은 표시를 지운다.
  //
  // UPMEM 과 달리 여기서는 **누적**한다. stage / invoke / unstage 가 tile loop 안에서
  // 여러 번 돌기 때문에 덮어쓰면 마지막 tile 한 번만 남는다.
  //
  // 칸 순서는 driver.py 의 _STAGE_NAMES["aimx"] 와 같아야 한다. layout 변환(host 조립)과
  // DMA 를 다른 칸에 둔다: stage = layout_in + h2d, unstage = d2h + layout_out,
  // descriptor 는 build(생성) + desc_h2d(업로드).
  enum TimingSlot {
    kAlloc, kBuild, kDescH2D, kLayoutIn, kH2D, kInvoke, kD2H, kLayoutOut, kTimingSlots
  };

  void emitTimingPreamble() {
    os() << "// PIM_TIMING: 단계별 누적 시간(ms). launcher 가 이름으로 읽어 간다.\n"
         << "// PIM Device 를 여럿 쓰면 스레드마다 자기 시계를 들고 한 배열에 더하므로\n"
         << "// 합계가 wall-clock 을 넘을 수 있다 — device 시간의 총합이라는 뜻이다.\n"
         << "#include <time.h>\n"
         << "extern \"C\" double pim_timing_ms[" << kTimingSlots << "];\n"
         << "double pim_timing_ms[" << kTimingSlots << "];\n"
         << "static double _pim_now() {\n"
         << "  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);\n"
         << "  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;\n"
         << "}\n"
         << "static __thread double _pim_last;\n"
         << "static void _pim_add(int slot) {\n"
         << "  double t = _pim_now(), d = t - _pim_last; _pim_last = t;\n"
         << "  _Pragma(\"omp atomic\") pim_timing_ms[slot] += d;\n"
         << "}\n"
         // invoke 수를 host 쪽에서 다시 세면 tile 산술과 어긋날 수 있다. 세는 쪽이 실제로
         // 부르는 쪽이어야 한다.
         << "extern \"C\" uint64_t pim_invoke_count;\n"
         << "uint64_t pim_invoke_count;\n\n";
  }

  void mark(int slot, unsigned depth) {
    e.indent(depth);
    os() << "_pim_add(" << slot << ");  // PIM_TIMING\n";
  }
  // ── PIM_TIMING 끝 ─────────────────────────────────────────────────────────

  LogicalResult run(ModuleOp module) {
    WalkResult legality = module.walk([](Operation *op) {
      StringRef dialect = op->getName().getDialectNamespace();
      if (dialect == "pim" || dialect == "linalg" || dialect == "tensor") {
        op->emitError("aimx-to-runtime-stub: incomplete lowering; run convert-pim-to-aimx before translation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (legality.wasInterrupted())
      return failure();
    func::FuncOp host;
    unsigned hostCount = 0;
    module.walk([&](func::FuncOp f) {
      ++hostCount;
      if (!host)
        host = f;
    });
    if (hostCount != 1 || host.isExternal())
      return module.emitError("aimx-to-runtime-stub: expected exactly one defined host function");

    uint64_t requiredDevices = 0;
    host.walk([&](aimx::AcquireOp acquire) {
      requiredDevices = std::max(requiredDevices, acquire.getNumDevices());
    });
    if (requiredDevices == 0 || requiredDevices > INT32_MAX)
      return host.emitError("aimx-to-runtime-stub: invalid acquired device count");
    hostFunc = host;
    os() << "// generated by triton-shared-translate ("
         << (model ? "aimx-to-model-stub" : "aimx-to-runtime-stub") << ")\n"
         << "#include \"aimx_runtime.h\"\n"
         << "#include \"aimx_descriptors.h\"\n";
    if (model)
      os() << "#include <memory>\n";
    os() << "\n";
    emitTimingPreamble();
    for (auto global : module.getOps<memref::GlobalOp>())
      if (failed(emitGlobal(global))) return failure();

    os() << "extern \"C\" void " << host.getName() << "(";
    if (model)
      os() << "aimx_model_call *_model, ";
    for (auto [i, arg] : llvm::enumerate(host.getArguments())) {
      if (i)
        os() << ", ";
      std::string n = "a" + std::to_string(i);
      e.bind(arg, n);
      if (auto mr = dyn_cast<BaseMemRefType>(arg.getType())) {
        std::string ct = CEmitter::scalarType(mr.getElementType());
        if (ct.empty())
          return host->emitError("aimx-to-runtime-stub: unsupported buffer "
                                 "element type");
        os() << ct << "* " << n;
      } else {
        std::string ct = CEmitter::scalarType(arg.getType());
        if (ct.empty())
          return host->emitError("aimx-to-runtime-stub: unsupported argument "
                                 "type");
        os() << ct << " " << n;
      }
    }
    os() << ") {\n";
    os() << "  static std::mutex _host_mutex;\n"
         << "  std::lock_guard<std::mutex> _host_lock(_host_mutex);\n";
    // PIM_TIMING: 시계는 device loop 안에서 스레드마다 따로 시작한다. init_pim_drv 는
    // 어느 칸에도 들어가지 않는다 — 첫 launch 말고는 싸고, 스레드에 속하지 않는다.
    os() << "  memset(pim_timing_ms, 0, sizeof(pim_timing_ms));  // PIM_TIMING\n";
    os() << "  pim_invoke_count = 0;  // PIM_TIMING\n";
    // 반환값이 이 머신의 FPGA 개수다. 이것을 넘는 fpga_id 는 런타임 배열 밖을 읽는다 —
    // 오류를 돌려주지 않고 쓰레기 fd 로 ioctl 을 걸므로 여기서 막는다.
    if (!model) {
      os() << "  aimx_runtime_init(AIMX_BASE_FPGA, "
           << requiredDevices << ");\n";
      if (failed(emitBlock(host.getBody().front(), 1)))
        return failure();
      os() << "}\n";
      return success();
    }
    // model stage: FPGA 는 PIMSession 이 열었다. instance 를 batch_size 개씩 돌린다.
    os() << "  for (_model->batch_base = 0; _model->batch_base < _model->instances; "
            "_model->batch_base += _model->batch_size) {\n";
    if (failed(emitBlock(host.getBody().front(), 2)))
      return failure();
    os() << "  }\n}\n";
    if (modelPlans != 1)
      return host.emitError("aimx-to-model-stub: a model stage has exactly one device plan");
    emitModelTables(host.getNumArguments());
    return success();
  }

  void emitModelTables(unsigned arguments) {
    auto list = [&](ArrayRef<int64_t> values) {
      for (auto [i, value] : llvm::enumerate(values))
        os() << (i ? ", " : "") << value;
    };
    os() << "\n// model stage 표. Python adapter 가 slot 역할과 상주 크기를 여기서 읽는다.\n"
         << "extern \"C\" const uint32_t pim_model_slot_count = " << slotBytes.size() << ";\n"
         << "extern \"C\" const uint32_t pim_model_argument_count = " << arguments << ";\n"
         << "extern \"C\" const uint32_t pim_model_device_count = " << modelDevices << ";\n"
         << "extern \"C\" const int64_t pim_model_slot_bytes[" << slotBytes.size() << "] = {";
    list(slotBytes);
    os() << "};\n// slot 을 stage 로 채우는 host 인자 번호, 없으면 -1\n"
         << "extern \"C\" const int64_t pim_model_slot_input[" << slotBytes.size() << "] = {";
    list(slotInput);
    os() << "};\n// slot 을 unstage 로 비우는 host 인자 번호, 없으면 -1\n"
         << "extern \"C\" const int64_t pim_model_slot_output[" << slotBytes.size() << "] = {";
    list(slotOutput);
    os() << "};\n// slot 의 aimx::AllocKind: 0 data, 1 descriptor chain, 2 PMM\n"
         << "extern \"C\" const int64_t pim_model_slot_kind[" << slotBytes.size() << "] = {";
    list(slotKinds);
    os() << "};\n// slot 마다 [useful_rows, useful_cols, physical_rows, physical_cols, row_groups, "
         << "col_groups, transpose] (aimx.stage group_layout), stage 가 없으면 0\n"
         << "extern \"C\" const int64_t pim_model_slot_layout[" << slotBytes.size() * 7 << "] = {";
    for (auto [i, layout] : llvm::enumerate(slotLayouts)) {
      os() << (i ? ", " : "");
      list(layout);
    }
    os() << "};\n";
  }

  // model stage 에서 host 인자 번호. 함수 인자가 아니면 -1.
  int64_t hostArgument(Value value) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument || argument.getOwner()->getParentOp() != hostFunc.getOperation())
      return -1;
    return argument.getArgNumber();
  }

  LogicalResult emitGlobal(memref::GlobalOp global) {
    auto initial = global.getInitialValue();
    auto data = initial ? dyn_cast<DenseFPElementsAttr>(*initial) : DenseFPElementsAttr();
    if (!data || data.getNumElements() != 1 || !global.getConstant())
      return global.emitError("AiMX host constants require one immutable floating-point element");
    Type type = global.getType().getElementType();
    uint64_t bits = (*data.getValues<APFloat>().begin()).bitcastToAPInt().getZExtValue();
    os() << "static const " << CEmitter::scalarType(type) << " aimx_data_" << global.getSymName()
         << " = ";
    if (type.isF32()) {
      // Preserve the exact source bits without C++ type-punning or decimal rounding.
      os() << "[] { uint32_t bits = 0x";
      os().write_hex(bits);
      os() << "u; float value; memcpy(&value, &bits, sizeof(value)); return value; }()";
    } else if (type.isBF16() || type.isF16()) {
      os() << "0x";
      os().write_hex(bits);
      os() << "u";
    } else {
      return global.emitError("AiMX host constant dtype is not implemented");
    }
    os() << ";\n";
    return success();
  }

  LogicalResult emitBlock(Block &block, unsigned depth, bool patchAddresses = false) {
    for (Operation &op : block)
      if (failed(emitOp(&op, depth, patchAddresses)))
        return failure();
    return success();
  }

  LogicalResult emitOp(Operation *op, unsigned depth, bool patchAddresses) {
    if (isa<func::ReturnOp>(op) || isa<scf::YieldOp>(op))
      return success();
    if (isa<aimx::AcquireOp>(op))
      return success(); // The acquired range is initialized once at entry.
    if (auto require = dyn_cast<aimx::RequireOp>(op)) {
      e.indent(depth);
      os() << "aimx_require(" << e.name(require.getCondition()) << ", "
           << require.getMessageAttr() << ");\n";
      return success();
    }
    if (auto global = dyn_cast<memref::GetGlobalOp>(op)) {
      e.bind(global.getResult(), ("(&aimx_data_" + global.getName() + ")").str());
      return success();
    }
    if (auto region = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (!model)
        return emitConstruction(region, depth);
      // model stage 의 pack(2) 호출은 host image 만 채운다. device 에 아무것도 보내지 않아야
      // 다른 stage 가 PIM 연산을 하는 동안 background 에서 돌 수 있다 (연산과 DMA 는 겹칠 수 없다).
      // chain(3) 호출은 이 construction 만 한다.
      e.indent(depth);
      os() << "if (_model->upload != 2) {\n";
      if (failed(emitConstruction(region, depth + 1)))
        return failure();
      e.indent(depth);
      os() << "}\n";
      return success();
    }
    if (auto fd = dyn_cast<aimx::ForeachDeviceOp>(op))
      return emitForeachDevice(fd, depth);
    // PIM_TIMING: 단계 하나가 끝날 때마다 그 칸에 더한다. 칸 이름은 driver.py 에 있다.
    if (auto alloc = dyn_cast<aimx::AllocOp>(op)) {
      if (failed(emitAlloc(alloc, depth)))
        return failure();
      mark(kAlloc, depth);
      return success();
    }
    if (auto build = dyn_cast<aimx::BuildDescriptorsOp>(op)) {
      if (failed(emitBuild(build, depth, patchAddresses)))
        return failure();
      mark(kBuild, depth);
      return success();
    }
    if (auto st = dyn_cast<aimx::StageOp>(op)) {
      if (!model) {
        if (failed(emitStage(st, depth)))
          return failure();
        mark(kH2D, depth);
        return success();
      }
      // weight 는 upload(1)·pack(2) 호출에서만, activation 과 상수는 실행 호출에서만 올린다.
      // chain(3) 호출은 아무것도 stage 하지 않는다.
      int64_t argument = hostArgument(st.getSrc());
      int slot = buffers.lookup(st.getDst()).slot;
      if (slot < 0)
        return st.emitError("aimx-to-model-stub: a staged buffer must be a resident slot");
      slotInput[slot] = argument;
      if (auto layout = st.getGroupLayoutAttr()) {
        auto shape = layout.asArrayRef();
        slotLayouts[slot] = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                             st.getTranspose() ? 1 : 0};
      }
      e.indent(depth);
      // weight 인자: upload(1)·pack(2). activation 인자와 상수: 실행(0). chain(3): 둘 다 아님.
      if (argument >= 0)
        os() << "if (_model->weights[" << argument
             << "] ? (_model->upload == 1 || _model->upload == 2) : !_model->upload) {\n";
      else
        os() << "if (!_model->upload) {\n";
      if (failed(emitStage(st, depth + 1)))
        return failure();
      mark(kH2D, depth + 1);
      e.indent(depth);
      os() << "}\n";
      return success();
    }
    if (auto inv = dyn_cast<aimx::InvokeOp>(op)) {
      unsigned guarded = model ? depth + 1 : depth;
      if (model) {
        e.indent(depth);
        os() << "if (!_model->upload) {\n";
      }
      if (failed(emitInvoke(inv, guarded)))
        return failure();
      mark(kInvoke, guarded);
      if (model) {
        e.indent(depth);
        os() << "}\n";
      }
      return success();
    }
    if (auto un = dyn_cast<aimx::UnstageOp>(op)) {
      unsigned guarded = model ? depth + 1 : depth;
      if (model) {
        int slot = buffers.lookup(un.getSrc()).slot;
        if (slot < 0)
          return un.emitError("aimx-to-model-stub: an unstaged buffer must be a resident slot");
        slotOutput[slot] = hostArgument(un.getDst());
        e.indent(depth);
        os() << "if (!_model->upload) {\n";
      }
      // p_k device의 partial은 device 순서(k 좌표 0이 대입, 나머지 누산)로 host에 더한다.
      bool ordered = un->hasAttr("aimx.device_reduction");
      if (ordered) {
        e.indent(guarded);
        os() << "#pragma omp ordered\n";
      }
      if (failed(emitUnstage(un, guarded)))
        return failure();
      mark(kLayoutOut, guarded);
      if (model) {
        e.indent(depth);
        os() << "}\n";
      }
      return success();
    }
    if (auto loop = dyn_cast<scf::ForOp>(op))
      return e.emitFor(loop, depth, [&](Block &b, unsigned d) {
        return emitBlock(b, d, patchAddresses);
      });
    return e.emitArith(op, depth);
  }

  //===--------------------------------------------------------------------===//
  // device loop
  //===--------------------------------------------------------------------===//

  LogicalResult emitForeachDevice(aimx::ForeachDeviceOp fd, unsigned depth) {
    ArrayRef<int64_t> counts = fd.getCounts();
    Block &body = fd.getBody().front();
    if (persistentBody)
      return fd.emitError("aimx-to-runtime-stub: nested device plans are unsupported");
    persistentBody = fd->hasAttr("aimx.persistent");
    residentSlot = 0;
    unsigned allocationCount = 0;
    if (model) {
      if (!persistentBody)
        return fd.emitError("aimx-to-model-stub: a model stage needs a persistent device plan");
      if (modelPlans++)
        return fd.emitError("aimx-to-model-stub: a model stage has exactly one device plan");
    } else if (persistentBody) {
      auto allocations = fd.getBody().front().getOps<aimx::AllocOp>();
      allocationCount = std::distance(allocations.begin(), allocations.end());
      os() << "#ifndef AIMX_KERNEL_KEY\n"
           << "#error \"Persistent AiMX requires an immutable compiled-source AIMX_KERNEL_KEY\"\n"
           << "#endif\n";
    }
    // One host worker per device. The shared resident lease protects the entire
    // stage/invoke/unstage sequence across kernel DSOs, not just one vendor call.
    // Direct vendor calls, other processes, and the legacy transient path do
    // not participate in this ownership protocol.
    int64_t devices = 1;
    for (int64_t n : counts)
      devices *= n;
    modelDevices = devices;
    // model stage 는 batch slot 마다 다른 instance 를 돌리므로 공유 slot 도 batch slot 만큼 둔다.
    std::string batchFactor = model ? " * (size_t)_model->batch_size" : "";
    // broadcast 입력: 공유 축을 뺀 좌표 조합마다 slot 하나. slot마다 처음 도착한
    // thread 하나만 변환하고, 나머지는 그 변환이 끝난 뒤 같은 slot을 전송한다.
    // round마다 영역이 바뀌므로 once flag는 이 foreach_device마다 새로 만든다.
    SmallVector<SharedPool> pools;
    for (aimx::StageOp st : body.getOps<aimx::StageOp>()) {
      ArrayAttr shared = st.getSharedOverAttr();
      if (!shared)
        continue;
      auto alloc = st.getDst().getDefiningOp<aimx::AllocOp>();
      if (!alloc)
        return st.emitError("aimx-to-runtime-stub: shared stage needs an aimx.alloc destination");
      int64_t slots = 1;
      for (auto [d, axis] : llvm::enumerate(fd.getAxes()))
        if (!llvm::is_contained(shared, axis))
          slots *= counts[d];
      SharedPool pool{e.temp("_pool"), e.temp("_once"), slots,
                      static_cast<int64_t>(alloc.getBytes())};
      e.indent(depth);
      os() << "char *" << pool.pool << " = (char*)aimx_host_buffer((size_t)"
           << slots << batchFactor << " * " << pool.bytes << "ull);\n";
      e.indent(depth);
      if (model)
        os() << "std::unique_ptr<std::once_flag[]> " << pool.once << "(new std::once_flag[(size_t)"
             << slots << batchFactor << "]);\n";
      else
        os() << "std::once_flag " << pool.once << "[" << slots << "];\n";
      sharedPools[st] = pool;
      pools.push_back(pool);
    }
    unsigned inner;
    std::string linear;
    // p_k partial을 누산하는 unstage가 있으면 그 unstage만 반복 순서대로 실행한다.
    bool ordered = false;
    body.walk([&](aimx::UnstageOp un) { ordered |= un->hasAttr("aimx.device_reduction"); });
    std::string orderedClause = ordered ? "ordered " : "";
    if (model) {
      // job = batch slot × device. job 마다 FPGA 하나와 context 하나를 쓴다.
      e.indent(depth);
      os() << "#pragma omp parallel for " << orderedClause << "num_threads(_model->batch_size * "
           << devices << ") schedule(static)\n";
      e.indent(depth);
      os() << "for (int64_t _job = 0; _job < (int64_t)_model->batch_size * " << devices
           << "; _job++) {\n";
      inner = depth + 1;
      e.indent(inner);
      os() << "uint32_t _slot = (uint32_t)(_job / " << devices << ");\n";
      e.indent(inner);
      os() << "int64_t _u = _job % " << devices << ";\n";
      e.indent(inner);
      os() << "uint32_t _instance = _model->batch_base + _slot;\n";
      e.indent(inner);
      os() << "if (_instance >= _model->instances) continue;\n";
      int64_t stride = devices;
      for (auto [d, n] : llvm::enumerate(counts)) {
        stride /= n;
        e.indent(inner);
        os() << "int64_t " << e.name(body.getArgument(d + 1)) << " = (_u / " << stride
             << ") % " << n << ";\n";
      }
      // instance 마다 host tensor 가 다르다. 함수 인자 이름을 이 block 안에서 가린다.
      for (auto [i, arg] : llvm::enumerate(hostFunc.getArguments())) {
        auto memref = dyn_cast<BaseMemRefType>(arg.getType());
        if (!memref)
          continue;
        std::string ct = CEmitter::scalarType(memref.getElementType());
        e.indent(inner);
        os() << ct << " *" << e.name(arg) << " = (" << ct
             << "*)_model->pointers[(size_t)_instance * _model->arguments + " << i << "];\n";
      }
      e.indent(inner);
      os() << "int _fpga = _model->fpgas[_job];\n";
      e.indent(inner);
      os() << "_pim_last = _pim_now();  // PIM_TIMING\n";
      e.indent(inner);
      os() << "aimx_resident_context *_resident = _model->contexts[_job];\n";
    } else {
      if (!counts.empty()) {
        e.indent(depth);
        os() << "#pragma omp parallel for " << orderedClause << "collapse(" << counts.size()
             << ") num_threads(" << devices << ") schedule(static)\n";
      }
      for (auto [d, n] : llvm::enumerate(counts)) {
        std::string iv = e.name(body.getArgument(d + 1));
        e.indent(depth + d);
        os() << "for (int64_t " << iv << " = 0; " << iv << " < " << n << "; "
             << iv << "++) {\n";
      }
      inner = depth + counts.size();
      // 좌표를 카드 번호 하나로 편다. counts 의 곱이 aimx.acquire 의 개수다.
      linear = counts.empty() ? "0" : e.name(body.getArgument(1));
      for (unsigned d = 1, n = counts.size(); d < n; d++)
        linear = "(" + linear + ") * " + std::to_string(counts[d]) + " + " +
                 e.name(body.getArgument(d + 1));
      e.indent(inner);
      // 카드 번호. AIMX_BASE_FPGA 는 aimx_runtime.h 가 정의하고 기본값이 1 이다 —
      // 지금 동작하는 카드가 1 번이기 때문이다. PIM Device 가 여럿이면 foreach_device 가
      // 만든 좌표가 여기에 더해진다.
      os() << "int _fpga = AIMX_BASE_FPGA + (int)(" << linear << ");\n";
      // PIM_TIMING: 이 스레드의 시계를 여기서 시작한다. _pim_last 는 thread-local 이다.
      e.indent(inner);
      os() << "_pim_last = _pim_now();  // PIM_TIMING\n";
      if (persistentBody) {
        e.indent(inner);
        os() << "aimx_resident_context *_resident = aimx_resident_acquire("
             << "AIMX_KERNEL_KEY \":site" << residentSite++ << "\", _fpga, "
             << allocationCount << ");\n";
      }
    }

    if (failed(emitBlock(body, inner)))
      return failure();
    // 이 카드 몫을 다 쓴 뒤에 돌려준다. chain 이 자기 물리 주소를 싣고 있으므로
    // launch 가 끝나기 전에는 어떤 buffer 도 놓을 수 없다.
    if (persistentBody && !model) {
      e.indent(inner);
      os() << "aimx_resident_release(_resident);\n";
    } else if (!persistentBody) {
      for (Value v : llvm::reverse(allocated))
        emitFree(buffers[v], inner);
    }
    // PIM_TIMING: 놓는 것도 메모리 관리이므로 잡는 것과 같은 칸에 넣는다.
    mark(kAlloc, inner);
    buffers.clear();
    allocated.clear();
    havePMM = false;
    persistentBody = false;

    if (model) {
      e.indent(depth);
      os() << "}\n";
    } else {
      for (unsigned d = counts.size(); d > 0; d--) {
        e.indent(depth + d - 1);
        os() << "}\n";
      }
    }
    for (const SharedPool &pool : pools) {
      e.indent(depth);
      os() << "aimx_host_free(" << pool.pool << ", (size_t)" << pool.slots
           << batchFactor << " * " << pool.bytes << "ull);\n";
    }
    return success();
  }

  void emitFree(const Buffer &b, unsigned depth) {
    // PMM 은 pim_private_malloc 이 잡았으므로 짝이 되는 것으로 놓는다. host 조립본도 없다.
    if (b.kind == aimx::AllocKind::PMM) {
      e.indent(depth);
      os() << "pim_private_free((void*)" << b.va << ", _fpga);\n";
      return;
    }
    e.indent(depth);
    os() << "pim_free((void*)" << b.va << ", _fpga);\n";
    e.indent(depth);
    os() << "aimx_host_free(" << b.host << ", " << b.bytes << ");\n";
  }

  //===--------------------------------------------------------------------===//
  // 할당
  //===--------------------------------------------------------------------===//

  LogicalResult emitAlloc(aimx::AllocOp op, unsigned depth) {
    Buffer b;
    b.bytes = op.getBytes();
    b.kind = op.getKind();
    std::string n = e.name(op.getBuffer());
    if (persistentBody) {
      b.slot = residentSlot;
      if (model) {
        slotBytes.push_back(b.bytes);
        slotInput.push_back(-1);
        slotOutput.push_back(-1);
        slotKinds.push_back(static_cast<int64_t>(b.kind));
        slotLayouts.push_back({});
      }
      e.indent(depth);
      os() << "aimx_resident_buffer &" << n
           << " = *aimx_resident_alloc(_resident, " << residentSlot++ << ", "
           << b.bytes << ", " << op.getAlignment() << ", "
           << static_cast<int>(b.kind) << ");\n";
      b.va = n + ".va";
      b.pa = n + ".pa";
      b.host = n + ".host";
      b.idx = n + ".idx";
      b.next = n + ".next";
      if (b.kind == aimx::AllocKind::PMM) {
        if (havePMM)
          return op.emitError("persistent plan has more than one PMM");
        pmm = b;
        havePMM = true;
      } else if (!havePMM) {
        return op.emitError("persistent PMM allocation must come first");
      }
      buffers[op.getBuffer()] = b;
      return success();
    }
    b.va = n + "_va";
    b.pa = n + "_pa";
    b.host = n + "_h";

    if (b.kind == aimx::AllocKind::PMM) {
      if (havePMM)
        return op->emitError("aimx-to-runtime-stub: more than one pmm "
                             "allocation on one device");
      e.indent(depth);
      // 세 번째 인자는 참조 구현이 쓰는 값 그대로다 (matmul.c: pim_private_malloc).
      os() << "uint64_t " << b.va << " = (uint64_t)pim_private_malloc("
           << b.bytes << ", _fpga, 7);\n";
      e.indent(depth);
      os() << "uint64_t " << b.pa << " = VA2PA(" << b.va << ", _fpga);\n";
      e.indent(depth);
      os() << "aimx_require((" << b.pa << " % " << op.getAlignment()
           << "ull) == 0, \"the pmm region is not aligned\");\n";
      pmm = b;
      havePMM = true;
      buffers[op.getBuffer()] = b;
      allocated.push_back(op.getBuffer());
      return success();
    }

    if (!havePMM)
      return op->emitError("aimx-to-runtime-stub: the pmm allocation must come "
                           "first — pim_private_malloc can hand back an address "
                           "pim_malloc already gave out");
    e.indent(depth);
    os() << "uint64_t " << b.va << " = (uint64_t)pim_malloc(" << b.bytes
         << ", _fpga);\n";
    e.indent(depth);
    os() << "uint64_t " << b.pa << " = VA2PA(" << b.va << ", _fpga);\n";
    e.indent(depth);
    os() << "aimx_require((" << b.pa << " % " << op.getAlignment()
         << "ull) == 0, \"a PIM DRAM buffer is not aligned\");\n";
    e.indent(depth);
    os() << "aimx_require(" << b.pa << " + " << b.bytes << "ull <= " << pmm.pa
         << " || " << pmm.pa << " + " << pmm.bytes << "ull <= " << b.pa
         << ", \"a PIM DRAM buffer overlaps the PMM region\");\n";
    e.indent(depth);
    os() << "void *" << b.host << " = aimx_host_buffer(" << b.bytes << ");\n";

    if (b.kind == aimx::AllocKind::Descriptor) {
      b.idx = n + "_idx";
      b.next = n + "_next";
      e.indent(depth);
      os() << "uint32_t " << b.idx << " = 0;\n";
      e.indent(depth);
      os() << "uint64_t " << b.next << " = " << b.pa << " + "
           << aimx::kGroupBytes << "ull;\n";
    }
    buffers[op.getBuffer()] = b;
    allocated.push_back(op.getBuffer());
    return success();
  }

  //===--------------------------------------------------------------------===//
  // chain
  //===--------------------------------------------------------------------===//

  LogicalResult emitBuild(aimx::BuildDescriptorsOp op, unsigned depth, bool patchAddresses) {
    if (!op->getParentOfType<scf::ExecuteRegionOp>())
      return op.emitError("AiMX descriptor builds need a complete scf.execute_region construction boundary");
    auto it = buffers.find(op.getChain());
    if (it == buffers.end() || it->second.kind != aimx::AllocKind::Descriptor)
      return op->emitError("aimx-to-runtime-stub: the chain is not an "
                           "aimx.alloc of kind descriptor");
    const Buffer &chain = it->second;
    e.indent(depth);
    // A patch visits the same records that already passed the initial build's
    // capacity bound. It changes addresses, not chain links or record count.
    os() << chain.idx << " = " << (patchAddresses ? "aimx_patch_" : "aimx_prog_")
         << op.getProgram() << "((pim_isa_32_t*)" << chain.host << ", ";
    if (patchAddresses)
      os() << chain.idx;
    else
      os() << "(uint32_t)(" << chain.bytes << "ull / sizeof(pim_isa_32_t)), "
           << chain.idx << ", &" << chain.next << ", _fpga";
    for (Value arg : op.getArgs()) {
      if (isa<aimx::BufferType>(arg.getType())) {
        auto buffer = buffers.find(arg);
        if (buffer == buffers.end())
          return op.emitError("AiMX descriptor argument has no physical allocation");
        os() << ", " << buffer->second.pa;
      } else {
        os() << ", " << e.name(arg);
      }
    }
    os() << ");\n";

    return success();
  }

  LogicalResult emitConstruction(scf::ExecuteRegionOp region, unsigned depth) {
    if (region.getNumResults() != 0 || !region.getRegion().hasOneBlock())
      return region.emitError("AiMX chain construction requires a resultless single block");
    SmallVector<Value> chains;
    region.walk([&](aimx::BuildDescriptorsOp build) {
      if (!llvm::is_contained(chains, build.getChain()))
        chains.push_back(build.getChain());
    });
    DescriptorReusePlan plan;
    if (persistentBody) {
      auto classified = classifyDescriptorReuse(region);
      if (failed(classified)) return failure();
      plan = *classified;
    }
    std::string fullTile = "true";
    for (auto [bound, full] : plan.fullBounds) {
      if (fullTile == "true") fullTile.clear();
      else fullTile += " && ";
      fullTile += "(" + e.name(bound) + " == " + std::to_string(full) + ")";
    }
    // 'ready' denotes a full-tile chain when topology is conditional. A tail
    // overwrites the same storage, so invalidate before building it and leave
    // ready false afterward. The next full tile must build before it can reuse.
    if (!plan.fullBounds.empty()) {
      e.indent(depth);
      os() << "if (!(" << fullTile << ")) _resident->ready = false;\n";
    }
    DescriptorReuse reuse = plan.kind;
    bool guarded = reuse == DescriptorReuse::Unchanged;
    e.indent(depth);
    os() << (guarded ? "if (!_resident->ready) {\n" : "{\n");
    ++depth;
    for (Value value : chains) {
      auto found = buffers.find(value);
      if (found == buffers.end() || found->second.kind != aimx::AllocKind::Descriptor)
        return region.emitError("AiMX construction requires an allocated descriptor chain");
      const Buffer &chain = found->second;
      e.indent(depth);
      os() << chain.idx << " = 0;\n";
    }
    if (reuse == DescriptorReuse::Addresses) {
      e.indent(depth);
      os() << "if (_resident->ready) {\n";
      if (failed(emitBlock(region.getRegion().front(), depth + 1, true))) return failure();
      e.indent(depth);
      os() << "} else {\n";
      ++depth;
    }
    for (Value value : chains) {
      const Buffer &chain = buffers[value];
      e.indent(depth);
      os() << chain.next << " = " << chain.pa << " + " << aimx::kGroupBytes << "ull;\n";
    }
    if (failed(emitBlock(region.getRegion().front(), depth))) return failure();
    if (reuse == DescriptorReuse::Addresses) {
      --depth;
      e.indent(depth);
      os() << "}\n";
    }
    // 여기까지가 descriptor 생성(patch 포함), 아래가 그 업로드다.
    mark(kBuild, depth);
    // Address patches update the retained host records. The current transfer
    // API uploads one contiguous chain; this is not sparse device-side DMA.
    for (Value value : chains) {
      const Buffer &chain = buffers[value];
      e.indent(depth);
      os() << "if (" << chain.idx << " != 0)\n";
      e.indent(depth + 1);
      os() << "aimx_require(aimx_copy_to_pim((void*)" << chain.va << ", " << chain.host
           << ", sizeof(pim_isa_32_t) * (size_t)" << chain.idx
           << ", _fpga) >= 0, \"descriptor upload failed\");\n";
    }
    // 그대로 재사용하는 chain 은 올린 뒤 host record 를 다시 읽지 않으므로 page 를 돌려준다.
    // 다시 만들 때는 0 page 에 처음부터 쓴다. 주소만 고치는 chain 은 record 를 고쳐 다시 올린다.
    if (persistentBody && reuse == DescriptorReuse::Unchanged) {
      for (Value value : chains) {
        const Buffer &chain = buffers[value];
        e.indent(depth);
        os() << "aimx_require(madvise(" << chain.host << ", " << chain.bytes
             << "ull, MADV_DONTNEED) == 0, \"descriptor host release failed\");\n";
      }
    }
    if (persistentBody) {
      e.indent(depth);
      os() << "_resident->ready = " << fullTile << ";\n";
    }
    e.indent(depth - 1);
    os() << "}\n";
    mark(kDescH2D, depth - 1);
    return success();
  }

  LogicalResult emitInvoke(aimx::InvokeOp op, unsigned depth) {
    auto it = buffers.find(op.getChain());
    if (it == buffers.end() || it->second.kind != aimx::AllocKind::Descriptor)
      return op->emitError("aimx-to-runtime-stub: the chain is not an "
                           "aimx.alloc of kind descriptor");
    const Buffer &chain = it->second;
    e.indent(depth);
    os() << "if (" << chain.idx << " != 0) {\n";
    e.indent(depth + 1);
    os() << "pim_args _args;\n";
    e.indent(depth + 1);
    os() << "memset(&_args, 0, sizeof(_args));\n";
    e.indent(depth + 1);
    os() << "_args.pmm_base_addr = " << pmm.va << ";\n";
    e.indent(depth + 1);
    os() << "_args.desc_base_va = " << chain.va << ";\n";
    e.indent(depth + 1);
    os() << "_args.desc_base_pa = " << chain.pa << ";\n";
    e.indent(depth + 1);
    os() << "_args.desc_idx = " << chain.idx << " - 1;\n";
    e.indent(depth + 1);
    // 마지막 group 의 시작 주소. next 는 이미 그 다음을 가리킨다.
    os() << "_args.desc_last = " << chain.next << " - "
         << (aimx::kGroupBytes * 2) << "ull;\n";
    e.indent(depth + 1);
    os() << "aimx_require(pim_exec(&_args, _fpga) >= 0, \"pim_exec failed\");\n";
    e.indent(depth + 1);
    os() << "_Pragma(\"omp atomic\") pim_invoke_count++;  // PIM_TIMING\n";
    e.indent(depth);
    os() << "}\n";
    return success();
  }

  //===--------------------------------------------------------------------===//
  // 자료 이동
  //===--------------------------------------------------------------------===//

  // host 원소 하나 ↔ PIM 의 bfloat16 word. host 타입을 여기서 흡수한다.
  // host 가 이미 bf16 이면 옮기기만 하면 된다. f16 은 지수 폭이 달라 원시 16 bit 를
  // 그대로 옮기면 값이 완전히 달라지므로 반드시 float 를 거친다.
  // 다루지 못하는 타입은 빈 문자열이고 호출자가 진단을 낸다.
  static std::string toBF16(Type t, StringRef expr) {
    if (t.isBF16())
      return expr.str();
    if (t.isF32())
      return ("aimx_to_bf16(" + expr + ")").str();
    // f16 은 표로 간다. 산술로 하면 컴파일러가 벡터화해 줄 때만 빠르다 — aimx_runtime.h.
    if (t.isF16())
      return ("aimx_f16_bf16.t[" + expr + "]").str();
    return "";
  }
  static std::string fromBF16(Type t, StringRef word) {
    if (t.isBF16())
      return word.str();
    if (t.isF32())
      return ("aimx_from_bf16(" + word + ")").str();
    if (t.isF16())
      return ("aimx_to_fp16(aimx_from_bf16(" + word + "))").str();
    return "";
  }
  // host 원소 하나에 PIM 의 bf16 word 를 더한 결과. 덧셈은 언제나 float 에서 한다 —
  // host 가 f32 면 그것이 곧 fp32 누산이고, f16/bf16 이면 host buffer 의 정밀도가
  // 그것뿐이라 tile 마다 한 번씩 더 반올림된다. 다루는 타입은 fromBF16 과 같다.
  static std::string addBF16(Type t, StringRef dst, StringRef word) {
    if (t.isF32())
      return (dst + " + aimx_from_bf16(" + word + ")").str();
    if (t.isBF16())
      return ("aimx_to_bf16(aimx_from_bf16(" + dst + ") + aimx_from_bf16(" +
              word + "))")
          .str();
    if (t.isF16())
      return ("aimx_to_fp16(aimx_from_fp16(" + dst + ") + aimx_from_bf16(" +
              word + "))")
          .str();
    return "";
  }

  // 이 tile 의 i 번째 행이 원본에서 시작하는 선형 위치. origin 은 tile 의 시작이고
  // strides 는 원본의 축 간격이다. 행마다 한 번만 계산하고 안쪽에서는 j*stride 만
  // 더한다 — 원소마다 곱셈 둘을 하면 gcc 가 접근 패턴을 gather 로 보고 벡터화를
  // 포기한다. -O3 의 stride-1 loop versioning 도 이 모양이라야 걸린다.
  std::string hostRow(ValueRange origin, ValueRange strides, StringRef i) {
    unsigned rank = origin.size();
    std::string offset = "0";
    for (unsigned d = 0; d + 2 < rank; ++d)
      offset += " + " + e.name(origin[d]) + " * " + e.name(strides[d]);
    offset += " + (" + e.name(origin[rank - 2]) + " + " + i.str() +
              ") * " + e.name(strides[rank - 2]);
    offset += " + " + e.name(origin[rank - 1]) + " * " +
              e.name(strides[rank - 1]);
    return offset;
  }

  LogicalResult emitStage(aimx::StageOp op, unsigned depth) {
    auto it = buffers.find(op.getDst());
    if (it == buffers.end())
      return op->emitError("aimx-to-runtime-stub: stage writes a buffer that "
                           "was not allocated here");
    const Buffer &b = it->second;
    if (op.getTranspose() && !op.getGroupLayoutAttr())
      return op->emitError("aimx-to-runtime-stub: transpose is emitted only for "
                           "contiguous group layouts");
    int64_t bytes = CEmitter::elementBytes(op.getElementType());
    if (bytes != aimx::kTypeSize)
      return op->emitError("aimx-to-runtime-stub: PIM DRAM holds bfloat16, so "
                           "element_type must be 16 bit");
    Type srcElem = cast<BaseMemRefType>(op.getSrc().getType()).getElementType();
    std::string srcTy = CEmitter::scalarType(srcElem);
    if (srcTy.empty() || toBF16(srcElem, "x").empty())
      return op->emitError("aimx-to-runtime-stub: a host buffer must be bf16, "
                           "f32, or f16 to be staged");
    std::string i = e.temp("_i"), j = e.temp("_j"), sr = e.temp("_s"),
                dr = e.temp("_d"), base = e.temp("_b");
    unsigned rank = op.getSizes().size();
    std::string rows = e.name(op.getSizes()[rank - 2]);
    std::string cols = e.name(op.getSizes()[rank - 1]);
    std::string padRows = e.name(op.getPadTo()[0]), pitch = e.name(op.getPitch());
    // block group 주기 배치. 없는 축은 항등이고, 둘 다 없으면 아래에서 예전 코드를
    // 그대로 낸다 — padding 없는 조합의 생성 코드가 바뀌지 않게 한다.
    DenseI64ArrayAttr gRows = op.getGroupRowsAttr();
    DenseI64ArrayAttr gCols = op.getGroupColsAttr();

    e.indent(depth);
    os() << "{  // aimx.stage\n";
    // broadcast: 공유 slot 하나를 처음 도착한 thread만 채운다. scatter: 자기 host buffer.
    std::string host = b.host;
    unsigned pd = depth;
    auto shared = sharedPools.find(op);
    if (shared != sharedPools.end()) {
      auto fd = cast<aimx::ForeachDeviceOp>(op->getParentOp());
      Block &body = fd.getBody().front();
      std::string key = e.temp("_key");
      e.indent(depth + 1);
      os() << "int64_t " << key << " = 0;\n";
      for (auto [d, axis] : llvm::enumerate(fd.getAxes())) {
        if (llvm::is_contained(op.getSharedOverAttr(), axis))
          continue;
        e.indent(depth + 1);
        os() << key << " = " << key << " * " << fd.getCounts()[d] << " + "
             << e.name(body.getArgument(d + 1)) << ";\n";
      }
      if (model) {
        e.indent(depth + 1);
        os() << key << " += (int64_t)_slot * " << shared->second.slots << ";\n";
      }
      host = e.temp("_slot");
      e.indent(depth + 1);
      os() << "char *" << host << " = " << shared->second.pool << " + " << key
           << " * " << shared->second.bytes << "ll;\n";
      e.indent(depth + 1);
      os() << "std::call_once(" << shared->second.once << "[" << key << "], [&] {\n";
      pd = depth + 1;
    } else if (model) {
      // pack(2) 호출이면 host 조립본 대신 job 의 pack image 안 이 buffer 자리에 바로 조립한다.
      host = e.temp("_dst");
      e.indent(depth + 1);
      os() << "char *" << host << " = _model->upload == 2 ? (char*)aimx_model_pack_target(_model, "
           << "(uint32_t)_job, (void*)" << b.va << ") : (char*)" << b.host << ";\n";
    }
    e.indent(pd + 1);
    os() << "aimx_require(" << rows << " >= 0 && " << cols << " >= 0 && "
         << padRows << " > 0 && " << pitch << " > 0 && "
         << rows << " <= " << b.bytes / bytes << "ll && "
         << cols << " <= " << b.bytes / bytes
         << "ll, \"invalid staged extents\");\n";
    // tile 모양은 런타임 값이고 buffer 크기만 compile time 상수다. 그래서 검사해야 할
    // 것은 "tile 을 넘었는가" 가 아니라 **"이 buffer 를 넘었는가"** 다. k 가 compile
    // time 상한을 넘으면 여기서 잡힌다 — 조용히 잘리지 않는다.
    e.indent(pd + 1);
    os() << "aimx_require((size_t)" << padRows << " * (size_t)" << pitch << " * "
         << bytes << "u <= (size_t)" << b.bytes
         << "ull, \"the problem does not fit the PIM DRAM buffer "
         << "chosen at compile time — raise tile-k or tile-n\");\n";
    if (auto packed = op.getGroupLayoutAttr()) {
      auto shape = packed.asArrayRef();
      int64_t ur = shape[0], uc = shape[1], pr = shape[2], pc = shape[3];
      int64_t rowGroups = shape[4], colGroups = shape[5];
      std::string gr = e.temp("_gr"), gc = e.temp("_gc");
      std::string validRows = e.temp("_vr"), validCols = e.temp("_vc");
      std::string panel = e.temp("_panel");
      // group_layout은 물리 순서다. 전치 stage에서는 host 열이 물리 행, host 행이 물리 열이다.
      bool transpose = op.getTranspose();
      std::string physRows = transpose ? cols : rows;
      std::string physCols = transpose ? rows : cols;
      std::string rowStride = e.name(op.getStrides()[rank - 2]);
      std::string colStride = e.name(op.getStrides()[rank - 1]);
      e.indent(pd + 1);
      os() << "aimx_require(" << physRows << " <= " << ur * rowGroups << "ll && "
           << physCols << " <= " << uc * colGroups << "ll && " << padRows << " == "
           << pr * rowGroups << "ll && " << pitch << " == " << pc * colGroups
           << "ll, \"stage does not match the contiguous group layout\");\n";
      e.indent(pd + 1);
      os() << "uint16_t *" << base << " = (uint16_t*)" << host << ";\n";
      e.indent(pd + 1);
      os() << "for (int64_t " << gr << " = 0; " << gr << " < " << rowGroups
           << "ll; ++" << gr << ") {\n";
      e.indent(pd + 2);
      os() << "int64_t " << validRows << " = " << physRows << " - " << gr << " * " << ur << "ll;\n";
      e.indent(pd + 2);
      os() << "if (" << validRows << " < 0) " << validRows << " = 0;\n";
      e.indent(pd + 2);
      os() << "if (" << validRows << " > " << ur << "ll) " << validRows << " = " << ur << "ll;\n";
      e.indent(pd + 2);
      os() << "for (int64_t " << gc << " = 0; " << gc << " < " << colGroups
           << "ll; ++" << gc << ") {\n";
      e.indent(pd + 3);
      os() << "int64_t " << validCols << " = " << physCols << " - " << gc << " * " << uc << "ll;\n";
      e.indent(pd + 3);
      os() << "if (" << validCols << " < 0) " << validCols << " = 0;\n";
      e.indent(pd + 3);
      os() << "if (" << validCols << " > " << uc << "ll) " << validCols << " = " << uc << "ll;\n";
      e.indent(pd + 3);
      os() << "uint16_t *" << panel << " = " << base << " + (" << gr << " * "
           << colGroups << "ll + " << gc << ") * " << pr * pc << "ll;\n";
      if (!transpose) {
        e.indent(pd + 3);
        os() << "for (int64_t " << i << " = 0; " << i << " < " << validRows
             << "; ++" << i << ") {\n";
        e.indent(pd + 4);
        os() << "const " << srcTy << " *" << sr << " = " << e.name(op.getSrc()) << " + "
             << hostRow(op.getOrigin(), op.getStrides(),
                        "(" + gr + " * " + std::to_string(ur) + "ll + " + i + ")") << ";\n";
        e.indent(pd + 4);
        os() << "uint16_t *" << dr << " = " << panel << " + " << i << " * " << pc << "ll;\n";
        e.indent(pd + 4);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << validCols << "; ++" << j << ")\n";
        e.indent(pd + 5);
        os() << dr << "[" << j << "] = " << toBF16(srcElem, sr + "[(" + gc + " * " +
            std::to_string(uc) + "ll + " + j + ") * " + e.name(op.getStrides()[rank - 1]) + "]") << ";\n";
        e.indent(pd + 4);
        os() << "memset(" << dr << " + " << validCols << ", 0, (size_t)(" << pc
             << "ll - " << validCols << ") * " << bytes << "u);\n";
        e.indent(pd + 3);
        os() << "}\n";
      } else {
        // panel (k, m)에 host (m, k)를 쓴다. 어느 순서로 채워도 결과는 같으므로 host stride로
        // 순회 순서만 고른다. host 행 stride가 1이면 host가 이미 물리 순서라 연속 복사다.
        // 아니면 panel loop 뒤의 block 전치가 채운다. 여기서는 padding만 지운다.
        e.indent(pd + 3);
        os() << "for (int64_t " << i << " = 0; " << i << " < " << validRows << "; ++" << i << ")\n";
        e.indent(pd + 4);
        os() << "memset(" << panel << " + " << i << " * " << pc << "ll + " << validCols
             << ", 0, (size_t)(" << pc << "ll - " << validCols << ") * " << bytes << "u);\n";
        e.indent(pd + 3);
        os() << "if (" << rowStride << " == 1) {\n";
        e.indent(pd + 4);
        os() << "for (int64_t " << i << " = 0; " << i << " < " << validRows << "; ++" << i << ") {\n";
        e.indent(pd + 5);
        os() << "const " << srcTy << " *" << sr << " = " << e.name(op.getSrc()) << " + "
             << hostRow(op.getOrigin(), op.getStrides(), "(" + gc + " * " + std::to_string(uc) + "ll)")
             << " + (" << gr << " * " << ur << "ll + " << i << ") * " << colStride << ";\n";
        e.indent(pd + 5);
        os() << "uint16_t *" << dr << " = " << panel << " + " << i << " * " << pc << "ll;\n";
        e.indent(pd + 5);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << validCols << "; ++" << j << ")\n";
        e.indent(pd + 6);
        os() << dr << "[" << j << "] = " << toBF16(srcElem, sr + "[" + j + "]") << ";\n";
        e.indent(pd + 4);
        os() << "}\n";
        e.indent(pd + 3);
        os() << "}\n";
      }
      e.indent(pd + 3);
      os() << "memset(" << panel << " + " << validRows << " * " << pc
           << "ll, 0, (size_t)(" << pr << "ll - " << validRows << ") * "
           << pc * bytes << "ull);\n";
      e.indent(pd + 2);
      os() << "}\n";
      e.indent(pd + 1);
      os() << "}\n";
      if (transpose) {
        // host 행 stride가 1이 아니면 32×32 block으로 전치한다. host 32행을 바깥에 두고
        // K-Group과 K를 안쪽에서 끝까지 돌아, 행마다 순차로 읽게 한다(prefetch, TLB).
        // panel에는 block 행마다 64 B(한 cache line)씩 쓴다.
        std::string k0 = e.temp("_k0"), m0 = e.temp("_m0"), kn = e.temp("_kn"),
                    mn = e.temp("_mn"), block = e.temp("_blk");
        e.indent(pd + 1);
        os() << "if (" << rowStride << " != 1) {\n";
        e.indent(pd + 2);
        os() << "for (int64_t " << gc << " = 0; " << gc << " < " << colGroups
             << "ll; ++" << gc << ") {\n";
        e.indent(pd + 3);
        os() << "int64_t " << validCols << " = " << physCols << " - " << gc << " * " << uc << "ll;\n";
        e.indent(pd + 3);
        os() << "if (" << validCols << " < 0) " << validCols << " = 0;\n";
        e.indent(pd + 3);
        os() << "if (" << validCols << " > " << uc << "ll) " << validCols << " = " << uc << "ll;\n";
        e.indent(pd + 3);
        os() << "for (int64_t " << m0 << " = 0; " << m0 << " < " << validCols << "; " << m0
             << " += 32) {\n";
        e.indent(pd + 4);
        os() << "int64_t " << mn << " = " << validCols << " - " << m0 << " < 32 ? " << validCols
             << " - " << m0 << " : 32;\n";
        e.indent(pd + 4);
        os() << "for (int64_t " << gr << " = 0; " << gr << " < " << rowGroups
             << "ll; ++" << gr << ") {\n";
        e.indent(pd + 5);
        os() << "int64_t " << validRows << " = " << physRows << " - " << gr << " * " << ur << "ll;\n";
        e.indent(pd + 5);
        os() << "if (" << validRows << " < 0) " << validRows << " = 0;\n";
        e.indent(pd + 5);
        os() << "if (" << validRows << " > " << ur << "ll) " << validRows << " = " << ur << "ll;\n";
        e.indent(pd + 5);
        os() << "uint16_t *" << panel << " = " << base << " + (" << gr << " * "
             << colGroups << "ll + " << gc << ") * " << pr * pc << "ll;\n";
        e.indent(pd + 5);
        os() << "for (int64_t " << k0 << " = 0; " << k0 << " < " << validRows << "; " << k0
             << " += 32) {\n";
        e.indent(pd + 6);
        os() << "int64_t " << kn << " = " << validRows << " - " << k0 << " < 32 ? " << validRows
             << " - " << k0 << " : 32;\n";
        e.indent(pd + 6);
        os() << "uint16_t " << block << "[32][32];\n";
        e.indent(pd + 6);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << mn << "; ++" << j << ") {\n";
        e.indent(pd + 7);
        os() << "const " << srcTy << " *" << sr << " = " << e.name(op.getSrc()) << " + "
             << hostRow(op.getOrigin(), op.getStrides(),
                        "(" + gc + " * " + std::to_string(uc) + "ll + " + m0 + " + " + j + ")")
             << " + (" << gr << " * " << ur << "ll + " << k0 << ") * " << colStride << ";\n";
        e.indent(pd + 7);
        os() << "for (int64_t " << i << " = 0; " << i << " < " << kn << "; ++" << i << ")\n";
        e.indent(pd + 8);
        os() << block << "[" << i << "][" << j << "] = "
             << toBF16(srcElem, sr + "[" + i + " * " + colStride + "]") << ";\n";
        e.indent(pd + 6);
        os() << "}\n";
        e.indent(pd + 6);
        os() << "for (int64_t " << i << " = 0; " << i << " < " << kn << "; ++" << i << ")\n";
        e.indent(pd + 7);
        os() << "memcpy(" << panel << " + (" << k0 << " + " << i << ") * " << pc << "ll + " << m0
             << ", " << block << "[" << i << "], (size_t)" << mn << " * " << bytes << "u);\n";
        e.indent(pd + 5);
        os() << "}\n";
        e.indent(pd + 4);
        os() << "}\n";
        e.indent(pd + 3);
        os() << "}\n";
        e.indent(pd + 2);
        os() << "}\n";
        e.indent(pd + 1);
        os() << "}\n";
      }
    } else if (!gRows && !gCols) {
      // 아래 두 memset 의 길이가 여기서 음수가 아님을 확정한다. size_t 로 계산하므로
      // 어긋나면 조용히 buffer 밖을 지운다.
      e.indent(pd + 1);
      os() << "aimx_require(" << cols << " <= " << pitch << " && " << rows
           << " <= " << padRows
           << ", \"the staged tile is wider or taller than the padded shape\");\n";
      e.indent(pd + 1);
      os() << "uint16_t *" << base << " = (uint16_t*)" << host << ";\n";
      // padding 은 성능이 아니라 정확성 요구사항이다 — 남는 자리의 0 이 누산에 그대로
      // 더해진다. 그러나 **남는 자리만** 채우면 된다. buffer 전체를 memset 하면 B 는
      // launch 마다 수십 MiB 를 한 번 더 훑는 셈이고, padding 이 없을 때는 그것이 전부
      // 헛일이다. 아래 둘은 padding 이 없으면 길이 0 이다.
      e.indent(pd + 1);
      os() << "for (int64_t " << i << " = 0; " << i << " < " << rows << "; " << i
           << "++)\n";
      e.indent(pd + 2);
      os() << "memset(" << base << " + " << i << " * " << pitch << " + " << cols
           << ", 0, (size_t)(" << pitch << " - " << cols << ") * " << bytes
           << ");\n";
      e.indent(pd + 1);
      os() << "memset(" << base << " + " << rows << " * " << pitch
           << ", 0, (size_t)(" << padRows << " - " << rows << ") * (size_t)"
           << pitch << " * " << bytes << ");\n";
      // 행 주소를 밖으로 뺀다. 안쪽은 연속(또는 고정 stride) 접근이라 벡터화가 걸린다.
      e.indent(pd + 1);
      os() << "for (int64_t " << i << " = 0; " << i << " < " << rows << "; " << i
           << "++) {\n";
      e.indent(pd + 2);
      os() << "const " << srcTy << " *" << sr << " = " << e.name(op.getSrc())
           << " + " << hostRow(op.getOrigin(), op.getStrides(), i) << ";\n";
      e.indent(pd + 2);
      os() << "uint16_t *" << dr << " = " << base << " + " << i << " * " << pitch
           << ";\n";
      e.indent(pd + 2);
      os() << "for (int64_t " << j << " = 0; " << j << " < " << cols << "; " << j
           << "++)\n";
      e.indent(pd + 3);
      os() << dr << "[" << j << "] = "
           << toBF16(srcElem, sr + "[" + j + " * " +
                                 e.name(op.getStrides()[rank - 1]) +
                                 "]")
           << ";\n";
      e.indent(pd + 1);
      os() << "}\n";
    } else {
      // block group 경로. 논리(유효) index 가 물리 칸 (idx/useful)*phys + idx%useful
      // 로 가고, 각 phys 구간의 나머지는 0 이다 — Triton block 이 하드웨어 단위를 딱
      // 채우지 못한 어긋남이 이 0 이 된다. sizes 는 유효 단위, pad_to·pitch 는 물리
      // 단위다.
      int64_t ur = gRows ? gRows.asArrayRef()[0] : 1;
      int64_t pr = gRows ? gRows.asArrayRef()[1] : 1;
      int64_t uc = gCols ? gCols.asArrayRef()[0] : 1;
      int64_t pc = gCols ? gCols.asArrayRef()[1] : 1;
      auto physOf = [](const std::string &v, int64_t u, int64_t p) {
        return "((" + v + " + " + std::to_string(u - 1) + ") / " +
               std::to_string(u) + ") * " + std::to_string(p);
      };
      std::string physRows = gRows ? physOf(rows, ur, pr) : rows;
      std::string physCols = gCols ? physOf(cols, uc, pc) : cols;
      e.indent(pd + 1);
      os() << "aimx_require(" << physCols << " <= " << pitch << " && "
           << physRows << " <= " << padRows
           << ", \"the staged tile is wider or taller than the padded shape\");\n";
      e.indent(pd + 1);
      os() << "uint16_t *" << base << " = (uint16_t*)" << host << ";\n";
      // 유효 행이 아닌 물리 행(행 group 의 pad 자리, 유효 범위 밖)은 통째로 0 이다.
      // 그 0 이 누산에 그대로 더해지므로 정확성 요구사항이다.
      std::string r = e.temp("_r");
      e.indent(pd + 1);
      os() << "for (int64_t " << r << " = 0; " << r << " < " << padRows << "; "
           << r << "++)\n";
      e.indent(pd + 2);
      if (gRows)
        os() << "if (" << r << " % " << pr << " >= " << ur << " || (" << r
             << " / " << pr << ") * " << ur << " + " << r << " % " << pr
             << " >= " << rows << ")\n";
      else
        os() << "if (" << r << " >= " << rows << ")\n";
      e.indent(pd + 3);
      os() << "memset(" << base << " + " << r << " * " << pitch
           << ", 0, (size_t)" << pitch << " * " << bytes << ");\n";
      e.indent(pd + 1);
      os() << "for (int64_t " << i << " = 0; " << i << " < " << rows << "; " << i
           << "++) {\n";
      e.indent(pd + 2);
      os() << "const " << srcTy << " *" << sr << " = " << e.name(op.getSrc())
           << " + " << hostRow(op.getOrigin(), op.getStrides(), i) << ";\n";
      std::string rowIdx =
          gRows ? "((" + i + " / " + std::to_string(ur) + ") * " +
                      std::to_string(pr) + " + " + i + " % " +
                      std::to_string(ur) + ")"
                : i;
      e.indent(pd + 2);
      os() << "uint16_t *" << dr << " = " << base << " + " << rowIdx << " * "
           << pitch << ";\n";
      if (gCols) {
        // group 단위로 자른다 — 안쪽 loop 이 연속 접근이라야 벡터화가 걸린다.
        std::string j0 = e.temp("_g"), p0 = e.temp("_p"), w = e.temp("_w");
        e.indent(pd + 2);
        os() << "int64_t " << p0 << " = 0;\n";
        e.indent(pd + 2);
        os() << "for (int64_t " << j0 << " = 0; " << j0 << " < " << cols << "; "
             << j0 << " += " << uc << ") {\n";
        e.indent(pd + 3);
        os() << "int64_t " << w << " = " << cols << " - " << j0 << " < " << uc
             << " ? " << cols << " - " << j0 << " : " << uc << ";\n";
        e.indent(pd + 3);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << w << "; " << j
             << "++)\n";
        e.indent(pd + 4);
        os() << dr << "[" << p0 << " + " << j << "] = "
             << toBF16(srcElem, sr + "[(" + j0 + " + " + j + ") * " +
                                    e.name(op.getStrides()[rank - 1]) + "]")
             << ";\n";
        e.indent(pd + 3);
        os() << "memset(" << dr << " + " << p0 << " + " << w << ", 0, (size_t)("
             << pc << " - " << w << ") * " << bytes << ");\n";
        e.indent(pd + 3);
        os() << p0 << " += " << pc << ";\n";
        e.indent(pd + 2);
        os() << "}\n";
        e.indent(pd + 2);
        os() << "memset(" << dr << " + " << p0 << ", 0, (size_t)(" << pitch
             << " - " << p0 << ") * " << bytes << ");\n";
      } else {
        e.indent(pd + 2);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << cols << "; "
             << j << "++)\n";
        e.indent(pd + 3);
        os() << dr << "[" << j << "] = "
             << toBF16(srcElem, sr + "[" + j + " * " +
                                    e.name(op.getStrides()[rank - 1]) + "]")
             << ";\n";
        e.indent(pd + 2);
        os() << "memset(" << dr << " + " << cols << ", 0, (size_t)(" << pitch
             << " - " << cols << ") * " << bytes << ");\n";
      }
      e.indent(pd + 1);
      os() << "}\n";
    }
    if (shared != sharedPools.end()) {
      e.indent(depth + 1);
      os() << "});\n";
    }
    // 공유 slot 이면 먼저 온 thread 가 조립하는 동안 기다린 시간도 여기 들어간다.
    mark(kLayoutIn, depth + 1);
    e.indent(depth + 1);
    // model stage 의 공유 pool 은 pack 호출이면 device 대신 job 의 pack image 로 옮긴다.
    // 자기 buffer 는 pack 이면 이미 image 에 조립했으므로 보내지 않는다.
    if (model && shared != sharedPools.end())
      os() << "aimx_require(aimx_model_copy_to_pim(_model, (uint32_t)_job, (void*)" << b.va
           << ", " << host << ", (size_t)" << padRows << " * (size_t)" << pitch << " * "
           << bytes << ", _fpga) >= 0, \"input upload failed\");\n";
    else if (model)
      os() << "if (_model->upload != 2)\n" << std::string(2 * (depth + 2), ' ')
           << "aimx_require(aimx_copy_to_pim((void*)" << b.va << ", " << host
           << ", (size_t)" << padRows << " * (size_t)" << pitch << " * "
           << bytes << ", _fpga) >= 0, \"input upload failed\");\n";
    else
      os() << "aimx_require(aimx_copy_to_pim((void*)" << b.va << ", " << host
           << ", (size_t)" << padRows << " * (size_t)" << pitch << " * "
           << bytes << ", _fpga) >= 0, \"input upload failed\");\n";
    e.indent(depth);
    os() << "}\n";
    return success();
  }

  LogicalResult emitUnstage(aimx::UnstageOp op, unsigned depth) {
    auto it = buffers.find(op.getSrc());
    if (it == buffers.end())
      return op->emitError("aimx-to-runtime-stub: unstage reads a buffer that "
                           "was not allocated here");
    const Buffer &b = it->second;
    if (op.getTranspose() && !op.getGroupLayoutAttr())
      return op->emitError("aimx-to-runtime-stub: transpose is emitted only for "
                           "contiguous group layouts");
    int64_t bytes = CEmitter::elementBytes(op.getElementType());
    if (bytes != aimx::kTypeSize)
      return op->emitError("aimx-to-runtime-stub: PIM DRAM holds bfloat16, so "
                           "element_type must be 16 bit");
    Type dstElem = cast<BaseMemRefType>(op.getDst().getType()).getElementType();
    std::string dstTy = CEmitter::scalarType(dstElem);
    if (dstTy.empty() || fromBF16(dstElem, "x").empty())
      return op->emitError("aimx-to-runtime-stub: a host buffer must be bf16, "
                           "f32, or f16 to be unstaged");
    std::string i = e.temp("_i"), j = e.temp("_j"), sr = e.temp("_s"),
                dr = e.temp("_d");
    unsigned rank = op.getSizes().size();
    std::string rows = e.name(op.getSizes()[rank - 2]);
    std::string cols = e.name(op.getSizes()[rank - 1]);
    std::string pitch = e.name(op.getPitch());

    if (auto packed = op.getGroupLayoutAttr()) {
      auto shape = packed.asArrayRef();
      int64_t ur = shape[0], uc = shape[1], pr = shape[2], pc = shape[3];
      int64_t rowGroups = shape[4], colGroups = shape[5];
      int64_t panelElements = pr * pc;
      std::string j0 = e.temp("_g"), width = e.temp("_w");
      // group_layout은 물리 순서다. 전치 unstage에서는 host 행이 물리 열, host 열이 물리 행이다.
      bool transpose = op.getTranspose();
      std::string physRows = transpose ? cols : rows;
      std::string physCols = transpose ? rows : cols;
      e.indent(depth);
      os() << "{  // aimx.unstage contiguous groups\n";
      e.indent(depth + 1);
      os() << "aimx_require(" << rows << " >= 0 && " << cols << " >= 0 && "
           << physRows << " <= " << ur * rowGroups << "ll && " << physCols << " <= "
           << uc * colGroups << "ll && " << pitch << " == " << pc * colGroups
           << "ll && " << panelElements * rowGroups * colGroups * bytes << "ull <= "
           << b.bytes << "ull, \"unstage does not match the contiguous group layout\");\n";
      e.indent(depth + 1);
      os() << "aimx_require(aimx_copy_from_pim(" << b.host << ", (void*)" << b.va
           << ", " << panelElements * rowGroups * colGroups * bytes
           << "ull, _fpga) >= 0, \"output download failed\");\n";
      mark(kD2H, depth + 1);
      auto scatter = [&](unsigned d, bool accumulate) {
        e.indent(d);
        os() << "for (int64_t " << i << " = 0; " << i << " < " << rows << "; ++" << i << ") {\n";
        e.indent(d + 1);
        os() << dstTy << " *" << dr << " = " << e.name(op.getDst()) << " + "
             << hostRow(op.getOrigin(), op.getStrides(), i) << ";\n";
        e.indent(d + 1);
        os() << "for (int64_t " << j0 << " = 0; " << j0 << " < " << cols
             << "; " << j0 << " += " << uc << "ll) {\n";
        e.indent(d + 2);
        os() << "const uint16_t *" << sr << " = (const uint16_t*)" << b.host
             << " + ((" << i << " / " << ur << "ll) * " << colGroups << "ll + "
             << j0 << " / " << uc << "ll) * " << panelElements << "ll + ("
             << i << " % " << ur << "ll) * " << pc << "ll;\n";
        e.indent(d + 2);
        os() << "int64_t " << width << " = " << cols << " - " << j0 << " < " << uc
             << "ll ? " << cols << " - " << j0 << " : " << uc << "ll;\n";
        std::string lhs = dr + "[(" + j0 + " + " + j + ") * " +
                          e.name(op.getStrides()[rank - 1]) + "]";
        std::string word = sr + "[" + j + "]";
        e.indent(d + 2);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << width << "; ++" << j << ")\n";
        e.indent(d + 3);
        os() << lhs << " = " << (accumulate ? addBF16(dstElem, lhs, word) : fromBF16(dstElem, word)) << ";\n";
        e.indent(d + 1);
        os() << "}\n";
        e.indent(d);
        os() << "}\n";
      };
      // host (i, j)는 물리 (j, i)다. 출력은 한 물리 행(y)이라 host 쪽 j는 보통 1회다.
      auto scatterTransposed = [&](unsigned d, bool accumulate) {
        e.indent(d);
        os() << "for (int64_t " << i << " = 0; " << i << " < " << rows << "; ++" << i << ") {\n";
        e.indent(d + 1);
        os() << dstTy << " *" << dr << " = " << e.name(op.getDst()) << " + "
             << hostRow(op.getOrigin(), op.getStrides(), i) << ";\n";
        std::string lhs = dr + "[" + j + " * " + e.name(op.getStrides()[rank - 1]) + "]";
        std::string word = "((const uint16_t*)" + b.host + ")[((" + j + " / " +
                           std::to_string(ur) + "ll) * " + std::to_string(colGroups) + "ll + " +
                           i + " / " + std::to_string(uc) + "ll) * " +
                           std::to_string(panelElements) + "ll + (" + j + " % " +
                           std::to_string(ur) + "ll) * " + std::to_string(pc) + "ll + " + i +
                           " % " + std::to_string(uc) + "ll]";
        e.indent(d + 1);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << cols << "; ++" << j << ")\n";
        e.indent(d + 2);
        os() << lhs << " = " << (accumulate ? addBF16(dstElem, lhs, word) : fromBF16(dstElem, word)) << ";\n";
        e.indent(d);
        os() << "}\n";
      };
      e.indent(depth + 1);
      os() << "if (" << e.name(op.getAccumulate()) << ") {\n";
      if (transpose) scatterTransposed(depth + 2, true);
      else scatter(depth + 2, true);
      e.indent(depth + 1);
      os() << "} else {\n";
      if (transpose) scatterTransposed(depth + 2, false);
      else scatter(depth + 2, false);
      e.indent(depth + 1);
      os() << "}\n";
      e.indent(depth);
      os() << "}\n";
      return success();
    }

    // group_rows 가 있으면 buffer 의 행이 block group 주기로 놓여 있다 — 유효 행 i 가
    // 물리 행 (i/useful)*phys + i%useful 이고, 각 phys 구간의 나머지 행은 padding 이라
    // 읽지 않는다. 되읽는 길이도 물리 행 수다.
    DenseI64ArrayAttr gRows = op.getGroupRowsAttr();
    std::string physRows = rows;
    std::string srcRow = i;
    if (gRows) {
      int64_t ur = gRows.asArrayRef()[0], pr = gRows.asArrayRef()[1];
      physRows = "(((" + rows + " + " + std::to_string(ur - 1) + ") / " +
                 std::to_string(ur) + ") * " + std::to_string(pr) + ")";
      srcRow = "((" + i + " / " + std::to_string(ur) + ") * " +
               std::to_string(pr) + " + " + i + " % " + std::to_string(ur) +
               ")";
    }

    e.indent(depth);
    os() << "{  // aimx.unstage\n";
    e.indent(depth + 1);
    os() << "aimx_require(" << rows << " >= 0 && " << cols << " >= 0 && "
         << pitch << " > 0 && " << rows << " <= " << b.bytes / bytes
         << "ll && " << cols << " <= " << b.bytes / bytes
         << "ll, \"invalid unstaged extents\");\n";
    std::string physicalCols = cols;
    if (auto group = op.getGroupColsAttr()) {
      int64_t useful = group.asArrayRef()[0], physical = group.asArrayRef()[1];
      physicalCols = "((" + cols + " + " + std::to_string(useful - 1) + ") / " +
                     std::to_string(useful) + ") * " + std::to_string(physical);
    }
    e.indent(depth + 1);
    os() << "aimx_require(" << physicalCols << " <= " << pitch
         << ", \"unstage columns exceed the PIM row pitch\");\n";
    e.indent(depth + 1);
    os() << "aimx_require((size_t)" << physRows << " * (size_t)" << pitch
         << " * " << bytes << "u <= (size_t)" << b.bytes
         << "ull, \"unstage reads past the PIM DRAM buffer\");\n";
    // 실제로 쓴 행까지만 되읽는다. 행 group 밖의 padding 자리는 읽지 않는다.
    e.indent(depth + 1);
    os() << "aimx_require(aimx_copy_from_pim(" << b.host << ", (void*)" << b.va
         << ", (size_t)" << physRows << " * (size_t)" << pitch << " * "
         << bytes << ", _fpga) >= 0, \"output download failed\");\n";
    mark(kD2H, depth + 1);
    // stage 와 같은 이유로 행 주소를 밖으로 뺀다. group_cols 가 있으면 block group
    // 단위로 잘라 읽는다 — 유효 열 j 가 물리 칸 (j/useful)*phys + j%useful 에 있고,
    // pad 열은 계산 결과 자체가 0 이라 건너뛴다.
    DenseI64ArrayAttr gCols = op.getGroupColsAttr();
    std::string j0 = e.temp("_g"), p0 = e.temp("_p"), w = e.temp("_w");
    auto scatter = [&](unsigned d, bool accumulate) {
      e.indent(d);
      os() << "for (int64_t " << i << " = 0; " << i << " < " << rows << "; " << i
           << "++) {\n";
      e.indent(d + 1);
      os() << "const uint16_t *" << sr << " = (const uint16_t*)" << b.host
           << " + " << srcRow << " * " << pitch << ";\n";
      e.indent(d + 1);
      os() << dstTy << " *" << dr << " = " << e.name(op.getDst()) << " + "
           << hostRow(op.getOrigin(), op.getStrides(), i) << ";\n";
      if (!gCols) {
        std::string lhs = dr + "[" + j + " * " +
                          e.name(op.getStrides()[rank - 1]) + "]";
        std::string word = sr + "[" + j + "]";
        e.indent(d + 1);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << cols << "; "
             << j << "++)\n";
        e.indent(d + 2);
        os() << lhs << " = "
             << (accumulate ? addBF16(dstElem, lhs, word)
                            : fromBF16(dstElem, word))
             << ";\n";
      } else {
        int64_t uc = gCols.asArrayRef()[0], pc = gCols.asArrayRef()[1];
        std::string lhs =
            dr + "[(" + j0 + " + " + j + ") * " +
            e.name(op.getStrides()[rank - 1]) +
            "]";
        std::string word = sr + "[" + p0 + " + " + j + "]";
        e.indent(d + 1);
        os() << "int64_t " << p0 << " = 0;\n";
        e.indent(d + 1);
        os() << "for (int64_t " << j0 << " = 0; " << j0 << " < " << cols << "; "
             << j0 << " += " << uc << ") {\n";
        e.indent(d + 2);
        os() << "int64_t " << w << " = " << cols << " - " << j0 << " < " << uc
             << " ? " << cols << " - " << j0 << " : " << uc << ";\n";
        e.indent(d + 2);
        os() << "for (int64_t " << j << " = 0; " << j << " < " << w << "; " << j
             << "++)\n";
        e.indent(d + 3);
        os() << lhs << " = "
             << (accumulate ? addBF16(dstElem, lhs, word)
                            : fromBF16(dstElem, word))
             << ";\n";
        e.indent(d + 2);
        os() << p0 << " += " << pc << ";\n";
        e.indent(d + 1);
        os() << "}\n";
      }
      e.indent(d);
      os() << "}\n";
    };
    // 대입이냐 누산이냐는 이 tile 전체에 대해 한 번 정해진다. 원소마다 분기를 두면
    // 안쪽 loop 의 벡터화가 통째로 사라지므로 loop 을 두 벌 낸다.
    e.indent(depth + 1);
    os() << "if (" << e.name(op.getAccumulate()) << ") {\n";
    scatter(depth + 2, /*accumulate=*/true);
    e.indent(depth + 1);
    os() << "} else {\n";
    scatter(depth + 2, /*accumulate=*/false);
    e.indent(depth + 1);
    os() << "}\n";
    e.indent(depth);
    os() << "}\n";
    return success();
  }
};

LogicalResult translateAiMXToRuntimeStub(ModuleOp module, raw_ostream &os) {
  return StubEmitter(os, /*model=*/false).run(module);
}

LogicalResult translateAiMXToModelStub(ModuleOp module, raw_ostream &os) {
  return StubEmitter(os, /*model=*/true).run(module);
}

} // namespace

static void insertAiMXStubDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect, scf::SCFDialect,
                  aimx::AiMXDialect, pim::PIMDialect,
                  linalg::LinalgDialect, tensor::TensorDialect>();
}

void mlir::triton::registerAiMXToRuntimeStubTranslation() {
  TranslateFromMLIRRegistration reg(
      "aimx-to-runtime-stub", "emit the AiMX host driver from AiMX IR",
      translateAiMXToRuntimeStub, insertAiMXStubDialects);
}

void mlir::triton::registerAiMXToModelStubTranslation() {
  TranslateFromMLIRRegistration reg(
      "aimx-to-model-stub",
      "emit the AiMX model-session stage driver (session contexts, resident weights, FPGA batch)",
      translateAiMXToModelStub, insertAiMXStubDialects);
}
