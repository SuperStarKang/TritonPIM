// AiMX-only resource lifetime implementation. No computation is inferred here.
#include "aimx_runtime.h"

#include <limits.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct DeviceState {
  std::mutex mutex;
  int fpga = -1;
  std::string plan;
  std::vector<aimx_resident_buffer> buffers;
  aimx_resident_context context{};
};
struct Registry {
  std::mutex mutex;
  std::map<int, std::unique_ptr<DeviceState>> devices;
  pid_t process = getpid();
};
Registry &registry() {
  // Vendor allocator initialization lazily registers its own atexit cleanup.
  // Its ordering relative to C++ destructors/driver cleanup is not our contract.
  // Explicit reset/plan replacement frees live allocations; at process exit the
  // vendor owns device-memory cleanup and the OS owns host mappings/metadata.
  static Registry *value = new Registry;
  return *value;
}
void checkProcess(Registry &r) {
  if (r.process != getpid()) {
    fprintf(stderr, "AIMX] persistent state cannot be inherited across fork\n");
    // Do not run inherited vendor atexit handlers against the parent's devices.
    _exit(1);
  }
}
void clear(DeviceState &state) {
  for (auto it = state.buffers.rbegin(); it != state.buffers.rend(); ++it) {
    if (it->va)
      pim_free((void *)it->va, state.fpga);
    if (it->host)
      munmap(it->host, (size_t)it->bytes);
  }
  state.buffers.clear();
  state.plan.clear();
  state.context = {};
}
} // namespace

extern "C" int aimx_runtime_init(int base_fpga, int num_fpga) {
  Registry &r = registry();
  checkProcess(r);
  std::lock_guard<std::mutex> lock(r.mutex);
  aimx_require(base_fpga >= 0 && num_fpga > 0 && base_fpga <= INT_MAX - num_fpga,
               "invalid physical FPGA range");
  // The original vendor API initializes PL-DMA/QDMA for all physical FPGAs.
  // Call it only before the first device registration: repeated calls reopen
  // /dev/pim_addr. Plan replacement/reset keeps the registry entries alive.
  // Only the requested range below receives resident state and executes work.
  int available = r.devices.empty() ? init_mpim_drv() : totalDevNum;
  aimx_require(available >= 0 && base_fpga <= available - num_fpga,
               "AiMX initialization failed or requested FPGA range is unavailable");
  for (int fpga = base_fpga; fpga < base_fpga + num_fpga; ++fpga) {
    auto &entry = r.devices[fpga];
    if (!entry) {
      entry = std::make_unique<DeviceState>();
      entry->fpga = fpga;
    }
  }
  return available;
}

extern "C" aimx_resident_context *aimx_resident_acquire(
    const char *plan, int fpga, size_t buffers) {
  Registry &r = registry();
  checkProcess(r);
  DeviceState *state;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    auto found = r.devices.find(fpga);
    aimx_require(found != r.devices.end(), "acquiring an uninitialized FPGA");
    state = found->second.get();
  }
  state->mutex.lock();
  aimx_require(plan && buffers > 0, "invalid resident plan");
  if (state->plan != plan || state->buffers.size() != buffers) {
    // One cached plan per physical FPGA bounds retention during autotuning.
    // Replacement invalidates buffers AND descriptors as one transaction.
    clear(*state);
    state->plan = plan;
    state->buffers.resize(buffers);
    state->context = {state->buffers.data(), buffers, false, state};
  }
  return &state->context;
}

extern "C" aimx_resident_buffer *aimx_resident_alloc(
    aimx_resident_context *context, size_t slot, int64_t bytes,
    int64_t alignment, int kind) {
  aimx_require(context && slot < context->count && bytes > 0 &&
                   bytes <= 1023ll * 2 * 1024 * 1024 && alignment > 0 &&
                   (alignment & (alignment - 1)) == 0 && kind >= 0 && kind <= 2,
               "invalid resident allocation");
  auto &state = *static_cast<DeviceState *>(context->owner);
  auto &buffer = context->buffers[slot];
  if (buffer.va) {
    aimx_require(buffer.bytes == bytes && buffer.alignment == alignment && buffer.kind == kind,
                 "resident allocation signature changed without a new plan");
    return &buffer;
  }
  aimx_require(!context->ready, "adding a buffer to an already-built resident plan");
  buffer.bytes = bytes;
  buffer.alignment = alignment;
  buffer.kind = kind;
  // Own PMM explicitly. Vendor private_pool.c keys its cached PMM by thread ID,
  // which is not a stable ownership key for a persistent physical-device plan.
  buffer.va = (uint64_t)pim_malloc(bytes, state.fpga, kind == 2 ? 7 : 0);
  aimx_require(buffer.va != 0, "resident PIM allocation failed");
  buffer.pa = VA2PA(buffer.va, state.fpga);
  aimx_require(buffer.pa % (uint64_t)alignment == 0 &&
                   buffer.pa < (1ull << 33) && (uint64_t)bytes <= (1ull << 33) - buffer.pa,
               "resident physical address is misaligned or exceeds descriptor address width");
  // Base PA + offset is legal only for a physically linear allocation. Check
  // page boundaries once at allocation, never in the steady-state data path.
  long page = sysconf(_SC_PAGESIZE);
  aimx_require(page > 0, "cannot determine host page size");
  uint64_t offset = (uint64_t)page - buffer.va % (uint64_t)page;
  for (; offset < (uint64_t)bytes; offset += (uint64_t)page)
    aimx_require(VA2PA(buffer.va + offset, state.fpga) == buffer.pa + offset,
                 "resident allocation is not physically linear; segmented descriptors are required");
  aimx_require(VA2PA(buffer.va + bytes - 1, state.fpga) == buffer.pa + bytes - 1,
               "resident allocation ends in a non-linear physical range");
  for (size_t other = 0; other < context->count; ++other) {
    const auto &existing = context->buffers[other];
    if (other == slot || !existing.va)
      continue;
    aimx_require(buffer.pa + bytes <= existing.pa || existing.pa + existing.bytes <= buffer.pa,
                 "resident physical allocations overlap");
  }
  if (kind != 2) {
    buffer.host = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    aimx_require(buffer.host != MAP_FAILED, "resident host allocation failed");
  }
  if (kind == 1)
    buffer.next = buffer.pa + 512;
  return &buffer;
}

extern "C" void aimx_resident_release(aimx_resident_context *context) {
  auto &state = *static_cast<DeviceState *>(context->owner);
  state.mutex.unlock();
}

extern "C" void aimx_resident_reset() {
  Registry &r = registry();
  checkProcess(r);
  std::lock_guard<std::mutex> registryLock(r.mutex);
  for (auto &entry : r.devices) {
    std::lock_guard<std::mutex> lock(entry.second->mutex);
    clear(*entry.second);
  }
}

// ── Model-session contexts ───────────────────────────────────────────────────
// PIMSession owns these. Unlike aimx_resident_acquire there is no per-FPGA plan
// key, so contexts of different stages coexist on one FPGA until released.

extern "C" int aimx_model_open() {
  Registry &r = registry();
  checkProcess(r);
  std::lock_guard<std::mutex> lock(r.mutex);
  int available = r.devices.empty() ? init_mpim_drv() : totalDevNum;
  aimx_require(available > 0, "AiMX initialization failed or no FPGA is visible");
  for (int fpga = 0; fpga < available; ++fpga) {
    auto &entry = r.devices[fpga];
    if (!entry) {
      entry = std::make_unique<DeviceState>();
      entry->fpga = fpga;
    }
  }
  return available;
}

extern "C" aimx_resident_context *aimx_model_create(int fpga, size_t slots) {
  Registry &r = registry();
  checkProcess(r);
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    aimx_require(r.devices.count(fpga) == 1, "model context on an unopened FPGA");
  }
  aimx_require(slots > 0, "a model context needs at least one slot");
  auto *state = new DeviceState;
  state->fpga = fpga;
  state->plan = "model";
  state->buffers.resize(slots);
  state->context = {state->buffers.data(), slots, false, state};
  return &state->context;
}

extern "C" void aimx_model_destroy(aimx_resident_context *context) {
  auto *state = static_cast<DeviceState *>(context->owner);
  clear(*state);
  delete state;
}

extern "C" int64_t aimx_model_live_bytes(aimx_resident_context *context) {
  int64_t bytes = 0;
  for (size_t slot = 0; slot < context->count; ++slot)
    if (context->buffers[slot].va)
      bytes += context->buffers[slot].bytes;
  return bytes;
}
