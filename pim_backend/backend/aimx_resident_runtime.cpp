// AiMX-only resource lifetime implementation. No computation is inferred here.
#include "aimx_runtime.h"

#include <limits.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
// vendor multi_mmap_drv(int size)가 2 GiB에서 overflow하므로 pim_malloc 하나의 상한이다.
constexpr int64_t kMaxAllocationBytes = 1023ll * 2 * 1024 * 1024;

struct DeviceState {
  std::mutex mutex;
  int fpga = -1;
  std::string plan;
  std::vector<aimx_resident_buffer> buffers;
  aimx_resident_context context{};
  // model context만 쓴다. buffer i는 memory table 주소 va[i]에 놓이고 capacity[i]까지 쓰며,
  // host 조립본은 region host shadow의 같은 offset(host[i])이다.
  std::vector<uint64_t> placed_va;
  std::vector<int64_t> placed_capacity;
  std::vector<void *> placed_host;
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
    // region 안의 buffer는 따로 할당하지 않았으므로 region을 해제할 때 함께 사라진다.
    if (it->va && state.placed_va.empty())
      pim_free((void *)it->va, state.fpga);
    // region host shadow의 일부는 region과 함께 해제된다.
    if (it->host && state.placed_va.empty())
      munmap(it->host, (size_t)it->bytes);
  }
  state.buffers.clear();
  state.plan.clear();
  state.context = {};
}

// host 조립 buffer cache (aimx_runtime.h). thread마다 칸을 두어 get/put에는 lock이 없고,
// 모든 thread의 칸을 비울 수 있도록 처음 쓸 때 전역 목록에 등록한다.
constexpr int kHostBufferSlots = 8;
struct HostBufferCache {
  void *ptr[kHostBufferSlots] = {};
  size_t bytes[kHostBufferSlots] = {};
};
std::mutex &hostCacheMutex() {
  static auto *value = new std::mutex;
  return *value;
}
std::vector<HostBufferCache *> &hostCaches() {
  static auto *value = new std::vector<HostBufferCache *>;
  return *value;
}
HostBufferCache &hostCache() {
  thread_local HostBufferCache *cache = nullptr;
  if (!cache) {
    cache = new HostBufferCache;
    std::lock_guard<std::mutex> lock(hostCacheMutex());
    hostCaches().push_back(cache);
  }
  return *cache;
}
} // namespace

// 칸이 비었는지는 포인터로 판단한다. 크기로 판단하면 0 byte 요청이 빈 칸과 같아진다.
extern "C" void *aimx_host_buffer(size_t bytes) {
  HostBufferCache &cache = hostCache();
  for (int i = 0; i < kHostBufferSlots; i++)
    if (cache.ptr[i] && cache.bytes[i] == bytes) {
      void *p = cache.ptr[i];
      cache.ptr[i] = nullptr;
      return p;
    }
  void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  aimx_require(p != MAP_FAILED, "host buffer mmap failed");
  return p;
}

extern "C" void aimx_host_free(void *p, size_t bytes) {
  HostBufferCache &cache = hostCache();
  for (int i = 0; i < kHostBufferSlots; i++)
    if (!cache.ptr[i]) {
      cache.ptr[i] = p;
      cache.bytes[i] = bytes;
      return;
    }
  munmap(p, bytes);
}

extern "C" void aimx_host_cache_clear() {
  std::lock_guard<std::mutex> lock(hostCacheMutex());
  for (HostBufferCache *cache : hostCaches())
    for (int i = 0; i < kHostBufferSlots; i++)
      if (cache->ptr[i]) {
        munmap(cache->ptr[i], cache->bytes[i]);
        cache->ptr[i] = nullptr;
      }
}

extern "C" void aimx_host_cache_clear_thread() {
  HostBufferCache &cache = hostCache();
  for (int i = 0; i < kHostBufferSlots; i++)
    if (cache.ptr[i]) {
      munmap(cache.ptr[i], cache.bytes[i]);
      cache.ptr[i] = nullptr;
    }
}

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
                   bytes <= kMaxAllocationBytes && alignment > 0 &&
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
  if (!state.placed_va.empty()) {
    // memory table이 정한 자리. PMM을 포함한 모든 buffer가 region 안에 놓인다.
    aimx_require(bytes <= state.placed_capacity[slot],
                 "resident buffer exceeds its memory-table entry");
    buffer.va = state.placed_va[slot];
  } else {
    // Own PMM explicitly. Vendor private_pool.c keys its cached PMM by thread ID,
    // which is not a stable ownership key for a persistent physical-device plan.
    buffer.va = (uint64_t)pim_malloc(bytes, state.fpga, kind == 2 ? 7 : 0);
    aimx_require(buffer.va != 0, "resident PIM allocation failed");
  }
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
  if (kind != 2 && !state.placed_va.empty()) {
    buffer.host = state.placed_host[slot];
  } else if (kind != 2) {
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

extern "C" aimx_model_region *aimx_model_region_create(int fpga, int64_t bytes,
                                                       int64_t alignment) {
  Registry &r = registry();
  checkProcess(r);
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    aimx_require(r.devices.count(fpga) == 1, "memory-table region on an unopened FPGA");
  }
  aimx_require(bytes > 0 && bytes <= kMaxAllocationBytes,
               "a memory-table region must fit one vendor allocation (1023 x 2 MB)");
  aimx_require(alignment >= 4096 && alignment <= (4096 << 9) && (alignment & (alignment - 1)) == 0,
               "memory-table region alignment must be a power of two from 4 KB to 2 MB");
  auto *region = new aimx_model_region;
  region->fpga = fpga;
  region->bytes = bytes;
  // page_align_idx k는 2^k page(4 KB) 정렬이다. 2 MB를 넘는 할당은 hugepage 경계에서 시작한다.
  region->va = (uint64_t)pim_malloc(bytes, fpga, __builtin_ctzll(alignment) - 12);
  aimx_require(region->va != 0, "memory-table region allocation failed");
  region->pa = VA2PA(region->va, fpga);
  aimx_require(region->pa % (uint64_t)alignment == 0, "memory-table region is misaligned");
  // 쓰지 않은 page는 host memory를 쓰지 않는다.
  region->host = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  aimx_require(region->host != MAP_FAILED, "memory-table host shadow allocation failed");
  return region;
}

extern "C" void aimx_model_region_destroy(aimx_model_region *region) {
  munmap(region->host, (size_t)region->bytes);
  pim_free((void *)region->va, region->fpga);
  delete region;
}

extern "C" void *aimx_host_image_create(int64_t bytes) {
  aimx_require(bytes > 0, "a pack image needs bytes");
  void *image = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  aimx_require(image != MAP_FAILED, "pack image allocation failed");
  // 2 MB page로 받아 GB 단위 image를 채울 때의 page fault를 줄인다.
  aimx_require(madvise(image, (size_t)bytes, MADV_HUGEPAGE) == 0, "pack image huge page advice failed");
  return image;
}

extern "C" void aimx_host_image_destroy(void *image, int64_t bytes) {
  munmap(image, (size_t)bytes);
}

extern "C" void aimx_model_push(size_t count, aimx_model_region *const *regions,
                                void *const *hosts, const int64_t *bytes) {
  // FPGA마다 thread 하나가 자기 FPGA의 region들을 차례로 보낸다.
  std::map<int, std::vector<size_t>> byFpga;
  for (size_t i = 0; i < count; ++i) {
    aimx_require(bytes[i] > 0 && bytes[i] <= regions[i]->bytes, "push exceeds its region");
    byFpga[regions[i]->fpga].push_back(i);
  }
  std::vector<std::thread> threads;
  for (const auto &entry : byFpga) {
    int fpga = entry.first;
    const std::vector<size_t> &items = entry.second;
    threads.emplace_back([regions, hosts, bytes, fpga, &items] {
      for (size_t i : items)
        aimx_require(aimx_copy_to_pim((void *)regions[i]->va, hosts[i], (size_t)bytes[i], fpga) >= 0,
                     "weight push failed");
    });
  }
  for (auto &thread : threads)
    thread.join();
}

extern "C" aimx_resident_context *aimx_model_create(int fpga, size_t slots, const uint64_t *va,
                                                    const int64_t *capacity, void *const *host) {
  Registry &r = registry();
  checkProcess(r);
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    aimx_require(r.devices.count(fpga) == 1, "model context on an unopened FPGA");
  }
  aimx_require(slots > 0 && va && capacity && host, "a model context needs buffer addresses");
  auto *state = new DeviceState;
  state->fpga = fpga;
  state->plan = "model";
  state->placed_va.assign(va, va + slots);
  state->placed_capacity.assign(capacity, capacity + slots);
  state->placed_host.assign(host, host + slots);
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
