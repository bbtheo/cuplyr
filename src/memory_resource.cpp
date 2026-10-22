// src/memory_resource.cpp
//
// The RMM device memory resource cuplyr installs at load time (F57), and
// gpu_sync().
//
// Without one, every device allocation is a raw cudaMalloc/cudaFree
// (rmm's default cuda_memory_resource). cuplyr allocates many short-lived
// columns per call, so .onLoad installs either
//   "pool":  pool_memory_resource over a cuplyr-owned cuda_memory_resource, or
//   "async": cuda_async_memory_resource (cudaMallocAsync's own pool),
// wrapped in a statistics_resource_adaptor so `used` (bytes currently
// allocated from it) is exact and cheap to read.
//
// Lifetime: one resource per process, never freed. device_buffer deallocates
// through the resource it came from, long after the call that allocated it, and
// RMM's "current resource" is per copy of this library: every
// pkgload::load_all() dyn.load()s a fresh copy, with its own copy of RMM's
// per-device map, while libcudf stays bound to the map of the first copy loaded
// in the process. So a resource one copy made can still be referenced from
// another copy's map and from columns any copy created; no copy can know it is
// unreferenced. The first load with the switch on creates it; no load or unload
// ever deletes it (nor at process exit: the CUDA context may already be gone).
//
// Every later load makes that same resource current in its own copy's map, so
// its allocations go through it. It finds the resource through the copies
// already loaded, not through R state: each copy exports
// cuplyr_process_mr_v2(), which returns the resource it uses, and a new copy
// walks the loaded objects (dl_iterate_phdr()) for other `cuplyr.so` files and
// asks them. What it gets back is checked by magic number, layout version and
// size before use (a copy built from other sources may be loaded too). Nothing
// in the R session can forge, clobber or carry over such a pointer (round 2's
// R option could be all three), and copies are never unmapped (below), so a
// pointer returned is always live.
//
// Every copy of the library is pinned (RTLD_NODELETE) at its first install
// call: the resource's vtables, the map libcudf is bound to, and the finalizers
// of every column a copy created live in that copy's code and data, so a
// dyn.unload() (pkgload::unload()) must not unmap it.
//
// The switches (enabled, kind, initial size) are read by the first load that
// creates the resource; a later load reuses whatever exists.

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Rcpp.h>

namespace {

using base_mr = rmm::mr::device_memory_resource;
using pool_mr = rmm::mr::pool_memory_resource<base_mr>;
using stats_mr = rmm::mr::statistics_resource_adaptor<base_mr>;

constexpr std::uint64_t kMagic = 0x52'4d'72'79'6c'70'75'63ULL;  // "cuplyrMR"
// Layout version of Installed. Bump it, and the version in the name of
// cuplyr_process_mr_v2(), when the struct changes: a resource from a copy with
// another layout is not used.
constexpr std::uint32_t kAbi = 2;

struct Installed {
  // Keep these three first: they are read before trusting the rest.
  std::uint64_t magic = kMagic;
  std::uint32_t abi = kAbi;
  std::uint32_t size = 0;                                 // sizeof(Installed)
  std::string kind;                                       // "pool" or "async"
  std::unique_ptr<rmm::mr::cuda_memory_resource> cuda;    // the pool's upstream
  std::unique_ptr<pool_mr> pool;
  std::unique_ptr<rmm::mr::cuda_async_memory_resource> async;
  std::unique_ptr<stats_mr> stats;                        // the installed (current) resource
};

// The process's resource as seen by this copy of the library (nullptr: none
// yet, or this copy hasn't looked).
Installed* g_inst = nullptr;

constexpr std::size_t kAlign = 256;

std::size_t align_down(std::size_t n) { return n - (n % kAlign); }

// Keep this copy of the library mapped for the rest of the process (see the
// file comment). The handle is intentionally never closed.
void pin_this_library() {
  static bool pinned = false;
  if (pinned) return;
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&pin_this_library), &info) != 0 && info.dli_fname != nullptr) {
    pinned = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE) != nullptr;
  }
}

bool valid(const Installed* inst) {
  return inst != nullptr && inst->magic == kMagic && inst->abi == kAbi &&
         inst->size == sizeof(Installed);
}

// The resource another loaded copy of the library uses, or nullptr.
Installed* find_in_other_copies() {
  std::vector<std::string> paths;
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t, void* data) -> int {
        std::string name = info->dlpi_name != nullptr ? info->dlpi_name : "";
        const std::string so = "/cuplyr.so";
        if (name.size() > so.size() && name.compare(name.size() - so.size(), so.size(), so) == 0) {
          static_cast<std::vector<std::string>*>(data)->push_back(name);
        }
        return 0;
      },
      &paths);
  // dlopen() outside the dl_iterate_phdr() callback, which holds the loader lock.
  for (const auto& path : paths) {
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (handle == nullptr) continue;
    auto* fn = reinterpret_cast<void* (*)()>(dlsym(handle, "cuplyr_process_mr_v2"));
    auto* inst = fn != nullptr ? static_cast<Installed*>(fn()) : nullptr;
    dlclose(handle);  // drops only the reference dlopen() just added
    if (valid(inst)) return inst;
  }
  return nullptr;
}

Installed* create(const std::string& kind, double initial) {
  if (kind != "pool" && kind != "async") {
    Rcpp::stop("gpu_mr_install(): unknown resource kind '%s'", kind.c_str());
  }
  auto [free_b, total_b] = rmm::available_device_memory();
  (void)total_b;
  std::size_t cap = align_down(static_cast<std::size_t>(0.9 * static_cast<double>(free_b)));
  std::size_t want = initial > 0 ? static_cast<std::size_t>(std::min(initial, 9e18)) : 0;
  std::size_t init = align_down(std::min(want, cap));

  auto inst = std::make_unique<Installed>();
  inst->size = sizeof(Installed);
  inst->kind = kind;
  try {
    base_mr* top = nullptr;
    if (kind == "pool") {
      inst->cuda = std::make_unique<rmm::mr::cuda_memory_resource>();
      inst->pool = std::make_unique<pool_mr>(inst->cuda.get(), init);
      top = inst->pool.get();
    } else {
      // No release threshold is passed, so RMM's default (the device total)
      // applies: freed memory stays reserved for reuse.
      inst->async = std::make_unique<rmm::mr::cuda_async_memory_resource>(
          init > 0 ? std::optional<std::size_t>(init) : std::nullopt);
      top = inst->async.get();
    }
    inst->stats = std::make_unique<stats_mr>(top);
  } catch (const std::exception& e) {
    Rcpp::stop("could not create the %s memory resource: %s", kind.c_str(), e.what());
  }
  return inst.release();
}

}  // namespace

// The resource this copy uses, for later copies (see the file comment).
extern "C" __attribute__((visibility("default"))) void* cuplyr_process_mr_v2() {
  return g_inst;
}

// Make the process's resource current for this copy of the library, creating
// it first when none exists and `enabled` (see the file comment): `kind` is
// "pool" or "async", `initial` the initial size in bytes (clamped to 90% of the
// device's free memory). Returns the kind of the resource now current
// ("pool"/"async"), or "cuda" when there is none.
// [[Rcpp::export]]
std::string gpu_mr_install(bool enabled, std::string kind, double initial) {
  pin_this_library();
  if (g_inst == nullptr) {
    Installed* found = find_in_other_copies();
    if (found == nullptr && !enabled) return "cuda";
    g_inst = found != nullptr ? found : create(kind, initial);
    rmm::mr::set_current_device_resource(g_inst->stats.get());
  }
  return g_inst->kind;
}

// c(pool_bytes, pool_used_bytes) of the installed resource, NA, NA if none.
// pool_bytes: what the resource holds from the device (the pool's size, or the
// async pool's reserved memory). pool_used_bytes: bytes currently allocated
// from it.
// [[Rcpp::export]]
Rcpp::NumericVector gpu_mr_state() {
  if (g_inst == nullptr) {
    return Rcpp::NumericVector::create(NA_REAL, NA_REAL);
  }
  double held = 0;
  if (g_inst->pool) {
    held = static_cast<double>(g_inst->pool->pool_size());
  } else {
    std::uint64_t reserved = 0;
    if (cudaMemPoolGetAttribute(g_inst->async->pool_handle(), cudaMemPoolAttrReservedMemCurrent,
                                &reserved) == cudaSuccess) {
      held = static_cast<double>(reserved);
    }
  }
  double used = static_cast<double>(g_inst->stats->get_bytes_counter().value);
  return Rcpp::NumericVector::create(held, used);
}

// The kind of the resource current for this copy ("pool"/"async"), or "cuda"
// when none.
// [[Rcpp::export]]
std::string gpu_mr_kind() {
  return g_inst == nullptr ? std::string("cuda") : g_inst->kind;
}

// Wait until all work queued on the device has finished.
// [[Rcpp::export]]
void gpu_sync_device() {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    Rcpp::stop("CUDA error (gpu_sync): %s", cudaGetErrorString(err));
  }
}
