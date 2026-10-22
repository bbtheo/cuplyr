// src/oom_inject.cpp
//
// Test hook for the OOM safety net (F58): gpu_oom_inject(n) makes the next n
// RMM device allocations throw rmm::out_of_memory, the way a full pool does.
// While armed an adaptor sits on top of the current device resource; n = 0
// takes it off again, so nothing is installed (and nothing costs anything)
// when disarmed.
//
// Lifetime: same rule as alloc_tracking.cpp. A device_buffer deallocates
// through the resource it was allocated from, possibly long after the hook was
// disarmed, so an adaptor is never destroyed (a few bytes leaked per arm after
// a disarm); it only stops being current. The failure counter is global: an
// adaptor that is no longer current never allocates, and one that is passes
// everything through once the counter is 0.

#include <rmm/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <Rcpp.h>

namespace {

int g_remaining = 0;

class oom_inject_mr final : public rmm::mr::device_memory_resource {
 public:
  explicit oom_inject_mr(rmm::mr::device_memory_resource* upstream) : upstream_(upstream) {}
  rmm::mr::device_memory_resource* upstream() const { return upstream_; }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
    if (g_remaining > 0) {
      --g_remaining;
      throw rmm::out_of_memory("injected out_of_memory (gpu_oom_inject)");
    }
    return upstream_->allocate(bytes, stream);
  }
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override {
    upstream_->deallocate(ptr, bytes, stream);
  }
  bool do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept override {
    return this == &other;
  }

  rmm::mr::device_memory_resource* upstream_;
};

oom_inject_mr* g_top = nullptr;  // the adaptor while it is current, else nullptr

}  // namespace

// Arm the next `n` allocations to fail; n = 0 disarms.
// [[Rcpp::export]]
void gpu_oom_inject(int n) {
  if (n < 0) Rcpp::stop("n must be >= 0");
  g_remaining = n;
  if (n > 0) {
    if (g_top == nullptr || rmm::mr::get_current_device_resource() != g_top) {
      g_top = new oom_inject_mr(rmm::mr::get_current_device_resource());  // never destroyed
      rmm::mr::set_current_device_resource(g_top);
    }
  } else if (g_top != nullptr) {
    // Pop only if nothing was pushed on top meanwhile; otherwise it stays in
    // the chain as a pass-through.
    if (rmm::mr::get_current_device_resource() == g_top) {
      rmm::mr::set_current_device_resource(g_top->upstream());
    }
    g_top = nullptr;
  }
}

// Failures armed and not yet fired.
// [[Rcpp::export]]
int gpu_oom_inject_remaining() {
  return g_remaining;
}
