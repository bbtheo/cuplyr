// src/oom_inject.cpp
//
// Test hook for the OOM safety net (F58): gpu_oom_inject(n) makes the next n
// RMM device allocations throw rmm::out_of_memory, the way a full pool does.
// While armed a node (cuplyr_mr.hpp's Forwarder) sits on top of the current
// device resource; n = 0 takes it off again, so nothing is installed (and
// nothing costs anything) when disarmed.
//
// Lifetime: same rule as alloc_tracking.cpp. A device_buffer deallocates
// through the resource it was allocated from, possibly long after the hook was
// disarmed, so a node is never destroyed (a few bytes leaked per arm after
// a disarm); it only stops being current. The failure counter is global: a
// node that is no longer current never allocates, and one that is passes
// everything through once the counter is 0.

#include "cuplyr_mr.hpp"

#include <rmm/error.hpp>

#include <Rcpp.h>

namespace {

int g_remaining = 0;

class OomNode final : public cuplyr::mr::Forwarder {
 public:
  using Forwarder::Forwarder;

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t align) override {
    if (g_remaining > 0) {
      --g_remaining;
      throw rmm::out_of_memory("injected out_of_memory (gpu_oom_inject)");
    }
    return Forwarder::allocate(stream, bytes, align);
  }
};

OomNode* g_top = nullptr;  // the injector while it is current, else nullptr

}  // namespace

// Arm the next `n` allocations to fail; n = 0 disarms.
// [[Rcpp::export]]
void gpu_oom_inject(int n) {
  if (n < 0) Rcpp::stop("n must be >= 0");
  g_remaining = n;
  if (n > 0) {
    if (g_top == nullptr || !cuplyr::mr::is_current(g_top)) {
      g_top = new OomNode(cuplyr::mr::current_copy());  // never destroyed
      cuplyr::mr::make_current(g_top);
    }
  } else if (g_top != nullptr) {
    // Pop only if nothing was pushed on top meanwhile; otherwise it stays in
    // the chain as a pass-through.
    if (cuplyr::mr::is_current(g_top)) {
      cuplyr::mr::make_current_prev(g_top);
    }
    g_top = nullptr;
  }
}

// Failures armed and not yet fired.
// [[Rcpp::export]]
int gpu_oom_inject_remaining() {
  return g_remaining;
}
