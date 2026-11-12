// src/alloc_tracking.cpp
//
// Allocation tracking windows (F56, test infrastructure; off unless a window
// is open). A window installs a fresh rmm statistics_resource_adaptor as the
// current device resource, with the resource that was current at window start
// as its upstream, so windows nest and an outer window counts the inner one.
//
// Lifetime: rmm::device_buffer deallocates through the resource it was
// allocated from, which can be long after the window closed. So an adaptor is
// never destroyed while it has outstanding bytes or allocations, or while it
// is on the window stack. Closed adaptors whose counters are back to zero are
// reclaimed at the next push. An outer adaptor's outstanding count includes
// its inner windows', so a live inner adaptor keeps its upstream alive too.

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <Rcpp.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace {

using stats_mr = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

struct Window {
  int id;
  stats_mr* mr;
  rmm::mr::device_memory_resource* prev;
};

std::vector<std::unique_ptr<stats_mr>> g_owned;  // every adaptor not yet reclaimed
std::vector<Window> g_stack;                     // open windows, innermost last
int g_next_id = 0;

bool on_stack(const stats_mr* mr) {
  for (const auto& w : g_stack) if (w.mr == mr) return true;
  return false;
}

// Reclaim closed adaptors with nothing outstanding. An outer adaptor counts its
// inner windows, so it is never zero while an inner one is still live.
void reclaim_closed() {
  g_owned.erase(
    std::remove_if(g_owned.begin(), g_owned.end(), [](const std::unique_ptr<stats_mr>& p) {
      return !on_stack(p.get()) &&
             p->get_bytes_counter().value == 0 &&
             p->get_allocations_counter().value == 0;
    }),
    g_owned.end());
}

}  // namespace

// [[Rcpp::export]]
int gpu_alloc_window_push() {
  reclaim_closed();
  auto* prev = rmm::mr::get_current_device_resource();
  auto owned = std::make_unique<stats_mr>(prev);
  stats_mr* mr = owned.get();
  g_owned.push_back(std::move(owned));
  rmm::mr::set_current_device_resource(mr);
  int id = ++g_next_id;
  g_stack.push_back({id, mr, prev});
  return id;
}

// Returns c(total, value, peak) in bytes, as doubles.
// [[Rcpp::export]]
Rcpp::NumericVector gpu_alloc_window_pop(int id) {
  if (g_stack.empty() || g_stack.back().id != id) {
    Rcpp::stop("allocation tracking windows must close in LIFO order");
  }
  Window w = g_stack.back();
  g_stack.pop_back();
  rmm::mr::set_current_device_resource(w.prev);
  auto b = w.mr->get_bytes_counter();
  return Rcpp::NumericVector::create(static_cast<double>(b.total),
                                     static_cast<double>(b.value),
                                     static_cast<double>(b.peak));
}

// [[Rcpp::export]]
int gpu_alloc_window_count() {
  return static_cast<int>(g_stack.size());
}
