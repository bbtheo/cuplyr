// src/alloc_tracking.cpp
//
// Allocation tracking windows (F56, test infrastructure; off unless a window
// is open). A window installs a fresh counting node (cuplyr_mr.hpp's
// StatsNode) as the current device resource, with a copy of the resource that
// was current at window start as its upstream, so windows nest and an outer
// window counts the inner one.
//
// Lifetime: rmm::device_buffer deallocates through the resource it was
// allocated from, which can be long after the window closed. So a node is
// never destroyed while it has outstanding bytes or allocations, or while it
// is on the window stack. Closed nodes whose counters are back to zero are
// reclaimed at the next push. An outer node's outstanding count includes
// its inner windows', so a live inner node keeps its upstream alive too.

#include "cuplyr_mr.hpp"

#include <Rcpp.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace {

using cuplyr::mr::StatsNode;

struct Window {
  int id;
  StatsNode* node;
};

std::vector<std::unique_ptr<StatsNode>> g_owned;  // every node not yet reclaimed
std::vector<Window> g_stack;                      // open windows, innermost last
int g_next_id = 0;

bool on_stack(const StatsNode* node) {
  for (const auto& w : g_stack) if (w.node == node) return true;
  return false;
}

// Reclaim closed nodes with nothing outstanding. An outer node counts its
// inner windows, so it is never zero while an inner one is still live.
void reclaim_closed() {
  g_owned.erase(
    std::remove_if(g_owned.begin(), g_owned.end(), [](const std::unique_ptr<StatsNode>& p) {
      return !on_stack(p.get()) && p->bytes().value == 0 && p->allocations() == 0;
    }),
    g_owned.end());
}

}  // namespace

// [[Rcpp::export]]
int gpu_alloc_window_push() {
  reclaim_closed();
  auto owned = std::make_unique<StatsNode>(cuplyr::mr::current_copy());
  StatsNode* node = owned.get();
  g_owned.push_back(std::move(owned));
  cuplyr::mr::make_current(node);
  int id = ++g_next_id;
  g_stack.push_back({id, node});
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
  cuplyr::mr::make_current_prev(w.node);
  auto b = w.node->bytes();
  return Rcpp::NumericVector::create(static_cast<double>(b.total),
                                     static_cast<double>(b.value),
                                     static_cast<double>(b.peak));
}

// [[Rcpp::export]]
int gpu_alloc_window_count() {
  return static_cast<int>(g_stack.size());
}
