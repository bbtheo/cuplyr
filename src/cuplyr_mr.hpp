// src/cuplyr_mr.hpp
//
// cuplyr's own device memory resources, on RMM's `cuda::mr::resource_ref`
// model (RMM 26.02 and later). The same source builds against RMM 26.02 and
// 26.10, whose APIs differ
// in how a resource is installed and what the stock adaptors look like, so the
// resources cuplyr needs are written once here instead of using RMM's adaptors:
//
//   StatsNode: forwards to an upstream and counts the bytes/allocations that
//              pass through it (the installed resource's `used` figure, and
//              every allocation tracking window, src/alloc_tracking.cpp).
//   OomNode:   (src/oom_inject.cpp) fails allocations on demand.
//
// Model. A resource is a `Node`: a heap object that is never destroyed (a
// device_buffer deallocates through the resource it came from, long after any
// window or injector that was current at the time; see memory_resource.cpp and
// alloc_tracking.cpp for the lifetime rules, which are unchanged). What RMM
// stores and hands out is a `Handle`: a copyable, pointer-sized value that
// satisfies `cuda::mr::resource_with<device_accessible>` and forwards to its
// node. A node forwards to an upstream, held either as a non-owning ref (the
// base pool/async resource, owned by memory_resource.cpp's `Installed`) or as
// an owning `any_mr` copy of whatever was current when the node was made.
//
// The two compat blocks are the places that know how the RMM generations
// build and install a resource:
//   26.02: pool_memory_resource<Upstream> is a template over a ref. libcudf
//          26.02's default resource still comes from RMM's legacy per-device
//          pointer map, which the ref setters do not touch, so a node is
//          installed through a stock callback_memory_resource bridge and the
//          legacy setter (the only call that updates both maps); see
//          make_current().
//   26.10: set_current_device_resource(any_resource) takes it by value (one
//          map); pool_memory_resource is a refcounted, non-template class.
// (The boundary is the 26.06 release, where RMM removed the legacy
// rmm::mr::device_memory_resource; releases between 26.02 and 26.10 are not
// tested.) libcudf's own API differences are in cuplyr_compat.hpp.
#pragma once

#include <rmm/version_config.hpp>

#define CUPLYR_RMM_NEW_API (RMM_VERSION_MAJOR > 26 || (RMM_VERSION_MAJOR == 26 && RMM_VERSION_MINOR >= 6))

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>
#if !CUPLYR_RMM_NEW_API
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#endif
#include <rmm/version_config.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace cuplyr::mr {

// An owning, type-erased copy of a resource (what RMM's per-device map holds).
using any_mr = cuda::mr::any_resource<cuda::mr::device_accessible>;

// ---- compat (1/2): building resources -------------------------------------

#if CUPLYR_RMM_NEW_API

using pool_mr = rmm::mr::pool_memory_resource;

inline std::unique_ptr<pool_mr> make_pool(rmm::mr::cuda_memory_resource& upstream,
                                          std::size_t initial) {
  return std::make_unique<pool_mr>(any_mr{upstream}, initial);
}

// The stream an upstream's allocate()/deallocate() takes.
inline cuda::stream_ref upstream_stream(cuda::stream_ref s) { return s; }

#else

using pool_mr = rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>;

// `upstream` must outlive the pool (memory_resource.cpp's Installed owns both).
inline std::unique_ptr<pool_mr> make_pool(rmm::mr::cuda_memory_resource& upstream,
                                          std::size_t initial) {
  return std::make_unique<pool_mr>(rmm::device_async_resource_ref{upstream}, initial);
}

inline rmm::cuda_stream_view upstream_stream(cuda::stream_ref s) {
  return rmm::cuda_stream_view{s.get()};
}

// The type of the legacy per-device pointer RMM 26.02 still keeps next to its
// resource_ref map, and which libcudf 26.02 reads for its default resource.
// Named by decltype so no source line spells the legacy base class.
using legacy_ptr = decltype(cudf::set_current_device_resource(nullptr));

#endif

// An owning copy of the resource that is current on this copy of the library.
inline any_mr current_copy() {
  return static_cast<any_mr>(rmm::mr::get_current_device_resource_ref());
}

// ---- nodes and handles ----------------------------------------------------

constexpr std::size_t kDefaultAlign = alignof(std::max_align_t);

class Node {
 public:
  virtual ~Node() = default;
  virtual void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t align) = 0;
  virtual void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                          std::size_t align) noexcept = 0;

#if !CUPLYR_RMM_NEW_API
  // RMM 26.02 only; see make_current().
  std::unique_ptr<rmm::mr::callback_memory_resource> bridge;
  legacy_ptr prev_legacy = nullptr;
#endif
};

// What RMM holds and passes around for a Node. Copies are interchangeable;
// two handles are equal when they name the same node.
class Handle {
 public:
  explicit Handle(Node* node) noexcept : node_(node) {}

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t align = kDefaultAlign) {
    return node_->allocate(stream, bytes, align);
  }
  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                  std::size_t align = kDefaultAlign) noexcept {
    node_->deallocate(stream, ptr, bytes, align);
  }
  void* allocate_sync(std::size_t bytes, std::size_t align = kDefaultAlign) {
    cuda::stream_ref stream{cudaStream_t{}};
    void* ptr = node_->allocate(stream, bytes, align);
    cudaStreamSynchronize(stream.get());
    return ptr;
  }
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t align = kDefaultAlign) noexcept {
    cuda::stream_ref stream{cudaStream_t{}};
    node_->deallocate(stream, ptr, bytes, align);
    cudaStreamSynchronize(stream.get());
  }

  friend bool operator==(Handle const& a, Handle const& b) noexcept { return a.node_ == b.node_; }
  friend void get_property(Handle const&, cuda::mr::device_accessible) noexcept {}

 private:
  Node* node_;
};

static_assert(cuda::mr::resource_with<Handle, cuda::mr::device_accessible>,
              "Handle must satisfy the cuda::mr::resource concept");

inline any_mr as_any(Node* node) { return any_mr{Handle{node}}; }

// A node that passes everything to an upstream.
class Forwarder : public Node {
 public:
  // Upstream: a copy of a resource (typically what was current). Held by value, so
  // the node keeps it alive and its address is stable (the node never moves).
  explicit Forwarder(any_mr upstream) : keep_(std::move(upstream)), up_(*keep_) {}
  // Upstream: a resource somebody else keeps alive for longer than this node.
  explicit Forwarder(rmm::device_async_resource_ref upstream) : up_(upstream) {}

  // The owning copy of the upstream, to put back as current (nullptr when the
  // upstream is a non-owning ref).
  any_mr* upstream_copy() { return keep_ ? &*keep_ : nullptr; }

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t align) override {
    return up_.allocate(upstream_stream(stream), bytes, align);
  }
  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                  std::size_t align) noexcept override {
    up_.deallocate(upstream_stream(stream), ptr, bytes, align);
  }

 private:
  std::optional<any_mr> keep_;  // declared before up_: up_ may refer into it
  rmm::device_async_resource_ref up_;
};

// ---- compat (2/2): making a node current -----------------------------------

#if CUPLYR_RMM_NEW_API

// RMM 26.10 keeps one per-device map of owning any_resource values; the map
// takes its own copy of the handle and releases the previous resource.
inline void make_current(Node* node) { (void)rmm::mr::set_current_device_resource(as_any(node)); }

// Put back what `node` forwards to (it must hold an owning upstream copy).
inline void make_current_prev(Forwarder* node) {
  (void)rmm::mr::set_current_device_resource(*node->upstream_copy());
}

// Whether `node` is the resource that is current on this copy of the library.
inline bool is_current(Node* node) { return current_copy() == as_any(node); }

#else

// RMM 26.02 keeps two per-device maps: the resource_ref one, and a legacy
// pointer one that libcudf 26.02's cudf::get_current_device_resource_ref()
// reads, so a resource installed through the ref map alone is invisible to
// libcudf's own default allocations (they would go to plain cudaMalloc, past
// the pool, the statistics and any tracking window). Only the legacy setter
// updates both. Its argument is RMM's callback_memory_resource, a stock adaptor
// that forwards to the node; the node's bridge is made on first use and never
// freed, like the node.
inline void make_current(Node* node) {
  if (!node->bridge) {
    node->bridge = std::make_unique<rmm::mr::callback_memory_resource>(
        [](std::size_t bytes, rmm::cuda_stream_view stream, void* arg) -> void* {
          return static_cast<Node*>(arg)->allocate(cuda::stream_ref{stream.value()}, bytes,
                                                   rmm::CUDA_ALLOCATION_ALIGNMENT);
        },
        [](void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void* arg) {
          static_cast<Node*>(arg)->deallocate(cuda::stream_ref{stream.value()}, ptr, bytes,
                                              rmm::CUDA_ALLOCATION_ALIGNMENT);
        },
        node, node);
  }
  node->prev_legacy = cudf::set_current_device_resource(node->bridge.get());
}

inline void make_current_prev(Forwarder* node) {
  (void)cudf::set_current_device_resource(node->prev_legacy);
}

inline bool is_current(Node* node) {
  if (!node->bridge) return false;
  any_mr mine{static_cast<any_mr>(rmm::device_async_resource_ref{*node->bridge})};
  return current_copy() == mine;
}

#endif

// Counts what passes through. One relaxed atomic add per allocation and per
// free (plus a CAS only when a new peak is set): cheap enough for the async
// default, which goes through it on every column.
class StatsNode : public Forwarder {
 public:
  struct Counter {
    std::int64_t value;  // currently outstanding
    std::int64_t peak;   // highest `value` seen
    std::int64_t total;  // ever allocated
  };

  using Forwarder::Forwarder;

  void* allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t align) override {
    void* ptr = Forwarder::allocate(stream, bytes, align);  // throws: nothing counted
    auto b = static_cast<std::int64_t>(bytes);
    std::int64_t now = bytes_.fetch_add(b, std::memory_order_relaxed) + b;
    std::int64_t peak = bytes_peak_.load(std::memory_order_relaxed);
    while (now > peak &&
           !bytes_peak_.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
    }
    bytes_total_.fetch_add(b, std::memory_order_relaxed);
    allocs_.fetch_add(1, std::memory_order_relaxed);
    return ptr;
  }
  void deallocate(cuda::stream_ref stream, void* ptr, std::size_t bytes,
                  std::size_t align) noexcept override {
    Forwarder::deallocate(stream, ptr, bytes, align);
    bytes_.fetch_sub(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
    allocs_.fetch_sub(1, std::memory_order_relaxed);
  }

  Counter bytes() const {
    return {bytes_.load(std::memory_order_relaxed), bytes_peak_.load(std::memory_order_relaxed),
            bytes_total_.load(std::memory_order_relaxed)};
  }
  std::int64_t allocations() const { return allocs_.load(std::memory_order_relaxed); }

 private:
  std::atomic<std::int64_t> bytes_{0};
  std::atomic<std::int64_t> bytes_peak_{0};
  std::atomic<std::int64_t> bytes_total_{0};
  std::atomic<std::int64_t> allocs_{0};
};

}  // namespace cuplyr::mr
