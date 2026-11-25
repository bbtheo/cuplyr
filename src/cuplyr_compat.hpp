// src/cuplyr_compat.hpp
//
// The few places where libcudf 26.02 and 26.10 differ in a way no single call
// spelling covers (the memory-resource differences live in cuplyr_mr.hpp).
// Everything else builds unchanged on both. 26.10 is the development series;
// 26.02 is the supported minimum.
#pragma once

#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <memory>

// libcudf 26.10 moved distinct_count() out of cudf/stream_compaction.hpp into
// its own header; 26.02 still declares it in stream_compaction.hpp. The new
// header is also the marker for the 26.10 filtered_join constructor below.
#if __has_include(<cudf/reduction/distinct_count.hpp>)
#include <cudf/reduction/distinct_count.hpp>
#define CUPLYR_CUDF_NEW_API 1
#else
#include <cudf/stream_compaction.hpp>
#define CUPLYR_CUDF_NEW_API 0
#endif

namespace cuplyr {

// The raw CUDA stream behind what cudf::get_default_stream() returns:
// rmm::cuda_stream_view on 26.02 (`value()`), cuda::stream_ref on 26.10 (`get()`).
template <class Stream>
inline cudaStream_t raw_stream(Stream const& stream) {
  if constexpr (requires { stream.get(); }) {
    return stream.get();
  } else {
    return stream.value();
  }
}

// A filtered_join whose hash set is built from `build` (probed with semi_join()
// / anti_join()). 26.02 takes a set_as_build_table argument (cuplyr always
// passes RIGHT: LEFT is unimplemented there); 26.10 removed it and requires the
// stream.
inline std::unique_ptr<cudf::filtered_join> make_filtered_join(cudf::table_view const& build,
                                                               cudf::null_equality compare_nulls,
                                                               rmm::cuda_stream_view stream) {
#if CUPLYR_CUDF_NEW_API
  return std::make_unique<cudf::filtered_join>(build, compare_nulls, stream);
#else
  return std::make_unique<cudf::filtered_join>(build, compare_nulls,
                                               cudf::set_as_build_table::RIGHT, stream);
#endif
}

}  // namespace cuplyr
