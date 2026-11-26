// src/cuplyr_compat.hpp
//
// The few places where libcudf 26.02 and 26.10 differ in a way no single call
// spelling covers (the memory-resource differences live in cuplyr_mr.hpp).
// Everything else builds unchanged on both. 26.10 is the development series;
// 26.02 is the supported minimum.
#pragma once

#include <variant>  // libcudf 26.10 headers using std::variant (cudf/ast/expressions.hpp) do not include it
#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/transform.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
                                                               cuda::stream_ref stream) {
#if CUPLYR_CUDF_NEW_API
  return std::make_unique<cudf::filtered_join>(build, compare_nulls, stream);
#else
  return std::make_unique<cudf::filtered_join>(build, compare_nulls,
                                               cudf::set_as_build_table::RIGHT, stream);
#endif
}

// ---- libcudf 26.10 deprecation ports (26.02 lacks the replacements) ----
//
// Everything below calls the 26.10 spelling on 26.10 and the 26.02 spelling on
// 26.02; the rest of the source calls these wrappers instead of the cudf names.

// cudf::apply_boolean_mask() -> cudf::apply_retention_mask() (26.10). Same
// semantics: rows whose mask is null or false are dropped.
inline std::unique_ptr<cudf::table> apply_retention_mask(
    cudf::table_view const& input, cudf::column_view const& mask,
    cuda::stream_ref stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) {
#if CUPLYR_CUDF_NEW_API
  return cudf::apply_retention_mask(input, mask, stream, mr);
#else
  return cudf::apply_boolean_mask(input, mask, stream, mr);
#endif
}

// A copy of the floating column `col` whose null mask also marks NaN as null
// (cudf::column_nans_to_nulls() on 26.10; the deprecated mask-only
// cudf::nans_to_nulls() plus a copy on 26.02).
inline std::unique_ptr<cudf::column> nan_to_null_column(cudf::column_view const& col) {
#if CUPLYR_CUDF_NEW_API
  return cudf::column_nans_to_nulls(col);
#else
  auto mask = cudf::nans_to_nulls(col);
  auto out = std::make_unique<cudf::column>(col);
  out->set_null_mask(std::move(*mask.first), mask.second);
  return out;
#endif
}

// Only the NaN-as-null mask of `col` and its null count, for callers that
// build a view over the original data. On 26.10 this releases the mask of
// column_nans_to_nulls()'s result (the data copy is freed).
inline std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> nan_null_mask(
    cudf::column_view const& col) {
#if CUPLYR_CUDF_NEW_API
  auto out = cudf::column_nans_to_nulls(col);
  auto count = out->null_count();
  auto contents = out->release();
  return {std::move(contents.null_mask), count};
#else
  return cudf::nans_to_nulls(col);
#endif
}

// substr(): the characters [start, stop) of every string (26.10 takes
// std::optional positions, 26.02 numeric_scalars).
inline std::unique_ptr<cudf::column> slice_strings_range(cudf::strings_column_view const& input,
                                                         cudf::size_type start,
                                                         cudf::size_type stop) {
#if CUPLYR_CUDF_NEW_API
  return cudf::strings::slice_strings(input, std::optional<cudf::size_type>{start},
                                      std::optional<cudf::size_type>{stop});
#else
  cudf::numeric_scalar<cudf::size_type> start_scalar(start, true);
  cudf::numeric_scalar<cudf::size_type> stop_scalar(stop, true);
  return cudf::strings::slice_strings(input, start_scalar, stop_scalar);
#endif
}

// parquet_reader_options_builder::columns() -> column_names() (26.10).
inline void parquet_builder_column_names(cudf::io::parquet_reader_options_builder& builder,
                                         std::vector<std::string> names) {
#if CUPLYR_CUDF_NEW_API
  builder.column_names(std::move(names));
#else
  builder.columns(std::move(names));
#endif
}

}  // namespace cuplyr
