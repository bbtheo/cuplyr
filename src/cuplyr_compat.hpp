// src/cuplyr_compat.hpp
//
// The few places where libcudf 26.02 and 26.10 differ in a way no single call
// spelling covers (the memory-resource differences live in cuplyr_mr.hpp).
// Everything else builds unchanged on both. 26.10 is the development series;
// 26.02 is the supported minimum. The blocks below call the 26.10 spelling of
// an API that 26.10 deprecated or moved, and the 26.02 one otherwise; the rest
// of the source calls the wrappers, never the deprecated names.
#pragma once

#include <variant>  // libcudf 26.10 headers using std::variant (cudf/ast/expressions.hpp) do not include it
#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// libcudf 26.10 moved distinct_count() out of cudf/stream_compaction.hpp into
// its own header; 26.02 still declares it in stream_compaction.hpp. The new
// header is also the marker for CUPLYR_CUDF_NEW_API, the one macro for every
// 26.02/26.10 difference below: filtered_join's constructor, apply_retention_mask(),
// the NaN-as-null mask, slice_strings() and the parquet reader's column_names().
// The two series are the only ones supported (26.02 is the declared minimum, 26.10
// the default), so one marker is enough; a series that splits these APIs apart
// would need one macro per API.
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

// The NaN-as-null mask of the floating column `col`: valid where `col` is valid
// and not NaN, with its null count. Only the mask is built, never a copy of the
// data, so a caller keeps the original data under a view or sets the mask on a
// column it owns, and copies only when a NaN is present. cudf::nans_to_nulls()
// (26.02) is deprecated in 26.04; its replacement, column_nans_to_nulls(),
// returns a whole copy of the column (all of it thrown away when only the mask
// is wanted: 8 B per row on every double comparison), so on 26.10 the mask is
// built from the 1-byte is_not_nan() column by bools_to_mask(), which reads a
// null as false. is_not_nan() is true under a null, and the data under a null
// can be NaN (R's NA_real_ payload), so the flags are read under `col`'s own
// null mask: through a view when `col` starts at row 0 (bools_to_mask() then
// ANDs it in, with no bitmask_and() pass), bitmask_and() otherwise. Against
// nans_to_nulls()'s single pass this costs one is_not_nan() pass; measured at
// 1e3-1e6 rows, column_nans_to_nulls() is no faster at any size (its copy
// costs about what that pass does).
inline std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> nan_null_mask(
    cudf::column_view const& col) {
#if CUPLYR_CUDF_NEW_API
  if (col.is_empty()) return {std::make_unique<rmm::device_buffer>(), 0};
  auto not_nan = cudf::is_not_nan(col);
  if (col.null_count() == 0) return cudf::bools_to_mask(not_nan->view());
  if (col.offset() == 0) {
    cudf::column_view flags(not_nan->type(), col.size(), not_nan->view().head(), col.null_mask(),
                            col.null_count());
    return cudf::bools_to_mask(flags);
  }
  auto mask = cudf::bools_to_mask(not_nan->view());
  not_nan.reset();
  std::vector<cudf::bitmask_type const*> masks{
      static_cast<cudf::bitmask_type const*>(mask.first->data()), col.null_mask()};
  std::vector<cudf::size_type> begin_bits{0, col.offset()};
  auto both = cudf::bitmask_and(masks, begin_bits, col.size());
  return {std::make_unique<rmm::device_buffer>(std::move(both.first)), both.second};
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
