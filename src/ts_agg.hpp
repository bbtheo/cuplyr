// src/ts_agg.hpp
//
// F41: mean() / median() of a Date or POSIXct column, shared by gpu_summarise()
// (src/ops_groupby.cpp) and gpu_window() (src/ops_window.cpp).
//
// cudf has no mean/median over a timestamp, so the aggregate runs on the
// integer representation (days / microseconds) cast to FLOAT64 and the
// FLOAT64 result is converted back:
//   - POSIXct: rounded to whole microseconds, half away from zero, like the
//     upload's llround() (posixct_to_gpu(), src/transfer_io.cpp).
//   - Date: floored, like the Date upload (date_to_gpu()); `fractional` is set
//     when any value had a fraction, so the R side can warn
//     (`cuplyr_warning_fractional_date`).
//   - NaN (mean of an all-NA group with na.rm = TRUE, or of no rows) -> null:
//     a timestamp can't hold NaN, and R prints it as NA.
#ifndef CUPLYR_TS_AGG_HPP
#define CUPLYR_TS_AGG_HPP

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/round.hpp>

#include <memory>
#include <string>

namespace cuplyr {

// EMPTY = not a timestamp mean/median; otherwise the timestamp type id the
// result converts back to.
inline cudf::type_id ts_agg_target(const std::string& fn, cudf::type_id in) {
    if (fn != "mean" && fn != "median") return cudf::type_id::EMPTY;
    if (in == cudf::type_id::TIMESTAMP_DAYS || in == cudf::type_id::TIMESTAMP_MICROSECONDS) return in;
    return cudf::type_id::EMPTY;
}

// The timestamp column as FLOAT64 days / microseconds (nulls kept).
inline std::unique_ptr<cudf::column> ts_to_float64(const cudf::column_view& col) {
    bool days = col.type().id() == cudf::type_id::TIMESTAMP_DAYS;
    auto ints = cudf::bit_cast(col, cudf::data_type{days ? cudf::type_id::INT32 : cudf::type_id::INT64});
    return cudf::cast(ints, cudf::data_type{cudf::type_id::FLOAT64});
}

// A FLOAT64 aggregate back to `target` (TIMESTAMP_DAYS / TIMESTAMP_MICROSECONDS).
inline std::unique_ptr<cudf::column> float64_to_ts(std::unique_ptr<cudf::column> f,
                                                   cudf::type_id target, bool& fractional) {
    fractional = false;
    {
        auto mask = cudf::nans_to_nulls(f->view());
        f->set_null_mask(std::move(*mask.first), mask.second);
    }
    if (f->size() == 0) {
        return cudf::make_empty_column(cudf::data_type{target});
    }

    std::unique_ptr<cudf::column> whole;
    cudf::data_type int_type{target == cudf::type_id::TIMESTAMP_DAYS ? cudf::type_id::INT32
                                                                      : cudf::type_id::INT64};
    if (target == cudf::type_id::TIMESTAMP_DAYS) {
        whole = cudf::unary_operation(f->view(), cudf::unary_operator::FLOOR);
        auto differs = cudf::binary_operation(f->view(), whole->view(),
                                              cudf::binary_operator::NOT_EQUAL,
                                              cudf::data_type{cudf::type_id::BOOL8});
        auto any = cudf::reduce(differs->view(),
                                *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                cudf::data_type{cudf::type_id::BOOL8});
        fractional = any->is_valid() &&
                     static_cast<cudf::numeric_scalar<bool>*>(any.get())->value();
    } else {
        // Half away from zero, like llround(). cudf::round() (deprecated, but
        // the only floating-point rounding) does it in one kernel and exactly
        // past 2^52 microseconds (2112), where floor(|f| + 0.5) ties to even.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        whole = cudf::round(f->view(), 0, cudf::rounding_method::HALF_UP);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
    }
    auto ints = cudf::cast(whole->view(), int_type);
    return std::make_unique<cudf::column>(cudf::bit_cast(ints->view(), cudf::data_type{target}));
}

}  // namespace cuplyr

#endif  // CUPLYR_TS_AGG_HPP
