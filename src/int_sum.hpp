// src/int_sum.hpp
//
// F34/F42: dplyr's sum() of an integer or logical column is an R integer. The
// GPU reduces in INT64 (cudf's groupby/reduce sum type for INT32/BOOL8); this
// narrows the result column afterwards. Shared by gpu_summarise() and the
// window "agg" family of gpu_window().
#ifndef CUPLYR_INT_SUM_HPP
#define CUPLYR_INT_SUM_HPP

#include <cudf/column/column.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <memory>
#include <string>

namespace cuplyr {

inline bool is_int_sum(const std::string& agg_type, cudf::type_id input_id) {
    return agg_type == "sum" &&
           (input_id == cudf::type_id::INT32 || input_id == cudf::type_id::BOOL8);
}

// If every value fits R's integer range (+-(2^31 - 1); -2^31 is NA_integer_
// in R) the column becomes INT32. Otherwise base R's sum() returns a double
// and one overflowing group makes the whole column double, so the whole
// column becomes FLOAT64 and the function returns true so the R side can
// reconcile (eager) or raise (lazy).
// Cost: one minmax reduction (two scalars to the host) plus one cast.
inline bool narrow_int_sum(std::unique_ptr<cudf::column>& col) {
    constexpr int64_t kRIntMax = 2147483647;  // R's .Machine$integer.max
    if (col->type().id() != cudf::type_id::INT64) {
        col = cudf::cast(col->view(), cudf::data_type{cudf::type_id::INT64});
    }
    bool overflow = false;
    if (col->size() > 0) {
        auto [lo, hi] = cudf::minmax(col->view());
        auto* lo_s = static_cast<cudf::numeric_scalar<int64_t>*>(lo.get());
        auto* hi_s = static_cast<cudf::numeric_scalar<int64_t>*>(hi.get());
        // An all-null column has no valid extremes and can't overflow.
        overflow = lo_s->is_valid() && hi_s->is_valid() &&
                   (lo_s->value() < -kRIntMax || hi_s->value() > kRIntMax);
    }
    col = cudf::cast(col->view(), cudf::data_type{
        overflow ? cudf::type_id::FLOAT64 : cudf::type_id::INT32});
    return overflow;
}

} // namespace cuplyr

#endif // CUPLYR_INT_SUM_HPP
