// src/zero_sign.hpp
//
// F67: the zero sign of a double sum(), mean() or cumsum(). R accumulates
// these from +0, and +0 + -0 is +0, so sum() and cumsum() never return -0
// (their values are whole multiples of 2^-1074, so they never round to zero
// either). mean() divides the sum by n, so it is -0 when a negative sum (a
// negative subnormal) rounds to zero on the division, and +0 when the sum is
// zero. cudf's reductions and scans start from the first value and give -0
// for an all -0 group or prefix. gpu_summarise() and gpu_window() unsign the
// SUM and, for a mean(), divide afterwards, never the mean itself; the
// rowwise sum()/mean() fold (rw_agg_f64 in expr_eval.hpp) adds the +0 to its
// first operand instead.
#ifndef CUPLYR_ZERO_SIGN_HPP
#define CUPLYR_ZERO_SIGN_HPP

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/types.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

namespace cuplyr {

// Whether a valid row of a FLOAT64 column holds -0. Its bits read as INT64
// are INT64_MIN, and no other double has them, so the search runs on a
// zero-copy INT64 view. (A double -0.0 needle wouldn't do: FLOAT64 equality
// matches +0, and cudf's numeric_scalar stores any value == 0 as +0.)
// contains() only compares valid rows. Cost: one read of the column and a
// host sync.
inline bool has_negative_zero(cudf::column_view const& v) {
    if (v.type().id() != cudf::type_id::FLOAT64) return false;
    if (v.size() == 0 || v.null_count() == v.size()) return false;
    auto bits = cudf::bit_cast(v, cudf::data_type{cudf::type_id::INT64});
    cudf::numeric_scalar<int64_t> needle(std::numeric_limits<int64_t>::min(), true);
    return cudf::contains(bits, needle);
}

// Adds +0 to every row, which turns -0 into +0 and changes nothing else
// (NaN, +-Inf and nulls pass through). Allocates a new column.
inline void add_plus_zero(std::unique_ptr<cudf::column>& col) {
    col = cudf::binary_operation(col->view(), cudf::numeric_scalar<double>(0.0, true),
                                 cudf::binary_operator::ADD, col->type());
}

// Turns every -0 of a FLOAT64 sum/cumsum result into +0 (or of the sum a
// mean() is about to divide; never of a mean itself, which can be a true -0).
// The pass allocates a column, so a result longer than kUnsignAlways rows (a
// row-length scan, a summarise() over many groups) is searched first and
// left alone when it has no -0; a short per-group result takes the pass
// unconditionally, which is cheaper than the search's host sync. A
// non-FLOAT64 column is left alone.
inline void unsign_zero_sum(std::unique_ptr<cudf::column>& col) {
    constexpr cudf::size_type kUnsignAlways = 1 << 16;
    if (col->type().id() != cudf::type_id::FLOAT64) return;
    if (col->size() > kUnsignAlways && !has_negative_zero(col->view())) return;
    add_plus_zero(col);
}

// The same for the cumsum() scan of ONE group (an ungrouped window), in scan
// order and before the sticky-NA rule nulls every row after the first NA.
// A running sum is -0 only where every value so far is -0 (x + y rounds to
// -0 only for -0 + -0), so a -0 at row i needs rows 0..i all -0: the first
// row decides. A first row that is NA makes every later row NA, so nothing
// needs fixing then either. Cost: one element to the host.
inline void unsign_zero_scan_one_group(std::unique_ptr<cudf::column>& col) {
    if (col->type().id() != cudf::type_id::FLOAT64 || col->size() == 0) return;
    auto head = cudf::get_element(col->view(), 0);
    auto* h = static_cast<cudf::numeric_scalar<double>*>(head.get());
    if (!h->is_valid()) return;
    const double v = h->value();
    if (v == 0.0 && std::signbit(v)) add_plus_zero(col);
}

} // namespace cuplyr

#endif // CUPLYR_ZERO_SIGN_HPP
