// src/ops_reframe.cpp
//
// GPU reshape for native reframe() (F22). reframe() is computed in two steps:
// a WIDE table with one row per group (the group keys, then one scalar
// aggregate per element of every multi-row result: `quantile(x, c(.1, .9))`
// becomes two columns), built by the ordinary summarise path, and this
// reshape to the LONG layout dplyr returns: k rows per group, groups in order,
// elements in expression order.
//
// Everything stays on the device, with no host-side n*k index vector:
// - a "rep" output (a group key, or a size-1 result) is the wide column with
//   every row repeated k times (`cudf::repeat(table, k)`);
// - an "interleave" output (a size-k result) takes its k wide columns and
//   interleaves them row-major (`cudf::interleave_columns()`:
//   r0c0, r0c1, ..., r1c0, ...), which is exactly "group by group, element by
//   element";
// - a "literal" output (`prob = c(.25, .5)`) is a tiny uploaded column of m
//   rows (m == 1 or k) gathered with the map `sequence(n * k) % m`, built on
//   the device.

#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <climits>
#include <memory>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

// xptr:    the wide table
// src:     one integer vector per output column, in output order. Length 1 =
//          repeat that (0-based) wide column; length k = interleave those wide
//          columns in order; length 0 = a literal (see lit_idx).
// lit_idx: per output column, the 1-based position in `lit_ptrs` of its
//          literal column (m rows, m == 1 or k), NA for a non-literal.
// k:       rows per group
// [[Rcpp::export]]
SEXP gpu_reframe_expand(SEXP xptr, List src, List lit_ptrs, IntegerVector lit_idx, int k) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view wide = get_table_view(ptr);
    if (k < 1) {
        Rcpp::stop("gpu_reframe_expand(): k must be at least 1");
    }
    const cudf::size_type n = wide.num_rows();
    if (static_cast<double>(n) * k > static_cast<double>(INT_MAX)) {
        Rcpp::stop("reframe() result has too many rows (max ~2.1 billion)");
    }
    const cudf::size_type total = n * k;
    const int n_out = src.size();

    std::vector<std::unique_ptr<cudf::column>> out(n_out);

    // --- rep outputs, repeated in one call -----------------------------------
    std::vector<cudf::column_view> rep_cols;
    std::vector<int> rep_out;
    for (int o = 0; o < n_out; ++o) {
        IntegerVector s = src[o];
        if (s.size() == 1) {
            if (s[0] < 0 || s[0] >= wide.num_columns()) {
                Rcpp::stop("Column index out of bounds: %d", s[0]);
            }
            rep_cols.push_back(wide.column(s[0]));
            rep_out.push_back(o);
        }
    }
    if (!rep_cols.empty()) {
        auto repeated = cudf::repeat(cudf::table_view(rep_cols), k)->release();
        for (size_t i = 0; i < rep_out.size(); ++i) {
            out[rep_out[i]] = std::move(repeated[i]);
        }
    }

    // --- interleave outputs --------------------------------------------------
    for (int o = 0; o < n_out; ++o) {
        IntegerVector s = src[o];
        if (s.size() <= 1) continue;
        if (s.size() != k) {
            Rcpp::stop("gpu_reframe_expand(): an output has %d sources, expected %d",
                       static_cast<int>(s.size()), k);
        }
        std::vector<cudf::column_view> cols;
        for (int j = 0; j < s.size(); ++j) {
            if (s[j] < 0 || s[j] >= wide.num_columns()) {
                Rcpp::stop("Column index out of bounds: %d", s[j]);
            }
            cols.push_back(wide.column(s[j]));
        }
        // The k columns can differ only by an INT32 vs FLOAT64 drift (an
        // integer min()/max() that met an empty group): widen to FLOAT64.
        bool same = true;
        for (const auto& c : cols) {
            if (c.type() != cols[0].type()) same = false;
        }
        std::vector<std::unique_ptr<cudf::column>> casted;
        if (!same) {
            for (auto& c : cols) {
                if (!cudf::is_numeric(c.type())) {
                    Rcpp::stop("reframe(): the elements of one result have different types");
                }
                casted.push_back(cudf::cast(c, cudf::data_type{cudf::type_id::FLOAT64}));
                c = casted.back()->view();
            }
        }
        out[o] = (n == 0) ? cudf::empty_like(cols[0])
                          : cudf::interleave_columns(cudf::table_view(cols));
    }

    // --- literal outputs -----------------------------------------------------
    std::unique_ptr<cudf::column> seq;
    for (int o = 0; o < n_out; ++o) {
        IntegerVector s = src[o];
        if (s.size() != 0) continue;
        if (IntegerVector::is_na(lit_idx[o])) {
            Rcpp::stop("gpu_reframe_expand(): output %d has neither a source nor a literal", o + 1);
        }
        Rcpp::XPtr<GpuTablePtr> lp(lit_ptrs[lit_idx[o] - 1]);
        cudf::table_view lit = get_table_view(lp);
        const cudf::size_type m = lit.num_rows();
        if (m != 1 && m != k) {
            Rcpp::stop("Can't recycle a literal of size %d to size %d", m, k);
        }
        if (total == 0) {
            out[o] = cudf::empty_like(lit.column(0));
            continue;
        }
        if (!seq) {
            cudf::numeric_scalar<cudf::size_type> zero(0);
            seq = cudf::sequence(total, zero);
        }
        cudf::numeric_scalar<cudf::size_type> mod(m);
        auto map = cudf::binary_operation(seq->view(), mod, cudf::binary_operator::MOD,
                                          cudf::data_type{cudf::type_id::INT32});
        auto gathered = cudf::gather(lit, map->view(), cudf::out_of_bounds_policy::DONT_CHECK);
        out[o] = std::move(gathered->release()[0]);
    }

    return make_gpu_table_xptr(std::make_unique<cudf::table>(std::move(out)));
}
