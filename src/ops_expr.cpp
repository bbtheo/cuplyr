// src/ops_expr.cpp
//
// The three expression-engine entry points (section 2.2). Fully dormant in
// task T2: nothing in R/filter.R or R/mutate.R calls these yet. IR nodes
// arrive as nested Rcpp::List with `col` nodes carrying 0-based `index`
// fields -- ir_bind() is expected to have already run on the R side (T3/T5
// wire this up; T2's tests hand-build already-bound IR directly).
#include "expr_eval.hpp"
#include "gpu_table.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_compute_column(SEXP xptr, Rcpp::List ir) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    eval_ctx ctx;
    for (int i = 0; i < view.num_columns(); ++i) {
        ctx.cols.push_back(view.column(i));
    }

    cudf::size_type idx = materialize_top_level(ir, ctx);
    // materialize_top_level() always pushes a fresh entry, so idx is always
    // within ctx.owned's range (never a bare alias into the input view).
    size_t owned_idx = static_cast<size_t>(idx - view.num_columns());
    std::unique_ptr<cudf::column> result_col = std::move(ctx.owned[owned_idx]);

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(result_col));
    auto tbl = std::make_unique<cudf::table>(std::move(cols));
    return make_gpu_table_xptr(std::move(tbl));
}

// [[Rcpp::export]]
SEXP gpu_filter_expr(SEXP xptr, Rcpp::List ir) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    eval_ctx ctx;
    for (int i = 0; i < view.num_columns(); ++i) {
        ctx.cols.push_back(view.column(i));
    }

    cudf::size_type idx = materialize(ir, ctx);
    cudf::column_view mask = ctx.cols[idx];

    if (mask.type().id() != cudf::type_id::BOOL8) {
        Rcpp::stop("filter() expressions must be logical");
    }

    auto result = cudf::apply_boolean_mask(view, mask);
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_mutate_expr(SEXP xptr,
                     Rcpp::List exprs,
                     Rcpp::CharacterVector out_names,
                     Rcpp::CharacterVector out_types,
                     Rcpp::CharacterVector input_names) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int n_input = view.num_columns();
    int n_expr = exprs.size();

    if (out_names.size() != n_expr || out_types.size() != n_expr) {
        Rcpp::stop("gpu_mutate_expr: exprs/out_names/out_types must have the same length");
    }
    if (input_names.size() != n_input) {
        Rcpp::stop("gpu_mutate_expr: input_names must have one entry per input column");
    }

    eval_ctx ctx;
    for (int i = 0; i < n_input; ++i) {
        ctx.cols.push_back(view.column(i));
    }

    // Sequential evaluation with append-and-remap (section 2.2): each
    // expression's IR was bound (ir_bind(), R side) against a running schema
    // where a replaced name's index points at the *appended* position, not
    // the original column's position -- so every expression's top-level
    // result must land at ctx index (n_input + i), matching
    // materialize_top_level()'s "always push, even for a bare column copy"
    // contract. Later expressions that reference an earlier expression's
    // output therefore find it by that same appended index.
    std::map<std::string, cudf::size_type> output_col_indices;

    for (int i = 0; i < n_expr; ++i) {
        Rcpp::List node = exprs[i];
        cudf::size_type idx = materialize_top_level(node, ctx);

        std::string out_type_str = Rcpp::as<std::string>(out_types[i]);
        cudf::data_type want = expr_type_from_str(out_type_str);
        // The R schema is authoritative: cast if the computed type differs
        // from the declared output type (section 2.2).
        if (ctx.cols[idx].type().id() != want.id()) {
            auto casted = cudf::cast(ctx.cols[idx], want);
            idx = ctx.push(std::move(casted));
        }

        output_col_indices[Rcpp::as<std::string>(out_names[i])] = idx;
    }

    // Final assembly (replace-in-place, then append new), lifted from
    // ops_mutate_batch.cpp:141-202.
    auto take_col = [&](cudf::size_type idx) -> std::unique_ptr<cudf::column> {
        if (idx < n_input) {
            return std::make_unique<cudf::column>(ctx.cols[idx]);  // deep copy of an unmodified input
        }
        size_t owned_idx = static_cast<size_t>(idx - n_input);
        return std::move(ctx.owned[owned_idx]);
    };

    std::vector<std::unique_ptr<cudf::column>> final_columns;
    final_columns.reserve(static_cast<size_t>(n_input + n_expr));
    std::set<std::string> added;

    for (int i = 0; i < n_input; ++i) {
        std::string name = Rcpp::as<std::string>(input_names[i]);
        auto it = output_col_indices.find(name);
        if (it != output_col_indices.end()) {
            final_columns.push_back(take_col(it->second));
            added.insert(name);
        } else {
            final_columns.push_back(std::make_unique<cudf::column>(ctx.cols[i]));
        }
    }

    for (int i = 0; i < n_expr; ++i) {
        std::string name = Rcpp::as<std::string>(out_names[i]);
        if (!added.count(name)) {
            final_columns.push_back(take_col(output_col_indices[name]));
            added.insert(name);
        }
    }

    auto result = std::make_unique<cudf::table>(std::move(final_columns));
    return make_gpu_table_xptr(std::move(result));
}
