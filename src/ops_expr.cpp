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

// Remaps a "virtual" col-node index (an R-side ir_bind() index >= n_input,
// assuming expression k's output always lands at ctx slot n_input+k with no
// gaps -- see gpu_mutate_expr()'s own doc comment below) to the REAL ctx
// index that expression k's output actually landed at, as recorded by
// gpu_mutate_expr()'s main loop in `logical_to_real`.
//
// That "no gaps" assumption is only true when an expression's own
// evaluation pushes EXACTLY one ctx entry (its own final result). It breaks
// whenever an earlier expression's evaluation pushes extra internal scratch
// columns before its own top-level result -- e.g. a nested `%in%` or
// `if_else()`/`case_when()`/`coalesce()` handler materializing a
// sub-argument, or build_ast()'s rare column-level-cast fallback (see
// src/expr_eval.hpp). A concrete repro that predates Phase 4 and was caught
// by its oracle tests: `mutate(z = (x %in% c(1,2)) & (y > 3), w = z)` --
// evaluating `z` pushes ONE scratch column (the materialized `%in%`
// sub-result) before `z`'s own final AST result, so `z` actually lands at
// ctx index n_input+1, not n_input+0 as `w = z`'s R-bound `col` node
// (n_input + 0, since `z` is the first appended schema column) assumes.
// Without this remap, `w` would read the scratch `%in%` column instead of
// `z`'s real value.
//
// Returns a NEW Rcpp::List tree (never mutates `node` in place -- IR nodes
// are ordinary R lists, and the R side keeps its own references to the same
// objects across this call).
static Rcpp::List remap_virtual_col_indices(Rcpp::List node,
                                             int n_input,
                                             const std::vector<cudf::size_type>& logical_to_real) {
    using namespace cuplyr;
    std::string kind = ir_kind(node);

    if (kind == "lit") {
        return node;
    }

    if (kind == "col") {
        int idx = Rcpp::as<int>(node["index"]);
        if (idx >= n_input) {
            size_t logical_k = static_cast<size_t>(idx - n_input);
            if (logical_k < logical_to_real.size()) {
                Rcpp::List remapped = Rcpp::clone(node);
                remapped["index"] = logical_to_real[logical_k];
                return remapped;
            }
        }
        return node;
    }

    // kind == "call": remap every argument recursively.
    Rcpp::List args = node["args"];
    Rcpp::List new_args(args.size());
    for (int i = 0; i < args.size(); ++i) {
        new_args[i] = remap_virtual_col_indices(args[i], n_input, logical_to_real);
    }
    Rcpp::List new_node = Rcpp::clone(node);
    new_node["args"] = new_args;
    return new_node;
}

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
    // the original column's position -- i.e. R ASSUMES every expression's
    // top-level result lands at ctx index (n_input + i), with no gaps.
    //
    // That assumption holds whenever an expression's own evaluation pushes
    // EXACTLY one ctx entry, which is the common case -- but NOT always: a
    // nested handler op (`%in%`, `if_else()`/`case_when()`/`coalesce()`, or
    // build_ast()'s rare column-cast fallback) can push extra internal
    // scratch columns into ctx before an expression's own top-level result,
    // shifting every SUBSEQUENT expression's real index away from what R
    // assumed. `logical_to_real[k]` records expression k's REAL final ctx
    // index (after materialize_top_level() and any output-type cast) as
    // each is evaluated; `remap_virtual_col_indices()` rewrites any `col`
    // node in a LATER expression's IR that refers to an earlier expression's
    // (virtual, R-assumed) appended position to that real index before
    // materializing it. See remap_virtual_col_indices()'s own doc comment
    // for a concrete repro this fixes.
    std::map<std::string, cudf::size_type> output_col_indices;
    std::vector<cudf::size_type> logical_to_real;
    logical_to_real.reserve(static_cast<size_t>(n_expr));

    for (int i = 0; i < n_expr; ++i) {
        Rcpp::List node = remap_virtual_col_indices(exprs[i], n_input, logical_to_real);
        cudf::size_type idx = materialize_top_level(node, ctx);

        std::string out_type_str = Rcpp::as<std::string>(out_types[i]);
        cudf::data_type want = expr_type_from_str(out_type_str);
        // The R schema is authoritative: cast if the computed type differs
        // from the declared output type (section 2.2).
        if (ctx.cols[idx].type().id() != want.id()) {
            auto casted = cudf::cast(ctx.cols[idx], want);
            idx = ctx.push(std::move(casted));
        }

        logical_to_real.push_back(idx);
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
