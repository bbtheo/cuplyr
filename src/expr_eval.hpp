// src/expr_eval.hpp
//
// The shared expression evaluator: rlang-derived IR (crossed from R as a
// nested Rcpp::List, see R/ir.R and scratchpad/phase1_expression_engine.md
// section 2) -> cudf::compute_column() / dedicated cuDF calls.
//
// Design authority: scratchpad/phase1_expression_engine.md section 2.2 (the
// eval_ctx / ast_expressible / build_ast / materialize mutual recursion) and
// section 2.3 (the exact op-name -> cudf::ast::ast_operator / handler-name
// mapping). This file is dormant in task T2: nothing in ops_expr.cpp is
// wired into any dplyr verb yet.
//
// --- Deviations from the section 2.2 pseudocode, and why ---
//
// 1. `ast_expressible(Rcpp::List node)` in the design doc takes only the
//    node. In practice, deciding "is this col node's type STRING" requires
//    looking at the *actual* column type, and a `col` IR node only carries
//    `name`/`index` (its GPU type lives in the R-side schema, never
//    serialized onto the node itself, see R/ir.R). So `ast_expressible()`
//    here also takes the `eval_ctx&` and reads `ctx.cols[index].type()`.
//
// 2. `gpu_mutate_expr()`'s signature in the design doc omits the input
//    table's column *names*, but "final assembly (replace-in-place, then
//    append)" (lifted from ops_mutate_batch.cpp:141-202) is a name-keyed
//    operation, and a `cudf::table` carries no column-name metadata at all
//    (names are pure R-side schema state). `ops_expr.cpp::gpu_mutate_expr()`
//    therefore takes one extra `Rcpp::CharacterVector input_names`
//    parameter alongside `out_names`/`out_types`.
//
// 3. The design's pseudocode has `build_ast()` return `ast::expression
//    const&` given just `(node, tree, scalars, ctx)`. Implemented here with
//    one more parameter, `type_hint` (a GPU type string, or "" for none):
//    needed for the two places section 1.2/2.2 say a bare, untyped `NA`
//    literal ("kind=lit, na=TRUE, type=NULL") must "adopt the sibling
//    operand's type at build time" -- and for unary `-x`'s zero literal,
//    which must match `x`'s type (INT32 stays INT32, not promoted to
//    FLOAT64). Both are resolved via `resolve_static_type()` below, a
//    best-effort static type inference over the *op-class* rules in
//    section 1.3 (mirroring `ir_infer_type()`/`infer_mutate_output_type()`
//    on the R side, but not calling into R): exact for col/lit nodes and
//    for the single-level op classes (comparisons/logic -> BOOL8, sqrt et
//    al -> FLOAT64, abs/round -> arg type), an approximation (falls back to
//    a promotion-hierarchy scan of argument types) for nested arithmetic.
//    The R-supplied `output_type` is always the authoritative final answer
//    (enforced by the cast-to-declared-type step in ops_expr.cpp); this
//    resolver is *only* ever used to type an intermediate literal that has
//    no declared type of its own.
#ifndef CUPLYR_EXPR_EVAL_HPP
#define CUPLYR_EXPR_EVAL_HPP

#include "gpu_table.hpp"
#include "ops_common.hpp"
#include "r_to_column.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/replace.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <Rcpp.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace cuplyr {

// -----------------------------------------------------------------------------
// GPU type string <-> cudf::data_type
// -----------------------------------------------------------------------------

inline cudf::data_type expr_type_from_str(const std::string& type_str) {
    if (type_str == "FLOAT64") return cudf::data_type{cudf::type_id::FLOAT64};
    if (type_str == "FLOAT32") return cudf::data_type{cudf::type_id::FLOAT32};
    if (type_str == "INT64")   return cudf::data_type{cudf::type_id::INT64};
    if (type_str == "INT32")   return cudf::data_type{cudf::type_id::INT32};
    if (type_str == "INT16")   return cudf::data_type{cudf::type_id::INT16};
    if (type_str == "INT8")    return cudf::data_type{cudf::type_id::INT8};
    if (type_str == "BOOL8")   return cudf::data_type{cudf::type_id::BOOL8};
    if (type_str == "STRING")  return cudf::data_type{cudf::type_id::STRING};
    Rcpp::stop("Unsupported GPU type string: %s", type_str.c_str());
}

inline std::string expr_type_to_str(cudf::data_type type) {
    switch (type.id()) {
        case cudf::type_id::FLOAT64: return "FLOAT64";
        case cudf::type_id::FLOAT32: return "FLOAT32";
        case cudf::type_id::INT64:   return "INT64";
        case cudf::type_id::INT32:   return "INT32";
        case cudf::type_id::INT16:   return "INT16";
        case cudf::type_id::INT8:    return "INT8";
        case cudf::type_id::BOOL8:   return "BOOL8";
        case cudf::type_id::STRING:  return "STRING";
        default: return "FLOAT64";
    }
}

// -----------------------------------------------------------------------------
// eval_ctx: the working table for one evaluation. `cols` starts as the input
// table's column views; every intermediate result (AST-fused subtree or
// handler dispatch) is appended by push(), which keeps `owned` (storage) and
// `cols` (views into that storage, used by column_reference indices and by
// ast_expressible()) in lockstep. Indices only ever grow -- nothing is ever
// reordered or removed -- so any column_reference built earlier stays valid
// for the lifetime of the ctx.
// -----------------------------------------------------------------------------

struct eval_ctx {
    std::vector<cudf::column_view> cols;
    std::vector<std::unique_ptr<cudf::column>> owned;

    cudf::table_view view() const { return cudf::table_view(cols); }

    cudf::size_type push(std::unique_ptr<cudf::column> col) {
        cudf::column_view v = col->view();
        owned.push_back(std::move(col));
        cols.push_back(v);
        return static_cast<cudf::size_type>(cols.size() - 1);
    }
};

// -----------------------------------------------------------------------------
// IR node field accessors (Rcpp::List, see R/ir.R node shapes)
// -----------------------------------------------------------------------------

inline std::string ir_kind(Rcpp::List node) {
    return Rcpp::as<std::string>(node["kind"]);
}

inline bool ir_lit_is_na(Rcpp::List node) {
    SEXP na_sexp = node["na"];
    return !Rf_isNull(na_sexp) && Rcpp::as<bool>(na_sexp);
}

// -----------------------------------------------------------------------------
// Section 2.3 op tables: IR canonical op name -> cudf AST operator, or ->
// non-AST handler name. Adding a new AST-native op per the section 6
// extension contract is one line here; adding a handler op is one line here
// plus one `apply_handler()` case.
// -----------------------------------------------------------------------------

inline bool ast_op_from_name(const std::string& op, cudf::ast::ast_operator& out) {
    static const std::map<std::string, cudf::ast::ast_operator> table = {
        {"+", cudf::ast::ast_operator::ADD},
        {"-", cudf::ast::ast_operator::SUB},  // unary handled specially in build_ast()
        {"*", cudf::ast::ast_operator::MUL},
        {"/", cudf::ast::ast_operator::TRUE_DIV},
        {"^", cudf::ast::ast_operator::POW},
        {"%%", cudf::ast::ast_operator::PYMOD},
        {"%/%", cudf::ast::ast_operator::FLOOR_DIV},
        {"==", cudf::ast::ast_operator::EQUAL},
        {"!=", cudf::ast::ast_operator::NOT_EQUAL},
        {"<", cudf::ast::ast_operator::LESS},
        {"<=", cudf::ast::ast_operator::LESS_EQUAL},
        {">", cudf::ast::ast_operator::GREATER},
        {">=", cudf::ast::ast_operator::GREATER_EQUAL},
        {"&", cudf::ast::ast_operator::NULL_LOGICAL_AND},
        {"|", cudf::ast::ast_operator::NULL_LOGICAL_OR},
        {"!", cudf::ast::ast_operator::NOT},
        {"xor", cudf::ast::ast_operator::BITWISE_XOR},
        {"is.na", cudf::ast::ast_operator::IS_NULL},
        {"sqrt", cudf::ast::ast_operator::SQRT},
        {"log", cudf::ast::ast_operator::LOG},
        {"exp", cudf::ast::ast_operator::EXP},
        {"abs", cudf::ast::ast_operator::ABS},
        {"floor", cudf::ast::ast_operator::FLOOR},
        {"ceil", cudf::ast::ast_operator::CEIL},
        {"sin", cudf::ast::ast_operator::SIN},
        {"cos", cudf::ast::ast_operator::COS},
        {"tan", cudf::ast::ast_operator::TAN},
    };
    auto it = table.find(op);
    if (it == table.end()) return false;
    out = it->second;
    return true;
}

// STRING comparisons are lowered by a non-AST handler (section 0: "the AST:
// decided, no empirical risk taken"); %in% and round() likewise never have
// an ast_op entry above, so ast_op_from_name() returning false for them is
// exactly the "needs a handler" signal used throughout this file.
inline bool ir_op_is_comparison(const std::string& op) {
    static const std::set<std::string> cmp = {"==", "!=", "<", "<=", ">", ">="};
    return cmp.count(op) > 0;
}

// op_dispatches_to_ast(): materialize()'s top-level "build_ast+compute_column
// vs apply_handler" decision for a call node. This is deliberately NOT the
// same question as ast_expressible() (which asks "does this WHOLE subtree,
// recursively, collapse into one kernel" -- used only to decide, per
// argument, whether build_ast() should inline a child or materialize() it
// separately). A node can dispatch to the AST path even when one of its
// *children* is a non-expressible handler op (e.g. `(x %in% c(1,2)) & (y >
// 0)`: `&` is genuinely AST-native, and build_ast() handles the
// non-expressible `%in%` child itself by materializing it and referencing it
// via column_reference) -- reusing ast_expressible() here would wrongly send
// such nodes to apply_handler(), which has no case for "&". The one
// exception: a *comparison* op ("==" etc.) whose own immediate operands are
// STRING must go to the handler even though the op name has an ast_op entry
// (section 0's STRING carve-out) -- checked shallowly here (col type from
// ctx, lit's own declared type; a "call" operand producing STRING doesn't
// occur in Phase 1's scope, so it isn't checked).
inline bool op_dispatches_to_ast(const std::string& op, Rcpp::List node, const eval_ctx& ctx) {
    cudf::ast::ast_operator dummy;
    if (!ast_op_from_name(op, dummy)) return false;

    if (ir_op_is_comparison(op)) {
        Rcpp::List args = node["args"];
        for (int i = 0; i < args.size(); ++i) {
            Rcpp::List a = args[i];
            std::string ak = ir_kind(a);
            if (ak == "col") {
                int idx = Rcpp::as<int>(a["index"]);
                if (ctx.cols[idx].type().id() == cudf::type_id::STRING) return false;
            } else if (ak == "lit") {
                SEXP t = a["type"];
                if (!Rf_isNull(t) && Rcpp::as<std::string>(t) == "STRING") return false;
            }
        }
    }
    return true;
}

// The only two widening AST cast targets the ast_operator enum offers
// (CAST_TO_UINT64 is not used here: this schema's type universe never needs
// an unsigned target). See build_ast()'s "AST operand type matching" note.
inline bool ast_cast_op_for_type(cudf::type_id target, cudf::ast::ast_operator& out) {
    switch (target) {
        case cudf::type_id::INT64:
            out = cudf::ast::ast_operator::CAST_TO_INT64;
            return true;
        case cudf::type_id::FLOAT64:
            out = cudf::ast::ast_operator::CAST_TO_FLOAT64;
            return true;
        default:
            return false;
    }
}

// -----------------------------------------------------------------------------
// Forward declarations (mutual recursion, section 2.2)
// -----------------------------------------------------------------------------

inline bool ast_expressible(Rcpp::List node, const eval_ctx& ctx);
inline cudf::ast::expression const& build_ast(Rcpp::List node,
                                               cudf::ast::tree& tree,
                                               std::vector<std::unique_ptr<cudf::scalar>>& scalars,
                                               eval_ctx& ctx,
                                               const std::string& type_hint);
inline cudf::size_type materialize(Rcpp::List node, eval_ctx& ctx);
inline std::unique_ptr<cudf::column> apply_handler(Rcpp::List node, eval_ctx& ctx);

// Promotion hierarchy (D5 / section 1.3): BOOL8 < INT32 < INT64 < FLOAT64.
// Shared by resolve_static_type()'s arithmetic-op branch and by build_ast()'s
// operand type-matching (see the "AST operand type matching" note below).
inline int promotion_rank(cudf::type_id id) {
    switch (id) {
        case cudf::type_id::BOOL8: return 0;
        case cudf::type_id::INT32: return 1;
        case cudf::type_id::INT64: return 2;
        default: return 3;  // FLOAT64 and anything else
    }
}

inline cudf::type_id type_by_promotion_rank(int rank) {
    static const cudf::type_id table[4] = {
        cudf::type_id::BOOL8, cudf::type_id::INT32, cudf::type_id::INT64, cudf::type_id::FLOAT64
    };
    return table[rank];
}

// -----------------------------------------------------------------------------
// resolve_static_type(): best-effort static type inference, see the "1." /
// "3." deviation notes at the top of this file. Mirrors the op-class table
// in scratchpad/phase1_expression_engine.md section 1.3 / R/ir.R's registry
// `type` functions, without calling back into R.
// -----------------------------------------------------------------------------

inline cudf::data_type resolve_static_type(Rcpp::List node, const eval_ctx& ctx) {
    std::string kind = ir_kind(node);

    if (kind == "col") {
        int idx = Rcpp::as<int>(node["index"]);
        return ctx.cols[idx].type();
    }

    if (kind == "lit") {
        SEXP type_sexp = node["type"];
        if (Rf_isNull(type_sexp)) return cudf::data_type{cudf::type_id::FLOAT64};
        return expr_type_from_str(Rcpp::as<std::string>(type_sexp));
    }

    // kind == "call"
    std::string op = Rcpp::as<std::string>(node["op"]);
    Rcpp::List args = node["args"];

    static const std::set<std::string> bool_ops = {
        "==", "!=", "<", "<=", ">", ">=", "&", "|", "!", "xor", "is.na", "%in%", "between"
    };
    if (bool_ops.count(op)) return cudf::data_type{cudf::type_id::BOOL8};

    static const std::set<std::string> float_ops = {
        "/", "^", "sqrt", "log", "exp", "floor", "ceil", "sin", "cos", "tan"
    };
    if (float_ops.count(op)) return cudf::data_type{cudf::type_id::FLOAT64};

    if (op == "-" && args.size() == 1) {
        // unary minus preserves the operand's type
        return resolve_static_type(args[0], ctx);
    }

    if (op == "abs" || op == "round") {
        return resolve_static_type(args[0], ctx);
    }

    if (op == "+" || op == "-" || op == "*" || op == "%%" || op == "%/%") {
        // Approximates infer_mutate_output_type() for the purpose of typing an
        // intermediate literal only -- the real output_type always comes from R.
        int best = 0;
        for (int i = 0; i < args.size(); ++i) {
            cudf::data_type t = resolve_static_type(args[i], ctx);
            int r = promotion_rank(t.id());
            if (r > best) best = r;
        }
        return cudf::data_type{type_by_promotion_rank(best)};
    }

    // %in%'s first argument type, or any other unrecognized op: safe FLOAT64
    // default (matches ir_infer_type()'s bare-NA-in-isolation fallback).
    return cudf::data_type{cudf::type_id::FLOAT64};
}

// -----------------------------------------------------------------------------
// make_literal(): the scalar-ownership pattern (section 2.2). `scalars`
// owns the concrete cudf::scalar; `tree` owns the ast::literal referencing
// it. Both must outlive the compute_column() call that uses the resulting
// tree. NA handling: `force_na` (the IR node's own `na` flag) OR the R
// value itself being NA-for-its-type both produce a scalar constructed with
// is_valid = false (`set_valid`-equivalent via the scalar's own ctor).
// -----------------------------------------------------------------------------

inline std::unique_ptr<cudf::scalar> build_scalar(const std::string& type_str,
                                                   SEXP value_sexp,
                                                   bool force_na) {
    cudf::data_type dtype = expr_type_from_str(type_str);
    bool is_na = force_na || Rf_isNull(value_sexp);

    switch (dtype.id()) {
        case cudf::type_id::FLOAT64: {
            double v = 0.0;
            if (!is_na) {
                v = Rcpp::as<double>(value_sexp);
                if (Rcpp::NumericVector::is_na(v)) is_na = true;
            }
            return std::make_unique<cudf::numeric_scalar<double>>(v, !is_na);
        }
        case cudf::type_id::INT32: {
            int32_t v = 0;
            if (!is_na) {
                int rv = Rcpp::as<int>(value_sexp);
                if (rv == NA_INTEGER) {
                    is_na = true;
                } else {
                    v = static_cast<int32_t>(rv);
                }
            }
            return std::make_unique<cudf::numeric_scalar<int32_t>>(v, !is_na);
        }
        case cudf::type_id::INT64: {
            int64_t v = 0;
            if (!is_na) {
                double rv = Rcpp::as<double>(value_sexp);
                if (Rcpp::NumericVector::is_na(rv)) {
                    is_na = true;
                } else {
                    v = static_cast<int64_t>(rv);
                }
            }
            return std::make_unique<cudf::numeric_scalar<int64_t>>(v, !is_na);
        }
        case cudf::type_id::BOOL8: {
            bool v = false;
            if (!is_na) {
                int rv = Rcpp::as<int>(value_sexp);
                if (rv == NA_LOGICAL) {
                    is_na = true;
                } else {
                    v = (rv != 0);
                }
            }
            return std::make_unique<cudf::numeric_scalar<bool>>(v, !is_na);
        }
        case cudf::type_id::STRING: {
            std::string v;
            if (!is_na) {
                if (Rcpp::CharacterVector::is_na(value_sexp)) {
                    is_na = true;
                } else {
                    v = Rcpp::as<std::string>(value_sexp);
                }
            }
            return std::make_unique<cudf::string_scalar>(v, !is_na);
        }
        default:
            Rcpp::stop("Unsupported literal type: %s", type_str.c_str());
    }
}

// Wraps a just-built scalar in an ast::literal and hands ownership of the
// scalar to `scalars` so it outlives the tree/compute_column() call.
inline cudf::ast::literal const& emplace_literal(std::unique_ptr<cudf::scalar> s,
                                                  cudf::ast::tree& tree,
                                                  std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
    scalars.push_back(std::move(s));
    cudf::scalar& ref = *scalars.back();
    switch (ref.type().id()) {
        case cudf::type_id::FLOAT64:
            return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<double>&>(ref));
        case cudf::type_id::INT32:
            return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int32_t>&>(ref));
        case cudf::type_id::INT64:
            return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int64_t>&>(ref));
        case cudf::type_id::BOOL8:
            return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<bool>&>(ref));
        case cudf::type_id::STRING:
            return tree.emplace<cudf::ast::literal>(static_cast<cudf::string_scalar&>(ref));
        default:
            Rcpp::stop("Unsupported literal scalar type");
    }
}

inline cudf::ast::literal const& make_literal(const std::string& type_str,
                                               SEXP value_sexp,
                                               bool force_na,
                                               cudf::ast::tree& tree,
                                               std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
    return emplace_literal(build_scalar(type_str, value_sexp, force_na), tree, scalars);
}

// -----------------------------------------------------------------------------
// ast_expressible(): true iff `node` collapses into (part of) a single
// compute_column() kernel. See deviation note "1." at the top of this file
// for why `ctx` is required (col nodes carry no type of their own).
// -----------------------------------------------------------------------------

inline bool ast_expressible(Rcpp::List node, const eval_ctx& ctx) {
    std::string kind = ir_kind(node);

    if (kind == "col") {
        int idx = Rcpp::as<int>(node["index"]);
        return ctx.cols[idx].type().id() != cudf::type_id::STRING;
    }

    if (kind == "lit") {
        SEXP type_sexp = node["type"];
        if (Rf_isNull(type_sexp)) return true;  // bare NA: type resolved at build time, assumed non-string
        return Rcpp::as<std::string>(type_sexp) != "STRING";
    }

    // kind == "call"
    std::string op = Rcpp::as<std::string>(node["op"]);
    cudf::ast::ast_operator dummy;
    if (!ast_op_from_name(op, dummy)) return false;

    Rcpp::List args = node["args"];
    for (int i = 0; i < args.size(); ++i) {
        if (!ast_expressible(args[i], ctx)) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// build_ast(): builds `node` (assumed ast_expressible(), possibly with
// non-expressible descendants that get materialize()'d and referenced by
// column_reference) into `tree`, returning a reference into it.
// -----------------------------------------------------------------------------

inline cudf::ast::expression const& build_ast(Rcpp::List node,
                                               cudf::ast::tree& tree,
                                               std::vector<std::unique_ptr<cudf::scalar>>& scalars,
                                               eval_ctx& ctx,
                                               const std::string& type_hint) {
    std::string kind = ir_kind(node);

    if (kind == "col") {
        int idx = Rcpp::as<int>(node["index"]);
        return tree.emplace<cudf::ast::column_reference>(idx);
    }

    if (kind == "lit") {
        bool na = ir_lit_is_na(node);
        std::string type_str;
        if (!type_hint.empty()) {
            // Caller wants this literal built as a specific type -- either a
            // bare NA adopting its sibling's type (section 1.2), or a typed
            // literal that must be *retyped* to match a mismatched sibling's
            // type (see the "AST operand type matching" note on build_ast()
            // above the call-handling code below).
            type_str = type_hint;
        } else if (!Rf_isNull(node["type"])) {
            type_str = Rcpp::as<std::string>(node["type"]);
        } else {
            type_str = "FLOAT64";  // isolated bare NA, no hint available
        }
        return make_literal(type_str, node["value"], na, tree, scalars);
    }

    // kind == "call"
    std::string op = Rcpp::as<std::string>(node["op"]);
    Rcpp::List args = node["args"];

    cudf::ast::ast_operator ast_op;
    if (!ast_op_from_name(op, ast_op)) {
        // Not actually ast-native (shouldn't happen if the caller checked
        // ast_expressible() first) -- fall back to materializing it whole.
        cudf::size_type idx = materialize(node, ctx);
        return tree.emplace<cudf::ast::column_reference>(idx);
    }

    // --- AST operand type matching ---
    // Header-signature surprise (verified empirically, not documented in
    // section 0): libcudf's AST evaluator requires a binary operation's two
    // operands to have EXACTLY matching cudf types -- it throws "An AST
    // expression was provided non-matching operand types" otherwise, even
    // for combinations that would freely promote in R (e.g. an INT32 column
    // and a FLOAT64 literal for `/`). Matching-type inputs DO still promote
    // correctly on the *output* side (confirmed empirically: POW(INT32,
    // INT32) -> FLOAT64; TRUE_DIV(INT32, INT32) -> FLOAT64) -- the AST layer
    // just never promotes *inputs* for you.
    //
    // Fix: resolve each argument's static type (treating a bare-NA literal
    // as adopting its sibling's type, per section 1.2), and for a binary op
    // whose two operand types differ, retype/cast the narrower one up to
    // the wider one (promotion_rank(), the same BOOL8<INT32<INT64<FLOAT64
    // hierarchy as resolve_static_type()'s arithmetic branch) before
    // building it:
    //   - a literal operand is simply rebuilt at the target type (no cast
    //     op needed -- see the lit branch above using `type_hint`).
    //   - a column/expression operand is wrapped in an AST CAST_TO_INT64 /
    //     CAST_TO_FLOAT64 operation() when the target is one of those two
    //     (the only two widening targets the ast_operator enum offers; this
    //     covers every mismatch this schema's type universe can produce
    //     except a bare BOOL8-vs-INT32 mismatch promoting to INT32, which
    //     has no AST cast target at all -- for that one case only, fall
    //     back to materializing the operand as a real column and casting it
    //     with cudf::cast() (unary.hpp), which supports any target type.
    bool is_binary = args.size() == 2;
    std::vector<cudf::data_type> resolved(args.size());
    std::vector<bool> is_unresolved_na(args.size(), false);
    for (int i = 0; i < args.size(); ++i) {
        Rcpp::List a = args[i];
        is_unresolved_na[i] = ir_kind(a) == "lit" && Rf_isNull(a["type"]) && ir_lit_is_na(a);
        if (!is_unresolved_na[i]) resolved[i] = resolve_static_type(a, ctx);
    }
    for (int i = 0; i < args.size(); ++i) {
        if (!is_unresolved_na[i]) continue;
        bool found = false;
        for (int j = 0; j < args.size(); ++j) {
            if (j == i || is_unresolved_na[j]) continue;
            resolved[i] = resolved[j];
            found = true;
            break;
        }
        if (!found) resolved[i] = cudf::data_type{cudf::type_id::FLOAT64};
    }

    cudf::data_type target = resolved[0];
    if (is_binary && promotion_rank(resolved[1].id()) > promotion_rank(resolved[0].id())) {
        target = resolved[1];
    }

    std::vector<std::reference_wrapper<cudf::ast::expression const>> operand_refs;
    operand_refs.reserve(args.size());
    for (int i = 0; i < args.size(); ++i) {
        Rcpp::List a = args[i];
        bool needs_cast = is_binary && !is_unresolved_na[i] && resolved[i].id() != target.id();
        std::string hint = is_unresolved_na[i] || (needs_cast && ir_kind(a) == "lit")
                                ? expr_type_to_str(target)
                                : std::string();

        if (ir_kind(a) == "lit") {
            // Bare-NA adoption or a mismatched typed literal: both handled
            // by the lit branch's `hint` override -- rebuild at the target
            // type directly, no AST cast operation needed.
            operand_refs.push_back(build_ast(a, tree, scalars, ctx, hint));
            continue;
        }

        cudf::ast::expression const& built = ast_expressible(a, ctx)
            ? build_ast(a, tree, scalars, ctx, hint)
            : tree.emplace<cudf::ast::column_reference>(materialize(a, ctx));

        if (!needs_cast) {
            operand_refs.push_back(built);
            continue;
        }

        cudf::ast::ast_operator cast_op;
        if (ast_cast_op_for_type(target.id(), cast_op)) {
            operand_refs.push_back(tree.emplace<cudf::ast::operation>(cast_op, built));
        } else {
            // Rare fallback (e.g. BOOL8-vs-INT32 promoting to INT32, which
            // has no AST cast target): materialize the operand as a real
            // column and cast it at the column level instead, referencing
            // the result by column_reference.
            cudf::size_type idx = materialize(a, ctx);
            auto casted = cudf::cast(ctx.cols[idx], target);
            cudf::size_type casted_idx = ctx.push(std::move(casted));
            operand_refs.push_back(tree.emplace<cudf::ast::column_reference>(casted_idx));
        }
    }

    // Unary `-x`: no NEG in the ast_operator enum (section 2.3); lower to
    // SUB(typed-zero, x) so INT32 stays INT32 rather than promoting to
    // FLOAT64 the way a naive "0.0 - x" would.
    if (op == "-" && args.size() == 1) {
        std::string zero_type = expr_type_to_str(resolve_static_type(args[0], ctx));
        cudf::ast::expression const& zero_lit = make_literal(zero_type, Rcpp::wrap(0), false, tree, scalars);
        return tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::SUB, zero_lit, operand_refs[0].get());
    }

    if (operand_refs.size() == 1) {
        return tree.emplace<cudf::ast::operation>(ast_op, operand_refs[0].get());
    }
    return tree.emplace<cudf::ast::operation>(ast_op, operand_refs[0].get(), operand_refs[1].get());
}

// -----------------------------------------------------------------------------
// apply_handler(): the non-AST cuDF calls (section 2.3's second table).
// Column-producing arguments are materialize()'d; literal-only arguments
// (round()'s decimal places, %in%'s RHS set) are read directly off the IR
// node since they are host-side values, not per-row columns.
// -----------------------------------------------------------------------------

inline std::unique_ptr<cudf::column> apply_handler(Rcpp::List node, eval_ctx& ctx) {
    std::string op = Rcpp::as<std::string>(node["op"]);
    Rcpp::List args = node["args"];

    // ---- %in%: contains(haystack, needles) + the never-NA post-step ----
    if (op == "%in%") {
        Rcpp::List x_node = args[0];
        Rcpp::List set_node = args[1];

        cudf::size_type x_idx = materialize(x_node, ctx);
        cudf::column_view x_col = ctx.cols[x_idx];

        SEXP set_type_sexp = set_node["type"];
        std::string set_type = Rf_isNull(set_type_sexp) ? "FLOAT64" : Rcpp::as<std::string>(set_type_sexp);
        SEXP set_value = set_node["value"];
        cudf::data_type set_dtype = expr_type_from_str(set_type);

        std::unique_ptr<cudf::column> haystack;
        bool set_has_na = false;

        switch (set_dtype.id()) {
            case cudf::type_id::FLOAT64: {
                Rcpp::NumericVector v(set_value);
                for (R_xlen_t i = 0; i < v.size(); ++i) {
                    if (Rcpp::NumericVector::is_na(v[i])) set_has_na = true;
                }
                haystack = numeric_to_gpu(v);
                break;
            }
            case cudf::type_id::INT32: {
                Rcpp::IntegerVector v(set_value);
                for (R_xlen_t i = 0; i < v.size(); ++i) {
                    if (Rcpp::IntegerVector::is_na(v[i])) set_has_na = true;
                }
                haystack = integer_to_gpu(v);
                break;
            }
            case cudf::type_id::STRING: {
                Rcpp::CharacterVector v(set_value);
                for (R_xlen_t i = 0; i < v.size(); ++i) {
                    if (Rcpp::CharacterVector::is_na(v[i])) set_has_na = true;
                }
                haystack = character_to_gpu(v);
                break;
            }
            case cudf::type_id::BOOL8: {
                Rcpp::LogicalVector v(set_value);
                for (R_xlen_t i = 0; i < v.size(); ++i) {
                    if (Rcpp::LogicalVector::is_na(v[i])) set_has_na = true;
                }
                haystack = logical_to_gpu(v);
                break;
            }
            default:
                Rcpp::stop("Unsupported %%in%% set type: %s", set_type.c_str());
        }

        if (haystack->type().id() != x_col.type().id()) {
            haystack = cudf::cast(haystack->view(), x_col.type());
        }

        auto contains_result = cudf::contains(haystack->view(), x_col);

        // R's %in% never returns NA: NA %in% c(1,2) is FALSE, NA %in% c(NA,1)
        // is TRUE. contains()'s output null mask mirrors `needles`' (x_col's)
        // nulls; fill them with whether the haystack itself contained NA.
        cudf::numeric_scalar<bool> fill_value(set_has_na, true);
        return cudf::replace_nulls(contains_result->view(), fill_value);
    }

    // ---- round(x) / round(x, n): banker's rounding (HALF_EVEN) ----
    if (op == "round") {
        Rcpp::List x_node = args[0];
        cudf::size_type x_idx = materialize(x_node, ctx);
        cudf::column_view x_col = ctx.cols[x_idx];

        int32_t places = 0;
        if (args.size() > 1) {
            Rcpp::List places_node = args[1];
            places = static_cast<int32_t>(Rcpp::as<double>(places_node["value"]));
        }

        // Header-signature surprise (verified empirically, not just from the
        // docstring): cudf::round_decimal() only accepts integral or
        // fixed-point columns -- it throws "Only integral or fixed point
        // currently supported by round_decimal" for FLOAT32/FLOAT64, despite
        // section 0 treating it as a straightforward replacement for the
        // deprecated cudf::round() (whose own docstring examples ARE
        // floating-point). dplyr's round() is used overwhelmingly on
        // doubles, so: use round_decimal() for integral/fixed-point inputs
        // (where it is in fact the right, non-deprecated call and rounding
        // is a no-op anyway at decimal_places >= 0), and fall back to the
        // deprecated cudf::round() -- which does support HALF_EVEN on
        // floating-point columns -- for FLOAT32/FLOAT64.
        if (x_col.type().id() == cudf::type_id::FLOAT32 || x_col.type().id() == cudf::type_id::FLOAT64) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            auto result = cudf::round(x_col, places, cudf::rounding_method::HALF_EVEN);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
            return result;
        }

        return cudf::round_decimal(x_col, places, cudf::rounding_method::HALF_EVEN);
    }

    // ---- string ==/!=/</<=/>/>=: binary_operation + get_compare_op() ----
    if (ir_op_is_comparison(op)) {
        Rcpp::List lhs_node = args[0];
        Rcpp::List rhs_node = args[1];
        cudf::binary_operator bop = get_compare_op(op);
        cudf::data_type out_type{cudf::type_id::BOOL8};

        bool lhs_is_lit = ir_kind(lhs_node) == "lit";
        bool rhs_is_lit = ir_kind(rhs_node) == "lit";

        if (!lhs_is_lit && !rhs_is_lit) {
            cudf::size_type li = materialize(lhs_node, ctx);
            cudf::size_type ri = materialize(rhs_node, ctx);
            return cudf::binary_operation(ctx.cols[li], ctx.cols[ri], bop, out_type);
        }
        if (rhs_is_lit && !lhs_is_lit) {
            cudf::size_type li = materialize(lhs_node, ctx);
            bool na = ir_lit_is_na(rhs_node);
            std::string s = na ? std::string() : Rcpp::as<std::string>(rhs_node["value"]);
            cudf::string_scalar sc(s, !na);
            return cudf::binary_operation(ctx.cols[li], sc, bop, out_type);
        }
        if (lhs_is_lit && !rhs_is_lit) {
            cudf::size_type ri = materialize(rhs_node, ctx);
            bool na = ir_lit_is_na(lhs_node);
            std::string s = na ? std::string() : Rcpp::as<std::string>(lhs_node["value"]);
            cudf::string_scalar sc(s, !na);
            return cudf::binary_operation(sc, ctx.cols[ri], bop, out_type);
        }
        // both literal: degenerate, but still well-defined
        bool lna = ir_lit_is_na(lhs_node);
        bool rna = ir_lit_is_na(rhs_node);
        std::string ls = lna ? std::string() : Rcpp::as<std::string>(lhs_node["value"]);
        std::string rs = rna ? std::string() : Rcpp::as<std::string>(rhs_node["value"]);
        cudf::string_scalar lsc(ls, !lna);
        cudf::string_scalar rsc(rs, !rna);
        // No scalar-scalar binary_operation overload; broadcast lhs to a
        // 1-row column-esque comparison via make_column_from_scalar is
        // overkill for a constant fold that should never reach C++ (R
        // constant-folds subtrees with no column reference, section 1.2).
        Rcpp::stop("Both operands of a string comparison were literals; this should have been constant-folded on the R side.");
    }

    Rcpp::stop("No handler implemented for op: %s", op.c_str());
}

// -----------------------------------------------------------------------------
// materialize(): returns an index into ctx.cols for `node`.
//   - col:  the column's own index (no computation, no new entry)
//   - lit:  broadcast to a full-size column (make_column_from_scalar) -- the
//           rare case of a literal needing to stand alone as a column, e.g.
//           round()'s value argument being itself a bare literal
//   - call: op_dispatches_to_ast() -> one build_ast() + one compute_column()
//           call (build_ast() internally materializes any non-expressible
//           descendants and references them by column_reference, so this
//           branch is taken whenever the *node's own* op is AST-native, even
//           if some descendant isn't -- see op_dispatches_to_ast()'s comment
//           for why this differs from ast_expressible()); otherwise ->
//           apply_handler(). Either way, pushed as a new ctx column so later
//           references see it as a proper materialized column.
// -----------------------------------------------------------------------------

inline cudf::size_type materialize(Rcpp::List node, eval_ctx& ctx) {
    std::string kind = ir_kind(node);

    if (kind == "col") {
        return Rcpp::as<cudf::size_type>(node["index"]);
    }

    if (kind == "lit") {
        bool na = ir_lit_is_na(node);
        SEXP type_sexp = node["type"];
        std::string type_str = Rf_isNull(type_sexp) ? "FLOAT64" : Rcpp::as<std::string>(type_sexp);
        auto s = build_scalar(type_str, node["value"], na);
        auto col = cudf::make_column_from_scalar(*s, ctx.view().num_rows());
        return ctx.push(std::move(col));
    }

    // kind == "call"
    std::string op = Rcpp::as<std::string>(node["op"]);
    if (op_dispatches_to_ast(op, node, ctx)) {
        cudf::ast::tree tree;
        std::vector<std::unique_ptr<cudf::scalar>> scalars;
        cudf::ast::expression const& root = build_ast(node, tree, scalars, ctx, "");
        auto result = cudf::compute_column(ctx.view(), root);
        return ctx.push(std::move(result));
    }

    auto result = apply_handler(node, ctx);
    return ctx.push(std::move(result));
}

// -----------------------------------------------------------------------------
// materialize_top_level(): used only for a mutate/compute-column's top-level
// expression, where -- unlike materialize()'s generic "a bare col reference
// needs no new column" shortcut -- the result MUST always land at a fresh
// ctx index, even for a straight column copy (`mutate(y = x)`). This keeps
// gpu_mutate_expr()'s sequential index bookkeeping (section 2.2: "R side
// mirrors this with a running name -> index map") in lockstep with the
// running schema the R side binds each expression's `col` indices against:
// expression i's output always occupies ctx index `n_input + i`, matching
// what ops_mutate_batch.cpp's "copy" case did by deep-copying rather than
// aliasing (ops_mutate_batch.cpp:72-79).
// -----------------------------------------------------------------------------

inline cudf::size_type materialize_top_level(Rcpp::List node, eval_ctx& ctx) {
    if (ir_kind(node) == "col") {
        int idx = Rcpp::as<int>(node["index"]);
        auto copy = std::make_unique<cudf::column>(ctx.cols[idx]);
        return ctx.push(std::move(copy));
    }
    return materialize(node, ctx);
}

} // namespace cuplyr

#endif // CUPLYR_EXPR_EVAL_HPP
