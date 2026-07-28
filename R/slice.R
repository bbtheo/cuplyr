# GPU-native slice() family (Phase 3 task 2: ungrouped; Phase 5 task W7:
# grouped/`by=`/`.by=`; Phase 5 task W8: slice_sample())
#
# Covers slice()/slice_head()/slice_tail()/slice_min()/slice_max()/
# slice_sample(), for ungrouped input, `group_by()`-grouped input, and
# on-the-fly `by=`/`.by=` grouping. `slice_sample()`'s native path
# (bottom of this file) is architecturally different from the other four:
# there is no AST node at all -- indices are generated on the CPU (to
# reproduce dplyr's own R-level RNG stream bit-for-bit) and only the
# final gather runs on the GPU, via the existing `gpu_slice_indices()`
# primitive. `weight_by=` still always falls back to CPU (R/fallback.R
# no longer defines `slice_sample.tbl_gpu` -- it moved here).
#
# `slice_group_mode()` below is the single place that classifies a call's
# grouping ("none" / "group_by" / "by"), shared by all four verbs. "none"
# keeps using the dedicated ungrouped C++ kernels (`gpu_slice_head()`/
# `gpu_slice_tail()`/`gpu_slice_indices()`/`gpu_slice_rank()`,
# `src/ops_slice.cpp`) unchanged from Phase 3.2. "group_by"/"by" route
# through `slice_grouped_native()`, a pure-R *desugar* built with ZERO new
# GPU primitives: it composes the existing window/mutate/filter/arrange/
# select AST machinery (Phase 5 W1-W6) into the same handful of ops for
# every grouped slice_* shape (see that function's own docs for the full
# node sequence). Only genuinely unsupported grouped shapes -- an
# `order_by` expression the IR can't parse, or a `slice(i)` index vector
# with duplicates/reordering (dplyr's own arbitrary per-group gather can't
# be expressed as a single membership predicate) -- still fall back to CPU.
#
# Row-count-dependent resolution (the n=/prop= clamp for head/tail/rank,
# and slice()'s index-vector validation/negative-index handling) is
# deliberately NOT done here in R for the UNGROUPED path: for a lazy
# pipeline, the input's actual row count isn't known until upstream ops
# actually execute, so it can't be computed at parse time. Instead the
# *raw*, unresolved amount/indices are stored on the AST node (see
# R/ast.R::ast_slice()) and resolved in C++ at lowering time, once the real
# input table (and its real row count) exists -- see R/lower.R::lower_slice()
# and src/ops_common.hpp::compute_slice_size(). The GROUPED path (below)
# can't reuse that C++ resolution (there is no single scalar `nrow` -- every
# group has its own size), so it expresses the identical clamp rule as a
# per-row IR expression instead (`slice_grouped_native()`'s "size" mutate
# step), broadcasting each group's own row count (a real window-computed
# column, not a compile-time scalar) through the same formula.
#
# Semantics (verified empirically against dplyr 1.2.1 before implementation,
# cross-checked against dplyr's own `get_slice_size()`/`slice_rank_idx()`
# source):
#   - slice(1:3), slice(c(3,1,1)) (duplicates + reordering both allowed),
#     slice(-(1:2)) (negative = drop), slice(0) (empty), out-of-range
#     indices silently dropped, mixing positive/negative errors, NA dropped,
#     fractional indices error, slice() with no dots returns 0 rows (dplyr's
#     own behavior: `vec_c()` of zero chunks is `integer(0)`). This full
#     generality (arbitrary reordering/duplication of positive indices) is
#     ungrouped-only; the grouped desugar (below) only handles a strictly
#     increasing, duplicate-free positive index vector, or any set of
#     negative indices (order/duplicates never matter for a "drop these"
#     set) -- anything else falls back to CPU.
#   - slice_head()/slice_tail(): `n=`/`prop=` resolve via
#     clamp(0, floor(amount), nrow) when amount >= 0, else
#     clamp(0, ceiling(nrow + amount), nrow) -- prop is `amount * nrow`
#     substituted into that same rule. Default (neither given) is n = 1.
#     Grouped: `nrow` is each GROUP's own row count, not the whole table's.
#   - slice_min()/slice_max(): same n=/prop= rule as above (confirmed
#     negative `n` is accepted, not just for head/tail). `with_ties = TRUE`
#     (default) uses SQL-RANK()-style tie cutoff (the result can exceed
#     `n`); `with_ties = FALSE` uses first-occurrence/stable-position
#     cutoff (result is exactly `n`, or fewer if the table is smaller).
#     `na_rm = FALSE` (default): NA rows in `order_by` always sort *last*
#     regardless of min/max direction and can appear in the result once
#     `n`/`prop` reaches past the non-NA count. Result row order follows
#     `order_by` (ascending for min, descending for max), NOT original row
#     order -- confirmed empirically. A computed `order_by` (e.g.
#     `slice_min(df, x + y)`) never appears as an output column, matching
#     dplyr.
#   - Grouped (`group_by()`) output for EVERY slice_* variant is reordered
#     into ascending GROUP-KEY order (NA-key group sorted last -- matching
#     `arrange(group_cols)`'s own ascending-NA-last convention), with each
#     group's own surviving rows kept in their original relative order (or,
#     for slice_min()/slice_max(), in `order_by` order, NA last regardless
#     of direction). This is a genuine reordering relative to the input's
#     physical row order (confirmed empirically: unsorted/interleaved group
#     values still come back sorted 1, 2, 3, NA).
#   - `by=`/`.by=` output for EVERY slice_* variant -- including
#     slice_min()/slice_max(), NOT just the positional variants -- is
#     ordered by each GROUP's FIRST-APPEARANCE position in the original
#     (pre-slice) table, NOT ascending group-key order, and NOT the literal
#     original row order of the surviving rows either (confirmed
#     empirically with a case where a group's surviving row is physically
#     LATER in the table than another group's, yet the first group still
#     comes first in the output because ITS group label appeared earlier)
#     -- exactly analogous to `summarise(.by=)`'s already-documented
#     first-appearance rule (section 6 of
#     `scratchpad/phase5_window_design.md`), extended here to `slice()`.
#     This is a correction to that design doc's own assumption ("`.by`
#     preserves original row order") for the min/max case, and clarifies it
#     for the positional case: the row order that's "preserved" is the
#     group's first-appearance order, which only happens to coincide with
#     literal original row order when a `by=` column's groups are already
#     contiguous in the input (the common case, but not the general rule).
#     Result is always ungrouped, matching `.by=`'s general contract.
#   - (Historical note, resolved in Phase 6 task 6.1) `arrange(desc(x))` in
#     cuplyr used to diverge from real dplyr 1.2.1 (which sorts NA last for
#     BOTH directions) by sorting NA FIRST for descending order --
#     `slice_grouped_native()` was written against that buggy behavior and
#     so builds its OWN NA-last sort key for `order_by` (an `is.na()`
#     indicator column, `..slice_na..`, as an explicit sort key ahead of
#     `order_by` itself) rather than relying on a plain `desc(order_by)`
#     arrange. Phase 6 task 6.1 fixed `gpu_arrange()`
#     (src/ops_arrange.cpp) to flip `null_order` with direction, so a plain
#     `arrange(order_col, descending = descending)` now ALSO places NA last
#     on its own -- making the explicit `..slice_na..` sort key redundant
#     for ordering purposes (its presence ahead of `order_col` no longer
#     changes the result: `order_col`'s own null placement already groups
#     every NA row last, in original relative order, before this key would
#     even apply). It is deliberately NOT removed from `sort_specs` here:
#     `..slice_na..` is still independently required (unchanged) to build
#     `..slice_valid..`/`..slice_nn..` for the `with_ties` rank-boundary
#     count below, and folding the sort-key removal into this fix would
#     touch tie-breaking behavior in a hand-built multi-stage desugar with
#     no oracle coverage isolating that specific change -- lower risk to
#     leave the now-redundant key in place than to verify its removal is
#     truly a no-op across every mode/with_ties/na_rm combination.

#' Classify a call's effective grouping for the slice() family
#'
#' Shared by all five verbs (Phase 5, task W7): resolves `by=`/`.by=` via
#' `resolve_by()` (which itself raises dplyr's own "Can't supply `by`..."
#' error when `.data` is already grouped) and returns which of three
#' dispatch paths applies.
#'
#' @param .data A `tbl_gpu` object
#' @param by_quo A quosure for the verb's `by=`/`.by=` argument
#' @param arg_name Character scalar, `"by"` or `".by"` (interpolated into
#'   `resolve_by()`'s grouped-`.data` error message, see its own docs)
#' @return `list(mode = "none"|"group_by"|"by", group_cols = <chr>)`:
#'   `"none"` when `.data` is ungrouped and no `by=`/`.by=` was supplied
#'   (the existing Phase 3.2 ungrouped native path applies); `"group_by"`
#'   when `.data` is `group_by()`-grouped; `"by"` when `by=`/`.by=` was
#'   supplied (which implies `.data` itself is ungrouped, since
#'   `resolve_by()` would otherwise have already errored)
#' @keywords internal
slice_group_mode <- function(.data, by_quo, arg_name) {
  by_cols <- resolve_by(.data, by_quo, arg_name)
  if (length(by_cols) > 0) {
    return(list(mode = "by", group_cols = by_cols))
  }
  if (length(.data$groups) > 0) {
    return(list(mode = "group_by", group_cols = .data$groups))
  }
  list(mode = "none", group_cols = character())
}

# Resolve a verb's `n=`/`prop=` pair into `list(amount, is_prop)`, matching
# dplyr's `check_slice_n_prop()`/`get_slice_size()` contract exactly (shared
# by slice_head()/slice_tail()/slice_min()/slice_max(), which all use the
# identical rule -- verified empirically, including that slice_min()/
# slice_max() accept a negative `n` just like slice_head()/slice_tail() do).
# `n`/`prop` are `NULL` when the caller's argument was missing.
#
# @param n Numeric scalar or `NULL`
# @param prop Numeric scalar or `NULL`
# @return `list(amount, is_prop)`, or throws if both/neither are invalid
# @keywords internal
resolve_slice_amount <- function(n, prop) {
  if (!is.null(n) && !is.null(prop)) {
    stop("Must supply `n` or `prop`, but not both.", call. = FALSE)
  }
  if (is.null(n) && is.null(prop)) {
    return(list(amount = 1, is_prop = FALSE))
  }
  if (!is.null(n)) {
    if (!(is.numeric(n) && length(n) == 1) || is.na(n)) {
      stop("`n` must be a single number, not: ", rlang::as_label(n), call. = FALSE)
    }
    return(list(amount = as.double(n), is_prop = FALSE))
  }
  if (!(is.numeric(prop) && length(prop) == 1) || is.na(prop)) {
    stop("`prop` must be a single number, not: ", rlang::as_label(prop), call. = FALSE)
  }
  list(amount = as.double(prop), is_prop = TRUE)
}

# Evaluate slice()'s dots into a single raw (unresolved) numeric index
# vector, or throw if any dot can't be evaluated as a plain, data-independent
# index expression.
#
# Each dot is evaluated via `rlang::eval_tidy()` with NO data mask -- so a
# bare symbol resolves only against the quosure's own captured (lexical, R
# calling-frame) environment. This has exactly the effect we want: ordinary
# R variables the user has in scope (`slice(df, my_idx)`) resolve fine, but
# a bare *column* reference (`slice(df, which(x > 5))`) raises "object
# 'x' not found" (there is no column-name binding without a data mask),
# which the caller catches and routes to the CPU fallback -- including
# dplyr's own `n()` (not bound here, so `slice(df, 1:n())` also, correctly,
# falls back rather than silently resolving to something wrong).
#
# @param dots A list of quosures (from `rlang::enquos(...)`)
# @return A numeric vector (possibly empty), matching dplyr's own
#   `vec_c()`-concatenation-of-dots contract
# @keywords internal
resolve_slice_dots <- function(dots) {
  if (length(dots) == 0) {
    return(numeric(0))
  }

  vals <- lapply(dots, function(d) {
    v <- rlang::eval_tidy(d)
    if (!is.numeric(v)) {
      stop("slice() only supports plain numeric index expressions in the ",
           "GPU-native path.", call. = FALSE)
    }
    as.double(v)
  })

  unlist(vals, use.names = FALSE)
}

# Generate a column name guaranteed not to collide with `existing_names`,
# for the synthetic order_by column slice_min()/slice_max() add when
# `order_by` is a computed expression (mirrors distinct()'s identical need
# for computed key columns, though distinct keeps its temp columns around
# under the user's own name -- this one is always dropped again before the
# result is returned, see R/lower.R::lower_slice()'s `mode == "rank"` case).
#
# A thin wrapper around unique_temp_name() (R/window.R), which generalizes
# this exact collision-avoidance scheme (Phase 5 task W1: window
# decomposition needs the identical logic for its own `..win*..`/
# `..winarg*..` temp columns, so the one implementation now lives there).
#
# @param existing_names Character vector of names already in use
# @return A character scalar not in `existing_names`
# @keywords internal
unique_slice_order_name <- function(existing_names) {
  unique_temp_name("..slice_order_by..", existing_names)
}

# Resolve slice_min()/slice_max()'s `order_by` quosure against the IR
# (the same expression IR mutate()/distinct() use, see R/ir.R). A bare
# column reference is used directly (no extra mutate node); anything else
# recognized by the IR (arithmetic, function calls, ...) is computed via a
# wrapping `ast_mutate()` under a synthetic temp column name, matching
# distinct()'s pattern for computed key expressions (R/distinct.R).
#
# @param order_quo A quosure for `order_by`
# @param schema The current input schema (`list(names, types, ...)`)
# @return `list(extra_mutate = NULL or list(one mutate expr struct),
#   order_col = <character>)`, or `NULL` if the IR doesn't recognize the
#   expression (caller falls back to the CPU path)
# @keywords internal
resolve_slice_order <- function(order_quo, schema) {
  ir <- ir_parse_quo(order_quo, schema)
  if (is.null(ir)) {
    return(NULL)
  }

  if (identical(ir$kind, "col")) {
    return(list(extra_mutate = NULL, order_col = ir$name))
  }

  temp_name <- unique_slice_order_name(schema$names)
  step <- make_mutate_expr(temp_name, ir, schema)
  list(extra_mutate = list(step), order_col = temp_name)
}

# -----------------------------------------------------------------------------
# Grouped/`by=` slice() family (Phase 5, task W7)
# -----------------------------------------------------------------------------

# Classify slice()'s raw (unresolved) index vector for the GROUPED/`by=`
# native path. Unlike the ungrouped path (`gpu_slice_indices()`, which
# supports ANY combination of duplicates/reordering for positive indices
# via a direct per-row gather), the grouped desugar expresses "keep these
# positions" as an IR predicate over a per-group `row_number()` column
# (`rn %in% i` / `!(rn %in% -i)`), which can only reproduce a
# SUBSET-membership test, never an arbitrary reordering/duplication of the
# kept rows. So only two shapes are supported natively here: a strictly
# increasing, duplicate-free positive index vector (`rn %in% i` alone
# already preserves per-group row order -- identical to what dplyr
# produces for this shape), and any set of negative indices (order/
# duplicates never matter for a "drop these" set). Anything else --
# unsorted/duplicated positive indices, mixed signs, non-whole numbers --
# returns `NULL`, the caller's "fall back to CPU" signal: `gpu_fallback()`
# re-runs the real `dplyr::slice()` call, which raises the identical
# mixed-sign/fractional errors dplyr itself would, so this function does
# not need to replicate those error messages itself.
#
# @param raw_indices A numeric vector (may contain 0/NA/fractional/mixed
#   sign values), as returned by `resolve_slice_dots()`
# @return `list(kind = "positive"|"negative", vals = <integer>)`, or `NULL`
# @keywords internal
classify_slice_indices <- function(raw_indices) {
  vals <- raw_indices[!is.na(raw_indices) & raw_indices != 0]

  if (any(vals != floor(vals))) {
    return(NULL)
  }
  if (length(vals) == 0) {
    # Nothing left after dropping 0/NA (including the "no dots at all"
    # case): always native, regardless of sign convention -- `rn %in%
    # integer(0)` is FALSE for every row, giving the 0-row result dplyr's
    # own `vec_c()`-of-zero-chunks contract produces.
    return(list(kind = "positive", vals = integer(0)))
  }

  has_pos <- any(vals > 0)
  has_neg <- any(vals < 0)
  if (has_pos && has_neg) {
    return(NULL)
  }

  if (has_neg) {
    return(list(kind = "negative", vals = as.integer(-vals)))
  }

  if (length(vals) > 1 && !all(diff(vals) > 0)) {
    return(NULL)
  }

  list(kind = "positive", vals = as.integer(vals))
}

#' GPU-native grouped/`by=` slice() family desugar (Phase 5, task W7)
#'
#' Implements every `group_by()`-grouped or `by=`/`.by=` on-the-fly-grouped
#' `slice()`/`slice_head()`/`slice_tail()`/`slice_min()`/`slice_max()` call
#' by composing the existing window/mutate/filter/arrange/select AST
#' machinery (Phase 5 W1-W6) -- ZERO new GPU primitives. See this file's
#' module docs for the empirically-verified semantics this reproduces
#' (group order, NA handling, `by=`'s first-appearance rule).
#'
#' The node sequence, in order (only the parts relevant to `mode`/
#' `group_mode` are actually emitted):
#' \enumerate{
#'   \item (`mode == "rank"`, computed `order_by`) `ast_mutate`: materialize
#'     the computed `order_by` expression under a temp name.
#'   \item (`mode == "rank"`) `ast_mutate`: `..slice_na.. = is.na(order_col)`
#'     -- the NA-last sort key `order_by` needs (see module docs on why a
#'     plain `desc(order_col)` arrange can't be used here).
#'   \item (`group_mode == "by"`) `ast_window` (UNGROUPED): `..slice_rowid..
#'     = row_number()` over the whole table, then `ast_window` (grouped):
#'     `..slice_first.. = min(..slice_rowid..)` -- each group's own
#'     first-appearance position, broadcast to every row of that group.
#'   \item `ast_arrange`: group order key first (`group_cols` ascending for
#'     `group_mode == "group_by"`; `..slice_first..` ascending for `"by"`),
#'     then (`mode == "rank"` only) `..slice_na..` ascending, then
#'     `order_col` in the requested direction.
#'   \item (`mode == "rank" && with_ties`) `ast_mutate`:
#'     `..slice_valid.. = if_else(..slice_na.., 0L, 1L)`.
#'   \item `ast_window` (grouped by `group_cols`): whichever of
#'     `..slice_gs.. = n()`, `..slice_rn.. = row_number()` (frame:
#'     `group_cols`, no order -- physically pre-arranged above, so bare
#'     `row_number()` already reproduces `order_col`'s ranking for rank
#'     mode), `..slice_mr.. = min_rank(order_col)` (frame: `group_cols`,
#'     `order_col` in the requested direction), `..slice_nn.. =
#'     sum(..slice_valid..)` are needed for this `mode`/`with_ties`
#'     combination.
#'   \item (needs `..slice_gs..`) `ast_mutate`: `..slice_size.. =` the same
#'     `n=`/`prop=` clamp rule `compute_slice_size()` (C++) uses for the
#'     ungrouped path, expressed as IR (`gs` is a real per-group column
#'     here, not a compile-time scalar).
#'   \item `ast_filter`: the mode-specific predicate (see module docs'
#'     desugar table).
#'   \item `ast_select`: back down to the ORIGINAL column set (captured
#'     before any of the above), dropping every `..slice_*..`/computed-
#'     `order_by` temp column in one step.
#' }
#'
#' Two-pass (plan, then emit): every failure mode that triggers a CPU
#' fallback (`order_by` the IR can't parse, an unsupported `slice()` index
#' shape) is checked BEFORE any `push_op()` call, so a rejected call never
#' partially executes GPU work ("never half-lower").
#'
#' @param .data A `tbl_gpu` object (grouped, or ungrouped with `by=`/`.by=`
#'   already resolved by the caller)
#' @param group_mode `"group_by"` or `"by"`, as returned by
#'   `slice_group_mode()`
#' @param group_cols Character vector of group column names
#' @param mode One of `"head"`, `"tail"`, `"index"`, `"rank"`
#' @param amount,is_prop For `mode` `"head"`/`"tail"`/`"rank"`: as resolved
#'   by `resolve_slice_amount()`
#' @param raw_indices For `mode == "index"`: as returned by
#'   `resolve_slice_dots()`
#' @param order_quo For `mode == "rank"`: a quosure for `order_by`
#' @param descending For `mode == "rank"`: `FALSE` for `slice_min()`,
#'   `TRUE` for `slice_max()`
#' @param with_ties,na_rm For `mode == "rank"`: as documented on
#'   `slice_min()`/`slice_max()`
#' @return A new `tbl_gpu`, or `NULL` if this call needs the CPU fallback
#'   (caller's responsibility to re-run the real dplyr verb)
#' @keywords internal
slice_grouped_native <- function(.data, group_mode, group_cols, mode,
                                  amount = NULL, is_prop = NULL,
                                  raw_indices = NULL,
                                  order_quo = NULL, descending = NULL,
                                  with_ties = NULL, na_rm = NULL) {
  orig_names <- current_schema(.data)$names

  # ---- Plan phase: resolve everything that can fail, before any push_op() ----
  order_col <- NULL
  order_extra_mutate <- NULL
  if (identical(mode, "rank")) {
    resolved <- tryCatch(resolve_slice_order(order_quo, current_schema(.data)),
                         error = function(e) NULL)
    if (is.null(resolved)) {
      return(NULL)
    }
    order_col <- resolved$order_col
    order_extra_mutate <- resolved$extra_mutate
  }

  idx_plan <- NULL
  if (identical(mode, "index")) {
    idx_plan <- classify_slice_indices(raw_indices)
    if (is.null(idx_plan)) {
      return(NULL)
    }
  }

  # ---- Emit phase ----
  result <- .data

  if (identical(mode, "rank") && !is.null(order_extra_mutate)) {
    result <- push_op(result, ast_mutate(input_node(result), order_extra_mutate))
  }

  na_key_name <- NULL
  if (identical(mode, "rank")) {
    schema <- current_schema(result)
    na_key_name <- unique_temp_name("..slice_na..", schema$names)
    na_step <- make_mutate_expr(na_key_name, ir_call("is.na", list(ir_col(order_col))), schema)
    result <- push_op(result, ast_mutate(input_node(result), list(na_step)))
  }

  # Group-order key: group_by() sorts by group_cols ascending (NA-key group
  # last); by=/.by= sorts by each group's FIRST-APPEARANCE position instead
  # (verified empirically -- see module docs) via min(row_number()) per
  # group, never the raw group-key value.
  order_key_name <- NULL
  if (identical(group_mode, "by")) {
    schema <- current_schema(result)
    rowid_name <- unique_temp_name("..slice_rowid..", schema$names)
    rowid_spec <- window_spec(rowid_name, fn = "row_number", output_type = "INT32")
    result <- push_op(result, ast_window(input_node(result), list(rowid_spec), character()))

    schema <- current_schema(result)
    first_name <- unique_temp_name("..slice_first..", schema$names)
    first_spec <- window_spec(first_name, fn = "min", value_col = rowid_name, output_type = "INT32")
    result <- push_op(result, ast_window(input_node(result), list(first_spec), group_cols))

    order_key_name <- first_name
  }

  sort_specs <- if (!is.null(order_key_name)) {
    list(list(col_name = order_key_name, descending = FALSE))
  } else {
    lapply(group_cols, function(g) list(col_name = g, descending = FALSE))
  }

  if (identical(mode, "rank")) {
    sort_specs <- c(sort_specs,
                    list(list(col_name = na_key_name, descending = FALSE)),
                    list(list(col_name = order_col, descending = isTRUE(descending))))
  }

  result <- push_op(result, ast_arrange(input_node(result), sort_specs, groups = character()))

  # Window stage: n()/row_number() (frame: group_cols, no order) cover
  # head/tail/index/rank(with_ties=FALSE); min_rank(order_col)/count-valid
  # cover rank(with_ties=TRUE). Physically pre-arranging (above) rather
  # than passing order_col into THIS frame is what makes bare
  # row_number() reproduce order_col's ranking for rank mode.
  need_gs <- mode %in% c("head", "tail", "rank")
  need_rn <- mode %in% c("head", "tail", "index") || (identical(mode, "rank") && !isTRUE(with_ties))
  need_mr_nn <- identical(mode, "rank") && isTRUE(with_ties)

  schema <- current_schema(result)
  gs_name <- if (need_gs) unique_temp_name("..slice_gs..", schema$names) else NULL
  rn_name <- if (need_rn) {
    unique_temp_name("..slice_rn..", c(schema$names, gs_name))
  } else {
    NULL
  }

  valid_name <- NULL
  if (need_mr_nn) {
    schema <- current_schema(result)
    valid_name <- unique_temp_name("..slice_valid..", schema$names)
    valid_ir <- ir_call("if_else", list(ir_col(na_key_name), ir_lit_from_r(0L), ir_lit_from_r(1L)))
    valid_step <- make_mutate_expr(valid_name, valid_ir, schema)
    result <- push_op(result, ast_mutate(input_node(result), list(valid_step)))
  }

  window_specs <- list()
  if (need_gs) {
    window_specs <- c(window_specs, list(window_spec(gs_name, fn = "n", output_type = "INT32")))
  }
  if (need_rn) {
    window_specs <- c(window_specs, list(window_spec(rn_name, fn = "row_number", output_type = "INT32")))
  }

  mr_name <- NULL
  nn_name <- NULL
  if (need_mr_nn) {
    schema <- current_schema(result)
    mr_name <- unique_temp_name("..slice_mr..", schema$names)
    nn_name <- unique_temp_name("..slice_nn..", c(schema$names, mr_name))
    window_specs <- c(window_specs, list(
      window_spec(mr_name, fn = "min_rank", value_col = order_col,
                  order_cols = order_col, order_desc = isTRUE(descending),
                  output_type = "INT32"),
      window_spec(nn_name, fn = "sum", value_col = valid_name,
                  output_type = type_window_sum("INT32"))
    ))
  }

  result <- push_op(result, ast_window(input_node(result), window_specs, group_cols))

  # Size (IR, per group): the same n=/prop= clamp rule
  # src/ops_common.hpp::compute_slice_size() uses for the ungrouped path,
  # expressed here as a real mutate expression since each group's own size
  # (`gs`) is a real per-row (broadcast) column, not a scalar known at plan
  # time.
  size_name <- NULL
  if (need_gs) {
    schema <- current_schema(result)
    size_name <- unique_temp_name("..slice_size..", schema$names)
    amount_lit <- ir_lit_from_r(as.double(amount))
    raw_ir <- if (isTRUE(is_prop)) {
      ir_call("*", list(amount_lit, ir_col(gs_name)))
    } else {
      amount_lit
    }
    size_ir <- ir_call("if_else", list(
      ir_call(">=", list(raw_ir, ir_lit_from_r(0))),
      ir_call("floor", list(raw_ir)),
      ir_call("+", list(ir_col(gs_name), ir_call("ceil", list(raw_ir))))
    ))
    size_step <- make_mutate_expr(size_name, size_ir, schema)
    result <- push_op(result, ast_mutate(input_node(result), list(size_step)))
  }

  # Filter predicate, per mode (module docs' desugar table).
  pred <- switch(mode,
    "head" = ir_call("<=", list(ir_col(rn_name), ir_col(size_name))),
    "tail" = ir_call(">", list(ir_col(rn_name), ir_call("-", list(ir_col(gs_name), ir_col(size_name))))),
    "index" = {
      if (identical(idx_plan$kind, "positive")) {
        ir_call("%in%", list(ir_col(rn_name), ir_lit_from_r(idx_plan$vals, allow_vector = TRUE)))
      } else {
        ir_call("!", list(ir_call("%in%", list(ir_col(rn_name), ir_lit_from_r(idx_plan$vals, allow_vector = TRUE)))))
      }
    },
    "rank" = {
      base_pred <- if (isTRUE(with_ties)) {
        ir_call("<=", list(
          ir_call("coalesce", list(ir_col(mr_name), ir_call("+", list(ir_col(nn_name), ir_lit_from_r(1L))))),
          ir_col(size_name)
        ))
      } else {
        ir_call("<=", list(ir_col(rn_name), ir_col(size_name)))
      }
      if (isTRUE(na_rm)) {
        ir_call("&", list(base_pred, ir_call("!", list(ir_col(na_key_name)))))
      } else {
        base_pred
      }
    }
  )

  schema <- current_schema(result)
  result <- push_op(result, ast_filter(input_node(result), list(make_predicate(pred, schema))))

  push_op(result, ast_select(input_node(result), orig_names))
}

#' Select rows by position
#'
#' Keeps rows of a GPU table by integer position, similar to
#' `dplyr::slice()`. Ungrouped, `group_by()`-grouped, and `.by=` calls with
#' plain, data-independent index expressions all run entirely on the GPU
#' (Phase 5, task W7): ungrouped via a direct `cudf::gather()`; grouped/
#' `.by=` via a per-group `row_number()` desugar (R/slice.R's module docs),
#' restricted to a strictly increasing positive index vector or any set of
#' negative indices (arbitrary duplication/reordering is ungrouped-only --
#' see `...`'s docs below). Everything else (an index expression the
#' GPU-native path can't evaluate without a data mask, e.g. one that
#' references a column or `n()`, or a grouped/`.by=` index vector with
#' duplicates/reordering) transparently falls back to CPU evaluation (see
#' [gpu_fallback()]).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Integer row positions, e.g. `slice(df, 1:3)`. Duplicates and
#'   reordering are allowed for positive indices when `.data` is ungrouped
#'   and `.by=` isn't supplied (`slice(df, c(3, 1, 1))`); grouped/`.by=`
#'   calls only run natively for a strictly increasing, duplicate-free
#'   positive index vector (falling back to CPU otherwise); negative
#'   indices drop those rows instead (order/duplicates never matter here,
#'   grouped or not); `0`/`NA` are silently dropped; out-of-range positions
#'   are silently dropped; positive and negative indices cannot be mixed.
#' @param .by Optional on-the-fly grouping columns, GPU-native (Phase 5,
#'   task W7): result rows are ordered by each group's FIRST-APPEARANCE
#'   position in `.data` (see module docs), and the result is always
#'   ungrouped. Supplying `.by` when `.data` is already grouped (via
#'   `group_by()`) is an error, matching dplyr.
#' @param .preserve Passed through to the CPU fallback when triggered.
#'
#' @return A `tbl_gpu` object with the selected rows.
#'
#' @export
#' @importFrom dplyr slice
slice.tbl_gpu <- function(.data, ..., .by = NULL, .preserve = FALSE) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(.by)
  gm <- slice_group_mode(.data, by_quo, ".by")

  idx <- tryCatch(resolve_slice_dots(dots), error = function(e) NULL)
  if (!is.null(idx)) {
    result <- if (identical(gm$mode, "none")) {
      push_op(.data, ast_slice(input_node(.data), mode = "index", raw_indices = idx))
    } else {
      slice_grouped_native(.data, gm$mode, gm$group_cols, mode = "index", raw_indices = idx)
    }
    if (!is.null(result)) return(result)
  }

  gpu_fallback("slice", .data, function(tbl) {
    rlang::inject(dplyr::slice(tbl, !!!dots, .by = !!by_quo, .preserve = .preserve))
  })
}

#' Select the first rows
#'
#' Keeps the first `n` (or `prop` fraction of) rows of a GPU table, similar
#' to `dplyr::slice_head()`. Ungrouped, `group_by()`-grouped, and `by=`
#' calls all run entirely on the GPU (Phase 5, task W7): ungrouped via
#' `cudf::slice()`; grouped/`by=` via a per-group window desugar (see
#' R/slice.R's module docs for the exact group-ordering/NA-handling rules).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Not used; for extensibility, must be empty.
#' @param n Number of rows to keep. Negative `n` keeps all but the last
#'   `abs(n)` rows. Clamped to `[0, nrow(.data)]` (grouped: `[0,
#'   group_size]`, per group). Default `1` if neither `n` nor `prop` is
#'   supplied.
#' @param prop Proportion of rows to keep (`floor(prop * nrow(.data))`,
#'   clamped like `n`; grouped: `floor(prop * group_size)`, per group).
#'   Exactly one of `n`/`prop` may be supplied.
#' @param by Optional on-the-fly grouping columns, GPU-native (Phase 5,
#'   task W7): result rows are ordered by each group's FIRST-APPEARANCE
#'   position in `.data` (see module docs), and the result is always
#'   ungrouped. Supplying `by` when `.data` is already grouped (via
#'   `group_by()`) is an error, matching dplyr.
#'
#' @return A `tbl_gpu` object with the first rows selected.
#'
#' @export
#' @importFrom dplyr slice_head
slice_head.tbl_gpu <- function(.data, ..., n, prop, by = NULL) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  gm <- slice_group_mode(.data, by_quo, "by")
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0) {
    amt <- tryCatch(
      resolve_slice_amount(np$n, np$prop),
      error = function(e) NULL
    )
    if (!is.null(amt)) {
      result <- if (identical(gm$mode, "none")) {
        push_op(.data, ast_slice(input_node(.data), mode = "head",
                                 amount = amt$amount, is_prop = amt$is_prop))
      } else {
        slice_grouped_native(.data, gm$mode, gm$group_cols, mode = "head",
                              amount = amt$amount, is_prop = amt$is_prop)
      }
      if (!is.null(result)) return(result)
    }
  }

  gpu_fallback("slice_head", .data, function(tbl) {
    rlang::inject(dplyr::slice_head(tbl, !!!dots, !!!np, by = !!by_quo))
  })
}

#' Select the last rows
#'
#' Keeps the last `n` (or `prop` fraction of) rows of a GPU table, similar
#' to `dplyr::slice_tail()`. Ungrouped, `group_by()`-grouped, and `by=`
#' calls all run entirely on the GPU (Phase 5, task W7) -- see
#' [slice_head.tbl_gpu()] for the shared dispatch/grouping semantics.
#'
#' @inheritParams slice_head.tbl_gpu
#'
#' @return A `tbl_gpu` object with the last rows selected.
#'
#' @export
#' @importFrom dplyr slice_tail
slice_tail.tbl_gpu <- function(.data, ..., n, prop, by = NULL) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  gm <- slice_group_mode(.data, by_quo, "by")
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0) {
    amt <- tryCatch(
      resolve_slice_amount(np$n, np$prop),
      error = function(e) NULL
    )
    if (!is.null(amt)) {
      result <- if (identical(gm$mode, "none")) {
        push_op(.data, ast_slice(input_node(.data), mode = "tail",
                                 amount = amt$amount, is_prop = amt$is_prop))
      } else {
        slice_grouped_native(.data, gm$mode, gm$group_cols, mode = "tail",
                              amount = amt$amount, is_prop = amt$is_prop)
      }
      if (!is.null(result)) return(result)
    }
  }

  gpu_fallback("slice_tail", .data, function(tbl) {
    rlang::inject(dplyr::slice_tail(tbl, !!!dots, !!!np, by = !!by_quo))
  })
}

#' Select rows with the smallest values of a variable
#'
#' Keeps the `n` (or `prop` fraction of) rows with the smallest `order_by`
#' values, similar to `dplyr::slice_min()`. Ungrouped, `group_by()`-grouped,
#' and `by=` calls with a plain `order_by` (a bare column, or an expression
#' the mutate/filter IR recognizes) all run entirely on the GPU (Phase 5,
#' task W7): ungrouped via `cudf::rank()` + sort + filter; grouped/`by=`
#' via a per-group window desugar (see R/slice.R's module docs for the
#' exact group-ordering rules, including `by=`'s first-appearance-order
#' rule, which applies here too, not just to the positional slice_*()s).
#' An unsupported `order_by` expression (one the IR can't parse) still
#' transparently falls back to CPU evaluation (see [gpu_fallback()]).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param order_by Column (or IR-supported expression) to rank by.
#' @param ... Not used; for extensibility, must be empty.
#' @param n,prop As for [slice_head.tbl_gpu()].
#' @param by Optional on-the-fly grouping columns, GPU-native (Phase 5,
#'   task W7) -- see [slice_head.tbl_gpu()]'s `by` docs.
#' @param with_ties If `TRUE` (the default), ties at the cutoff are all
#'   kept (the result can have more than `n` rows); if `FALSE`, ties are
#'   broken by original row position and the result has exactly `n` rows
#'   (or fewer if `.data` is smaller).
#' @param na_rm If `FALSE` (the default), `NA` values in `order_by` sort
#'   last (for both `slice_min()` and `slice_max()`) and can appear in the
#'   result once `n`/`prop` reaches past the non-`NA` count. If `TRUE`,
#'   `NA` rows are always excluded.
#'
#' @return A `tbl_gpu` object with the selected rows, in ascending
#'   `order_by` order (grouped: ascending `order_by` order within each
#'   group's own block).
#'
#' @export
#' @importFrom dplyr slice_min
slice_min.tbl_gpu <- function(.data, order_by, ..., n, prop, by = NULL,
                               with_ties = TRUE, na_rm = FALSE) {
  order_quo <- rlang::enquo(order_by)
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  gm <- slice_group_mode(.data, by_quo, "by")
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0) {
    result <- if (identical(gm$mode, "none")) {
      slice_rank_native(.data, order_quo, np, descending = FALSE,
                        with_ties = with_ties, na_rm = na_rm)
    } else {
      slice_rank_grouped(.data, gm, order_quo, np, descending = FALSE,
                         with_ties = with_ties, na_rm = na_rm)
    }
    if (!is.null(result)) return(result)
  }

  gpu_fallback("slice_min", .data, function(tbl) {
    rlang::inject(dplyr::slice_min(tbl, order_by = !!order_quo, !!!dots, !!!np,
                                    by = !!by_quo, with_ties = with_ties, na_rm = na_rm))
  })
}

#' Select rows with the largest values of a variable
#'
#' Keeps the `n` (or `prop` fraction of) rows with the largest `order_by`
#' values, similar to `dplyr::slice_max()`. See [slice_min.tbl_gpu()] for
#' the shared native-vs-fallback dispatch and `order_by`/`with_ties`/
#' `na_rm` semantics (identical, except the ranking direction and the
#' result's row order, which is descending here).
#'
#' @inheritParams slice_min.tbl_gpu
#'
#' @return A `tbl_gpu` object with the selected rows, in descending
#'   `order_by` order.
#'
#' @export
#' @importFrom dplyr slice_max
slice_max.tbl_gpu <- function(.data, order_by, ..., n, prop, by = NULL,
                               with_ties = TRUE, na_rm = FALSE) {
  order_quo <- rlang::enquo(order_by)
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  gm <- slice_group_mode(.data, by_quo, "by")
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0) {
    result <- if (identical(gm$mode, "none")) {
      slice_rank_native(.data, order_quo, np, descending = TRUE,
                        with_ties = with_ties, na_rm = na_rm)
    } else {
      slice_rank_grouped(.data, gm, order_quo, np, descending = TRUE,
                         with_ties = with_ties, na_rm = na_rm)
    }
    if (!is.null(result)) return(result)
  }

  gpu_fallback("slice_max", .data, function(tbl) {
    rlang::inject(dplyr::slice_max(tbl, order_by = !!order_quo, !!!dots, !!!np,
                                    by = !!by_quo, with_ties = with_ties, na_rm = na_rm))
  })
}

# Shared grouped/`by=` native-path builder for slice_min()/slice_max():
# resolves n=/prop= (the only part `slice_grouped_native()` itself can't
# resolve without knowing `mode` up front) and delegates to
# `slice_grouped_native(mode = "rank")`, or returns `NULL` (caller falls
# back to CPU) if `n=`/`prop=` are invalid.
#
# @keywords internal
slice_rank_grouped <- function(.data, gm, order_quo, np, descending, with_ties, na_rm) {
  amt <- tryCatch(resolve_slice_amount(np$n, np$prop), error = function(e) NULL)
  if (is.null(amt)) {
    return(NULL)
  }

  slice_grouped_native(.data, gm$mode, gm$group_cols, mode = "rank",
                       amount = amt$amount, is_prop = amt$is_prop,
                       order_quo = order_quo, descending = descending,
                       with_ties = with_ties, na_rm = na_rm)
}

# Shared ungrouped native-path builder for slice_min()/slice_max(): resolves
# order_by/n/prop and builds+pushes the `ast_slice(mode = "rank")` node, or
# returns `NULL` (caller falls back to CPU) if `order_by` isn't something
# the IR recognizes, or `n=`/`prop=` are invalid.
#
# @keywords internal
slice_rank_native <- function(.data, order_quo, np, descending, with_ties, na_rm) {
  amt <- tryCatch(resolve_slice_amount(np$n, np$prop), error = function(e) NULL)
  if (is.null(amt)) {
    return(NULL)
  }

  schema <- current_schema(.data)
  resolved <- tryCatch(resolve_slice_order(order_quo, schema), error = function(e) NULL)
  if (is.null(resolved)) {
    return(NULL)
  }

  input <- input_node(.data)
  if (!is.null(resolved$extra_mutate)) {
    input <- ast_mutate(input, resolved$extra_mutate)
  }

  node <- ast_slice(input, mode = "rank",
                    order_col = resolved$order_col,
                    order_is_temp = !is.null(resolved$extra_mutate),
                    descending = descending,
                    amount = amt$amount, is_prop = amt$is_prop,
                    with_ties = with_ties, na_rm = na_rm)

  push_op(.data, node)
}

#' First rows of a GPU table
#'
#' `utils::head()` method for `tbl_gpu`, implemented as `slice_head(x, n =
#' n)` (see [slice_head.tbl_gpu()] for the native-vs-fallback dispatch and
#' negative-`n` semantics).
#'
#' @param x A `tbl_gpu` object.
#' @param n Number of rows to keep (default `6`, matching `utils::head()`).
#'   Negative `n` keeps all but the last `abs(n)` rows.
#' @param ... Not used.
#'
#' @return A `tbl_gpu` object with the first `n` rows.
#'
#' @export
head.tbl_gpu <- function(x, n = 6L, ...) {
  slice_head.tbl_gpu(x, n = n)
}

#' Last rows of a GPU table
#'
#' `utils::tail()` method for `tbl_gpu`, implemented as `slice_tail(x, n =
#' n)` (see [slice_tail.tbl_gpu()] for the native-vs-fallback dispatch and
#' negative-`n` semantics).
#'
#' @param x A `tbl_gpu` object.
#' @param n Number of rows to keep (default `6`, matching `utils::tail()`).
#'   Negative `n` keeps all but the first `abs(n)` rows.
#' @param ... Not used.
#'
#' @return A `tbl_gpu` object with the last `n` rows.
#'
#' @export
tail.tbl_gpu <- function(x, n = 6L, ...) {
  slice_tail.tbl_gpu(x, n = n)
}

# -----------------------------------------------------------------------------
# slice_sample() (Phase 5, task W8)
# -----------------------------------------------------------------------------

# Resolve `slice_sample()`'s per-table/per-group sample size, matching
# dplyr's own `get_slice_size(allow_outsize = replace)` contract exactly
# (`dplyr:::get_slice_size`, verified empirically against dplyr 1.2.1):
# for a NON-NEGATIVE `amount`, `replace = TRUE` skips the `[0, n]` clamp
# entirely (oversampling with replacement is allowed to produce a result
# LARGER than `n`); for a NEGATIVE `amount`, the clamp always applies,
# `replace` or not -- this asymmetry is real dplyr behavior, not a cuplyr
# simplification (`get_slice_size()`'s `n < 0`/`prop < 0` branches never
# consult `allow_outsize` at all). `prop`'s raw amount is `prop * n`
# (computed against the SAME `n` that gets clamped against, matching
# dplyr's `!!slice_input$prop * n` substitution); `n`'s raw amount is
# `amount` itself.
#
# @param amount Numeric scalar, as resolved by `resolve_slice_amount()`
# @param is_prop Logical scalar, as resolved by `resolve_slice_amount()`
# @param n Integer scalar: the table's (or group's) actual row count
# @param replace Logical scalar, `slice_sample()`'s own `replace=` argument
# @return An integer scalar, the number of rows to draw for this
#   table/group
# @keywords internal
compute_slice_sample_size <- function(amount, is_prop, n, replace) {
  raw <- if (isTRUE(is_prop)) amount * n else amount

  if (amount >= 0) {
    if (isTRUE(replace)) {
      floor(raw)
    } else {
      max(0, min(floor(raw), n))
    }
  } else {
    max(0, min(ceiling(n + raw), n))
  }
}

#' GPU-native `slice_sample()` (Phase 5, task W8)
#'
#' Reproduces dplyr's own RNG stream bit-for-bit by generating row indices
#' on the CPU with the exact same `sample.int()` call sequence real dplyr
#' makes, then gathering those indices on the GPU via the existing
#' `gpu_slice_indices()` primitive (Phase 3.2) -- there is no AST node for
#' `slice_sample()` at all, unlike every other verb in this package: the
#' index vector is the whole point (it must be generated on the CPU to
#' consume R's global RNG state the same way real dplyr does), so there is
#' nothing left for a lazy AST/optimizer to defer.
#'
#' Semantics verified empirically against dplyr 1.2.1 before implementing
#' (`dplyr:::slice_sample.data.frame`, `dplyr:::sample_int`,
#' `dplyr:::get_slice_size`, `dplyr:::compute_by_groups`, source read via
#' `getAnywhere()`/`print()`):
#'   - Ungrouped: exactly one `sample.int(nrow, size, replace = replace)`
#'     call (`dplyr:::sample_int`'s own body: `if (size == 0L) integer(0)
#'     else sample.int(n, size, prob = wt, replace = replace)`, `wt` always
#'     `NULL` here since `weight_by=` never reaches this function -- see
#'     below).
#'   - `group_by()`-grouped: dplyr iterates groups in ASCENDING GROUP-KEY
#'     order (NA-key group last) -- the same order `dplyr::group_by()`
#'     itself produces (confirmed by literally calling
#'     `dplyr::group_by()`/`dplyr::group_rows()` on the collected group
#'     columns here, so this can never drift from real dplyr's own
#'     ordering). One `sample.int()` call per group, in that order,
#'     consuming the RNG stream in that same order; the resulting rows are
#'     concatenated in the SAME order (group-key order), matching real
#'     dplyr's own output row order exactly (confirmed empirically: this
#'     is genuinely a reordering relative to input row order, same as the
#'     other grouped `slice_*()` verbs' W7 finding). Output stays grouped
#'     by the same columns, matching real dplyr.
#'   - `by=`: dplyr's own `compute_by_groups()` groups via
#'     `vctrs::vec_group_loc()`, which orders groups by FIRST-APPEARANCE
#'     position in the input -- NOT ascending group-key order (confirmed
#'     empirically with a case where ascending-key order and
#'     first-appearance order differ: the RNG draw order AND the output
#'     row order both follow first-appearance order, not ascending key,
#'     and NOT a final sort-by-original-row-position either). This
#'     CORRECTS `scratchpad/phase5_window_design.md` §7's own text ("`by=`
#'     => sort() final indices (original row order)"), which was written
#'     before this empirical check and turns out to describe neither the
#'     RNG order nor the output order dplyr actually uses -- the real rule
#'     is exactly the grouped case's rule, just with a different (first-
#'     appearance, not ascending-key) group ordering source, i.e. exactly
#'     analogous to `summarise(.by=)`'s and W7's grouped-`slice()`'s
#'     already-documented first-appearance rule. Implemented here via
#'     `vctrs::vec_group_loc()` directly (the identical function
#'     `compute_by_groups()` itself calls), rather than reimplementing
#'     first-appearance grouping by hand, to remove any group-order
#'     divergence risk. Output is always ungrouped, matching `by=`'s
#'     general contract.
#'   - `n=`/`prop=` resolution: see `compute_slice_sample_size()`'s own
#'     docs for the exact (and slightly asymmetric) `replace=`-dependent
#'     clamp rule, verified against `dplyr:::get_slice_size()`'s source.
#'     Oversampling `n` without `replace` does NOT error -- it silently
#'     clamps to the table's (or group's) own row count, same as
#'     `slice_head()`/`slice_tail()`.
#'   - `weight_by=` always falls back to CPU (see `slice_sample.tbl_gpu()`
#'     below) -- `gpu_fallback()` re-runs the real `dplyr::slice_sample()`
#'     call, which already reproduces the identical `sample.int(...,
#'     prob = wt)` behavior, so there is nothing to gain (and real
#'     complexity to lose, since `wt` needs the ACTUAL data values, not
#'     just group columns) from a native path for this case.
#'
#' @param .data A `tbl_gpu` object (already materialized if it had pending
#'   lazy ops -- the caller is responsible for calling `compute()` first,
#'   since CPU-side index generation needs a real row count/grouping,
#'   which a lazy AST doesn't have)
#' @param gm A `list(mode, group_cols)` as returned by `slice_group_mode()`
#' @param amount,is_prop As resolved by `resolve_slice_amount()`
#' @param replace Logical scalar, `slice_sample()`'s own `replace=`
#' @return A new (always eager) `tbl_gpu`
#' @keywords internal
slice_sample_native <- function(.data, gm, amount, is_prop, replace) {
  materialized <- if (has_pending_ops(.data)) compute(.data) else .data
  schema <- materialized$schema

  if (identical(gm$mode, "none")) {
    nrow <- gpu_dim(materialized$ptr)[1]
    size <- compute_slice_sample_size(amount, is_prop, nrow, replace)
    idx <- if (size == 0L) integer(0) else sample.int(nrow, size, replace = replace)
    out_groups <- character()
  } else {
    group_cols <- gm$group_cols

    # "select() + collect() ONLY the group columns" -- built directly from
    # the existing gpu_select()/collect() primitives rather than routing
    # through the dplyr::select() S3 generic, since this is an internal
    # projection with no user-facing meaning of its own.
    key_col_idx <- match(group_cols, schema$names) - 1L
    key_ptr <- gpu_select(materialized$ptr, key_col_idx)
    key_schema <- list(names = group_cols, types = schema$types[key_col_idx + 1L])
    key_tbl <- collect(new_tbl_gpu(ptr = key_ptr, schema = key_schema))

    rows_list <- if (identical(gm$mode, "group_by")) {
      dplyr::group_rows(dplyr::group_by(key_tbl, !!!rlang::syms(group_cols)))
    } else {
      vctrs::vec_group_loc(key_tbl)$loc
    }

    idx <- integer(0)
    for (rows in rows_list) {
      gsize <- length(rows)
      size <- compute_slice_sample_size(amount, is_prop, gsize, replace)
      local_idx <- if (size == 0L) integer(0) else sample.int(gsize, size, replace = replace)
      idx <- c(idx, rows[local_idx])
    }

    out_groups <- if (identical(gm$mode, "group_by")) group_cols else character()
  }

  new_ptr <- gpu_slice_indices(materialized$ptr, as.double(idx))
  new_tbl_gpu(ptr = new_ptr, schema = schema, groups = out_groups)
}

#' Sample rows
#'
#' Keeps a random sample of rows of a GPU table, similar to
#' `dplyr::slice_sample()`. Ungrouped, `group_by()`-grouped, and `by=`
#' calls all run entirely on the GPU for the actual row gather (Phase 5,
#' task W8): row indices are generated on the CPU (to reproduce dplyr's
#' own RNG stream bit-for-bit, see `slice_sample_native()`'s own docs for
#' the full recipe/group-ordering rules), then gathered via
#' `gpu_slice_indices()`. `weight_by=` still transparently falls back to
#' CPU evaluation (see [gpu_fallback()]) -- weighted sampling needs the
#' actual (potentially computed) column values, not just row counts/group
#' columns.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Not used; for extensibility, must be empty.
#' @param n Number of rows to sample. Default `1` if neither `n` nor
#'   `prop` is supplied. Negative values are clamped like
#'   [slice_head.tbl_gpu()]'s `n`. With `replace = TRUE`, a non-negative
#'   `n`/`prop` is NOT clamped to the table's (or group's) row count --
#'   oversampling with replacement is allowed and produces a larger
#'   result; without `replace`, oversampling silently clamps (no error).
#' @param prop Proportion of rows to sample (`n = prop * nrow(.data)`,
#'   subject to the same clamp rule as `n`).
#' @param by Optional on-the-fly grouping columns, GPU-native (Phase 5,
#'   task W8): row indices are drawn (and the result ordered) by each
#'   group's FIRST-APPEARANCE position in `.data` (see
#'   `slice_sample_native()`'s own docs), and the result is always
#'   ungrouped. Supplying `by` when `.data` is already grouped (via
#'   `group_by()`) is an error, matching dplyr.
#' @param weight_by Sampling weights. Always falls back to CPU evaluation
#'   (see [gpu_fallback()]) -- not GPU-native.
#' @param replace Whether to sample with replacement.
#'
#' @return A `tbl_gpu` object with the sampled rows.
#'
#' @export
#' @importFrom dplyr slice_sample
slice_sample.tbl_gpu <- function(.data, ..., n, prop, by = NULL,
                                  weight_by = NULL, replace = FALSE) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  weight_quo <- rlang::enquo(weight_by)
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  # resolve_by() (via slice_group_mode()) raises dplyr's own
  # "Can't supply `by`..." error when `.data` is already grouped -- that
  # error is real, correct behavior and must propagate, not be swallowed
  # into a fallback, so this call is deliberately OUTSIDE any tryCatch()
  # (same pattern as slice_head.tbl_gpu() etc.).
  gm <- slice_group_mode(.data, by_quo, "by")

  if (length(dots) == 0 && rlang::quo_is_null(weight_quo)) {
    amt <- tryCatch(resolve_slice_amount(np$n, np$prop), error = function(e) NULL)
    if (!is.null(amt)) {
      was_lazy <- identical(.data$exec_mode, "lazy")
      result <- slice_sample_native(.data, gm, amt$amount, amt$is_prop, replace)
      if (was_lazy) result <- as_lazy(result)
      return(result)
    }
  }

  gpu_fallback("slice_sample", .data, function(tbl) {
    rlang::inject(dplyr::slice_sample(tbl, !!!dots, !!!np, by = !!by_quo,
                                       weight_by = !!weight_quo, replace = replace))
  })
}
