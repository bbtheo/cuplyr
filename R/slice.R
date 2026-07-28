# GPU-native slice() family (Phase 3, task 2)
#
# Covers ungrouped slice()/slice_head()/slice_tail()/slice_min()/slice_max().
# `slice_sample()` and every *grouped* variant (including on-the-fly
# `by=`/`.by=` grouping) stay on the CPU fallback (R/fallback.R):
#   - grouped slice needs per-group row offsets (group boundary/segment
#     machinery) that doesn't exist yet -- a later, Phase-5-adjacent task.
#   - slice_sample() needs dplyr's own RNG stream reproduced bit-for-bit
#     (CPU-side index generation + GPU gather) -- a separate task.
#
# `slice_use_native()` below is the single place that decides native vs.
# fallback for all four verbs, so the dispatch logic isn't duplicated
# four times.
#
# Row-count-dependent resolution (the n=/prop= clamp for head/tail/rank,
# and slice()'s index-vector validation/negative-index handling) is
# deliberately NOT done here in R: for a lazy pipeline, the input's actual
# row count isn't known until upstream ops actually execute, so it can't be
# computed at parse time. Instead the *raw*, unresolved amount/indices are
# stored on the AST node (see R/ast.R::ast_slice()) and resolved in C++ at
# lowering time, once the real input table (and its real row count) exists
# -- see R/lower.R::lower_slice() and src/ops_common.hpp::compute_slice_size().
#
# Semantics (verified empirically against dplyr 1.2.1 before implementation,
# cross-checked against dplyr's own `get_slice_size()`/`slice_rank_idx()`
# source):
#   - slice(1:3), slice(c(3,1,1)) (duplicates + reordering both allowed),
#     slice(-(1:2)) (negative = drop), slice(0) (empty), out-of-range
#     indices silently dropped, mixing positive/negative errors, NA dropped,
#     fractional indices error, slice() with no dots returns 0 rows (dplyr's
#     own behavior: `vec_c()` of zero chunks is `integer(0)`).
#   - slice_head()/slice_tail(): `n=`/`prop=` resolve via
#     clamp(0, floor(amount), nrow) when amount >= 0, else
#     clamp(0, ceiling(nrow + amount), nrow) -- prop is `amount * nrow`
#     substituted into that same rule. Default (neither given) is n = 1.
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

# Decide whether the ungrouped GPU-native slice path applies. Shared by
# slice()/slice_head()/slice_tail()/slice_min()/slice_max() (never by
# slice_sample(), which always uses the fallback regardless of grouping).
#
# @param .data A `tbl_gpu` object
# @param by_quo A quosure for the verb's `by=`/`.by=` argument (on-the-fly
#   grouping forces the fallback exactly like pre-existing `group_by()`
#   grouping does, since neither has group-offset machinery yet)
# @return `TRUE` if the native path should be attempted
# @keywords internal
slice_use_native <- function(.data, by_quo) {
  length(.data$groups) == 0 && rlang::quo_is_null(by_quo)
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

#' Select rows by position
#'
#' Keeps rows of a GPU table by integer position, similar to
#' `dplyr::slice()`. Ungrouped calls with plain, data-independent index
#' expressions (no `.by=`) run entirely on the GPU (`cudf::gather()`);
#' everything else (grouped input, `.by=`, or an index expression the
#' GPU-native path can't evaluate without a data mask, e.g. one that
#' references a column or `n()`) transparently falls back to CPU
#' evaluation (see [gpu_fallback()]).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Integer row positions, e.g. `slice(df, 1:3)`. Duplicates and
#'   reordering are allowed for positive indices (`slice(df, c(3, 1, 1))`);
#'   negative indices drop those rows instead; `0`/`NA` are silently
#'   dropped; out-of-range positions are silently dropped; positive and
#'   negative indices cannot be mixed.
#' @param .by Optional on-the-fly grouping columns (forces the CPU
#'   fallback; use [dplyr::group_by()] beforehand for the GPU-native
#'   ungrouped path).
#' @param .preserve Passed through to the CPU fallback when triggered.
#'
#' @return A `tbl_gpu` object with the selected rows.
#'
#' @export
#' @importFrom dplyr slice
slice.tbl_gpu <- function(.data, ..., .by = NULL, .preserve = FALSE) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(.by)

  if (slice_use_native(.data, by_quo)) {
    idx <- tryCatch(resolve_slice_dots(dots), error = function(e) NULL)
    if (!is.null(idx)) {
      node <- ast_slice(input_node(.data), mode = "index", raw_indices = idx)
      return(push_op(.data, node))
    }
  }

  gpu_fallback("slice", .data, function(tbl) {
    rlang::inject(dplyr::slice(tbl, !!!dots, .by = !!by_quo, .preserve = .preserve))
  })
}

#' Select the first rows
#'
#' Keeps the first `n` (or `prop` fraction of) rows of a GPU table, similar
#' to `dplyr::slice_head()`. Ungrouped calls (no `by=`) run entirely on the
#' GPU (`cudf::slice()`); grouped input or `by=` transparently falls back
#' to CPU evaluation (see [gpu_fallback()]).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Not used; for extensibility, must be empty.
#' @param n Number of rows to keep. Negative `n` keeps all but the last
#'   `abs(n)` rows. Clamped to `[0, nrow(.data)]`. Default `1` if neither
#'   `n` nor `prop` is supplied.
#' @param prop Proportion of rows to keep (`floor(prop * nrow(.data))`,
#'   clamped like `n`). Exactly one of `n`/`prop` may be supplied.
#' @param by Optional on-the-fly grouping columns (forces the CPU fallback).
#'
#' @return A `tbl_gpu` object with the first rows selected.
#'
#' @export
#' @importFrom dplyr slice_head
slice_head.tbl_gpu <- function(.data, ..., n, prop, by = NULL) {
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0 && slice_use_native(.data, by_quo)) {
    amt <- tryCatch(
      resolve_slice_amount(np$n, np$prop),
      error = function(e) NULL
    )
    if (!is.null(amt)) {
      node <- ast_slice(input_node(.data), mode = "head",
                        amount = amt$amount, is_prop = amt$is_prop)
      return(push_op(.data, node))
    }
  }

  gpu_fallback("slice_head", .data, function(tbl) {
    rlang::inject(dplyr::slice_head(tbl, !!!dots, !!!np, by = !!by_quo))
  })
}

#' Select the last rows
#'
#' Keeps the last `n` (or `prop` fraction of) rows of a GPU table, similar
#' to `dplyr::slice_tail()`. Ungrouped calls (no `by=`) run entirely on the
#' GPU (`cudf::slice()`); grouped input or `by=` transparently falls back
#' to CPU evaluation (see [gpu_fallback()]).
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
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0 && slice_use_native(.data, by_quo)) {
    amt <- tryCatch(
      resolve_slice_amount(np$n, np$prop),
      error = function(e) NULL
    )
    if (!is.null(amt)) {
      node <- ast_slice(input_node(.data), mode = "tail",
                        amount = amt$amount, is_prop = amt$is_prop)
      return(push_op(.data, node))
    }
  }

  gpu_fallback("slice_tail", .data, function(tbl) {
    rlang::inject(dplyr::slice_tail(tbl, !!!dots, !!!np, by = !!by_quo))
  })
}

#' Select rows with the smallest values of a variable
#'
#' Keeps the `n` (or `prop` fraction of) rows with the smallest `order_by`
#' values, similar to `dplyr::slice_min()`. Ungrouped calls with a plain
#' `order_by` (a bare column, or an expression the mutate/filter IR
#' recognizes) run entirely on the GPU (`cudf::rank()` + sort + filter);
#' grouped input, `by=`, or an unsupported `order_by` expression
#' transparently falls back to CPU evaluation (see [gpu_fallback()]).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param order_by Column (or IR-supported expression) to rank by.
#' @param ... Not used; for extensibility, must be empty.
#' @param n,prop As for [slice_head.tbl_gpu()].
#' @param by Optional on-the-fly grouping columns (forces the CPU fallback).
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
#'   `order_by` order.
#'
#' @export
#' @importFrom dplyr slice_min
slice_min.tbl_gpu <- function(.data, order_by, ..., n, prop, by = NULL,
                               with_ties = TRUE, na_rm = FALSE) {
  order_quo <- rlang::enquo(order_by)
  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(by)
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0 && slice_use_native(.data, by_quo)) {
    result <- slice_rank_native(.data, order_quo, np, descending = FALSE,
                                with_ties = with_ties, na_rm = na_rm)
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
  np <- list()
  if (!missing(n)) np$n <- n
  if (!missing(prop)) np$prop <- prop

  if (length(dots) == 0 && slice_use_native(.data, by_quo)) {
    result <- slice_rank_native(.data, order_quo, np, descending = TRUE,
                                with_ties = with_ties, na_rm = na_rm)
    if (!is.null(result)) return(result)
  }

  gpu_fallback("slice_max", .data, function(tbl) {
    rlang::inject(dplyr::slice_max(tbl, order_by = !!order_quo, !!!dots, !!!np,
                                    by = !!by_quo, with_ties = with_ties, na_rm = na_rm))
  })
}

# Shared native-path builder for slice_min()/slice_max(): resolves
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
