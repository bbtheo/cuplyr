# GPU-native count()/tally()/add_count() (Phase 3, task 5)
#
# All three are sugar over the existing group_by()/summarise()/mutate()/
# arrange() machinery -- no new AST node, no new C++ binding. `add_tally()`
# is NOT implemented here at all: it's a plain (non-generic) function in
# dplyr 1.2 that calls tbl_vars()/group_vars()/mutate()/arrange() internally
# (all generics/functions tbl_gpu already supports, see R/group-by.R's
# tbl_vars.tbl_gpu() and mutate()'s general window/aggregate machinery,
# Phase 5, which natively handles its internal `mutate(x, name := n()-or-
# sum(wt))` call), so the real `dplyr::add_tally()` works transparently on a
# tbl_gpu with no tbl_gpu-specific code at all.
#
# Semantics verified empirically against dplyr 1.2.1 (see
# tests/testthat/test-dplyr-count.R and the session notes in
# scratchpad/workflow_state.md) before writing this file:
#
#   - count(x, ...) = group_by(x, ..., .add = TRUE) |> tally() |>
#     <restore x's ORIGINAL grouping>. The "restore" step is NOT tally()'s
#     own summarise()-style "drop the last grouping level" rule -- it's
#     dplyr_reconstruct(out, x), which recomputes
#     intersect(group_vars(x), names(out)). Since x's own group columns are
#     always still present in the output (group_by(.add=TRUE) never removes
#     them), this intersect is *always* exactly x's original groups --
#     i.e. count() unconditionally restores the pre-call grouping in full,
#     even when x was grouped by more than one column (verified:
#     count(group_by(df, a, b), g) restores groups = c("a","b"), not
#     drop-last-style c("a")).
#   - tally(x) (called directly, no dots) has NO such reconstruction step:
#     it's exactly one summarise(x, name := n()-or-sum(wt)) call, so it
#     inherits summarise()'s default `.groups` behavior of dropping only
#     the LAST grouping level (verified: tally(group_by(df, a, b)) has
#     groups = "a"; tally(group_by(df, g)) is ungrouped). Phase 6, task 3
#     implemented real `.groups=` semantics in summarise() itself (see
#     R/summarise.R's `resolve_summarise_groups()`), so tally.tbl_gpu()'s
#     plain `summarise()` call now returns the correctly drop-last-grouped
#     result on its own -- the manual `orig_groups[-length(orig_groups)]`
#     restoration this file used to need (back when summarise() always
#     dropped ALL groups, a known tracked gap) is gone; only a defensive
#     `dplyr.summarise.inform = FALSE` override remains, matching real
#     dplyr's own `tally.data.frame()`.
#   - Name collisions (no explicit `name=`): dplyr's default name is "n",
#     bumped to "nn"/"nnn"/... only while it collides -- but the vars it
#     checks against differ between the two entry points, a genuinely
#     non-obvious empirical asymmetry:
#       - tally()'s check (used by count() too, via its internal tally()
#         call) is against `group_vars()` of the (post-.add) grouped table
#         ONLY -- an existing ordinary data column named "n" does NOT
#         trigger the bump (verified: count(tibble(g=.., n=..), g) still
#         names the output column "n", silently dropping the original "n"
#         data column, since summarise() only keeps group + aggregate
#         columns anyway).
#       - add_tally()'s check (used by add_count() too) is against
#         `tbl_vars()`, i.e. ALL columns -- an existing ordinary data
#         column named "n" DOES trigger the bump to "nn" (verified:
#         add_count(tibble(g=.., n=..), g) produces columns g, n, nn).
#   - wt = : `tally_n()` is `sum(wt, na.rm = TRUE)` when `wt` is given,
#     `n()` otherwise -- our own aggregation already always excludes NA
#     (see R/summarise.R), so a bare `sum(<wt-expr>)` (no na.rm needed) is
#     the exact equivalent; verified with NA in the weight column.
#   - Expression dots (`count(df, g = x %/% 10)`): dplyr's group_by()
#     supports both tidyselect-style column selection AND creating a new
#     column via a data-masked expression in the same `...` -- but our own
#     group_by.tbl_gpu() only supports the former (parse_group_cols() has
#     no mutate step). count()/add_count() need the latter, so this file
#     resolves dots the same way distinct.tbl_gpu() does (R/distinct.R):
#     parse each dot through the mutate IR, only emitting an actual
#     ast_mutate step for genuinely new/computed columns, leaving bare
#     existing-column references as a no-op. (Also verified: dplyr's own
#     count()/group_by() do NOT support tidyselect helpers like
#     starts_with() in this position -- group_by()'s `...` is data-masked,
#     not selecting, so `count(df, starts_with("x"))` errors in real dplyr
#     too. Not supported here either, for the same reason.)
#   - add_count()'s per-row broadcast goes through mutate()'s general
#     window/aggregate machinery (Phase 5, R/window.R/src/ops_window.cpp) --
#     as of Phase 5 task W9, this REPLACED the narrow join-back-based
#     `grouped_aggregate_mutate()` this used to route through, once the
#     general window path covered the exact same `n()`/`sum(<expr>)` shape
#     natively. NA keys are grouped together correctly (cudf's groupby uses
#     `null_policy::INCLUDE`, see src/ops_groupby.cpp) and original row
#     order is preserved (the window kernel's scatter-back step,
#     src/ops_window.cpp). See test-dplyr-count.R's dedicated NA-key tests.
#   - `.drop` (count()'s factor-levels arg): accepted, ignored -- same
#     "compatibility only" stance as group_by.tbl_gpu()'s own `.drop`;
#     factor group fidelity is Phase 11 (see scratchpad/todo.md).
#     add_count()'s `.drop` was deprecated in dplyr 1.0.0 and errors if
#     supplied at all (replicated here without a `lifecycle` dependency).

# Resolve dplyr:count()'s `...` dots into concrete grouping column names,
# building an ast_mutate step for any dot that isn't already a bare
# existing-column reference under its own name (mirrors distinct.tbl_gpu()'s
# own dot-resolution loop, R/distinct.R -- see that file for the identical
# pattern applied to distinct()'s key columns).
#
# @param .data A tbl_gpu object
# @param dots A list of quosures, as returned by rlang::enquos(...)
#   (NOT yet named -- this function names unnamed dots itself, silently,
#   matching dplyr's count()/group_by(), which don't warn for an unnamed
#   computed grouping expression)
# @return `list(data = <tbl_gpu, with any needed mutate step applied>,
#   names = <character vector of resolved column names, in dot order>)`
# @keywords internal
resolve_grouping_dots <- function(.data, dots) {
  if (length(dots) == 0) {
    return(list(data = .data, names = character(0)))
  }

  dots <- name_dots_silently(dots)
  schema <- current_schema(.data)
  expressions <- list()
  names_out <- character(length(dots))

  for (i in seq_along(dots)) {
    new_name <- names(dots)[i]
    expr <- dots[[i]]

    ir <- ir_parse_quo(expr, schema)

    if (is.null(ir)) {
      stop(
        "count()/add_count() only support column references and expressions ",
        "mutate() also supports (arithmetic, comparisons, logical operators, ",
        "is.na(), between(), %in%, and the usual math functions).\n",
        "Expression: ", rlang::quo_text(expr),
        call. = FALSE
      )
    }

    is_trivial_copy <- identical(ir$kind, "col") && identical(ir$name, new_name)

    if (!is_trivial_copy) {
      step <- make_mutate_expr(new_name, ir, schema)
      expressions <- c(expressions, list(step))
      schema <- update_schema_for_expr(schema, step)
    }

    names_out[i] <- new_name
  }

  result <- .data
  if (length(expressions) > 0) {
    result <- push_op(.data, ast_mutate(input_node(.data), expressions))
  }

  list(data = result, names = unique(names_out))
}

# Pick the aggregation's output column name: the user's explicit `name=` if
# given, else "n" bumped to "nn"/"nnn"/... while it collides with `vars`
# (dplyr's own n_name()/check_n_name() rule). Callers pass different `vars`
# for count()/tally() (group columns only) vs add_count() (every column) --
# see this file's header comment for the empirically-verified asymmetry.
#
# @param name Character or `NULL` (the user's `name=` argument)
# @param vars Character vector to avoid colliding with
# @return Character, the resolved column name
# @keywords internal
resolve_agg_name <- function(name, vars) {
  if (!is.null(name)) {
    return(name)
  }
  candidate <- "n"
  while (candidate %in% vars) {
    candidate <- paste0("n", candidate)
  }
  candidate
}

# Build the (unevaluated) aggregation call for count()/tally()/add_count():
# `n()` if no weight column, `sum(<wt-expr>, na.rm = TRUE)` otherwise.
#
# `for_mutate` is now a no-op kept only for call-site clarity/back-compat:
# BOTH `count()`/`tally()` (via `summarise()`) and `add_count()` (via
# `mutate()`'s window/aggregate machinery) need na.rm=TRUE spelled out
# explicitly -- this is the exact equivalent of dplyr's own
# `dplyr:::tally_n()`'s unconditional `sum(wt, na.rm = TRUE)`. Before Phase
# 6 task 6.1 (Fix A), `summarise()`'s own real groupby aggregation always
# excluded NA unconditionally (a bug -- it didn't honor R's na.rm = FALSE
# default at all), so a bare `sum(<wt-expr>)` happened to already behave
# like na.rm=TRUE for the `count()`/`tally()` call site; now that
# `summarise()` honors na.rm=FALSE by default (matching R and dplyr), that
# implicit behavior is gone, and na.rm=TRUE must be explicit here too, same
# as `add_count()` already required (see `ir_parse_sum_call()`, R/ir.R, for
# the mutate()-side na.rm=TRUE desugar this exercises there).
#
# @param wt_quo A quosure, as returned by rlang::enquo(wt) -- possibly NULL
#   (rlang::quo_is_null())
# @param for_mutate Logical, unused (kept for call-site clarity) -- both
#   branches now build the identical `sum(<wt-expr>, na.rm = TRUE)` call
# @return A language object: `quote(n())`, or `sum(<wt-expr>, na.rm = TRUE)`
# @keywords internal
build_count_agg_call <- function(wt_quo, for_mutate = FALSE) {
  if (rlang::quo_is_null(wt_quo)) {
    return(quote(n()))
  }
  rlang::call2("sum", rlang::quo_get_expr(wt_quo), na.rm = TRUE)
}

#' Count observations by group
#'
#' Counts the number of rows (or sums a weight column) for each combination
#' of `...`, similar to `dplyr::count()`. Sugar over [group_by.tbl_gpu()],
#' [summarise.tbl_gpu()], and [arrange.tbl_gpu()] -- runs entirely on the
#' GPU, with no CPU round-trip.
#'
#' @param x A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Column names or expressions to group by (e.g. `count(df, x)`
#'   or `count(df, g = x %/% 10)`), parsed through the same expression IR
#'   [mutate.tbl_gpu()] uses for any computed expression. If `x` is already
#'   grouped, these are added to the existing grouping (matching
#'   `group_by(x, ..., .add = TRUE)`).
#' @param wt Optional weight column/expression. If supplied, each group's
#'   count is `sum(wt)` (NA-excluding) instead of the row count.
#' @param sort If `TRUE`, sorts the result in descending order of the count
#'   column.
#' @param name The name of the count column in the output. If `NULL`
#'   (default), uses `"n"`, or `"nn"`/`"nnn"`/... if `"n"` already names one
#'   of the (post-`...`) grouping columns.
#' @param .drop Ignored. Included for compatibility with the dplyr generic
#'   (factor group level dropping is deferred to Phase 11 -- see
#'   `scratchpad/todo.md`).
#'
#' @return A `tbl_gpu` object with one row per group, containing the
#'   grouping columns and the count column. Grouping: if `x` was ungrouped,
#'   the result is ungrouped; if `x` was grouped, the result's grouping is
#'   restored to `x`'s ORIGINAL grouping in full (verified against dplyr
#'   1.2.1 -- see this file's header comment for why this is not simply
#'   "drop the last grouping level").
#'
#' @seealso
#' \code{\link{tally.tbl_gpu}}, \code{\link{group_by.tbl_gpu}},
#' \code{\link{summarise.tbl_gpu}}
#'
#' @export
#' @importFrom dplyr count
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   gpu_mtcars |> count(cyl) |> collect()
#'   gpu_mtcars |> count(cyl, sort = TRUE) |> collect()
#' }
count.tbl_gpu <- function(x, ..., wt = NULL, sort = FALSE, name = NULL, .drop = TRUE) {
  dots <- rlang::enquos(...)
  wt_quo <- rlang::enquo(wt)
  orig_groups <- x$groups

  resolved <- resolve_grouping_dots(x, dots)
  working <- resolved$data
  dot_names <- resolved$names

  grouped <- if (length(dot_names) > 0) {
    dplyr::group_by(working, !!!rlang::syms(dot_names), .add = TRUE)
  } else {
    working
  }

  agg_name <- resolve_agg_name(name, grouped$groups)
  agg_call <- build_count_agg_call(wt_quo)

  # Defensive `dplyr.summarise.inform = FALSE` override (Phase 6, task 3),
  # matching real dplyr's own count()->tally() path (see
  # tally.tbl_gpu()'s identical guard below): count()'s own grouping
  # restoration is unconditional (the `new_tbl_gpu()` call just below always
  # overrides whatever `summarise()` itself computed for `$groups`), so this
  # is purely about not emitting a spurious regroup message when the
  # underlying summarise() peels a multi-column grouping.
  old_opt <- getOption("dplyr.summarise.inform")
  on.exit(options(dplyr.summarise.inform = old_opt), add = TRUE)
  options(dplyr.summarise.inform = FALSE)

  result <- rlang::inject(dplyr::summarise(grouped, !!agg_name := !!agg_call))

  if (isTRUE(sort)) {
    result <- dplyr::arrange(result, dplyr::desc(!!rlang::sym(agg_name)))
  }

  new_tbl_gpu(
    ptr = result$ptr,
    schema = result$schema,
    lazy_ops = result$lazy_ops,
    groups = intersect(orig_groups, result$schema$names),
    exec_mode = result$exec_mode
  )
}

#' Count the rows in each group (no additional grouping)
#'
#' Like [count.tbl_gpu()], but without any `...`: aggregates exactly the
#' groups `x` already has (or all rows, if ungrouped). Sugar over
#' [summarise.tbl_gpu()].
#'
#' @param x A `tbl_gpu` object created by [tbl_gpu()].
#' @param wt Optional weight column/expression -- as [count.tbl_gpu()].
#' @param sort If `TRUE`, sorts the result in descending order of the count
#'   column.
#' @param name The name of the count column. If `NULL` (default), uses
#'   `"n"`, or `"nn"`/`"nnn"`/... if `"n"` already names one of `x`'s group
#'   columns (note: unlike [add_count.tbl_gpu()]/`dplyr::add_tally()`, this
#'   check is against the GROUP columns only, not every column -- verified
#'   empirically against dplyr 1.2.1, see this file's header comment).
#'
#' @return A `tbl_gpu` with the count column appended. If `x` was grouped by
#'   more than one column, the result's grouping drops only the LAST
#'   grouping column (matching `summarise()`'s default `.groups =
#'   "drop_last"` behavior for a real `grouped_df` -- this is `tally()`'s
#'   own rule, and is NOT the same as [count.tbl_gpu()]'s "restore original
#'   grouping in full" rule; see this file's header comment).
#'
#' @seealso \code{\link{count.tbl_gpu}}
#'
#' @export
#' @importFrom dplyr tally
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   gpu_mtcars |> group_by(cyl) |> tally() |> collect()
#' }
tally.tbl_gpu <- function(x, wt = NULL, sort = FALSE, name = NULL) {
  wt_quo <- rlang::enquo(wt)
  orig_groups <- x$groups

  agg_name <- resolve_agg_name(name, orig_groups)
  agg_call <- build_count_agg_call(wt_quo)

  # Matches real dplyr's own `tally.data.frame()`, which wraps its internal
  # `summarise()` call in `local_options(dplyr.summarise.inform = FALSE)` --
  # a defensive suppression regardless of the caller's own
  # `dplyr.summarise.inform` setting (verified against dplyr 1.2.1's actual
  # source). Phase 6, task 3: now that `summarise()` itself implements the
  # real `.groups = "drop_last"` default, this plain `summarise()` call
  # ALREADY returns the correctly drop-last-grouped result (peeling only the
  # LAST grouping level) with no extra group-restoration step needed here --
  # unlike before this task, when `summarise()` always dropped ALL groups
  # and `tally()` had to manually recompute
  # `orig_groups[-length(orig_groups)]` itself.
  old_opt <- getOption("dplyr.summarise.inform")
  on.exit(options(dplyr.summarise.inform = old_opt), add = TRUE)
  options(dplyr.summarise.inform = FALSE)

  result <- rlang::inject(dplyr::summarise(x, !!agg_name := !!agg_call))

  if (isTRUE(sort)) {
    result <- dplyr::arrange(result, dplyr::desc(!!rlang::sym(agg_name)))
  }

  result
}

#' Add a count column without collapsing rows
#'
#' Like [count.tbl_gpu()], but keeps every row and column of `x`, appending
#' the per-group count as a new column (broadcast to every row in the
#' group) instead of collapsing to one row per group. Sugar over
#' [group_by.tbl_gpu()] and a grouped-aggregate [mutate.tbl_gpu()] (which
#' natively broadcasts the per-group count/sum to every row via `mutate()`'s
#' general window/aggregate machinery, Phase 5 -- NA keys are grouped
#' together correctly and original row order is preserved, see
#' `R/window.R`/`src/ops_window.cpp`).
#'
#' @param x A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Column names or expressions to group by, as
#'   [count.tbl_gpu()]. If `x` is already grouped, these are added to the
#'   existing grouping for computing the count, but the OUTPUT's grouping
#'   is restored to `x`'s original grouping (matching [count.tbl_gpu()]'s
#'   own reconstruction rule).
#' @param wt Optional weight column/expression, as [count.tbl_gpu()].
#' @param sort If `TRUE`, sorts the result in descending order of the count
#'   column (note: since no rows are collapsed, ties -- and non-tied rows
#'   within the same group -- are far more common than in `count()`).
#' @param name The name of the count column. If `NULL` (default), uses
#'   `"n"`, or `"nn"`/`"nnn"`/... if `"n"` already names ANY existing
#'   column of `x` (not just a group column -- verified empirically against
#'   dplyr 1.2.1 to differ from `count()`'s own narrower check; see this
#'   file's header comment).
#' @param .drop Deprecated in dplyr 1.0.0 (and here); errors if supplied.
#'
#' @return A `tbl_gpu` with the same rows as `x`, plus the count column,
#'   grouped the same as `x` was originally.
#'
#' @seealso \code{\link{count.tbl_gpu}}
#'
#' @export
#' @importFrom dplyr add_count
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   gpu_mtcars |> add_count(cyl) |> collect()
#' }
add_count.tbl_gpu <- function(x, ..., wt = NULL, sort = FALSE, name = NULL, .drop = NULL) {
  if (!is.null(.drop)) {
    stop("`add_count(.drop = )` was deprecated in dplyr 1.0.0 and is no ",
         "longer accepted.", call. = FALSE)
  }

  dots <- rlang::enquos(...)
  wt_quo <- rlang::enquo(wt)
  orig_groups <- x$groups

  resolved <- resolve_grouping_dots(x, dots)
  working <- resolved$data
  dot_names <- resolved$names

  grouped <- if (length(dot_names) > 0) {
    dplyr::group_by(working, !!!rlang::syms(dot_names), .add = TRUE)
  } else {
    working
  }

  all_cols <- current_schema(grouped)$names
  agg_name <- resolve_agg_name(name, all_cols)
  agg_call <- build_count_agg_call(wt_quo, for_mutate = TRUE)

  result <- rlang::inject(dplyr::mutate(grouped, !!agg_name := !!agg_call))

  if (isTRUE(sort)) {
    result <- dplyr::arrange(result, dplyr::desc(!!rlang::sym(agg_name)))
  }

  new_tbl_gpu(
    ptr = result$ptr,
    schema = result$schema,
    lazy_ops = result$lazy_ops,
    groups = intersect(orig_groups, result$schema$names),
    exec_mode = result$exec_mode
  )
}
