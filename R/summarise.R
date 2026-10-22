#' Summarise groups in a GPU table
#'
#' Computes aggregations on groups defined by [group_by()]. Operations are
#' performed entirely on the GPU for maximum performance.
#'
#' @param .data A grouped `tbl_gpu` object created by [group_by()].
#' @param ... Name-value pairs of summary functions. The name will be the
#'   name of the variable in the result. The value must be a single aggregation
#'   expression in the form `fun(column)`, where `column` can also be any
#'   expression the internal expression IR understands (see "Aggregation
#'   sub-expressions" below).
#' @param .groups Controls grouping structure of the result (Phase 6, task
#'   3; matches `dplyr::summarise()`'s own contract exactly, verified
#'   empirically against dplyr 1.2.1). One of:
#'   \itemize{
#'     \item `NULL` (default) -- `"drop_last"` (see below), PLUS an
#'       informational message when the result is still grouped (see
#'       "The regroup message" below).
#'     \item `"drop_last"` -- drop the LAST grouping variable, keeping the
#'       rest (e.g. `group_by(a, b, c)` -> result grouped by `a, b`); with
#'       only one grouping variable, the result is ungrouped. Same as the
#'       default, minus the message.
#'     \item `"drop"` -- the result is always ungrouped.
#'     \item `"keep"` -- every original grouping variable is retained.
#'     \item `"rowwise"` -- the summary runs on the GPU keeping every
#'       grouping variable, and the result is a rowwise `tbl_gpu` by those
#'       variables ([dplyr::rowwise()]); `collect()` returns dplyr's own
#'       `rowwise_df`.
#'   }
#'   On a rowwise `tbl_gpu` (see [rowwise.tbl_gpu()]) `summarise()` computes one
#'   row per input row, the rowwise variables first. `.groups = NULL` (and
#'   `"keep"`) groups the result by those variables (with dplyr's message
#'   when `.groups` is not given), `"drop"` ungroups it, `"rowwise"` keeps it
#'   rowwise; `"drop_last"` is an error, as in dplyr.
#'   `.groups=` is only validated against an already-grouped `.data` --
#'   matching a genuine (if surprising) dplyr quirk verified empirically:
#'   any value other than `"rowwise"` (including a typo) is silently a no-op on an
#'   UNGROUPED `.data`, since there is nothing to drop/keep/peel from an
#'   empty group set.
#'
#'   ## The regroup message
#'   When `.groups` is left `NULL` (the default) and the pre-summarise
#'   `.data` has MORE THAN ONE grouping variable, an informational message
#'   is emitted (loosely matching dplyr's own wording -- this is
#'   informational, not a value under parity test). It is suppressed by
#'   `options(dplyr.summarise.inform = FALSE)` (an explicit `TRUE`/`FALSE`
#'   value for this option always wins outright); with the option unset,
#'   dplyr's own rule applies: the message only fires when the CALLING
#'   code's top environment is the global environment (i.e. top-level/
#'   interactive use) -- it is silent when `summarise()` is called from
#'   inside a function/package (as in a `testthat` test, or one verb
#'   calling another, e.g. [tally.tbl_gpu()]).
#' @param .by Optional on-the-fly grouping columns (tidyselect), GPU-native
#'   (Phase 5, task W9). Unlike `mutate()`/`filter()`'s `.by=` (which behave
#'   exactly like `group_by()`), `summarise(.by=)` has one genuinely
#'   different rule verified empirically against dplyr 1.2.1: groups are
#'   emitted in FIRST-APPEARANCE order (the order each distinct `.by`
#'   combination first appears in `.data`), not sorted group-key order --
#'   see "`.by` group order" below. The result is always ungrouped -- unlike
#'   a `group_by()`-driven `summarise()`, whose result's grouping now
#'   depends on `.groups=` (Phase 6, task 3); `.groups=` can't be combined
#'   with `.by=` at all (rejected by dplyr's own `summarise()` generic).
#'   Supplying `.by` when `.data` is already grouped (via `group_by()`) is
#'   an error, matching dplyr exactly.
#'
#' @return A `tbl_gpu` object with one row per group containing the grouping
#'   columns and computed aggregations.
#'
#' @details
#' ## Supported aggregation functions
#' \itemize{
#'   \item `sum(x)` - Sum of values. Over an integer or logical `x` the result
#'     is an integer, as in dplyr; if it leaves R's integer range
#'     (`+-(2^31 - 1)`) dplyr returns a double, and so does an eager `tbl_gpu`.
#'     A lazy plan has a fixed schema and raises a
#'     `cuplyr_error_sum_overflow` error instead (use `sum(as.numeric(x))`),
#'     except that a lazy `summarise()` which also takes the `min()`/`max()`
#'     of an integer or logical column runs immediately and then behaves like
#'     an eager one (double on overflow).
#'   \item `mean(x)` - Arithmetic mean
#'   \item `min(x)` - Minimum value
#'   \item `max(x)` - Maximum value
#'   \item `n()` - Count of rows in each group
#'   \item `sd(x)` - Standard deviation
#'   \item `var(x)` - Variance
#'   \item `median(x)` - Median value (Phase 6, task 6.2)
#'   \item `quantile(x, probs)` - A single quantile, `probs` a scalar in
#'     `[0, 1]` (Phase 6, task 6.2; only one probability per call is
#'     supported)
#'   \item `n_distinct(x)` - Count of distinct values, `NA` counted as its
#'     own distinct value by default (Phase 6, task 6.2)
#'   \item `first(x)` / `last(x)` / `nth(x, n)` - The first/last/nth value
#'     in the group's current row order (Phase 6, task 6.2)
#'   \item `any(x)` / `all(x)` - Logical reductions with R's own
#'     three-valued logic (Phase 6, task 6.2; `x` must be logical)
#' }
#'
#' ## Aggregation sub-expressions
#' `column` is not limited to a bare column name: any expression the same
#' expression IR that powers [filter.tbl_gpu()]/[mutate.tbl_gpu()] understands
#' is computed as a hidden temporary column first, then aggregated. This
#' covers comparisons and logic (`sum(carb > 3 & wt < 4)`, `any(carb > 3)`),
#' arithmetic (`mean(hp / wt)`), and math functions (`mean(sqrt(hp))`,
#' `sum(carb %% 2 == 0)`), arbitrarily nested -- the same surface documented
#' in \code{\link{mutate.tbl_gpu}}. A comparison/logical sub-expression is
#' summed as `TRUE`/`FALSE` -> `1`/`0`, matching R's own coercion
#' (`sum(c(TRUE, TRUE))` is `2L`) -- except inside `any()`/`all()`, which
#' keep it logical. An expression shape the IR doesn't recognize raises the
#' same "Invalid aggregation expression" error as any other unsupported
#' aggregation shape.
#'
#' `column` may also be (or contain) a WINDOW-function call (Phase 6, task
#' 4), e.g. `last(cumsum(v))`, `sum(v - lag(v, default = 0))`,
#' `mean(cumsum(v) - lag(cumsum(v)))`: the window function is computed PER
#' GROUP first (over each group's own row order, using the same
#' window-decomposition engine [mutate.tbl_gpu()] uses), and the
#' aggregation then reduces that per-row window result -- matching dplyr's
#' own two-step semantics exactly. Only ONE aggregation's own argument tree
#' is decomposed this way: combining TWO SEPARATE aggregation calls at the
#' dot level (`sum(cumsum(x)) - mean(y)`) is not supported (the same
#' "Invalid aggregation expression" error as any other dot that isn't a
#' bare `fn(single_arg)` shape).
#'
#' ## NA handling
#' `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()`/`median()` all match R's
#' own `na.rm = FALSE` default (Phase 6, tasks 6.1/6.2): if a group contains
#' ANY `NA` in the aggregated column, the WHOLE group's result is `NA` for
#' that aggregation -- e.g. `summarise(m = mean(x))` is `NA` for a group
#' with even one `NA` value of `x`, matching real dplyr exactly. Pass
#' `na.rm = TRUE` explicitly (e.g. `sum(x, na.rm = TRUE)`) to exclude `NA`
#' values instead (cudf's own native aggregation behavior). `n()` is
#' unaffected by `na.rm=` (it counts every row regardless of nulls, like
#' dplyr's own `n()`).
#'
#' When `na.rm = TRUE` and a group's aggregated column is ENTIRELY `NA`
#' (Phase 6, task 4), the result matches base R's own empty-vector
#' reduction identity, exactly like real dplyr: `sum()` -> `0`, `mean()` ->
#' `NaN`, `min()` -> `Inf`, `max()` -> `-Inf` (with base R's warning "no
#' non-missing arguments to min; returning Inf" / "... max; returning -Inf",
#' as dplyr surfaces it, for double and integer input alike; it is only
#' raised when some group actually hits the identity), `sd()`/`var()`/
#' `median()` -> `NA` (their natural result already, no special-casing
#' needed).
#'
#' Over an INTEGER column, dplyr promotes the WHOLE `min()`/`max()` result
#' column to double the instant any group needs the identity, and keeps it
#' integer otherwise; cuplyr does the same. Because that type depends on the
#' data, a `summarise()` with an integer `min()`/`max()` runs immediately even
#' on a lazy table (pending operations are computed first), and the result's
#' schema reflects the real type. A 0-row input also counts as an empty
#' group: ungrouped, `summarise()` returns one row of empty-vector reductions
#' (`n()` 0, `mean()` `NaN`, `sum()` 0, `min()` `Inf`, ...); grouped, the
#' result has no rows but an integer `min()`/`max()` is typed double, as in
#' dplyr.
#'
#' `quantile()` diverges from the mean/sum/.../median family: matching base
#' R's own `quantile.default()` exactly, `na.rm = FALSE` (the default) with
#' an actual `NA` present is a hard ERROR ("missing values and NaN's not
#' allowed if 'na.rm' is FALSE"), not a `NA` result -- pass
#' `na.rm = TRUE` to compute the quantile over the non-`NA` values instead.
#'
#' `n_distinct()` counts `NA` as its own distinct value when
#' `na.rm = FALSE` (the default, matching dplyr's own `n_distinct()`
#' exactly) -- e.g. `n_distinct(c(1, 2, NA, 2))` is `3`. Pass
#' `na.rm = TRUE` to exclude `NA` from the distinct count instead.
#'
#' `first()`/`last()`/`nth()` use dplyr's own `na_rm=` spelling
#' (underscore, not `na.rm`): `na_rm = FALSE` (the default) never skips a
#' `NA` when picking the first/last/nth value (so the result can itself be
#' `NA`); `na_rm = TRUE` skips `NA` values, returning the first/last/nth
#' non-`NA` value in the group (or `NA` if the group has none).
#'
#' `any()`/`all()` implement R's exact three-valued logic: `any(x)` is
#' `TRUE` if any element is `TRUE` (regardless of `NA`s present), `NA` if
#' there's no `TRUE` but at least one `NA` (with `na.rm = FALSE`, the
#' default), else `FALSE`; `all(x)` is `FALSE` if any element is `FALSE`
#' (regardless of `NA`s present), `NA` if there's no `FALSE` but at least
#' one `NA` (with `na.rm = FALSE`), else `TRUE`. `na.rm = TRUE` drops `NA`
#' values first, matching base R's `any()`/`all()` exactly (an all-`NA`
#' group with `na.rm = TRUE` is `FALSE` for `any()`, `TRUE` for `all()`,
#' the empty-vector identity in both cases).
#'
#' ## Ungrouped summarise
#' If `.data` is not grouped, summarise will compute aggregations over all
#' rows, returning a single-row table.
#'
#' ## No expressions
#' `summarise(.data)` with no expressions follows dplyr: ungrouped, it
#' returns a one-row table with no columns; grouped, it returns the distinct
#' group keys (one row per group, in `group_by()`'s ascending, `NA`-last
#' order) and peels one grouping layer per `.groups`; with `.by`, it returns
#' the distinct `.by` combinations in first-appearance order.
#'
#' ## Tables with no columns
#' cudf cannot hold rows without columns, so a zero-column `tbl_gpu` keeps its
#' row count on the R side. What that covers: `tbl_gpu()`, `dim()`,
#' `collect()`/`as_tibble()`, `compute()`/`collapse()`,
#' `group_by()`/`ungroup()`, zero-dot `summarise()` (1 x 0), `summarise()`
#' with dots (a CPU fallback, so `n()` counts correctly), ungrouped
#' `slice()`/`slice_head()`/`slice_tail()`, `bind_rows()`/`bind_cols()` of
#' zero-column tables only, a literal-mask `filter()`, `dplyr_row_slice()`,
#' and adding a column with `$<-`/`[[<-`. Every other verb, joins with a zero-column side,
#' vctrs operations (`vec_size()`, `vec_slice()`, ...), and binding a
#' zero-column table that has rows to one with columns raise an error instead
#' of returning a wrong row count (full support is F14).
#'
#' ## Redefining columns
#' An expression named like a grouping column replaces that key in place
#' (same position, the expression's type; the table stays grouped by that
#' name). An expression that mentions a name an earlier expression created
#' (or a name created twice) sees the new value, as in dplyr; such calls run
#' through the CPU fallback.
#'
#' ## Group order
#' `group_by(g) |> summarise(...)` returns groups in dplyr's `group_by()`
#' order: ascending, `NA` last, strings in the C locale, `FALSE < TRUE`,
#' factors in level order (`gpu_summarise()` sorts the small aggregated result
#' by the keys). `count()`/`tally()` inherit it.
#'
#' ## `.drop = FALSE`
#' On a `group_by(.drop = FALSE)` grouping with a factor key, every group of
#' `group_data()` gets a row, empty ones included, holding each expression
#' evaluated on an empty vector (`n()` 0, `sum()` 0, `mean()` `NaN`, `min()`
#' `Inf` with base R's warning, ...); with no expressions the result is the
#' keys of every group. Such a call materializes a lazy table on the GPU (the
#' row count is data-dependent).
#'
#' ## `.by` group order
#' Verified empirically against dplyr 1.2.1: `summarise(.by=)` emits one row
#' per distinct `.by` combination in the order that combination FIRST
#' appears in `.data` -- e.g. `summarise(tibble(g = c("b","a","b")), n =
#' n(), .by = g)` returns `g` in the order `c("b", "a")`, not `c("a", "b")`.
#' This differs from `group_by(g) |> summarise(...)`, whose groups are
#' sorted. Implemented via a
#' desugar (`summarise_by_desugar()`): an ungrouped `row_number()` marks
#' each row's original position, a grouped summarise computes the user's
#' aggregations plus `min()` of that position per group, and the result is
#' sorted by that minimum (each group's first-appearance position) before
#' the helper column is dropped.
#'
#' @seealso
#' \code{\link{group_by.tbl_gpu}} for grouping data,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr summarise summarize
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Simple aggregation over all rows
#'   total <- gpu_mtcars |>
#'     summarise(avg_mpg = mean(mpg)) |>
#'     collect()
#'
#'   # Grouped aggregation
#'   by_cyl <- gpu_mtcars |>
#'     group_by(cyl) |>
#'     summarise(
#'       avg_mpg = mean(mpg),
#'       max_hp = max(hp),
#'       count = n()
#'     ) |>
#'     collect()
#'
#'   # Multiple grouping columns
#'   by_cyl_gear <- gpu_mtcars |>
#'     group_by(cyl, gear) |>
#'     summarise(
#'       mean_mpg = mean(mpg),
#'       min_wt = min(wt)
#'     ) |>
#'     collect()
#' }
summarise.tbl_gpu <- function(.data, ..., .groups = NULL, .by = NULL) {
  # Captured at this call depth so it matches exactly what dplyr's own
  # `summarise.grouped_df()` captures via its own `caller_env()` call --
  # needed by `resolve_summarise_groups()`'s regroup-message default (see
  # `summarise_should_inform()` below). Note: dplyr's exported `summarise()`
  # generic itself already rejects supplying BOTH `.by` and `.groups`
  # (checked before `UseMethod()` dispatch, so it applies to `tbl_gpu`
  # automatically -- verified empirically, no code needed here for that
  # specific conflict).
  caller_env <- rlang::caller_env()
  check_unique_input_names(.data)

  # A rowwise table summarises once per row (F49, R/rowwise.R).
  if (is_rowwise_gpu(.data)) {
    return(summarise_rowwise(.data, rlang::enquos(...), .groups, rlang::enquo(.by), caller_env))
  }

  # `.groups = "rowwise"`: summarise keeping every group, then the result is
  # rowwise by those groups (F49: a rowwise tbl_gpu; collect() gives dplyr's
  # rowwise_df). The host rowwise_df of a CPU route is rowwise already.
  if (identical(.groups, "rowwise")) {
    res <- summarise_impl_dots(.data, rlang::enquos(...), "keep", rlang::enquo(.by),
                               caller_env)
    if (is_tbl_gpu(res)) {
      return(rowwise_mark(res, res$groups))
    }
    return(dplyr::rowwise(dplyr::ungroup(res), dplyr::all_of(dplyr::group_vars(res))))
  }

  by_quo <- rlang::enquo(.by)

  # Phase 5, task W9: `.by=` resolution -- checked up front (before the
  # zero-dots branch just below), matching mutate()/filter()'s own `.by=`
  # resolution order: resolve_by() (R/execute.R) errors immediately if
  # `.data` is already grouped, and this must fire even when `...` is empty
  # (verified empirically: `summarise(group_by(df, g), .by = g)` raises the
  # grouped-`.data` error).
  by_cols <- resolve_by(.data, by_quo, ".by")
  by_given <- !rlang::quo_is_null(by_quo)

  dots <- rlang::enquos(...)
  summarise_impl(.data, dots, .groups, by_quo, by_cols, by_given, caller_env)
}

# Internal: the notice for a summarise step that computes the pending lazy ops
# early (a temp column, a window argument, an integer min()/max()). It is
# reported like a CPU fallback (`cuplyr.fallback`), except inside the wide step
# of a native reframe() (`with_reframe_wide()`), whose caller never wanted
# laziness there: the wide table is computed right away anyway.
# @keywords internal
notify_materialization <- function(verb, expr_text) {
  if (!isTRUE(reframe_state$wide)) {
    cuplyr_fallback_notify(verb, expr_text)
  }
  invisible(NULL)
}

# Internal: summarise_impl() with `.by=` resolved from its quosure.
# @keywords internal
summarise_impl_dots <- function(.data, dots, .groups, by_quo, caller_env) {
  by_cols <- resolve_by(.data, by_quo, ".by")
  by_given <- !rlang::quo_is_null(by_quo)
  summarise_impl(.data, dots, .groups, by_quo, by_cols, by_given, caller_env)
}

# Internal: the body of summarise.tbl_gpu() on captured dots, so the
# cur_group_id() path can run it again on a reduced set of dots.
# @keywords internal
summarise_impl <- function(.data, dots, .groups, by_quo, by_cols, by_given, caller_env) {
  # slice(.preserve = TRUE) fixed the group keys: the empty groups get a row
  # too (n() = 0), which only the CPU summarise (seeing the keys) produces.
  if (!is.null(fixed_keys_of(.data))) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!dots, .groups = .groups))
    }))
  }

  # F07: zero dots is a valid call (dplyr returns the group keys).
  if (length(dots) == 0) {
    return(summarise_zero_dots(.data, by_cols, by_given, .groups, caller_env))
  }

  # F07: a zero-column table has rows but no cudf column to count them with,
  # so even n() can't run on the GPU. The CPU fallback sees the real row
  # count (collect() honours `$nrow`).
  if (length(current_schema(.data)$names) == 0L) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!dots, .groups = .groups, .by = !!by_quo))
    }))
  }

  # Phase 10: across()/if_any()/if_all()/pick()/cur_*() -- see R/across.R's
  # own module docs. Must run on the ORIGINAL (not yet auto_name_dots()'d)
  # dots: a top-level `across(...)` dot is unnamed by construction, and
  # auto_name_dots() would otherwise stamp it with its own deparsed text as
  # a "name", which expand_across_dots() would then (wrongly) treat as a
  # user-supplied name. Other uses of `cur_group_id()` are deferred shapes
  # here (unlike mutate()/filter()): summarise()'s per-GROUP (not per-row)
  # shape means the row-level join-based substitution those two verbs use
  # doesn't produce a valid `fn(col)` aggregation dot on its own (see
  # substitute_cur_group_id()'s own docs) -- routed to the whole-call CPU
  # fallback instead, which real dplyr already handles correctly. The one
  # native shape, a dot that is exactly `cur_group_id()`, is handled just above.
  # length(cur_group_rows()) is n(); the CPU fallbacks keep the user's dots.
  user_dots <- dots
  dots <- colwise_rewrite_dots(dots, current_schema(.data))
  dots <- rewrite_context_dots(dots, character(), name_unnamed = TRUE)
  # A dot that is exactly cur_group_id() is the output row number: the rows of
  # a summarise result are in group_data() order. Anything more involved with
  # cur_group_id() (or a reduced call that would reach the CPU) stays on dplyr.
  if (summarise_has_cur_group_id(dots)) {
    reduced <- summarise_group_id_split(.data, dots, by_cols, by_given)
    if (!is.null(reduced)) {
      res <- summarise_with_group_id(
        .data, dots, reduced, .groups, by_quo, by_cols, by_given, caller_env
      )
      if (!is.null(res)) return(res)
    }
  }

  if (dots_need_fallback(dots, treat_cur_group_id_as_deferred = TRUE)) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!user_dots, .groups = .groups, .by = !!by_quo))
    }))
  }

  # F34: the fallbacks below must re-run what the user wrote, not the
  # across()-expanded, auto-named dots (that would drop dplyr's unnamed
  # data-frame splice and evaluate across() lambdas sequentially).
  orig_dots <- user_dots

  summarise_groups <- if (by_given) by_cols else .data$groups
  expanded <- expand_across_dots(dots, current_schema(.data), summarise_groups, "summarise")
  if (isTRUE(attr(expanded, "needs_fallback"))) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!user_dots, .groups = .groups, .by = !!by_quo))
    }))
  }
  # Named silently here: the "will use expression as column name" warning is
  # only true on the GPU path, so it is raised below, after the fallback
  # checks (dplyr itself does not warn for an unnamed splice or expression).
  dots <- name_dots_silently(expanded)

  # F53: a reducer whose arguments read no column (`sum(1L)`, `max(th)`) has
  # the same value in every group, but how often dplyr evaluates it (once per
  # group), and what it warns, is observable. The call goes to dplyr
  # unevaluated; it is never evaluated here.
  if (summarise_has_constant_agg(dots, current_schema(.data))) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!user_dots, .groups = .groups, .by = !!by_quo))
    }))
  }

  # Phase 11, task L1: universal expression-level CPU fallback. Runs AFTER
  # across()-expansion/auto_name_dots() (so an ordinary, expandable
  # across() dot is screened normally, as its expanded per-column dots --
  # never mistaken for an unrecognized shape) but BEFORE any GPU work at
  # all (summarise_dots_parseable() touches no GPU pointer, only the
  # schema) -- "never half-lower". A dot whose aggregation shape
  # parse_aggregations() would eventually reject as "Invalid aggregation
  # expression"/"Unsupported aggregation function" (an unknown reducer
  # like cor()/weighted.mean(), or an unrecognized sub-expression like
  # sum(paste0(x))) now falls back to real dplyr::summarise() instead of
  # hard-erroring -- see summarise_dots_parseable()'s own docs for exactly
  # which cases trigger this vs. propagate a genuine error.
  # F07: dots that redefine a name another dot assigned, or reference one
  # (`summarise(x = max(x), m = mean(x))`: the second dot sees the NEW x), are
  # evaluated sequentially by dplyr. The GPU path computes every aggregate
  # from the input columns at once, so these go to the CPU fallback.
  if (summarise_dots_are_sequential(dots) ||
      !summarise_dots_parseable(dots, current_schema(.data))) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!orig_dots, .groups = .groups, .by = !!by_quo))
    }))
  }

  auto_name_dots(expanded, "summarise")  # only for its unnamed-expression warning

  cpu <- function() {
    gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(.data = tbl, !!!orig_dots, .groups = .groups, .by = !!by_quo))
    })
  }
  # F42: an integer window sum() in an aggregation's argument that overflows
  # in some group reruns the whole call on the CPU (resolve_agg_window_arg()),
  # in both modes, as the window stages run at summarise(). The GPU attempt's
  # warnings are dropped then: dplyr raises its own.
  gpu_or_cpu <- function(attempt) {
    retry_on_sum_overflow(attempt, cpu, class = "cuplyr_summarise_rerun")
  }

  if (by_given) {
    # `.by=` summarise is always ungrouped (Phase 5, task W9); `.groups=`
    # can never reach here alongside `.by=` (dplyr's own generic already
    # rejects that combination before this method is even dispatched to).
    return(gpu_or_cpu(function() summarise_by_desugar(.data, dots, by_cols)))
  }

  retained_groups <- resolve_summarise_groups(.data$groups, .groups, caller_env)

  # F33: a `.drop = FALSE` grouping with a factor key also emits the empty
  # groups. The test reads the schema only, so the default path pays nothing.
  if (summarise_wants_empty_groups(.data)) {
    # An aggregate named like a key replaces it; rare enough for the CPU.
    if (any(names(dots) %in% .data$groups)) return(cpu())
    return(gpu_or_cpu(function() summarise_with_empty_groups(.data, dots, retained_groups, cpu)))
  }

  gpu_or_cpu(function() summarise_core(.data, dots, retained_groups))
}

# Internal (F53): TRUE iff some dot is a reducer call (`sum(1L)`, `max(th)`)
# with no column reference. dplyr evaluates such a dot once per group, so its
# side effects and warnings are per group; the caller sends the whole call to
# dplyr without evaluating anything. Nothing is evaluated here. A reducer
# over a context call (`max(row_number())`, `sum(n())`) is not constant and
# keeps its native route.
# @keywords internal
summarise_has_constant_agg <- function(dots, schema) {
  reducers <- c("sum", "mean", "min", "max", "sd", "var", "median", "quantile",
                "n_distinct", "first", "last", "nth", "any", "all")
  for (i in seq_along(dots)) {
    expr <- rlang::quo_get_expr(dots[[i]])
    head <- call_head_name(expr)
    if (!is.null(head) && head %in% reducers && length(expr) >= 2L &&
        !expr_has_column_ref(expr, schema) && !expr_has_mask_only_call(expr)) {
      return(TRUE)
    }
  }
  FALSE
}

# Internal: TRUE iff summarise() of `.data` has to add rows for empty groups
# (F33): `group_by(.drop = FALSE)` with some factor key. Schema checks only.
# @keywords internal
summarise_wants_empty_groups <- function(.data) {
  length(.data$groups) > 0L && isFALSE(.data$group_drop) &&
    any(.data$groups %in% names(current_schema(.data)$factor_levels))
}

# Internal: grouped summarise() on a `.drop = FALSE` grouping (F33). The
# observed groups are summarised as usual; the empty groups of group_data()
# take the values of the same aggregations over an empty slice (F07's
# ungrouped 0-row summarise: n() 0, sum() 0, mean() NaN, min() Inf, ...). The
# table is assembled on the GPU (one gather over [observed; empty-slice row]
# plus the uploaded key columns), so aggregate values never pass through R
# (they keep exactly what the GPU computed). Needs the data's row layout, so a
# lazy input's pending ops are computed first.
# @param dots auto_name_dots()'d aggregation quosures
# @param fallback Function running the whole call on the CPU, for the
#   (unexpected) aggregate type mismatches
# @keywords internal
summarise_with_empty_groups <- function(.data, dots, retained_groups, fallback) {
  was_lazy <- identical(.data$exec_mode, "lazy")
  # A GPU materialization of the pending ops, not a CPU fallback: no notice.
  base <- compute(.data)
  base$exec_mode <- "eager"
  keys <- base$groups

  finish <- function(result) {
    if (was_lazy) result$exec_mode <- "lazy"
    result
  }

  gd <- compute_group_data(base)
  if (!any(lengths(gd$.rows) == 0L)) {
    return(finish(summarise_core(base, dots, retained_groups)))
  }

  observed <- summarise_core(base, dots, character())
  empty_slice <- new_tbl_gpu(
    ptr = wrap_gpu_call("summarise", gpu_gather_rows(list(base$ptr), numeric(0))),
    schema = base$schema, exec_mode = "eager"
  )
  identity_row <- summarise_core(empty_slice, dots, character())

  n_keys <- length(keys)
  agg_names <- observed$schema$names[-seq_len(n_keys)]
  obs_types <- observed$schema$types[-seq_len(n_keys)]
  id_types <- identity_row$schema$types
  if (!identical(identity_row$schema$names, agg_names)) {
    return(fallback())
  }

  # INT32 vs FLOAT64 (an integer min()/max() that did or didn't meet an empty
  # group) is the one drift that can occur: cast to the common type.
  common <- ifelse(obs_types == id_types, obs_types, "FLOAT64")
  if (any(obs_types != id_types & !(obs_types %in% c("INT32", "FLOAT64") &
                                   id_types %in% c("INT32", "FLOAT64")))) {
    return(fallback())
  }
  casted <- wrap_gpu_call("summarise", {
    obs_ptr <- gpu_select(observed$ptr, n_keys + seq_along(agg_names) - 1L)
    id_ptr <- identity_row$ptr
    for (j in which(obs_types != common)) obs_ptr <- gpu_cast_column(obs_ptr, j - 1L, common[[j]])
    for (j in which(id_types != common)) id_ptr <- gpu_cast_column(id_ptr, j - 1L, common[[j]])
    list(obs = obs_ptr, id = id_ptr)
  })

  # Row of each group_data() group in the observed result, or the identity row
  # (index n_observed in the concatenation) when it is empty.
  obs_keys <- collect(dplyr::select(observed, dplyr::all_of(keys)))
  gd_keys <- gd[keys]
  n_obs <- nrow(obs_keys)
  idx <- vctrs::vec_match(gd_keys, obs_keys[keys])
  idx0 <- ifelse(is.na(idx), n_obs, idx - 1L)

  agg_ptr <- wrap_gpu_call("summarise", gpu_gather_rows(list(casted$obs, casted$id), as.numeric(idx0)))
  key_tbl <- tbl_gpu(tibble::as_tibble(gd_keys), lazy = FALSE)
  schema <- list(
    names = c(keys, agg_names),
    types = c(unname(key_tbl$schema$types), common),
    factor_levels = key_tbl$schema$factor_levels
  )
  schema$tz <- tz_from_positions(
    schema$names, schema$types,
    c(tz_by_position(key_tbl$schema), tz_by_position(observed$schema)[-seq_len(n_keys)])
  )
  finish(new_tbl_gpu(
    ptr = wrap_gpu_call("summarise", gpu_bind_cols_impl(list(key_tbl$ptr, agg_ptr))),
    schema = schema, groups = retained_groups,
    group_drop = length(retained_groups) == 0L,
    exec_mode = "eager"
  ))
}

# Internal: summarise() with no dots (F07), as dplyr's summarise.data.frame()
# / summarise.grouped_df() do it:
#   - ungrouped: a 1 x 0 table. It does not depend on the input's data, so it
#     is uploaded fresh (a lazy join result has no pointer to read); `$nrow`
#     carries the 1. A lazy input's pending ops are computed (and discarded)
#     first, since they can error.
#   - grouped: one row per group holding only the grouping columns, in
#     group_by()'s canonical order (ascending, NA last; gpu_summarise() sorts
#     the keys). A `.drop = FALSE` grouping with a factor key also emits its
#     empty groups, built from group_data()'s keys. `.groups` is resolved and
#     validated like every other summarise().
#   - `.by=`: the distinct `.by` combinations in first-appearance order,
#     ungrouped.
# @keywords internal
summarise_zero_dots <- function(.data, by_cols, by_given, groups_arg, caller_env) {
  if (by_given && length(by_cols) > 0) {
    return(strip_frame_attrs(dplyr::distinct(.data, !!!rlang::syms(by_cols))))
  }

  retained_groups <- resolve_summarise_groups(.data$groups, groups_arg, caller_env)

  if (length(.data$groups) == 0) {
    # The result is always the same 1 x 0 table, uploaded fresh (a lazy join
    # result has no `$ptr` to build on). Pending ops still run first, as in
    # dplyr, so their errors (`unmatched = "error"`, ...) surface.
    if (has_pending_ops(.data)) {
      cuplyr_fallback_notify("summarise", "no expressions (pending operations run first)")
      compute(.data)
    }
    return(copy_frame_meta(
      tbl_gpu(tibble::tibble(.rows = 1L), lazy = .data$exec_mode == "lazy"),
      .data, attrs = FALSE
    ))
  }

  if (summarise_wants_empty_groups(.data)) {
    return(summarise_group_keys(.data, retained_groups))
  }

  # gpu_summarise() sorts the keys itself (F33).
  push_op(.data, ast_summarise(input_node(.data), list(), .data$groups, retained_groups))
}

# Internal: zero-dots summarise() of a `.drop = FALSE` grouping (F33): the keys
# of every group_data() group, empty ones included, uploaded as the result.
# Needs the data's row layout, so a lazy input's pending ops are computed first
# on the GPU (not a CPU fallback: no notice).
# @keywords internal
summarise_group_keys <- function(.data, retained_groups) {
  was_lazy <- identical(.data$exec_mode, "lazy")
  base <- compute(.data)
  base$exec_mode <- "eager"
  # (group_data() carries a `.drop` attribute, which is not the table's)
  keys <- tbl_gpu(strip_frame_attrs(tibble::as_tibble(compute_group_data(base)[base$groups])),
                  lazy = FALSE)
  keys$groups <- retained_groups
  keys$group_drop <- length(retained_groups) == 0L
  if (was_lazy) keys$exec_mode <- "lazy"
  keys
}

# Internal: TRUE iff the dots must be evaluated one after another (F07): a
# name is assigned twice, or a dot mentions a name an EARLIER dot assigned
# (dplyr's later dot then sees the new value, for a grouping column as well as
# any other). Names are collected with `all.vars()` (deliberately generous: it
# also reports `.data$x`'s `x`) plus the string in `.data[["x"]]`, which
# `all.vars()` does not see. A false positive only costs the CPU fallback.
# @param dots auto_name_dots()'d quosures
# @keywords internal
summarise_dots_are_sequential <- function(dots) {
  nms <- names(dots)
  if (anyDuplicated(nms) > 0) {
    return(TRUE)
  }
  for (i in seq_along(dots)[-1]) {
    expr <- rlang::quo_get_expr(dots[[i]])
    if (any(c(all.vars(expr), expr_data_pronoun_names(expr)) %in% nms[seq_len(i - 1)])) {
      return(TRUE)
    }
  }
  FALSE
}

# Internal: every name read through `.data$name` / `.data[["name"]]` anywhere
# in `expr`.
# @keywords internal
expr_data_pronoun_names <- function(expr) {
  # An empty argument slot (`x[, 1]`) errors when forced: it names nothing.
  tryCatch({
    if (is_data_pronoun_call(expr)) {
      pronoun_call_name(expr)
    } else if (is.call(expr)) {
      unlist(lapply(as.list(expr), expr_data_pronoun_names), use.names = FALSE) %||% character()
    } else {
      character()
    }
  }, error = function(e) character())
}

# Internal: resolve `.groups=` into the retained group-column set, and fire
# dplyr's own conditional "has regrouped the output" message (Phase 6, task
# 3).
#
# Mirrors dplyr 1.2.1's own `summarise.data.frame()`/`summarise.grouped_df()`
# split EXACTLY (verified empirically):
#   - Ungrouped `.data` (`orig_groups` empty): `.groups=` is NOT VALIDATED AT
#     ALL except the `"rowwise"` special-case -- any value, including a
#     typo, is silently a no-op (verified: `df |> summarise(x, .groups =
#     "bogus")` on an ungrouped tibble does NOT error in real dplyr, while
#     the identical call on a grouped tibble does). The result is always
#     ungrouped, matching the fact that there's nothing to "keep"/
#     "drop_last"/"drop" from an empty group set.
#   - Grouped `.data`: `.groups=` IS validated (`NULL`/`"drop_last"`/
#     `"drop"`/`"keep"`/`"rowwise"` only, else a hard error naming the bad
#     value); the "has regrouped the output" message fires ONLY when
#     `.groups` is unset (`NULL`, the default) AND there is more than one
#     grouping column -- gated the same way dplyr gates it, see
#     `summarise_should_inform()`.
#
# `.groups = "rowwise"` never reaches this function: summarise.tbl_gpu()
# summarises with "keep" and marks the result rowwise (rowwise_mark()).
#
# @param orig_groups Character vector, `.data$groups` BEFORE this summarise
# @param groups_arg The raw `.groups=` value as passed by the caller (may be
#   `NULL`)
# @param caller_env The environment that called `summarise()` (captured via
#   `rlang::caller_env()` at the top of `summarise.tbl_gpu()`, the same call
#   depth dplyr's own method captures its own `caller_env()` at)
# @return Character vector, the retained group columns
# @keywords internal
resolve_summarise_groups <- function(orig_groups, groups_arg, caller_env) {
  if (length(orig_groups) == 0) {
    # Ungrouped input: dplyr's summarise.data.frame() never validates
    # `.groups=` (any non-"rowwise" value, valid or not, is a silent no-op)
    # and always returns ungrouped -- verified empirically, see this
    # function's own docs.
    return(character())
  }

  valid_values <- c("drop_last", "drop", "keep")
  if (!is.null(groups_arg) && !identical(groups_arg, character(0)) &&
      !groups_arg %in% valid_values) {
    stop(
      "`.groups` can't be \"", groups_arg, "\"\n",
      "Possible values are NULL (default), \"drop_last\", \"drop\", ",
      "\"keep\", and \"rowwise\"",
      call. = FALSE
    )
  }

  effective <- if (is.null(groups_arg)) "drop_last" else groups_arg

  if (identical(effective, "keep")) {
    return(orig_groups)
  }
  if (identical(effective, "drop")) {
    return(character())
  }

  # "drop_last" (either explicit, or the NULL default's effective value):
  # peel the LAST grouping variable, keeping the rest; with only one
  # grouping variable, the whole result is ungrouped.
  n <- length(orig_groups)
  retained <- if (n > 1) orig_groups[-n] else character()

  if (is.null(groups_arg) && n > 1 && summarise_should_inform(caller_env)) {
    inform_regrouped_output(orig_groups, retained)
  }

  retained
}

# Internal: dplyr's own `summarise_verbose()` rule (Phase 6, task 3) --
# whether the "has regrouped the output" message should fire, verified
# empirically against dplyr 1.2.1's own (unexported) `summarise_verbose()`:
# an explicit `options(dplyr.summarise.inform = TRUE/FALSE)` always wins;
# otherwise the message only fires when the CALLER's top environment is the
# global environment (i.e. top-level/interactive use -- e.g. `Rscript -e`
# or a script sourced at top level) and is silent when `summarise()` is
# called from inside a function/package's own code (a `testthat` test, or
# one cuplyr verb calling another internally, e.g. `tally.tbl_gpu()`).
# @keywords internal
summarise_should_inform <- function(caller_env) {
  inform <- getOption("dplyr.summarise.inform")
  if (isTRUE(inform) || isFALSE(inform)) {
    return(inform)
  }
  identical(topenv(caller_env), globalenv())
}

# Internal: emit (a loose paraphrase of) dplyr's own "has regrouped the
# output" message -- informational text, not a value under parity test
# (scratchpad/todo.md Phase 6, task 3 asked to "match message text loosely").
# @keywords internal
inform_regrouped_output <- function(old_groups, new_groups) {
  new_desc <- if (length(new_groups) == 0) "no groups (ungrouped)" else paste(new_groups, collapse = ", ")
  message(
    "`summarise()` has regrouped the output by dropping the last grouping ",
    "variable.\n",
    "Summaries were computed grouped by ", paste(old_groups, collapse = ", "), ".\n",
    "Output is grouped by ", new_desc, ".\n",
    "Use `summarise(.groups = \"drop_last\")` to silence this message, or ",
    "`.groups = \"keep\"`/`\"drop\"` to retain/drop every grouping level."
  )
}

# Internal: L1 pre-pass (Phase 11) -- TRUE iff every dot in `dots` (already
# across()-expanded/auto_name_dots()'d) is a shape parse_aggregations()
# will accept, checked WITHOUT any GPU side effects, so it's safe to call
# before touching `.data` at all ("never half-lower").
#
# Deliberately mirrors preprocess_agg_expressions()'s own decision tree
# exactly (same extract_na_rm()/extract_agg_shape_param()/
# decompose_agg_call()/ir_parse_quo() calls, same
# ir_has_window()-before-ir_is_const() ordering), rather than reusing that
# function directly, since preprocess_agg_expressions() has real
# side-effects (create_temp_column() mutates a GPU pointer, compute()
# materializes a lazy table) this pre-pass must never trigger -- this is a
# pure, repeatable dry run over quosures/schema only.
#
# Returns `FALSE` (a fallback trigger) for exactly the two shapes that
# would otherwise reach parse_aggregations()'s "Invalid aggregation
# expression" (decompose_agg_call() fails, or its arg doesn't parse at
# all) or "Unsupported aggregation function" (a function name outside the
# fixed cudf-backed list, e.g. cor()/weighted.mean()/an unknown reducer)
# errors -- i.e. "this SHAPE isn't understood" -- and for an NA flag the
# native path can't take exactly (the other spelling, a value that isn't a
# single TRUE/FALSE, see extract_na_rm()), which dplyr evaluates. A dot
# that's a genuine user mistake FLOWS THROUGH unchanged (not caught here)
# and still errors normally at column-not-found checks, or ir_parse_quo()
# throwing (e.g. a missing column inside the aggregation argument) --
# matching the "NULL-vs-throw trigger rule".
#
# @param dots Quosures from summarise(), already across()-expanded and
#   auto_name_dots()'d (exactly what summarise.tbl_gpu() itself is about to
#   hand to summarise_core()/summarise_by_desugar())
# @param schema `.data`'s current schema (list(names=, types=))
# @return Logical scalar
# @keywords internal
summarise_dots_parseable <- function(dots, schema) {
  known_agg_fns <- c("sum", "mean", "min", "max", "n", "sd", "var", "count",
                      "median", "quantile", "n_distinct", "first", "last", "nth",
                      "any", "all")

  for (i in seq_along(dots)) {
    # An NA flag the reducer doesn't take, or a value the native path can't
    # take exactly (extract_na_rm()), is dplyr's to evaluate: CPU.
    stripped <- extract_na_rm(dots[[i]], schema)
    if (is.null(stripped)) {
      return(FALSE)
    }
    quo <- stripped$quo
    # A shape the native path can't take (quantile() with type != 7 or a
    # probs it can't evaluate, a Period/Duration `n`/`probs` (F48), an nth()
    # it can't position) is not a user mistake until dplyr says so: CPU,
    # where dplyr gives its value or its own error.
    shaped <- tryCatch(extract_agg_shape_param(quo), error = function(e) NULL)
    if (is.null(shaped)) {
      return(FALSE)
    }
    quo <- shaped$quo

    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)

    if (grepl("^n\\(\\)$", strip_ns_prefix(rlang::quo_text(quo)))) {
      next  # bare n(): always valid, no column argument to check
    }

    decomposed <- decompose_agg_call(expr)
    if (is.null(decomposed) || is.null(decomposed$arg)) {
      return(FALSE)  # not a `fn(single_arg)` shape (and not bare n())
    }
    if (!decomposed$fn_name %in% known_agg_fns) {
      return(FALSE)  # unknown reducer, e.g. cor()/weighted.mean()
    }
    # n() takes no argument (`n(na.rm = TRUE)` is dplyr's "unused argument"),
    # and the one argument left must be the data: unnamed or `x =`, not a
    # misspelled control (`sum(na_rm = TRUE)`). n_distinct(...) takes its
    # data in `...` and rejects any name there (dplyr's check_dots_unnamed()),
    # where the base reducers' `...` ignore one.
    arg_name <- if (is.null(names(expr))) "" else names(expr)[[2]]
    if (identical(decomposed$fn_name, "n") || !arg_name %in% c("", "x") ||
        (identical(decomposed$fn_name, "n_distinct") && nzchar(arg_name))) {
      return(FALSE)
    }

    arg_expr <- decomposed$arg
    # F37/F41: of these, only min/max/first/last/nth/n_distinct and
    # mean/median take a timestamp on the GPU (the same set as
    # ir_ts_native_ops; mean/median compute on the integer rep and convert
    # back, ts_agg.hpp). cudf has no sd/var/sum/quantile over one, and dplyr
    # gives an error for sum/quantile/any/all: CPU, bare or computed.
    ts_agg <- !decomposed$fn_name %in% c("min", "max", "first", "last", "nth", "n_distinct",
                                         "mean", "median")
    if (ts_agg && is.symbol(arg_expr) && as.character(arg_expr) %in% schema$names &&
        unname(schema$types[match(as.character(arg_expr), schema$names)]) %in% cuplyr_timestamp_types) {
      return(FALSE)
    }
    if (is.symbol(arg_expr) && as.character(arg_expr) %in% schema$names) {
      # A bare factor column only aggregates natively under n_distinct();
      # first()/max()/sum()/... need dplyr's factor semantics (CPU)
      if (identical(ir_factor_rule(ir_col(as.character(arg_expr)), schema), "bare") &&
          !identical(decomposed$fn_name, "n_distinct")) {
        return(FALSE)
      }
      # F48: the typed allowlist (ir_native_arg_types) judges the column's type
      if (!ir_type_gate(decomposed$fn_name,
                        schema$types[match(as.character(arg_expr), schema$names)])) {
        return(FALSE)
      }
      next  # bare column reference: always valid, no temp column needed
    }

    arg_quo <- rlang::new_quosure(arg_expr, env)
    ir <- ir_parse_quo(arg_quo, schema)  # may throw a genuine error -- propagates

    status <- ir_factor_rule(ir, schema)
    if (status == "cpu" || (status == "bare" && !identical(decomposed$fn_name, "n_distinct"))) {
      return(FALSE)  # a factor outside the allowed places (F18's rule): CPU
    }
    if (is.null(ir)) {
      return(FALSE)  # sub-expression shape not understood at all
    }
    if (ts_agg && ir_type_is_timestamp(ir, schema)) {
      return(FALSE)
    }
    # F48: the same typed allowlist for a computed argument
    arg_type <- tryCatch(ir_infer_type(ir, schema), error = function(e) NA_character_)
    if (!is.na(arg_type) && !ir_type_gate(decomposed$fn_name, arg_type)) {
      return(FALSE)
    }
    # F34: dplyr evaluates the argument once per group. A length-1 argument
    # that contains a reducer (`sum(mean(n))`, `sum(n())`) is a reduction of a
    # scalar, which the broadcast-then-reduce window path would get wrong.
    if (ir_agg_arg_is_scalar_reduction(ir)) {
      return(FALSE)
    }
    if (ir_has_window(ir)) {
      next  # windows-inside-summarise: always valid (Phase 6, task 4)
    }
    # ir_is_const(ir) (e.g. sum(5)): left alone here, exactly like
    # preprocess_agg_expressions() -- not a "shape not understood" case,
    # so not a fallback trigger; parse_aggregations() raises its own
    # diagnosable error for this rare/degenerate shape.
  }

  TRUE
}

# Internal: per-group length analysis of an aggregate's argument (F34).
# dplyr evaluates `sum(<arg>)` once per group, so `<arg>` is either a
# length-n vector (n = the group's rows) or a length-1 value. A column is
# length n, a literal length 1; a reducing call (mean(), n(), sum(), ...)
# returns length 1; a row-level window call (cumsum(), lag(), rank family,
# rolling) takes its first argument's length (a bare `row_number()` is n);
# anything element-wise is n if some child is n (recycling).
#
# The GPU path broadcasts a reducer's result to n copies. That is harmless
# while the length-1 value is used element-wise, but wrong once it feeds a
# window (agg or row-level): dplyr runs the window over ONE value, the GPU
# over n copies (`cumsum(mean(n))`, `sum(mean(n))`, `lag(mean(n))`). The
# desugars of cummean()/consecutive_id() add a bare `row_number()` (length n)
# next to the user's argument; they are judged by the window that wraps the
# user's argument (cumsum()/lag()), so that injected node never hides the
# length-1 argument.
# @return `list(len = "1"|"n", reducer = <a reducer occurs inside>,
#   unsafe = <a window inside takes a length-1 value containing a reducer>)`
# @keywords internal
ir_agg_length <- function(ir) {
  kind <- ir$kind
  if (identical(kind, "lit")) {
    return(list(len = "1", reducer = FALSE, unsafe = FALSE))
  }
  if (!identical(kind, "call")) {
    return(list(len = "n", reducer = FALSE, unsafe = FALSE))  # col / vec
  }

  kids <- lapply(ir$args, ir_agg_length)
  any_reducer <- any(vapply(kids, `[[`, logical(1), "reducer"))
  any_unsafe <- any(vapply(kids, `[[`, logical(1), "unsafe"))
  window <- ir_call_registry[[ir$op]]$window
  if (!is.null(window)) {
    first <- if (length(kids) > 0) kids[[1]] else NULL
    unsafe <- any_unsafe ||
      (!is.null(first) && identical(first$len, "1") && isTRUE(first$reducer))
    if (identical(window$kind, "agg")) {
      return(list(len = "1", reducer = TRUE, unsafe = unsafe))
    }
    len <- if (is.null(first)) "n" else first$len
    return(list(len = len, reducer = any_reducer, unsafe = unsafe))
  }
  any_n <- any(vapply(kids, function(k) identical(k$len, "n"), logical(1)))
  list(len = if (any_n) "n" else "1", reducer = any_reducer, unsafe = any_unsafe)
}

# TRUE iff the broadcast-then-reduce path would get the aggregate's argument
# wrong: it is itself a length-1 value built from a reducer, or some window
# inside it takes one.
# @keywords internal
ir_agg_arg_is_scalar_reduction <- function(ir) {
  info <- ir_agg_length(ir)
  isTRUE(info$unsafe) || (identical(info$len, "1") && isTRUE(info$reducer))
}

# Internal: the shared aggregation pipeline -- temp-column preprocessing,
# aggregation parsing, one ast_summarise push -- used both by the ordinary
# group_by()-or-ungrouped path and by summarise_by_desugar()'s own internal
# grouped summarise call (with its extra `..summarise_first..` dot already
# appended).
#
# @param .data A tbl_gpu (its own `$groups` drive the aggregation)
# @param dots Already auto_name_dots()'d aggregation quosures
# @param retained_groups Character vector (Phase 6, task 3), the group
#   columns to retain on the OUTPUT's `$groups` -- as computed by
#   `resolve_summarise_groups()` for the ordinary `.groups=`-driven path, or
#   `character()` (the default -- always ungrouped) for
#   `summarise_by_desugar()`'s own internal grouped summarise call, which is
#   a pure computation device unrelated to user-visible `.groups=` semantics
#   (`.by=` summarise is always ungrouped, matching dplyr).
# @return A new tbl_gpu, grouped by `retained_groups` (see
#   propagate_groups(), R/execute.R, which reads `node$result_groups`)
# @keywords internal
summarise_core <- function(.data, dots, retained_groups = character(),
                           internal_cols = character(), sort_groups = TRUE) {
  # No wrapper of its own (F58): the work below is R logic (quosure parsing,
  # which may evaluate a `probs =`/`n =` argument) around wrapped GPU steps, so
  # a retry belongs to each step, not to a rerun of this parsing. The tables
  # its steps create are freed when the user-level call returns.
  # Phase 6, task 6.1 (Fix A): extract and strip any `na.rm = <literal>`
  # named argument BEFORE temp-column preprocessing -- this must happen
  # first so a sub-expression call like `sum(sqrt(x), na.rm = TRUE)` is
  # reduced to the `fn(single_arg)` shape preprocess_agg_expressions() (and
  # parse_aggregations()'s own regex) already understand, with na.rm
  # tracked out-of-band per dot instead of embedded in the call shape.
  # Each reducer's own spelling only (`na_rm` for first()/last()/nth()) --
  # see extract_na_rm()'s docs.
  extracted <- lapply(dots, extract_na_rm, schema = current_schema(.data))
  if (any(vapply(extracted, is.null, logical(1)))) {
    # Unreachable from summarise(): summarise_dots_parseable() sends these
    # dots to the CPU first.
    stop("internal error: summarise_core() given an NA flag the native path can't take",
         call. = FALSE)
  }
  na_rm_flags <- vapply(extracted, `[[`, logical(1), "na_rm")
  dots <- lapply(extracted, `[[`, "quo")

  # Phase 6, task 6.2: extract and strip quantile()'s `probs=` / nth()'s
  # `n=` scalar shape parameter -- run AFTER na.rm/na_rm extraction (so
  # this function's own argument-matching never has to understand that
  # control), reducing both to the same single-argument `fn(x)` shape
  # every other aggregation already produces.
  extracted2 <- lapply(dots, extract_agg_shape_param)
  agg_extra <- vapply(extracted2, `[[`, numeric(1), "extra")
  dots <- lapply(extracted2, `[[`, "quo")

  # Pre-process: create temporary columns for expressions inside agg functions
  # (e.g. sum(carb == 4)). Per D4, if a temp column is needed and .data is
  # lazy with pending ops, this materializes (compute()) before creating the
  # temp column and notifies via cuplyr_fallback_notify().
  preprocess_result <- preprocess_agg_expressions(.data, dots)
  working_data <- preprocess_result$data
  processed_dots <- preprocess_result$dots

  # Parse the aggregation expressions (now with simple column refs), emitting
  # aggregation structs that carry the cudf-accepted function name (fixes
  # lazy sd()/var(), which used to reach C++ as "sd"/"var" and error out).
  aggregations <- parse_aggregations(current_schema(working_data), processed_dots,
                                     na_rm_flags, agg_extra)

  # F07: internal helper aggregations (summarise_by_desugar()'s row-position
  # min()) must stay INT32 and never warn.
  for (i in seq_along(aggregations)) {
    if (aggregations[[i]]$output_col %in% internal_cols) {
      aggregations[[i]]$identity <- FALSE
    }
  }

  # F07: an INT32 min()/max() turns the WHOLE result column into FLOAT64
  # (Inf/-Inf) iff some group is empty -- a property of the data, which a
  # schema inferred before running can't know. Such a summarise runs right
  # away (a lazy table first materializes its pending ops, like the
  # temp-column case above) and the schema is corrected from the real result.
  if (any(vapply(aggregations, function(agg) isTRUE(agg$identity) &&
                 identical(unname(agg$input_type), "INT32"), logical(1)))) {
    return(summarise_run_now(working_data, aggregations, retained_groups, sort_groups))
  }

  # F34: an integer sum() that overflows is a double in dplyr. An eager
  # table reconciles its schema from the result; a lazy plan keeps INT32 and
  # errors at execution (see ast_summarise(on_sum_overflow=)).
  eager <- !identical(working_data$exec_mode, "lazy")
  result <- push_op(working_data, ast_summarise(
    input_node(working_data), aggregations, working_data$groups, retained_groups,
    on_sum_overflow = if (eager) "double" else "error", sort_groups = sort_groups
  ))
  if (eager) reconcile_summarise_types(result) else result
}

# Internal: run a summarise node immediately, then reconcile the schema with
# the result's real column types (F07, see summarise_core()). Returns a table
# in the input's exec_mode with no pending ops.
# @keywords internal
summarise_run_now <- function(.data, aggregations, retained_groups, sort_groups = TRUE) {
  was_lazy <- identical(.data$exec_mode, "lazy")

  if (was_lazy && has_pending_ops(.data)) {
    notify_materialization("summarise", paste0(
      "integer min()/max() (result type depends on the data): ",
      paste(vapply(aggregations, function(a) a$output_col, character(1)), collapse = ", ")
    ))
  }
  base <- compute(.data)
  base$exec_mode <- "eager"
  result <- push_op(base, ast_summarise(input_node(base), aggregations,
                                        base$groups, retained_groups,
                                        on_sum_overflow = "double",
                                        sort_groups = sort_groups))
  result <- reconcile_summarise_types(result)

  if (was_lazy) {
    result$exec_mode <- "lazy"
  }
  result
}

# Internal: correct an executed summarise result's schema from its real
# column types. The only possible drift is an INT32 column that came back
# double: an INT32 min()/max() promoted to FLOAT64 (F07), or an integer sum()
# that overflowed R's integer range (F34).
# @keywords internal
reconcile_summarise_types <- function(result) {
  actual <- gpu_col_types(result$ptr)
  promoted <- result$schema$types == "INT32" & actual == "dbl"  # gpu_col_types() spelling
  result$schema$types[promoted] <- "FLOAT64"
  result
}

# The NA flag each native reducer takes, by its exact name: base R's
# reducers and dplyr's n_distinct() spell it `na.rm`, dplyr's first()/last()/
# nth() `na_rm` (dplyr 1.2.1: `first(x, order_by = NULL, default = NULL,
# na_rm = FALSE)`, `n_distinct(..., na.rm = FALSE)`). n() takes none.
summarise_na_flag <- c(
  sum = "na.rm", mean = "na.rm", min = "na.rm", max = "na.rm", sd = "na.rm",
  var = "na.rm", median = "na.rm", quantile = "na.rm", n_distinct = "na.rm",
  any = "na.rm", all = "na.rm",
  first = "na_rm", last = "na_rm", nth = "na_rm"
)

# Internal: Extract and strip the NA flag from a single summarise()
# aggregation dot (Phase 6, task 6.1, Fix A; task 6.2 for first()/last()/
# nth(); per function since F19-a2)
#
# Only the reducer's own spelling (`summarise_na_flag`) is a flag. The other
# one is left in the call, where it is an extra argument the native shape
# doesn't take, so summarise_dots_parseable() sends the call to dplyr, which
# gives what R gives: one more `...` value for sum()/min()/max()/any()/all()
# (TRUE counts as 1), swallowed by mean()/median(), an "unused argument" error
# for sd()/var()/first()/last()/nth(). A partially matched name
# (`mean(x, na = TRUE)`) is left in the same way.
#
# Both defaults are `FALSE`, matching R's own `na.rm = FALSE` default: any NA
# in a group makes the WHOLE group's aggregation result NA for the mean/sum/
# .../median family. cudf's own groupby aggregations always exclude nulls, so
# this default requires an extra whole-group-null propagation step in
# `gpu_summarise()` (`src/ops_groupby.cpp`) for that family -- `na_rm = FALSE`
# requests that step; `na_rm = TRUE` opts back into cudf's native
# (NA-excluding) behavior directly. `n_distinct()`/`first()`/`last()`/`nth()`
# interpret the SAME flag differently (selecting cudf's own `null_policy`
# directly -- see `make_aggregation()`'s own docs), and `any()`/`all()`
# implement R's three-valued logic with it.
#
# The value must be one ir_na_flag_value() accepts (a single TRUE/FALSE that
# doesn't read the data mask). Anything else -- `na.rm = 1` (which base R
# coerces), `NA`, a vector, a column, the flag given twice -- returns NULL:
# not native, so summarise_dots_parseable() hands the user's dots to dplyr,
# which computes the value or raises its own error.
#
# @param quo A quosure, one summarise() dot (post `auto_name_dots()`)
# @param schema The input's schema (for the column-reference check)
# @return `list(quo = <possibly rewritten quosure, flag stripped>,
#   na_rm = logical(1))`, or NULL when the flag isn't native
# @keywords internal
extract_na_rm <- function(quo, schema) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)
  unchanged <- list(quo = quo, na_rm = FALSE)

  if (!is.call(expr) || is.null(names(expr))) {
    return(unchanged)
  }
  head <- expr[[1]]
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
    head <- head[[3]]  # strip namespace: pkg::fn -> fn
  }
  if (!is.symbol(head)) {
    return(unchanged)
  }
  flag <- summarise_na_flag[as.character(head)]
  if (is.na(flag)) {
    return(unchanged)  # n(), an unknown reducer: any flag stays an argument
  }

  flag_pos <- which(names(expr) == flag)
  if (length(flag_pos) == 0) {
    return(unchanged)
  }
  if (length(flag_pos) > 1) {
    return(NULL)  # R's "matched by multiple actual arguments"
  }
  na_rm_val <- ir_na_flag_value(expr[[flag_pos]], env, schema)
  if (is.null(na_rm_val)) {
    return(NULL)
  }

  list(quo = rlang::new_quosure(expr[-flag_pos], env), na_rm = na_rm_val)
}

# Internal: Extract and strip `quantile()`'s `probs=` / `nth()`'s `n=`
# scalar shape parameter from a single summarise() aggregation dot (Phase
# 6, task 6.2)
#
# Both `quantile(x, probs)` and `nth(x, n)` are two-(or-more)-argument
# calls whose SECOND argument is a scalar "which one" parameter, not a
# data column -- `decompose_agg_call()`'s generic 0-or-1-arg shape (used
# by every other aggregation) can't represent this, so it's extracted
# out-of-band here, run AFTER `extract_na_rm()` (so any na.rm=/na_rm=
# has already been stripped and doesn't need to be understood by this
# function's own argument matching), reducing the call to the same
# single-argument `fn(x)` shape every other aggregation produces --
# `preprocess_agg_expressions()`/`parse_aggregations()` need no
# quantile()/nth()-specific handling at all past this point.
#
# @param quo A quosure, one summarise() dot (post `extract_na_rm()`)
# @return `list(quo = <possibly rewritten quosure>, extra = <numeric
#   scalar, NA_real_ if not applicable>)`
# @keywords internal
extract_agg_shape_param <- function(quo) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)

  if (!is.call(expr)) {
    return(list(quo = quo, extra = NA_real_))
  }

  head <- expr[[1]]
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
    head <- head[[3]]  # strip namespace: pkg::fn -> fn
  }
  if (!is.symbol(head)) {
    return(list(quo = quo, extra = NA_real_))
  }
  fn_name <- as.character(head)

  if (identical(fn_name, "quantile")) {
    return(extract_quantile_probs(expr, env, quo))
  }
  if (identical(fn_name, "nth")) {
    return(extract_nth_position(expr, env, quo))
  }

  list(quo = quo, extra = NA_real_)
}

# Internal: `quantile()`'s own argument-matching, called from
# `extract_agg_shape_param()`
#
# Only a SINGLE probability is supported (scratchpad/todo.md's explicit
# scope cut for this wave): `probs=` must evaluate to one value in `[0,
# 1]`. `type=` (R's own quantile "algorithm" selector, default `7`) is
# validated but never threaded through -- cuplyr always computes cudf's
# `interpolation::LINEAR`, which is bit-for-bit R's own default `type = 7`
# for a plain (non-discrete) sample (verified empirically, see
# `gpu_summarise()`'s own docs and `test-dplyr-summarise.R`) -- so a
# `type=` OTHER than `7` is rejected outright (no silent divergence)
# rather than silently computing the wrong thing. `names=`/`digits=` only
# affect the OUTPUT NAME's label/rounding in real R (never the value
# itself), so both are accepted and ignored.
#
# @keywords internal
extract_quantile_probs <- function(expr, env, quo) {
  proto <- function(x, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = FALSE,
                     names = TRUE, type = 7, digits = 7) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    stop("Invalid quantile() call in aggregation expression: ",
         rlang::quo_text(quo), call. = FALSE)
  }
  present_args <- as.list(present)[-1]

  if (is.null(present_args$x)) {
    stop("quantile() requires an `x` argument: ", rlang::quo_text(quo), call. = FALSE)
  }
  # extract_na_rm() already took an exact `na.rm =`; one left here was
  # partially matched (`na = TRUE`), and the rewrite below would drop it.
  if (!is.null(present_args[["na.rm"]])) {
    stop("summarise()'s quantile() takes na.rm only by its full name: ",
         rlang::quo_text(quo), call. = FALSE)
  }

  if (!is.null(present_args$type)) {
    type_val <- tryCatch(
      rlang::eval_tidy(rlang::new_quosure(present_args$type, env)),
      error = function(e) NULL
    )
    if (is.null(type_val) || !is.numeric(type_val) || length(type_val) != 1 || type_val != 7) {
      stop("summarise()'s quantile() only supports type = 7 (R's own default): ",
           rlang::quo_text(quo), call. = FALSE)
    }
  }

  if (is.null(present_args$probs)) {
    stop("summarise()'s quantile() requires a single `probs` value: ",
         rlang::quo_text(quo), call. = FALSE)
  }
  probs_val <- tryCatch(
    rlang::eval_tidy(rlang::new_quosure(present_args$probs, env)),
    error = function(e) NULL
  )
  ir_check_timespan_lit(probs_val)  # F48: a Period/Duration `probs`: CPU
  if (is.null(probs_val) || !is.numeric(probs_val) || length(probs_val) != 1 ||
      is.na(probs_val) || probs_val < 0 || probs_val > 1) {
    stop("summarise()'s quantile() requires a single probability in [0, 1]: ",
         rlang::quo_text(quo), call. = FALSE)
  }

  list(
    quo = rlang::new_quosure(rlang::call2("quantile", present_args$x), env),
    extra = as.double(probs_val)
  )
}

# Internal: `nth()`'s own argument-matching, called from
# `extract_agg_shape_param()`
#
# `order_by=`/`default=` are rejected outright (not implemented this
# wave, mirroring `ir_parse_first_last_nth()`'s identical cut for the
# window-function `nth()` -- R/ir.R); `n` must be a single non-zero
# integer (dplyr's own 1-based/negative convention, translated to cudf's
# `NTH_ELEMENT` convention in `gpu_summarise()`, mirroring
# `src/ops_window.cpp`'s existing translation for the window path).
#
# @keywords internal
extract_nth_position <- function(expr, env, quo) {
  proto <- function(x, n, order_by = NULL, default = NULL, na_rm = FALSE) NULL
  present <- tryCatch(rlang::call_match(expr, proto, defaults = FALSE), error = function(e) NULL)
  if (is.null(present)) {
    stop("Invalid nth() call in aggregation expression: ", rlang::quo_text(quo), call. = FALSE)
  }
  present_args <- as.list(present)[-1]

  if (!is.null(present_args$order_by) || !is.null(present_args$default)) {
    stop("summarise()'s nth() does not support order_by=/default=: ",
         rlang::quo_text(quo), call. = FALSE)
  }
  # extract_na_rm() already took an exact `na_rm =`; one left here was
  # partially matched (`na = TRUE`), and the rewrite below would drop it.
  if (!is.null(present_args[["na_rm"]])) {
    stop("summarise()'s nth() takes na_rm only by its full name: ",
         rlang::quo_text(quo), call. = FALSE)
  }
  if (is.null(present_args[["x"]]) || is.null(present_args[["n"]])) {
    stop("nth() requires `x` and `n` arguments: ", rlang::quo_text(quo), call. = FALSE)
  }

  n_val <- tryCatch(
    rlang::eval_tidy(rlang::new_quosure(present_args$n, env)),
    error = function(e) NULL
  )
  ir_check_timespan_lit(n_val)  # F48: a Period/Duration `n`: CPU (parseable pre-pass)
  if (is.null(n_val) || !is.numeric(n_val) || length(n_val) != 1 || is.na(n_val) ||
      n_val != round(n_val) || n_val == 0) {
    stop("nth() requires a single non-zero integer position: ",
         rlang::quo_text(quo), call. = FALSE)
  }

  list(
    quo = rlang::new_quosure(rlang::call2("nth", present_args$x), env),
    extra = as.double(n_val)
  )
}

# Internal: `.by=` desugar for summarise() (Phase 5, task W9)
#
# scratchpad/phase5_window_design.md section 6:
#   1. mutate(..rowid.. = row_number()) -- an UNGROUPED window call (safe:
#      resolve_by() already guarantees `.data` itself carries no groups
#      when `.by=` is supplied, so this is never a grouped row_number()).
#   2. Group by `by_cols` and run the user's own aggregation dots PLUS one
#      extra `..first.. = min(..rowid..)` aggregation -- the smallest
#      original row index surviving in each group is exactly that group's
#      first-appearance position (every row of a group shares the same
#      grouped-summarise result, so `min()` over the whole group is safe
#      even though only one row's rowid "matters").
#   3. arrange(..first..) orders the one-row-per-group result by that
#      first-appearance position -- reproducing dplyr's empirically-verified
#      `.by=` group order (see summarise.tbl_gpu()'s own roxygen).
#   4. Drop `..first..`. The result is already ungrouped (summarise_core()
#      is called with its default `retained_groups = character()`, Phase 6
#      task 3 -- this internal grouped summarise is a pure computation
#      device, unrelated to `.groups=` semantics), matching dplyr's own
#      `.by=` contract.
#
# @param .data A tbl_gpu, confirmed ungrouped by resolve_by() before this is
#   ever called
# @param dots The already-auto_name_dots()'d aggregation quosures (`...`)
# @param by_cols Character vector, the resolved `.by=` columns
# @return A new (ungrouped) tbl_gpu, one row per distinct `by_cols`
#   combination, in first-appearance order
# @keywords internal
summarise_by_desugar <- function(.data, dots, by_cols) {
  reserved_names <- c(current_schema(.data)$names, names(dots))

  rowid_name <- unique_temp_name("..summarise_rowid..", reserved_names)
  with_rowid <- dplyr::mutate(.data, !!rowid_name := dplyr::row_number())

  # Set groups directly (equivalent to group_by(by_cols), but avoids a
  # round trip through the group_by() generic for a purely internal step).
  grouped <- new_tbl_gpu(
    ptr = with_rowid$ptr,
    schema = with_rowid$schema,
    lazy_ops = with_rowid$lazy_ops,
    groups = by_cols,
    exec_mode = with_rowid$exec_mode
  )

  first_name <- unique_temp_name("..summarise_first..", c(reserved_names, rowid_name))
  dots[[first_name]] <- rlang::new_quosure(
    rlang::call2("min", as.name(rowid_name)), rlang::base_env()
  )

  result <- summarise_core(grouped, dots, internal_cols = first_name, sort_groups = FALSE)
  result <- dplyr::arrange(result, !!rlang::sym(first_name))
  # the grouping above was internal: summarise() keeps only the bare class
  copy_frame_meta(dplyr::select(result, -dplyr::all_of(first_name)), .data, attrs = FALSE)
}

#' @rdname summarise.tbl_gpu
#' @export
summarize.tbl_gpu <- summarise.tbl_gpu

# Internal: Strip a leading `pkg::` qualifier from an aggregation call's
# text (e.g. `dplyr::n()` -> `n()`, `stats::sd(mpg)` -> `sd(mpg)`), so the
# shape-matching regexes below (and in parse_aggregations()) see the bare
# call regardless of how the user namespaced it.
#
# @param expr_text Character, expression text from rlang::quo_text()
# @return Character, with any leading `identifier::` removed
# @keywords internal
strip_ns_prefix <- function(expr_text) {
  sub("^[a-zA-Z_][a-zA-Z0-9_.]*::", "", trimws(expr_text))
}

# Internal: decompose an aggregation dot's raw expression into a bare
# `fn(arg)` (or no-arg `fn()`) shape, stripping one leading `pkg::`
# qualifier from the call head, e.g. `dplyr::n()` -> list(fn_name = "n",
# arg = NULL), `stats::sd(mpg)` -> list(fn_name = "sd", arg = quote(mpg)).
# Returns `NULL` if `expr` isn't a call, its head isn't a (possibly
# namespaced) bare symbol, or it has more than one argument (multi-arg
# aggregations like `quantile(x, 0.5)` aren't supported -- left alone here,
# parse_aggregations() raises the actual diagnosable error for these).
#
# @param expr A language object (rlang::quo_get_expr() of a summarise() dot)
# @return `list(fn_name, arg)` (`arg` is `NULL` for a no-arg call), or `NULL`
# @keywords internal
decompose_agg_call <- function(expr) {
  if (!is.call(expr)) {
    return(NULL)
  }

  head <- expr[[1]]
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
    head <- head[[3]]  # strip namespace: pkg::fn -> fn
  }
  if (!is.symbol(head)) {
    return(NULL)
  }

  n_args <- length(expr) - 1L
  if (n_args == 0) {
    return(list(fn_name = as.character(head), arg = NULL))
  }
  if (n_args == 1) {
    return(list(fn_name = as.character(head), arg = expr[[2]]))
  }
  NULL
}

# Internal: Pre-process aggregation expressions
#
# Detects expressions inside aggregation functions (e.g., sum(carb == 4),
# mean(sqrt(x)), sum(x %% 2 == 0)) and creates temporary columns for them
# via the expression IR (R/ir.R) -- the same evaluator filter()/mutate()
# use. Since the aggregation's argument is parsed with `ir_parse_quo()`,
# any expression shape the IR understands is now a valid aggregation
# sub-expression (T7 widens this from the previous single-operator-only
# regex whitelist to "ir_parse_quo() succeeds").
#
# Per D4: if a dot needs a temp column and `.data` is lazy with pending
# operations, this materializes (`compute()`) once, up front, before
# building any temp column, and notifies via `cuplyr_fallback_notify()` --
# `create_temp_column()` always operates on a materialized GPU pointer
# rather than reaching into an in-progress AST.
#
# @param .data A tbl_gpu object
# @param dots Quosures from summarise()
# @return List with modified data and simplified dots
# @keywords internal
preprocess_agg_expressions <- function(.data, dots) {
  working_data <- .data
  new_dots <- vector("list", length(dots))
  names(new_dots) <- names(dots)
  temp_cols <- list()

  for (i in seq_along(dots)) {
    quo <- dots[[i]]
    raw_expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)

    decomposed <- decompose_agg_call(raw_expr)

    if (is.null(decomposed) || is.null(decomposed$arg)) {
      # Not a `fn(single_arg)` shape (includes bare `n()`), or not a call
      # at all: nothing to preprocess. parse_aggregations() validates the
      # shape (and raises its own diagnosable error) from here.
      new_dots[[i]] <- quo
      next
    }

    arg_expr <- decomposed$arg

    # A bare column reference (the common case, e.g. sum(mpg)) needs no
    # temp column -- and is safe to check even while lazy ops are still
    # pending, since it's a pure schema lookup with no GPU work involved.
    if (is.symbol(arg_expr) && as.character(arg_expr) %in% current_schema(working_data)$names) {
      new_dots[[i]] <- quo
      next
    }

    arg_quo <- rlang::new_quosure(arg_expr, env)
    ir <- ir_parse_quo(arg_quo, current_schema(working_data))

    # A factor column spelled through the pronoun or parentheses (`.data$f`,
    # `.data[["f"]]`, `(f)`) parses to a bare `col` node: rewrite to fn(<f>)
    # like the bare-symbol branch above, so n_distinct() of a factor stays
    # native (F18's rule) instead of copying DICTIONARY32 into a temp column.
    # Limited to factors so every other type keeps its existing path.
    if (!is.null(ir) && identical(ir$kind, "col") &&
        identical(decomposed$fn_name, "n_distinct") &&
        identical(ir_factor_rule(ir, current_schema(working_data)), "bare")) {
      new_dots[[i]] <- rlang::new_quosure(
        as.call(list(as.name(decomposed$fn_name), as.name(ir$name))),
        rlang::base_env()
      )
      next
    }

    # any()/all() (Phase 6, task 6.2): keep a BOOL8-inferred sub-expression
    # (e.g. `any(x > 3)`) as BOOL8 rather than the sum()-oriented INT32
    # promotion every other aggregation wants -- see create_temp_column()'s
    # own `keep_bool` docs.
    keep_bool <- decomposed$fn_name %in% c("any", "all")

    # F22: dots that aggregate the same expression (reframe()'s k quantiles
    # of `x * 2` or `cumsum(x)`) share one temp column: the parsed IR already
    # holds any value the expression took from the environment, so equal IR is
    # an equal column. A window argument's column depends on the IR plus
    # working_data's groups and row order, which no dot changes. Each temp
    # column (and each window stage) copies the whole table, so k of them cost
    # k times the table.
    if (!is.null(ir)) {
      shared <- Find(function(t) identical(t$keep_bool, keep_bool) && identical(t$ir, ir), temp_cols)
      if (!is.null(shared)) {
        new_dots[[i]] <- rlang::new_quosure(
          as.call(list(as.name(decomposed$fn_name), as.name(shared$name))),
          rlang::base_env()
        )
        next
      }
    }

    # Phase 6, task 4: windows-inside-summarise (`sum(cumsum(v))`,
    # `last(cumsum(v))`, ...) -- checked BEFORE the ir_is_const() check below,
    # since a window call with no column argument at all (e.g. bare
    # `row_number()`) has no ir_cols() of its own and would otherwise be
    # (wrongly) treated as a constant subtree. See resolve_agg_window_arg()'s
    # own docs for the full derivation and scope boundary.
    if (!is.null(ir) && ir_has_window(ir)) {
      if (has_pending_ops(working_data)) {
        working_data <- compute(working_data)
        notify_materialization("summarise", rlang::quo_text(quo))
      }

      resolved <- resolve_agg_window_arg(working_data, ir, decomposed$fn_name, i)
      working_data <- resolved$data
      temp_cols[[length(temp_cols) + 1L]] <- list(ir = ir, keep_bool = keep_bool,
                                                  name = resolved$col_name)

      new_dots[[i]] <- rlang::new_quosure(
        as.call(list(as.name(decomposed$fn_name), as.name(resolved$col_name))),
        rlang::base_env()
      )
      next
    }

    if (is.null(ir) || ir_is_const(ir)) {
      # Either genuinely unparseable by the expression IR, or a constant
      # subtree with no column reference at all (e.g. `sum(5)`): neither is
      # a temp-column shape. Leave the dot untouched -- parse_aggregations()
      # raises the actual diagnosable error for these rare/degenerate
      # shapes (a literal that isn't a valid column reference).
      new_dots[[i]] <- quo
      next
    }

    # Materialize before building the temp column if lazy ops are pending.
    if (has_pending_ops(working_data)) {
      working_data <- compute(working_data)
      notify_materialization("summarise", rlang::quo_text(quo))
    }

    # Create a temporary column name (index-based, so it stays unique even
    # though not every dot ends up needing one).
    temp_col_name <- paste0(".temp_agg_", i)
    temp_cols[[length(temp_cols) + 1L]] <- list(ir = ir, keep_bool = keep_bool, name = temp_col_name)

    working_data <- create_temp_column(working_data, temp_col_name, ir, keep_bool = keep_bool)

    # Create a new quosure with the temp column name, e.g. `sum(.temp_agg_3)`.
    new_dots[[i]] <- rlang::new_quosure(
      as.call(list(as.name(decomposed$fn_name), as.name(temp_col_name))),
      rlang::base_env()
    )
  }

  list(data = working_data, dots = new_dots)
}

# Internal: windows-inside-summarise (Phase 6, task 4) -- resolve an
# aggregation's OWN argument tree, already confirmed to contain a window-
# function call (`ir_has_window(ir)`), into a plain materialized column
# reference, by reusing the exact window-decomposition machinery
# `mutate()`'s own window path drives (`plan_window_stages()`/
# `decompose_window_group()`, `R/window.R`, Phase 5): `summarise(z =
# last(cumsum(v)))` verified empirically against dplyr 1.2.1 to compute
# `cumsum(v)` PER GROUP first (over each group's own row order), THEN
# reduce that per-row result with `last()` -- exactly the two-step shape
# `plan_window_stages()` (window stage(s) materializing a temp column) +
# the aggregation's OWN reduction (parse_aggregations(), unchanged) already
# gives for free once the window stages are pushed onto `working_data`
# BEFORE the enclosing `summarise()` node.
#
# Windowing happens against `working_data`'s OWN current `$groups` -- the
# SAME grouping `gpu_summarise()` will reduce over moments later (whether
# real `group_by()` columns or, for `summarise(.by=)`, the internal
# `summarise_by_desugar()` call's own `by_cols` grouping) -- so the window
# "sees" exactly the rows/groups the aggregation itself will reduce.
#
# Scope boundary (documented, not a silent gap): only ONE aggregation's OWN
# argument tree is decomposed at a time. Arbitrary window nesting WITHIN
# that one argument is fully supported (same machinery `mutate()` uses for
# one dot) -- `sum(cumsum(x))`, `last(cumsum(v))`, `sum(v - lag(v, default =
# 0))`, `mean(cumsum(v) - lag(cumsum(v)))` (two window calls sharing a
# CSE'd subexpression, deduplicated by `ir_extract_windows()`'s own hash-
# keyed cache) all work. Combining TWO SEPARATE aggregation calls at the
# DOT level (`summarise(z = sum(cumsum(x)) - mean(y))`) is explicitly not
# attempted: `decompose_agg_call()` only ever recognizes a bare
# `fn(single_arg)` dot shape in the first place, so a dot like that never
# even reaches this function -- it falls straight through to
# `parse_aggregations()`'s pre-existing "Invalid aggregation expression"
# error, unchanged (summarise() has no CPU fallback at all, per this file's
# module docs, so there is nowhere else for such a dot to go).
#
# An unlowerable window spec inside the argument (currently only
# `roll_median()`, per `window_spec_lowerable()`) is rejected with a clear,
# summarise()-specific error BEFORE any stage is pushed (mirroring
# `mutate_window()`'s own "validate every spec before executing any of
# them" discipline, R/mutate.R) -- summarise() has no CPU fallback to drop
# back to the way `mutate_window_fallback()` can, so this is a hard error
# rather than a silent re-route.
#
# @param working_data A tbl_gpu (already materialized -- no pending lazy
#   ops; the caller guarantees this, same precondition as
#   `create_temp_column()`)
# @param ir A parsed (unbound) IR node for the aggregation's argument,
#   confirmed by the caller to contain a window call (`ir_has_window(ir)`)
# @param fn_name Character scalar, the aggregation function's OWN name
#   (e.g. `"sum"`, `"last"`) -- used only to decide `keep_bool` for
#   any()/all(), identically to `create_temp_column()`'s own parameter of
#   the same name, if the post-decomposition expression still needs its
#   own temp-column computation
# @param dot_index Integer, this dot's 1-based position in the summarise()
#   call -- used only for a readable, collision-free temp column name if a
#   `create_temp_column()` step ends up needed
# @return `list(data = <tbl_gpu, with every window stage already pushed>,
#   col_name = <character scalar, a real column name in the returned
#   data's schema, safe to reference directly as the aggregation's own
#   argument>)`
# @keywords internal
resolve_agg_window_arg <- function(working_data, ir, fn_name, dot_index) {
  exprs <- stats::setNames(list(ir), "..agg_arg..")
  schema <- current_schema(working_data)

  # F42: an integer window sum() inside the aggregation's argument is INT32, so
  # arithmetic on it is checked like dplyr's (`x * sum(x)`, also under a
  # row-level window: `last(cumsum(x * sum(x)))`). Such a sum can't become a
  # double mid-plan (later stages and the temp column below are typed from the
  # INT32), and dplyr makes only the overflowing groups' sums doubles: the
  # others stay integers, and arithmetic on them still overflows to NA. So an
  # overflow (raised here: resolve_agg_window_plan() computes the stages, and
  # the caller has already computed any earlier pending ops) asks
  # summarise.tbl_gpu() to rerun the whole call on the CPU.
  plan <- plan_window_stages(exprs, schema, working_data$groups)
  tryCatch(
    resolve_agg_window_plan(working_data, plan, fn_name, dot_index),
    cuplyr_error_sum_overflow = function(e) {
      rlang::abort("An integer window sum() overflowed inside summarise().",
                   class = "cuplyr_summarise_rerun")
    }
  )
}

# Internal: run a planned window decomposition for resolve_agg_window_arg()
# @keywords internal
resolve_agg_window_plan <- function(working_data, plan, fn_name, dot_index) {
  all_specs <- unlist(lapply(plan$stages, `[[`, "specs"), recursive = FALSE)
  unlowerable <- Filter(Negate(window_spec_lowerable), all_specs)
  if (length(unlowerable) > 0) {
    stop(
      "summarise() does not support ", unlowerable[[1]]$fn,
      "() inside an aggregation expression.",
      call. = FALSE
    )
  }

  for (stage in plan$stages) {
    if (length(stage$pre) > 0) {
      working_data <- push_op(working_data, ast_mutate(input_node(working_data), stage$pre))
    }
    working_data <- push_op(
      working_data, ast_window(input_node(working_data), stage$specs, working_data$groups)
    )
  }

  # create_temp_column() reads `$ptr`, which is stale while the window stages
  # above are still pending lazy ops: run them (silently -- the caller has
  # already reported its own materialization). Also for a bare column, so an
  # integer sum overflow is raised here, where resolve_agg_window_arg() can
  # replan, and not at collect().
  if (has_pending_ops(working_data)) {
    working_data <- compute(working_data)
  }

  post_ir <- plan$post[["..agg_arg.."]]

  if (identical(post_ir$kind, "col")) {
    return(list(data = working_data, col_name = post_ir$name))
  }

  temp_col_name <- paste0(".temp_agg_win_", dot_index)
  keep_bool <- fn_name %in% c("any", "all")
  working_data <- create_temp_column(working_data, temp_col_name, post_ir, keep_bool = keep_bool)
  list(data = working_data, col_name = temp_col_name)
}

# Internal: Create a temporary column from a parsed IR expression
#
# One IR-based implementation for every aggregation sub-expression shape
# (comparison, arithmetic, logical, math function, arbitrarily nested) --
# built via a single `ir_call`-equivalent step through `gpu_mutate_expr()`,
# the same evaluator entry point `mutate()` uses.
#
# @param .data A tbl_gpu object (already materialized -- no pending lazy
#   ops; `preprocess_agg_expressions()` guarantees this)
# @param col_name Name for the new column
# @param ir A parsed (not yet bound) IR node, non-const (references at
#   least one column)
# @param keep_bool Logical scalar (Phase 6, task 6.2). `FALSE` (default)
#   matches the original sum()-oriented behavior: a BOOL8-inferred
#   sub-expression is declared INT32. `any()`/`all()` need the OPPOSITE --
#   their argument (e.g. `any(x > 3)`) must stay BOOL8, since their
#   three-valued-logic implementation (`gpu_summarise()`) dispatches on
#   the aggregation's own MIN/MAX-of-BOOL8 semantics, not a sum-style
#   integer promotion.
# @return Modified tbl_gpu with the new column appended
# @keywords internal
create_temp_column <- function(.data, col_name, ir, keep_bool = FALSE) {
  schema <- .data$schema
  inferred_type <- ir_infer_type(ir, schema)

  # A boolean-producing expression (comparisons, &/|/!, is.na(), between(),
  # %in%, ...) is declared as INT32 here, matching R's own coercion of
  # logical values in arithmetic contexts (`sum(c(TRUE, TRUE))` is `2L`, an
  # integer, not logical) -- exactly what sum() (the overwhelmingly common
  # aggregation over a comparison, e.g. `sum(carb > 3)`) expects downstream
  # via make_aggregation()'s INT32->INT64 promotion rule. gpu_mutate_expr()
  # performs the actual GPU-side cast (BOOL8 -> INT32) since the declared
  # output type differs from the computed one (src/ops_expr.cpp). Skipped
  # for any()/all() (`keep_bool = TRUE`), which need the argument to stay
  # logical -- see this function's own `keep_bool` docs.
  new_type <- if (identical(inferred_type, "BOOL8") && !keep_bool) "INT32" else inferred_type

  bound_ir <- ir_bind(ir, schema)

  new_ptr <- warn_int_overflow(wrap_gpu_call(
    "summarise_mutate_expr",
    gpu_mutate_expr(.data$ptr, list(bound_ir), col_name, new_type, schema$names)
  ))

  # Everything else on the schema (notably `factor_levels`, which a factor
  # grouping key still needs) is carried over.
  new_schema <- append_schema_cols(schema, col_name, new_type, ir_infer_tz(ir, schema))

  copy_frame_meta(new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    lazy_ops = .data$lazy_ops,
    groups = .data$groups,
    group_drop = .data$group_drop %||% TRUE,
    exec_mode = .data$exec_mode
  ), .data)
}

# Internal: Parse aggregation expressions into make_aggregation() structs
#
# The one parser used by both eager and lazy scheduling (D1): given a
# schema and quosures already simplified by preprocess_agg_expressions()
# (so every dot's argument is either a bare column name or the no-arg
# n()), validates function/column names -- raising the same three
# diagnosable errors the pre-unification eager parser did, verbatim -- and
# emits aggregation structs (via make_aggregation(), which also folds in
# former get_agg_result_type()'s result-type inference) carrying the
# *cudf*-accepted function name. This is the fix for lazy sd()/var(),
# which used to reach C++ as "sd"/"var" (gpu_summarise()/get_groupby_agg()
# in src/ops_groupby.cpp only accept "std"/"variance") and error out with
# "Unknown aggregation type"; the eager path already translated these
# names and continues to do so here.
#
# A leading `pkg::` qualifier (`dplyr::n()`, `stats::sd(mpg)`, ...) is
# stripped before shape-matching so namespaced aggregation calls resolve
# the same way as their bare equivalents.
#
# @param schema Current schema (list(names=, types=))
# @param dots Quosures from summarise(), already temp-column-preprocessed
# @param na_rm_flags Logical vector, same length/order as `dots` (Phase 6,
#   task 6.1; extended 6.2 for the new functions): each dot's
#   already-extracted `na.rm`/`na_rm` value (see `extract_na_rm()`),
#   `FALSE` when not supplied by the user, matching R's own default
# @param agg_extra Numeric vector, same length/order as `dots` (Phase 6,
#   task 6.2): each dot's already-extracted shape parameter (see
#   `extract_agg_shape_param()`) -- `quantile()`'s probability, `nth()`'s
#   position, `NA_real_` for every other function
# @return List of aggregation structs (see make_aggregation())
# @keywords internal
parse_aggregations <- function(schema, dots, na_rm_flags = rep(FALSE, length(dots)),
                               agg_extra = rep(NA_real_, length(dots))) {
  # dplyr-facing aggregation function names (order matches the historical
  # error message from parse_agg_expressions()) and their cudf-accepted
  # equivalents (src/ops_groupby.cpp::get_groupby_agg(), plus the
  # dedicated non-get_groupby_agg() handling gpu_summarise() gives
  # quantile()/n_distinct()/first()/last()/nth()/any()/all() -- Phase 6,
  # task 6.2).
  agg_functions <- c("sum", "mean", "min", "max", "n", "sd", "var", "count",
                      "median", "quantile", "n_distinct", "first", "last", "nth",
                      "any", "all")
  agg_fn_map <- c(
    sum = "sum", mean = "mean", min = "min", max = "max",
    n = "n", sd = "std", var = "variance", count = "n",
    median = "median", quantile = "quantile", n_distinct = "n_distinct",
    first = "first", last = "last", nth = "nth", any = "any", all = "all"
  )

  # Aggregations restricted to a numeric-ish input (median()/quantile(): a
  # STRING column has no well-defined median/quantile) vs. restricted to a
  # logical input (any()/all(): dplyr's own three-valued-logic contract is
  # defined over logical vectors; a numeric/string column here would be a
  # user mistake this catches early with a clear message rather than a raw
  # cudf type error later). n_distinct()/first()/last()/nth() accept any
  # type (matching dplyr, which places no type restriction on these).
  numeric_only_fns <- c("median", "quantile")
  logical_only_fns <- c("any", "all")

  aggregations <- vector("list", length(dots))

  for (i in seq_along(dots)) {
    expr <- dots[[i]]
    expr_text <- rlang::quo_text(expr)

    # F34: read the call off the expression, not its deparsed text (a
    # backticked name like `my x` has no usable text form). A one-argument
    # call whose argument isn't a name keeps the old "Column '<text>' not
    # found" diagnosis.
    decomposed <- decompose_agg_call(rlang::quo_get_expr(expr))

    # n() has no column argument
    if (!is.null(decomposed) && identical(decomposed$fn_name, "n") && is.null(decomposed$arg)) {
      aggregations[[i]] <- make_aggregation(names(dots)[i], NA_character_, "n")
      next
    }

    if (is.null(decomposed) || is.null(decomposed$arg)) {
      # Defensive, internal-only path (Phase 11, task L1): every real
      # summarise() call is screened by summarise_dots_parseable() before
      # reaching here, which already routes this exact shape to the CPU
      # fallback -- unreachable from any dplyr-facing call, kept only as a
      # safety net.
      stop("Invalid aggregation expression: ", expr_text,
           "\nExpected format: function(column), e.g., mean(mpg)",
           call. = FALSE)
    }

    func_name <- decomposed$fn_name
    col_name <- if (is.symbol(decomposed$arg)) {
      as.character(decomposed$arg)
    } else {
      paste(deparse(decomposed$arg), collapse = "")
    }

    # Validate function
    if (!func_name %in% agg_functions) {
      # Defensive, internal-only path (Phase 11, task L1): same as above --
      # summarise_dots_parseable()'s `known_agg_fns` check already routes an
      # unknown reducer (e.g. cor()/weighted.mean()) to the CPU fallback
      # before this is ever reached.
      stop("Unsupported aggregation function: ", func_name,
           "\nSupported functions: ", paste(agg_functions, collapse = ", "),
           call. = FALSE)
    }

    # Validate column
    col_idx <- match(col_name, schema$names)
    if (is.na(col_idx)) {
      stop("Column '", col_name, "' not found.",
           "\nAvailable columns: ", paste(schema$names, collapse = ", "),
           call. = FALSE)
    }

    input_type <- schema$types[col_idx]

    # unname(): schema$types is a NAMED vector (named by column), and
    # identical() treats a "names" attribute as significant (identical(c(x =
    # "BOOL8"), "BOOL8") is FALSE even though the values are equal) -- same
    # landmine, same fix, as ir_check_type_operands()'s and na_if()'s
    # (R/ir.R) unname() calls.
    input_type_bare <- unname(input_type)

    if (func_name %in% numeric_only_fns && identical(input_type_bare, "STRING")) {
      stop(func_name, "() requires a numeric column; '", col_name,
           "' is STRING.", call. = FALSE)
    }
    if (func_name %in% logical_only_fns && !identical(input_type_bare, "BOOL8")) {
      stop(func_name, "() requires a logical column; '", col_name,
           "' is ", input_type_bare, ".", call. = FALSE)
    }

    cudf_fn <- unname(agg_fn_map[func_name])

    aggregations[[i]] <- make_aggregation(names(dots)[i], col_name, cudf_fn, input_type,
                                          na_rm = isTRUE(na_rm_flags[i]),
                                          extra = agg_extra[i])
  }

  aggregations
}

# Internal: TRUE iff some dot of a summarise() mentions cur_group_id().
# @keywords internal
summarise_has_cur_group_id <- function(dots) {
  any(vapply(
    dots, function(q) expr_contains_zero_arg_call(rlang::quo_get_expr(q), "cur_group_id"),
    logical(1)
  ))
}

# Internal: split the dots of a summarise() with cur_group_id() into the dots
# that are exactly `cur_group_id()` and the rest. `NULL` when the call is not
# of that shape (the whole call then goes to dplyr): an unnamed or clashing
# id dot, an id mentioned inside a longer expression, a later dot reading an
# id name, an unnamed dot beside the id, or an ungrouped call with nothing else
# to summarise.
# @return `NULL`, or `list(id = <indices of the id dots>, rest = <dots>)`
# @keywords internal
summarise_group_id_split <- function(.data, dots, by_cols, by_given) {
  nms <- names(dots) %||% rep("", length(dots))
  is_id <- vapply(dots, function(q) {
    e <- rlang::quo_get_expr(q)
    is.call(e) && length(e) == 1L && identical(call_head_name(e), "cur_group_id")
  }, logical(1))
  # cur_group_id() anywhere else in a dot
  mentions <- vapply(dots, function(q) {
    expr_contains_zero_arg_call(rlang::quo_get_expr(q), "cur_group_id")
  }, logical(1))
  if (any(mentions & !is_id)) return(NULL)
  id_names <- nms[is_id]
  if (any(!nzchar(id_names)) || anyDuplicated(nms[nzchar(nms)]) > 0L) return(NULL)
  # An unnamed other dot (`mean(x)`, `across()`) takes its name from dplyr's
  # naming rules, which the final select() by name can't reproduce
  if (any(!nzchar(nms[!is_id]))) return(NULL)
  keys <- if (by_given) by_cols else .data$groups
  if (any(id_names %in% c(keys, current_schema(.data)$names))) return(NULL)
  # A dot reading an id name sees the id (sequential evaluation)
  rest <- dots[!is_id]
  reads <- any(vapply(rest, function(q) {
    any(all.vars(rlang::quo_get_expr(q)) %in% id_names)
  }, logical(1)))
  if (reads) return(NULL)
  if (length(rest) == 0L && length(keys) == 0L) return(NULL)
  list(is_id = is_id, rest = rest, keys = keys)
}

# Internal: run a summarise() whose id dots are `cur_group_id()` on the
# remaining dots, then append the row number under each id name. `NULL` when
# the reduced call would reach the CPU (a "cuplyr_cpu_fallback" signal from
# gpu_fallback()), so the caller sends the whole original call there instead:
# a fallback result may have several rows per group.
# @keywords internal
summarise_with_group_id <- function(.data, dots, split, .groups, by_quo, by_cols, by_given,
                                    caller_env) {
  res <- tryCatch(
    summarise_impl(.data, split$rest, .groups, by_quo, by_cols, by_given, caller_env),
    cuplyr_cpu_fallback = function(cnd) NULL
  )
  if (is.null(res) || !is_tbl_gpu(res)) return(NULL)

  nms <- names(dots)
  id_names <- nms[split$is_id]
  out_groups <- res$groups
  out_drop <- res$group_drop
  # Number the rows ungrouped: row_number() over the whole result
  flat <- set_tbl_gpu_field(res, "groups", character())
  flat <- set_tbl_gpu_field(flat, "fixed_keys", NULL)
  id_exprs <- stats::setNames(
    rep(list(quote(dplyr::row_number())), length(id_names)), id_names
  )
  flat <- rlang::inject(dplyr::mutate(flat, !!!id_exprs))
  flat <- dplyr::select(flat, dplyr::all_of(c(split$keys, nms)))
  flat <- set_tbl_gpu_field(flat, "groups", out_groups)
  set_tbl_gpu_field(flat, "group_drop", out_drop)
}
