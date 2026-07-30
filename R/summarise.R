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
#'     \item `"rowwise"` -- NOT SUPPORTED on `tbl_gpu` (hard error): a
#'       `tbl_gpu` has no rowwise representation (`rowwise()` itself is a
#'       full CPU-fallback verb, see [dplyr::rowwise()]). `collect()` first,
#'       then call real `dplyr::rowwise()` on the plain data frame.
#'   }
#'   `.groups=` is only validated against an already-grouped `.data` --
#'   matching a genuine (if surprising) dplyr quirk verified empirically:
#'   any non-`"rowwise"` value (including a typo) is silently a no-op on an
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
#'   \item `sum(x)` - Sum of values
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
#' `NaN`, `min()` -> `Inf`, `max()` -> `-Inf` (real dplyr also surfaces a
#' base-R warning for `min()`/`max()` here -- cuplyr does not replicate that
#' warning, only the value), `sd()`/`var()`/`median()` -> `NA` (their
#' natural result already, no special-casing needed). One documented
#' boundary: `min()`/`max()`'s `Inf`/`-Inf` identity can only be
#' represented in a `FLOAT64` column; over an INTEGER column (where real
#' dplyr promotes the WHOLE result column to double the instant any group
#' needs this identity -- a data-dependent type decision this package's
#' parse-time schema inference can't make), the all-`NA`-group result stays
#' `NA` rather than promoting.
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
#' ## `.by` group order
#' Verified empirically against dplyr 1.2.1: `summarise(.by=)` emits one row
#' per distinct `.by` combination in the order that combination FIRST
#' appears in `.data` -- e.g. `summarise(tibble(g = c("b","a","b")), n =
#' n(), .by = g)` returns `g` in the order `c("b", "a")`, not `c("a", "b")`.
#' This differs from `group_by(g) |> summarise(...)`, whose group order is
#' cuDF's own (unspecified, hash-groupby-derived) order. Implemented via a
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

  by_quo <- rlang::enquo(.by)

  # Phase 5, task W9: `.by=` resolution -- checked up front (before the
  # zero-dots error just below), matching mutate()/filter()'s own `.by=`
  # resolution order: resolve_by() (R/execute.R) errors immediately if
  # `.data` is already grouped, and this must fire even when `...` is empty
  # (verified empirically: `summarise(group_by(df, g), .by = g)` raises the
  # grouped-`.data` error, it does NOT reach summarise()'s own "requires at
  # least one aggregation expression" check first).
  by_cols <- resolve_by(.data, by_quo, ".by")
  by_given <- !rlang::quo_is_null(by_quo)

  dots <- rlang::enquos(...)

  if (length(dots) == 0) {
    stop("summarise() requires at least one aggregation expression.",
         call. = FALSE)
  }

  # Phase 10: across()/if_any()/if_all()/pick()/cur_*() -- see R/across.R's
  # own module docs. Must run on the ORIGINAL (not yet auto_name_dots()'d)
  # dots: a top-level `across(...)` dot is unnamed by construction, and
  # auto_name_dots() would otherwise stamp it with its own deparsed text as
  # a "name", which expand_across_dots() would then (wrongly) treat as a
  # user-supplied name. `cur_group_id()` is treated as a deferred shape
  # here (unlike mutate()/filter()): summarise()'s per-GROUP (not per-row)
  # shape means the row-level join-based substitution those two verbs use
  # doesn't produce a valid `fn(col)` aggregation dot on its own (see
  # substitute_cur_group_id()'s own docs) -- routed to the whole-call CPU
  # fallback instead, which real dplyr already handles correctly.
  if (dots_need_fallback(dots, treat_cur_group_id_as_deferred = TRUE)) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(tbl, !!!dots, .groups = .groups, .by = !!by_quo))
    }))
  }

  summarise_groups <- if (by_given) by_cols else .data$groups
  dots <- expand_across_dots(dots, current_schema(.data), summarise_groups, "summarise")
  dots <- auto_name_dots(dots, "summarise")

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
  if (!summarise_dots_parseable(dots, current_schema(.data))) {
    return(gpu_fallback("summarise", .data, function(tbl) {
      rlang::inject(dplyr::summarise(tbl, !!!dots, .groups = .groups, .by = !!by_quo))
    }))
  }

  if (by_given) {
    # `.by=` summarise is always ungrouped (Phase 5, task W9); `.groups=`
    # can never reach here alongside `.by=` (dplyr's own generic already
    # rejects that combination before this method is even dispatched to).
    return(summarise_by_desugar(.data, dots, by_cols))
  }

  retained_groups <- resolve_summarise_groups(.data$groups, .groups, caller_env)

  summarise_core(.data, dots, retained_groups)
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
# `.groups = "rowwise"` is a HARD ERROR here regardless of grouped/ungrouped
# (cuplyr's own scope decision, not a literal dplyr-divergence in observable
# output -- just an unsupported input): a `tbl_gpu` has no rowwise
# representation at all (`rowwise()` itself is a full CPU-fallback verb,
# R/fallback.R), so faking it via a regular grouped `tbl_gpu` would silently
# misrepresent the result's actual semantics for later verbs (rowwise
# mutate/summarise treat each row as its own group, which a plain grouped
# `tbl_gpu` does not). Users needing a real rowwise result should
# `collect()` first, then call real `dplyr::rowwise()` on the plain data
# frame.
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
  if (identical(groups_arg, "rowwise")) {
    stop(
      "summarise(.groups = \"rowwise\") is not supported on tbl_gpu: a ",
      "tbl_gpu has no rowwise representation. collect() the result first, ",
      "then call dplyr::rowwise() on the plain data frame instead.",
      call. = FALSE
    )
  }

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
# errors -- i.e. "this SHAPE isn't understood". A dot that's a genuine
# user mistake FLOWS THROUGH unchanged (not caught here) and still errors
# normally once it reaches na.rm=/na_rm= validation, column-not-found
# checks, or ir_parse_quo() throwing (e.g. a missing column inside the
# aggregation argument) -- matching the "NULL-vs-throw trigger rule".
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
    quo <- extract_na_rm(dots[[i]])$quo
    quo <- extract_agg_shape_param(quo)$quo

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

    arg_expr <- decomposed$arg
    if (is.symbol(arg_expr) && as.character(arg_expr) %in% schema$names) {
      next  # bare column reference: always valid, no temp column needed
    }

    arg_quo <- rlang::new_quosure(arg_expr, env)
    ir <- ir_parse_quo(arg_quo, schema)  # may throw a genuine error -- propagates

    if (!is.null(ir) && ir_has_window(ir)) {
      next  # windows-inside-summarise: always valid (Phase 6, task 4)
    }
    if (is.null(ir)) {
      return(FALSE)  # sub-expression shape not understood at all
    }
    # ir_is_const(ir) (e.g. sum(5)): left alone here, exactly like
    # preprocess_agg_expressions() -- not a "shape not understood" case,
    # so not a fallback trigger; parse_aggregations() raises its own
    # diagnosable error for this rare/degenerate shape.
  }

  TRUE
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
summarise_core <- function(.data, dots, retained_groups = character()) {
  # Phase 6, task 6.1 (Fix A): extract and strip any `na.rm = <literal>`
  # named argument BEFORE temp-column preprocessing -- this must happen
  # first so a sub-expression call like `sum(sqrt(x), na.rm = TRUE)` is
  # reduced to the `fn(single_arg)` shape preprocess_agg_expressions() (and
  # parse_aggregations()'s own regex) already understand, with na.rm
  # tracked out-of-band per dot instead of embedded in the call shape.
  # Extended (Phase 6, task 6.2) to also recognize `na_rm = <literal>`
  # (first()/last()/nth()'s own spelling) -- see extract_na_rm()'s docs.
  extracted <- lapply(dots, extract_na_rm)
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

  push_op(working_data, ast_summarise(input_node(working_data), aggregations,
                                      working_data$groups, retained_groups))
}

# Internal: Extract and strip an `na.rm = <literal>` (or `na_rm = <literal>`)
# named argument from a single summarise() aggregation dot (Phase 6, task
# 6.1, Fix A; extended Phase 6, task 6.2 for the underscore spelling)
#
# Real dplyr's `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()`/`median()`/
# `quantile()`/`n_distinct()`/`any()`/`all()` all spell this control
# `na.rm` (dot); `first()`/`last()`/`nth()` spell it `na_rm` (underscore --
# confirmed via `args(dplyr::first)` against dplyr 1.2.1: `function (x,
# order_by = NULL, default = NULL, na_rm = FALSE)`). Both defaults to
# `FALSE`, matching R's own `na.rm = FALSE` default: any NA in a group
# makes the WHOLE group's aggregation result NA for the mean/sum/.../
# median family (verified empirically against dplyr 1.2.1). cudf's own
# groupby aggregations always exclude nulls, so this default requires an
# extra whole-group-null propagation step in `gpu_summarise()`
# (`src/ops_groupby.cpp`) for that family -- `na_rm = FALSE` (the R-side
# default emitted when no `na.rm=`/`na_rm=` is present) requests that
# step; `na_rm = TRUE` opts back into cudf's native (NA-excluding)
# behavior directly, no extra step. `n_distinct()`/`first()`/`last()`/
# `nth()` interpret the SAME flag differently (selecting cudf's own
# `null_policy` directly rather than a post-hoc propagation step -- see
# `make_aggregation()`'s own docs), and `any()`/`all()` implement R's
# three-valued logic with it -- but the R-side extraction (which spelling
# is present, its value) is identical for every one of these functions,
# so one shared implementation still covers all of them.
#
# Mirrors `ir_parse_sum_call()`'s (R/ir.R) validation style: the value is
# resolved via `eval_tidy()` in the dot's own environment (so a bound
# variable like `na.rm = drop_na` works, not just a literal `TRUE`/`FALSE`
# token), but unlike `ir_parse_sum_call()` (which returns `NULL` to signal
# "try a different shape" to its caller, since mutate()/filter() have
# fallback paths), an unevaluable or non-logical value -- or BOTH spellings
# present at once -- is a hard error here -- summarise()'s aggregation
# parsing has no CPU fallback (see this file's own module docs).
#
# @param quo A quosure, one summarise() dot (post `auto_name_dots()`)
# @return `list(quo = <possibly rewritten quosure, na.rm/na_rm stripped>,
#   na_rm = logical(1))`
# @keywords internal
extract_na_rm <- function(quo) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)

  if (!is.call(expr)) {
    return(list(quo = quo, na_rm = FALSE))
  }

  arg_names <- names(expr)
  if (is.null(arg_names)) {
    return(list(quo = quo, na_rm = FALSE))
  }

  na_rm_pos <- which(arg_names == "na.rm" | arg_names == "na_rm")
  if (length(na_rm_pos) == 0) {
    return(list(quo = quo, na_rm = FALSE))
  }
  if (length(na_rm_pos) > 1) {
    stop("Multiple na.rm=/na_rm= arguments in aggregation expression: ",
         rlang::quo_text(quo), call. = FALSE)
  }

  na_rm_val <- tryCatch(
    rlang::eval_tidy(rlang::new_quosure(expr[[na_rm_pos]], env)),
    error = function(e) NULL
  )
  if (!is.logical(na_rm_val) || length(na_rm_val) != 1 || is.na(na_rm_val)) {
    stop("na.rm=/na_rm= must evaluate to a single TRUE or FALSE in aggregation expression: ",
         rlang::quo_text(quo), call. = FALSE)
  }

  stripped_expr <- expr[-na_rm_pos]
  list(quo = rlang::new_quosure(stripped_expr, env), na_rm = isTRUE(na_rm_val))
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
  if (is.null(present_args$x) || is.null(present_args$n)) {
    stop("nth() requires `x` and `n` arguments: ", rlang::quo_text(quo), call. = FALSE)
  }

  n_val <- tryCatch(
    rlang::eval_tidy(rlang::new_quosure(present_args$n, env)),
    error = function(e) NULL
  )
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

  result <- summarise_core(grouped, dots)
  result <- dplyr::arrange(result, !!rlang::sym(first_name))
  dplyr::select(result, -dplyr::all_of(first_name))
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

    # Phase 6, task 4: windows-inside-summarise (`sum(cumsum(v))`,
    # `last(cumsum(v))`, ...) -- checked BEFORE the ir_is_const() check below,
    # since a window call with no column argument at all (e.g. bare
    # `row_number()`) has no ir_cols() of its own and would otherwise be
    # (wrongly) treated as a constant subtree. See resolve_agg_window_arg()'s
    # own docs for the full derivation and scope boundary.
    if (!is.null(ir) && ir_has_window(ir)) {
      if (has_pending_ops(working_data)) {
        working_data <- compute(working_data)
        cuplyr_fallback_notify("summarise", rlang::quo_text(quo))
      }

      resolved <- resolve_agg_window_arg(working_data, ir, decomposed$fn_name, i)
      working_data <- resolved$data

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
      cuplyr_fallback_notify("summarise", rlang::quo_text(quo))
    }

    # Create a temporary column name (index-based, so it stays unique even
    # though not every dot ends up needing one).
    temp_col_name <- paste0(".temp_agg_", i)

    # any()/all() (Phase 6, task 6.2): keep a BOOL8-inferred sub-expression
    # (e.g. `any(x > 3)`) as BOOL8 rather than the sum()-oriented INT32
    # promotion every other aggregation wants -- see create_temp_column()'s
    # own `keep_bool` docs.
    keep_bool <- decomposed$fn_name %in% c("any", "all")
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
  plan <- plan_window_stages(exprs, current_schema(working_data), working_data$groups)

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

  new_ptr <- wrap_gpu_call(
    "summarise_mutate_expr",
    gpu_mutate_expr(.data$ptr, list(bound_ir), col_name, new_type, schema$names)
  )

  new_tbl_gpu(
    ptr = new_ptr,
    schema = list(
      names = c(schema$names, col_name),
      types = c(schema$types, new_type)
    ),
    lazy_ops = .data$lazy_ops,
    groups = .data$groups,
    exec_mode = .data$exec_mode
  )
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
    match_text <- strip_ns_prefix(expr_text)

    # Check for n() which has no column argument
    if (grepl("^n\\(\\)$", match_text)) {
      aggregations[[i]] <- make_aggregation(names(dots)[i], NA_character_, "n")
      next
    }

    # Parse function(column) pattern
    match_result <- regmatches(
      match_text,
      regexec("^([a-zA-Z_][a-zA-Z0-9_]*)\\(([^)]+)\\)$", match_text)
    )[[1]]

    if (length(match_result) != 3) {
      # Defensive, internal-only path (Phase 11, task L1): every real
      # summarise() call is screened by summarise_dots_parseable() before
      # reaching here, which already routes this exact shape to the CPU
      # fallback -- unreachable from any dplyr-facing call, kept only as a
      # safety net.
      stop("Invalid aggregation expression: ", expr_text,
           "\nExpected format: function(column), e.g., mean(mpg)",
           call. = FALSE)
    }

    func_name <- match_result[2]
    col_name <- trimws(match_result[3])

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
    # landmine, same fix, as check_filter_comparison_types()'s (R/filter.R)
    # and na_if()'s (R/ir.R) existing unname() calls.
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
