#' Filter rows of a GPU table
#'
#' Selects rows from a GPU table where conditions are TRUE, similar to
#' `dplyr::filter()`. Filtering is performed entirely on the GPU for
#' maximum performance on large datasets, and matches dplyr's row-keeping
#' and three-valued-logic (`NA`) semantics exactly (see "NA semantics"
#' below).
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Logical expressions to filter by (see "Supported expressions"
#'   below for the full surface). Multiple expressions, whether passed as
#'   separate dots or combined with `&` within one dot, are always combined
#'   with AND (a row is kept only if every expression is TRUE for it).
#' @param .by Optional on-the-fly grouping columns (tidyselect), GPU-native
#'   (Phase 5, task W6): `filter(df, x > mean(x), .by = g)` behaves
#'   identically to `df |> group_by(g) |> filter(x > mean(x)) |> ungroup()`
#'   -- same window/aggregate-predicate machinery as `group_by()`-driven
#'   filtering, and the result is always ungrouped. Supplying `.by` when
#'   `.data` is already grouped (via `group_by()`) is an error, matching
#'   dplyr exactly.
#' @param .preserve Whether to keep the groups the filter emptied. With `TRUE`,
#'   a `group_by()`-grouped result keeps the input's group keys (empty groups
#'   included) for `group_data()`, `collect()` and `summarise()`, like dplyr
#'   (see [group_by.tbl_gpu()] for what carries them on). `FALSE` recomputes
#'   the groups from the remaining rows.
#'
#' @return A `tbl_gpu` object containing only rows where all conditions are TRUE.
#'   The GPU memory for the filtered result is newly allocated.
#'
#' @details
#' ## Supported expressions
#' `filter()` parses its arguments through an internal expression IR (see
#' `R/ir.R`) that lowers every predicate in the call to a **single fused GPU
#' kernel** (`cudf::compute_column()` under the hood) -- one kernel per
#' `filter()` call, no matter how many dots or how deeply the predicate is
#' nested. The IR understands:
#' \itemize{
#'   \item Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=` -- against a
#'     literal, another column, an environment variable, or an arithmetic
#'     expression (`(x - y) > 0`)
#'   \item Logic: `&`, `|`, `!`, `xor()`, arbitrarily nested
#'     (`x > 2 & (y < 5 | z == 1)`); `&&`/`||` are rejected with a message
#'     pointing at `&`/`|` (matching dplyr 1.1+)
#'   \item `is.na()`
#'   \item `%in%`, including a set containing `NA` (R's never-NA rule: a
#'     value that itself is `NA` is TRUE for `%in%` iff the set contains
#'     `NA`, never `NA` itself) and an empty set (`character(0)`, `c()`,
#'     etc. -- always FALSE, for every row, including `NA` rows)
#'   \item `between(x, lo, hi)` (dplyr's definition: `x >= lo & x <= hi`,
#'     including its `NA` propagation), `near(x, y, tol = )`
#'     (`abs(x - y) < tol`, matching `dplyr::near()` exactly)
#'   \item String-column comparisons (`==`/`!=`/`<`/`<=`/`>`/`>=`), against
#'     a literal or another string column
#'   \item Column-vs-column comparisons for every operator above
#'   \item Math on columns inside a predicate (`+ - * / %% %/% ^`, `sqrt`,
#'     `log`, `abs`, etc. -- the same set `mutate()` supports)
#'   \item `if_else()`, `case_when()`, `coalesce()`, `na_if()`,
#'     `case_match()`, `recode()` (scalar literal shapes), `recode_values()`, `replace_values()`,
#'     `replace_when()` -- the same conditional/vector functions `mutate()`
#'     supports (see its own docs for the exact type-unification rules),
#'     usable anywhere inside a predicate, including as the whole predicate
#'     itself (e.g. `filter(case_when(x > 1 ~ TRUE, .default = FALSE))`)
#'   \item `when_any(...)`/`when_all(...)` -- particularly useful inside
#'     `filter()` to combine several comma-separated conditions with `|`
#'     instead of `&` (e.g. `filter(when_any(x > 10, y < 0))`)
#'   \item The `.data`/`.env` pronouns, and ordinary R symbols (a bare name
#'     that matches a column resolves to that column; otherwise it's
#'     evaluated as an environment variable -- "columns shadow the
#'     environment")
#'   \item Window/aggregate functions (Phase 5, task W5), usable anywhere
#'     inside a predicate, including combined with plain column
#'     comparisons in the same dot (`filter(x > mean(x) & y < 5)`):
#'     `row_number()`/`row_number(x)`, `min_rank()`/`dense_rank()`/
#'     `percent_rank()`/`cume_dist()`, `lag()`/`lead()`,
#'     `cumsum()`/`cummax()`/`cummin()`/`cumprod()`/`cummean()`/
#'     `cumall()`/`cumany()`/`consecutive_id()`, `ntile()`,
#'     `mean()`/`sum()`/`min()`/`max()`/`n()`/`sd()`/`var()`/`median()`/
#'     `n_distinct()`/`first()`/`last()`/`nth()` -- the same set `mutate()`
#'     supports (see its own docs for exact semantics/argument support),
#'     evaluated per group when
#'     `.data` is grouped (via `group_by()` or the on-the-fly `.by=`
#'     argument, Phase 5 task W6 -- identical semantics either way). A
#'     grouped filter -- like dplyr's own -- preserves the ORIGINAL row
#'     order in the result, it does not reorder into group-key order.
#' }
#' Comparing a column against a literal of an incompatible type (e.g. a
#' numeric column against a character literal) errors immediately, naming
#' the column and its type, rather than silently falling back or producing
#' a raw GPU-side type error.
#'
#' ## Multi-dot semantics with window/aggregate predicates
#' Multiple dots (or multiple `&`-joined predicates within one dot) are
#' always combined with AND, and -- confirmed empirically against real
#' dplyr, since this is easy to get wrong -- every dot's predicate is
#' evaluated against the SAME, ORIGINAL (pre-`filter()`) data, never
#' against a progressively-narrowed intermediate result. Concretely,
#' `filter(x > 0, cumsum(x) < 10)` computes `cumsum(x)` over every row of
#' the original table, then ANDs that mask with `x > 0` -- it does NOT
#' filter to `x > 0` first and then compute `cumsum()` only over the
#' surviving rows (which would generally give a different, smaller
#' cumulative sum and therefore a different result). `cuplyr` reproduces
#' this exactly: every window-bearing dot's temp columns are computed
#' against the table as it stood when `filter()` was called, before any
#' dot's predicate (window-derived or plain) has removed a single row.
#'
#' ## NA semantics
#' `filter()` keeps a row iff its combined predicate is non-missing *and*
#' TRUE -- identical to dplyr (`cudf::apply_boolean_mask()`'s "non-null AND
#' true" rule matches `dplyr::filter()`'s own row-keeping rule exactly).
#' Concretely:
#' \itemize{
#'   \item `filter(x > 5)` drops rows where `x` is `NA` (the comparison is
#'     `NA`, and `NA` isn't kept), same as dplyr
#'   \item `&`/`|` use R's three-valued logic, not C's: `NA & FALSE` is
#'     `FALSE`, `NA | TRUE` is `TRUE`, `NA & TRUE`/`NA | FALSE`/`NA & NA`/
#'     `NA | NA` are all `NA` (`cudf::ast_operator::NULL_LOGICAL_AND`/
#'     `NULL_LOGICAL_OR`, chosen specifically because they match)
#'   \item `filter(x > 2 | is.na(x))` is the idiom for "keep NA rows too"
#'     -- works exactly as it does in dplyr
#'   \item `%in%` never produces `NA` (see "Supported expressions" above) --
#'     this is R's `%in%`/`match()` semantics, not null-propagating
#'   \item `filter(TRUE)` is a no-op (returns `.data` unchanged, no GPU
#'     work); `filter(FALSE)` and `filter(NA)` both return a zero-row
#'     result (a bare `NA` predicate is `FALSE`-equivalent for row-keeping,
#'     same as dplyr)
#' }
#' One documented, deliberate divergence from dplyr: cuDF's floating-point
#' `NaN` is not the same value as SQL/cuDF null, so `is.na()` on a *computed*
#' `NaN` (e.g. `0/0`) returns `FALSE` on the GPU where R's `is.na(NaN)` is
#' `TRUE`. `NA` values ingested from R (including in a `FLOAT64` column)
#' are unaffected -- they round-trip as genuine GPU nulls and behave
#' identically to dplyr in every case above.
#'
#' ## Not yet supported
#' \itemize{
#'   \item Arbitrary R functions with no cuDF equivalent -- these fall back
#'     to CPU evaluation, see below
#' }
#'
#' ## Expressions the IR doesn't understand (CPU fallback)
#' An expression shape the IR doesn't recognize (see `R/ir.R`'s
#' `ir_parse_quo()`) is handed to a CPU-side fallback: it's evaluated with
#' `rlang::eval_tidy()` and, if the result is a logical scalar or vector,
#' applied back to the GPU table as a boolean mask (materializing first,
#' via `compute()`, if `.data` is lazy with pending operations). Because
#' this evaluation happens with *no data mask*, it can only succeed for
#' expressions that evaluate to a logical value on their own without
#' referencing table columns -- e.g. a pre-computed logical vector or
#' `filter(rep(TRUE, n))`. An expression that references a table column
#' but doesn't parse (a rare shape given how much of dplyr's filter surface
#' the IR now covers) will still error with the "only supports comparisons"
#' message, since the column symbol can't resolve outside the table. This
#' fallback path is legitimate, supported behavior for the cases it does
#' cover, not an error condition -- see `getOption("cuplyr.fallback")`
#' below to be notified when it's used.
#'
#' ## `options(cuplyr.fallback = ...)`
#' Controls whether the CPU fallback described above (used by `filter()`
#' and by `summarise()`'s temp-column preprocessing) notifies the caller.
#' One of:
#' \itemize{
#'   \item `"silent"` (default) - no notification
#'   \item `"warn"` - emits a `warning()` naming the verb and expression
#'   \item `"error"` - `stop()`s instead of falling back
#' }
#'
#' ## Performance
#' Filtering on GPU is highly parallel and can process billions of rows
#' per second. For best performance, prefer expressions the IR parses
#' directly into a GPU predicate (the entire "Supported expressions" list
#' above) over expressions that require the CPU fallback.
#'
#' @seealso
#' \code{\link{mutate.tbl_gpu}} for creating new columns,
#' \code{\link{select.tbl_gpu}} for selecting columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr filter
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Filter with single condition
#'   efficient_cars <- gpu_mtcars |>
#'     filter(mpg > 25)
#'
#'   # Multiple conditions (combined with AND)
#'   result <- gpu_mtcars |>
#'     filter(mpg > 20) |>
#'     filter(cyl == 4) |>
#'     collect()
#'
#'   # Compare two columns
#'   gpu_cars <- tbl_gpu(cars)
#'   fast_stops <- gpu_cars |>
#'     filter(dist < speed) |>
#'     collect()
#' }
filter.tbl_gpu <- function(.data, ..., .by = NULL, .preserve = FALSE) {
  filter_impl(.data, rlang::enquos(...), rlang::enquo(.by), .preserve,
              complement = FALSE, call = rlang::current_env())
}

#' Drop rows of a GPU table where conditions are TRUE
#'
#' The complement of [filter.tbl_gpu()], like `dplyr::filter_out()`: all
#' conditions are combined with AND first, then the rows where that
#' combination is `TRUE` are dropped. Rows where it is `FALSE` or `NA` are
#' kept (so this is not `filter(!cond)`). With no conditions every row is
#' dropped. It runs on the same engine as `filter()` (same expression
#' surface, window predicates, `.by`, CPU fallbacks, eager and lazy): the
#' predicate pushed to the GPU is `is.na(P) | !P` for the combined `P`.
#'
#' @inheritParams filter.tbl_gpu
#' @param ... Logical expressions; rows where all of them are `TRUE` are
#'   dropped.
#' @return A `tbl_gpu` object without the rows where all conditions are TRUE.
#'
#' @export
#' @importFrom dplyr filter_out
filter_out.tbl_gpu <- function(.data, ..., .by = NULL, .preserve = FALSE) {
  filter_impl(.data, rlang::enquos(...), rlang::enquo(.by), .preserve,
              complement = TRUE, call = rlang::current_env())
}

# Internal: the engine behind filter() and filter_out(). `complement = TRUE`
# is filter_out(): every dot's predicate is ANDed into ONE expression P and the
# single pushed predicate is `is.na(P) | !P`.
#
# `.preserve = TRUE` on a group_by() table fixes the input's group keys on the
# result (as slice() does, see fixed_keys_of()); every other exit drops them.
# @keywords internal
filter_impl <- function(.data, dots, by_quo, .preserve, complement, call) {
  check_unique_input_names(.data, call)
  keys <- NULL
  # dplyr keeps a `.drop = FALSE` table's whole group structure whatever
  # `.preserve` says, so those tables take the same route.
  keep_groups <- isTRUE(.preserve) || !isTRUE(.data$group_drop %||% TRUE)
  if (keep_groups && rlang::quo_is_null(by_quo) && length(.data$groups) > 0L) {
    # taken now, from the computed input, so a lazy plan runs once
    if (has_pending_ops(.data)) .data <- compute(.data)
    keys <- slice_group_keys(.data)
  }
  groups <- .data$groups
  out <- filter_run(.data, dots, by_quo, .preserve, complement, call)
  if (is_tbl_gpu(out)) {
    if (!is.null(keys) && identical(out$groups, groups)) {
      out$fixed_keys <- keys
    } else if (!is.null(out[["fixed_keys"]]) && (is.null(keys) || !identical(out$groups, groups))) {
      out <- drop_fixed_keys(out)
    }
  }
  out
}

# A `tbl_gpu` without its fixed group keys.
# @keywords internal
drop_fixed_keys <- function(x) {
  cls <- class(x)
  x <- unclass(x)
  x$fixed_keys <- NULL
  structure(x, class = cls)
}

# filter_impl()'s body. Every exit may hand back its input unchanged; the
# caller sets or drops the fixed keys.
# @keywords internal
filter_run <- function(.data, dots, by_quo, .preserve, complement, call) {
  # dplyr's own checks and per-group semantics are reached by running the
  # real verb on the CPU: named inputs (a named dot is only fine when its
  # expression is a logical literal), `.by` together with `.preserve`, and a
  # table with no columns.
  cpu <- function() filter_cpu_fallback(.data, dots, by_quo, .preserve, complement)
  nms <- names(dots)
  if (!is.null(nms) && any(nzchar(nms) & !vapply(dots, function(q) is.logical(rlang::quo_get_expr(q)), logical(1)))) {
    return(cpu())
  }

  # Phase 5, task W6: `.by=` resolution -- checked up front (before the
  # zero-dots early return just below), matching dplyr's own behavior of
  # raising the grouped-`.data` error even for a zero-dot filter() call
  # (verified empirically). See resolve_by()'s own docs (R/execute.R) for
  # the exact error text and tidyselect semantics. `verb_groups` is `by_cols`
  # when `.by=` was supplied, `.data$groups` otherwise -- identical to
  # mutate.tbl_gpu()'s own resolution -- and threads straight into the
  # window-predicate planning below exactly like a `group_by()` column
  # would; the result is always ungrouped when `.by=` was used, which falls
  # out automatically since resolve_by() already guarantees `.data` itself
  # is ungrouped in that case.
  by_cols <- resolve_by(.data, by_quo, ".by", call = call)
  verb_groups <- if (!rlang::quo_is_null(by_quo)) by_cols else .data$groups

  if (!rlang::quo_is_null(by_quo) && isTRUE(.preserve)) return(cpu())
  if (complement && length(current_schema(.data)$names) == 0L) return(cpu())

  if (length(dots) == 0 && !complement) return(.data)

  # Phase 10: across()/if_any()/if_all()/pick()/cur_*() -- see R/across.R's
  # own module docs. Checked up front (before `.data` is touched at all),
  # same discipline as the window-predicate path below: anything genuinely
  # unsupported (pick()-as-data-frame, cur_group()/cur_group_rows()/
  # cur_data()/cur_data_all()/c_across(), a nested (non-top-level)
  # across()/if_any()/if_all(), a top-level `across()` dot at all -- real
  # dplyr itself rejects that shape in filter(), see
  # expand_across_dots()'s own docs) routes the WHOLE filter() call
  # through the CPU fallback. `cur_group_id()` gets its own GPU-native
  # substitution below instead (row-level, exactly like mutate()'s).
  # cur_group()$key and length(cur_group_rows()) are plain expressions; the
  # CPU fallbacks keep the dots as the user wrote them.
  user_dots <- dots
  dots <- rewrite_context_dots(dots, verb_groups)
  dots <- rewrite_top_n_rank_dots(dots, .data)
  if (dots_need_fallback(dots)) {
    return(filter_across_fallback(.data, user_dots, by_quo, .preserve, complement))
  }

  # Substitution runs before expansion here so the if_any()/if_all() piece
  # check can type cur_group_id() pieces; the helper column is excluded from
  # `.cols`'s selectable set alongside the group columns.
  fallback_data <- .data
  fallback_dots <- user_dots
  cgid <- substitute_cur_group_id(.data, dots, verb_groups, by = !rlang::quo_is_null(by_quo))
  .data <- cgid$data
  dots <- expand_across_dots(cgid$dots, current_schema(.data),
                             c(verb_groups, cgid$helper_name), "filter")
  # An unsupported `.fns` shape: dplyr runs the call on the table and dots as
  # the user wrote them (no cur_group_id() helper column)
  if (isTRUE(attr(dots, "needs_fallback"))) {
    return(filter_across_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement))
  }

  # Schema is stable across this whole call for every PLAIN dot: filter()
  # never adds/removes real columns. The one exception is Phase 5's
  # window-predicate handling directly below: when at least one dot's IR
  # contains a window-function call, the schema is temporarily extended
  # with `..win*..`/`..winarg*..` temp columns for the remainder of this
  # call; a trailing ast_select drops them again before returning, so the
  # table's real, user-visible schema is unaffected either way. Phase 10's
  # cur_group_id() helper column (if `cgid$used`) is dropped the SAME way:
  # `real_orig_names` (captured before that helper was ever added) is used
  # for the trailing select instead of `orig_names` whenever either
  # mechanism added a temp column.
  orig_data <- .data
  schema <- current_schema(.data)
  orig_names <- schema$names
  real_orig_names <- if (isTRUE(cgid$used)) setdiff(orig_names, cgid$helper_name) else orig_names

  # First pass: parse every dot up front (no push_op()/GPU work yet) so we
  # know, before touching `.data` at all, whether any dot needs the window
  # path below. Doing this classification BEFORE the per-dot loop's
  # ir_is_const() check (further down) matters, not just for staging: a
  # bare window call like `n()` has zero column ARGUMENTS of its own
  # (ir_cols() only walks `$args`), so `ir_is_const(ir_parse_quo(quo(n() >
  # 2)))` would itself read TRUE -- without routing window calls through
  # this dedicated path first, `filter(n() > 2)` fell into the
  # constant-expression branch instead (eval_tidy()-ing `n() > 2` with no
  # table in scope, which either errors or silently misbehaves) rather than
  # being planned as a real window predicate. This was the crash this task
  # fixes (see the regression test): checking ir_has_window() here, before
  # any const-folding logic ever runs, routes it correctly instead.
  parsed <- vector("list", length(dots))
  window_idx <- integer()

  for (i in seq_along(dots)) {
    ir <- parse_filter_ir(dots[[i]], schema)
    # `parsed[i] <- list(ir)`, NOT `parsed[[i]] <- ir`: the latter, when
    # `ir` is NULL (an opaque expression the IR doesn't understand -- a
    # perfectly ordinary, common case, see the opaque-dot branch below),
    # *deletes* that list element instead of storing NULL in it (R's
    # `x[[i]] <- NULL` list-element-removal special case), silently
    # shrinking `parsed` below `length(dots)` and desyncing every
    # subsequent index -- caught by the crash-regression/opaque-dot tests.
    parsed[i] <- list(ir)
    if (!is.null(ir) && ir_has_window(ir)) {
      window_idx <- c(window_idx, i)
    }
  }

  # Phase 11, task L1: universal expression-level CPU fallback. A dot that
  # `ir_parse_quo()` couldn't parse at all (`ir` is NULL -- as opposed to a
  # genuine thrown error, which already propagated out of the loop above)
  # AND that references a real schema column somewhere in its raw
  # expression tree (`dot_references_schema_col()`) can NEVER be handled by
  # `filter_eval_mask()`'s no-data-mask `eval_tidy()` -- that column symbol
  # either errors "object not found" or, worse, silently resolves to an
  # unrelated same-named object in the calling environment, giving a wrong
  # (not just missing) result. Route the WHOLE filter() call to the CPU
  # fallback instead, exactly like filter_window_fallback() already does
  # for an unlowerable window plan -- and before ANY push_op() has touched
  # `.data` (this check runs directly after the parse loop, ahead of the
  # window-handling block below), so "never half-lower" holds even for a
  # dot mixing a window call with an unknown function (e.g. `mean(x) > 0 &
  # toupper(y) == "A"` fails to parse as ONE unit, so it's never even added
  # to `window_idx` above -- it's caught here instead). A dot that's NULL
  # but references no schema column (a genuine constant/environment-vector
  # shape, e.g. `rep(TRUE, n)`) is untouched -- it still flows to
  # `filter_eval_mask()` below, unchanged from before L1.
  needs_expr_fallback <- vapply(seq_along(dots), function(i) {
    (is.null(parsed[[i]]) && dot_references_schema_col(dots[[i]], orig_names)) ||
      (!is.null(parsed[[i]]) && ir_factor_rule(parsed[[i]], schema) != "none") ||
      filter_ir_not_logical(parsed[[i]], schema, i %in% window_idx)
  }, logical(1))

  if (any(needs_expr_fallback)) {
    return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement))
  }

  # A dot with no column reference whose IR is NULL (opaque) or constant is
  # evaluated standalone in R (below) -- no data mask. If that errors, dplyr
  # may still succeed (the call can use pick()/n()/cur_*() inside a user
  # function, valid only in the mask), so hand the WHOLE call to dplyr,
  # decided here before any predicate is pushed. Each such dot is evaluated
  # exactly once: the value is cached for the loop below.
  standalone <- vector("list", length(dots))
  for (i in setdiff(seq_along(dots), window_idx)) {
    if (is.null(parsed[[i]]) || ir_is_const(parsed[[i]])) {
      res <- tryCatch(list(value = rlang::eval_tidy(dots[[i]], data = list())), error = function(e) NULL)
      if (is.null(res)) {
        return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement))
      }
      # Only a plain logical scalar is a constant predicate. A matrix/array
      # (dplyr's own error or deprecation warning), a vector on a grouped
      # table (recycled per group), a non-logical constant (dplyr's "must be a
      # logical vector" error), and anything at all under filter_out() are
      # dplyr's to evaluate.
      value <- res$value
      scalar_lgl <- is.logical(value) && length(value) == 1L && is.null(dim(value))
      if (!is.null(dim(value)) || (length(verb_groups) > 0L && length(value) != 1L) ||
          (complement && !scalar_lgl) || !is.logical(value)) {
        return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement))
      }
      standalone[i] <- list(res)
    }
  }

  # Phase 5, task W5: at least one dot's predicate contains a window call
  # (mean(x), n(), row_number(), cumsum(x), lag(x), ...). dplyr's multi-dot
  # semantics (verified empirically -- see this file's roxygen "Multi-dot
  # semantics" section): every dot's predicate is evaluated against the
  # SAME, ORIGINAL (pre-filter()) data, then ANDed together -- never
  # sequential/progressive narrowing, where a later dot's window aggregate
  # would "see" an earlier dot's already-filtered rows. We reproduce this by
  # planning and materializing EVERY window dot's temp columns up front,
  # against `.data` exactly as it stood at function entry -- before any
  # predicate (plain or window-derived) has removed a single row -- and
  # only afterward building/pushing ONE combined ast_filter node folding in
  # every dot's (possibly rewritten) predicate, in original dot order.
  if (length(window_idx) > 0) {
    window_exprs <- stats::setNames(
      parsed[window_idx],
      paste0("..filterwin", seq_along(window_idx), "..")
    )

    plan <- filter_plan_window(window_exprs, schema, verb_groups)

    if (is.null(plan)) {
      # The plan failed outright, or produced a window spec gpu_window()
      # can't lower (window_spec_lowerable(), R/lower.R -- unreachable for
      # any window kind currently in the registry, kept here defensively
      # for a future kind). "Never half-lower" (design doc section 1.3):
      # nothing has been pushed onto `.data` yet at this point, so falling
      # back to CPU evaluation of the WHOLE call, from the pristine
      # original table and dots, is safe.
      return(filter_window_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement))
    }

    # A non-logical predicate must raise dplyr's own "`..1` must be a logical
    # vector" error (filter_out()'s cudf NOT would even accept numerics).
    # Judge each window dot by its rewritten predicate's type, before any
    # stage is pushed, and let dplyr raise its error on the CPU.
    if (any(vapply(plan$post, filter_ir_not_logical, logical(1),
                                 schema = plan$schema, is_window = FALSE))) {
      return(filter_window_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement))
    }

    # F42: an integer window sum() is INT32, so arithmetic on it is checked as
    # in dplyr. dplyr makes only the overflowing groups' sums doubles; the
    # other groups' sums stay integers and arithmetic on them still overflows
    # to NA. An eager call therefore reruns on the CPU when any sum overflows
    # (exact, and overflow is rare). A lazy plan lets a top-level sum come back
    # FLOAT64 (inexact only when another group's in-range sum then meets
    # overflowing integer arithmetic), and a sum nested in another window call
    # (`max(x * sum(y))`), which a later stage reads as INT32, raises
    # `cuplyr_error_sum_overflow` at collect() (filter_plan_window()).
    eager <- identical(.data$exec_mode, "eager")
    push_stages <- function(d) {
      for (stage in plan$stages) {
        if (length(stage$pre) > 0) {
          d <- push_op(d, ast_mutate(input_node(d), stage$pre))
        }
        d <- push_op(d, ast_window(input_node(d), stage$specs, verb_groups,
                                   on_sum_overflow = if (eager) "error" else "double"))
      }
      d
    }
    if (eager) {
      staged <- retry_on_sum_overflow(function() list(data = push_stages(.data)),
                                      function() NULL)
      if (is.null(staged)) {
        return(filter_window_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement))
      }
      .data <- staged$data
    } else {
      .data <- push_stages(.data)
    }

    for (k in seq_along(window_idx)) {
      parsed[window_idx[k]] <- plan$post[k]
    }
    schema <- plan$schema
  }

  predicates <- list()

  flush_predicates <- function() {
    if (length(predicates) > 0) {
      .data <<- push_op(.data, ast_filter(input_node(.data), predicates))
      predicates <<- list()
    }
  }

  # filter_out(): only IR predicates are collected; they are ANDed into one P
  # and complemented once, after the loop. A constant FALSE/NA dot makes P
  # FALSE or NA, so every row is kept.
  comp_irs <- list()
  keep_all <- FALSE

  for (i in seq_along(dots)) {
    quo <- dots[[i]]
    ir <- parsed[[i]]

    if (complement) {
      if (is.null(ir) || ir_is_const(ir)) {
        if (!isTRUE(standalone[[i]]$value)) keep_all <- TRUE
      } else {
        comp_irs <- c(comp_irs, list(ir))
      }
      next
    }

    if (is.null(ir)) {
      # Opaque expression (the IR doesn't understand this shape): flush any
      # predicates accumulated so far into a real ast_filter node, then
      # fall back to CPU-side evaluation of this expression (D4).
      flush_predicates()
      .data <- filter_eval_mask(.data, quo, standalone[[i]]$value)
      next
    }

    if (ir_is_const(ir)) {
      # A constant expression (ir_cols() empty -- no column reference
      # anywhere in the subtree, e.g. TRUE, FALSE, `1 > 2`, `rep(TRUE, n)`):
      # section 3's design decision is that these never reach the IR/GPU
      # expression path. `compute_column()`/`gpu_filter_expr()` need a real
      # per-row column to broadcast an n-row mask against; a schema-less
      # constant subtree gives them nothing to anchor that broadcast to.
      # Evaluate in R instead (on the original quosure, not the already-
      # constant-folded IR) and route to the same no-op / impossible-
      # predicate / mask-apply paths the pre-IR parser used.
      value <- standalone[[i]]$value

      if (is.logical(value) && length(value) == 1 && isTRUE(value)) {
        next  # TRUE: genuine no-op, contributes nothing (D4/S4 semantics)
      }

      if (is.logical(value) && length(value) == 1) {
        # FALSE or NA: an impossible predicate. This is accumulated as a
        # normal predicate (preserving laziness exactly like any other
        # predicate, replacing the deleted "col != col" synthesis) rather
        # than executed eagerly: a bare FALSE literal never reaches
        # compute_column()'s AST-build path -- gpu_filter_expr()'s
        # materialize() has a dedicated `lit` branch that broadcasts a
        # literal scalar directly across the table's real row count, no
        # AST tree involved.
        predicates <- c(predicates, list(make_predicate(ir_lit_from_r(FALSE), schema)))
        next
      }

      # Anything else constant (a non-logical value, or a logical vector
      # such as `rep(TRUE, n)`): same CPU-eval fallback as an opaque
      # expression -- filter_eval_mask() has the exact error/mask-apply
      # logic for these shapes already.
      flush_predicates()
      .data <- filter_eval_mask(.data, quo, value)
      next
    }

    predicates <- c(predicates, list(make_predicate(ir, schema)))
  }

  flush_predicates()

  if (complement && !keep_all) {
    # keep iff !(P %in% TRUE) = is.na(P) | !P; no dot (or only TRUE dots)
    # leaves nothing to keep.
    p_ir <- Reduce(function(a, b) ir_call("&", list(a, b)), comp_irs)
    keep_ir <- if (is.null(p_ir)) {
      ir_lit_from_r(FALSE)
    } else {
      ir_call("|", list(ir_call("is.na", list(p_ir)), ir_call("!", list(p_ir))))
    }
    .data <- push_op(.data, ast_filter(input_node(.data), list(make_predicate(keep_ir, schema))))
  }

  if (length(window_idx) > 0 || isTRUE(cgid$used)) {
    # Drop every `..win*../..winarg*..` temp column introduced by the
    # window-planning block above, and/or Phase 10's `..cur_group_id..`
    # helper column (if used), restoring the table's real, user-visible
    # schema (section 1.3's "trailing select" step).
    .data <- push_op(.data, ast_select(input_node(.data), real_orig_names))
  }

  .data
}

# Internal: plan + validate the window stages for filter()'s window-bearing
# dots (Phase 5, task W5).
#
# A thin wrapper around plan_window_stages() (R/window.R) -- the exact same
# planning machinery mutate_window()/plan_mutate_window_batches() (R/
# mutate.R) drives -- that additionally validates every produced spec is
# lowerable (window_spec_lowerable(), R/lower.R), mirroring mutate_window()'s
# own two-pass "plan then validate, only emit if everything checks out"
# shape: `filter.tbl_gpu()` must never push a single AST node before it
# knows the WHOLE plan can be lowered (never half-lower).
#
# Unlike mutate()'s multi-dot chunking (window_batches()'s "does a later
# dot's window-call argument reference an earlier dot's OWN output column"
# rule), filter() dots never define a named output column for another dot
# to reference -- each dot is an independent boolean predicate -- so every
# window-bearing dot here is always planned jointly, in one batch (sharing
# CSE across dots, e.g. two dots both calling `mean(x)` still produce a
# single temp column). `exprs`' names are purely internal bookkeeping
# plan_window_stages() requires (never surfaced as a real column, since the
# trailing ast_select in filter.tbl_gpu() always restores the pre-window
# column set) -- they exist only so plan_window_stages()'s own chunking
# logic has something to compare against, which never actually splits
# anything here for exactly the reason above.
#
# @param exprs A *named* list of per-dot window-bearing IR nodes (only the
#   dots identified by `window_idx` in filter.tbl_gpu(), not every dot)
# @param schema `.data`'s schema before this filter() call
# @param group_cols Character vector, the effective group columns to window
#   over -- `verb_groups` from filter.tbl_gpu() (Phase 5 task W6): `by_cols`
#   when `.by=` was supplied, `.data$groups` otherwise
# @return `plan_window_stages()`'s result list, or `NULL` if planning
#   errored or produced an unlowerable spec
# @keywords internal
filter_plan_window <- function(exprs, schema, group_cols) {
  # sums = "lenient": an integer window sum() is INT32 (checked integer
  # arithmetic on it, as in dplyr). A lazy call pushes the window nodes with
  # on_sum_overflow = "double", so a top-level sum becomes a double only when
  # it overflows (F42); the temp columns never reach the output schema. A sum
  # nested in another window call can't change type mid-plan: its spec raises
  # on overflow instead. An eager call reruns on the CPU on any overflow (see
  # the caller).
  plan <- tryCatch(plan_window_stages(exprs, schema, group_cols, sums = "lenient"),
                   error = function(e) NULL)
  if (is.null(plan)) {
    return(NULL)
  }

  all_specs <- unlist(lapply(plan$stages, `[[`, "specs"), recursive = FALSE)
  if (length(all_specs) > 0 && !all(vapply(all_specs, window_spec_lowerable, logical(1)))) {
    return(NULL)
  }

  plan
}

# Internal: run the real dplyr::filter() (or dplyr::filter_out() when
# `complement`) on the CPU with the user's ORIGINAL dots, `.by` and
# `.preserve`. The shared body of every filter fallback below.
# @keywords internal
filter_cpu_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE) {
  verb <- if (complement) "filter_out" else "filter"
  fn <- if (complement) dplyr::filter_out else dplyr::filter
  gpu_fallback(verb, .data, function(tbl) {
    # Evaluated below the global environment: dplyr attributes a deprecation
    # (a one-column matrix) to the env of its caller, and an internal frame
    # would make lifecycle treat it as indirect use and warn only once.
    # The verb is bound under its own name so dplyr's error header reads
    # `filter()` / `filter_out()`.
    env <- rlang::new_environment(rlang::list2(!!verb := fn, tbl = tbl), parent = globalenv())
    call <- rlang::call2(verb, quote(tbl), !!!dots, .by = by_quo, .preserve = .preserve)
    rlang::eval_tidy(call, env = env)
  })
}

# Internal: whole-call CPU fallback for a window-bearing filter() call whose
# plan isn't (yet) fully lowerable (Phase 5, task W5).
#
# Mirrors mutate_window_fallback() (R/mutate.R): as of Phase 5 task W4,
# every window kind has C++ support (window_spec_lowerable()), so this is
# only reachable for a genuinely new, not-yet-implemented window kind added
# by some FUTURE phase -- there is currently no dplyr syntax that reaches
# it. Unlike filter_eval_mask() (which evaluates a single quosure with NO
# data mask, and so can never work for a window call referencing real table
# columns), this re-runs the real dplyr::filter() call, with every original
# dot, on the CPU via gpu_fallback() -- the only correct way to evaluate a
# window/aggregate predicate that cuplyr can't lower natively.
#
# @param .data The ORIGINAL tbl_gpu (BEFORE this filter() call -- nothing
#   has been pushed onto it yet, see the "never half-lower" comment at the
#   call site)
# @param dots The original (already rlang::enquos()'d) dots, every one of
#   them (not just the window-bearing ones), so the re-run dplyr::filter()
#   call reproduces the exact original semantics
# @param by_quo The `.by=` quosure, as captured by filter.tbl_gpu()
#   (Phase 5 task W6) -- re-injected so the CPU re-run honors on-the-fly
#   grouping too, exactly as the real call would have
# @return A new tbl_gpu (via gpu_fallback(), which also emits the
#   cuplyr_fallback_notify() notification)
# @keywords internal
filter_window_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE) {
  filter_cpu_fallback(.data, dots, by_quo, .preserve, complement)
}

# Internal: whole-call CPU fallback for a filter() call whose dots contain
# a deferred across()-family shape (Phase 10; dots_need_fallback(),
# R/across.R). Mirrors filter_window_fallback() exactly (re-runs the real
# dplyr::filter() call with the ORIGINAL dots, before any Phase 10
# substitution/expansion).
# @keywords internal
filter_across_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE) {
  filter_cpu_fallback(.data, dots, by_quo, .preserve, complement)
}

# Internal: whole-call CPU fallback for a filter() call with at least one dot
# whose shape the expression IR doesn't recognize at all, AND that
# references a real schema column (Phase 11, task L1 -- the "universal
# expression-level CPU fallback"; see scratchpad/phase11_design.md and
# dot_references_schema_col()'s own docs for why the column-reference check
# matters). Mirrors filter_window_fallback()/filter_across_fallback()
# exactly -- re-runs the real dplyr::filter() call, with the user's original
# dots (before across() expansion / cur_group_id() substitution), on the CPU.
# This is what subsumes grepl()/other base-R-string or arbitrary-function
# predicates referencing a real column, and closes the top_n()/top_frac()
# gap (both desugar to `filter(top_n_rank(n, wt))`). The common top_n_rank()
# shapes never get here: filter_run() rewrites them to a native
# `min_rank() <= n` first (rewrite_top_n_rank_dots()); this takes the rest.
# @keywords internal
filter_expr_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE) {
  filter_cpu_fallback(.data, dots, by_quo, .preserve, complement)
}

# Internal: TRUE iff a (non-window, non-constant) dot's IR is not a logical
# expression; filter_out() hands it to dplyr for its own error.
# @keywords internal
filter_ir_not_logical <- function(ir, schema, is_window) {
  if (is.null(ir) || is_window || ir_is_const(ir)) {
    return(FALSE)
  }
  !identical(unname(tryCatch(ir_infer_type(ir, schema), error = function(e) "BOOL8")), "BOOL8")
}

# Internal: TRUE iff `quo`'s raw expression references at least one symbol
# matching a current schema column name, anywhere in its tree (Phase 11,
# task L1). Used only to decide, for a dot `ir_parse_quo()` couldn't parse
# at all, whether it's safe to hand to `filter_eval_mask()`'s no-data-mask
# `eval_tidy()` (safe only when the expression is genuinely
# column-independent, e.g. a pre-computed logical vector or `rep(TRUE,
# n)`) or whether it must instead go to `filter_expr_fallback()` (any
# reference to a real column can never evaluate correctly with no data
# mask -- it either errors "object not found" or silently resolves to an
# unrelated same-named object in the calling environment).
#
# Deliberately simple: `all.vars()` (base R) walks the whole expression
# tree and returns every free variable symbol, with no need to
# special-case call heads/operators/`.data`-pronoun forms the way
# `ir_cols()`/`expr_has_column_ref()` do for parsed IR -- exactly the
# right tool here since the expression, by construction, did NOT parse
# into IR at all.
#
# @param quo A quosure (one filter() dot)
# @param col_names Character vector, the current schema's column names
# @return Logical scalar
# @keywords internal
dot_references_schema_col <- function(quo, col_names) {
  any(all.vars(rlang::quo_get_expr(quo)) %in% col_names)
}

# --- F29: dplyr's top_n()/top_frac() ------------------------------------------
#
# top_n() is a plain function that calls `filter(x, top_n_rank({{ n }}, !!wt))`,
# and `top_n_rank(n, wt)` is `min_rank(desc(wt)) <= n` for n > 0, else
# `min_rank(wt) <= abs(n)`. The dot reaching filter() is a quosure whose
# expression holds nested quosures for `n` and `wt` (`wt` is a plain symbol
# when top_n() picked the last column itself). A dot is rewritten to the
# equivalent native window predicate only for the shapes below; the CPU
# fallbacks keep rerunning the dots as the user wrote them.

# Inside an `n` expression, the pronouns (as symbols) and the functions (as
# call heads) that need the data mask. A bare symbol `n` is the user's own
# variable (`n <- 5; top_n(df, n, x)`); only a call to `n()` reads the mask.
top_n_mask_pronouns <- c(".data", ".env")
top_n_mask_fns <- c("n", "pick", "cur_group", "cur_group_id", "cur_group_rows",
                    "cur_column", "cur_data", "cur_data_all", "across",
                    "c_across", "if_any", "if_all")

# Look through nested quosures: list(expr, env) of a quosure or plain expr.
top_n_unwrap <- function(x, env) {
  while (rlang::is_quosure(x)) {
    env <- rlang::quo_get_env(x)
    x <- rlang::quo_get_expr(x)
  }
  list(expr = x, env = env)
}

# TRUE iff a quosure sits anywhere inside `expr`.
top_n_has_quosure <- function(expr) {
  if (rlang::is_quosure(expr)) return(TRUE)
  is.call(expr) && any(vapply(as.list(expr), top_n_has_quosure, logical(1)))
}

# The symbols of `expr`, through nested quosures, as list(vars =, fns =):
# `fns` are the called function names (`f` of `f()` and of `pkg::f()`),
# `vars` every other symbol.
top_n_symbols <- function(expr) {
  if (rlang::is_quosure(expr)) return(top_n_symbols(rlang::quo_get_expr(expr)))
  if (is.symbol(expr)) return(list(vars = as.character(expr), fns = character()))
  if (!is.call(expr)) return(list(vars = character(), fns = character()))
  head <- expr[[1]]
  out <- if (is.symbol(head)) {
    list(vars = character(), fns = as.character(head))
  } else if (is.call(head) && length(head) == 3L &&
             (identical(head[[1]], as.name("::")) || identical(head[[1]], as.name(":::")))) {
    list(vars = character(), fns = as.character(head[[3]]))
  } else {
    top_n_symbols(head)
  }
  for (arg in as.list(expr)[-1]) {
    if (rlang::is_missing(arg)) next
    sub <- top_n_symbols(arg)
    out$vars <- c(out$vars, sub$vars)
    out$fns <- c(out$fns, sub$fns)
  }
  out
}

# Evaluate `f()`, undoing any random-number draw it makes. Returns
# list(value =, rng =): `rng` is TRUE when `f()` moved the RNG, in which case
# `.Random.seed` has been put back (or removed again) and the caller must not
# use `value`, but leave the one real evaluation to dplyr.
# @keywords internal
top_n_rng_guard <- function(f) {
  seed_env <- globalenv()
  had <- exists(".Random.seed", envir = seed_env, inherits = FALSE)
  old <- if (had) get(".Random.seed", envir = seed_env, inherits = FALSE)
  value <- f()
  now <- if (exists(".Random.seed", envir = seed_env, inherits = FALSE)) {
    get(".Random.seed", envir = seed_env, inherits = FALSE)
  }
  moved <- !identical(old, now)
  if (moved) {
    if (had) assign(".Random.seed", old, envir = seed_env) else rm(".Random.seed", envir = seed_env)
  }
  list(value = value, rng = moved)
}

# The scalar number a constant `n` expression evaluates to, or NULL (it reads
# a column or the data mask, errors, draws random numbers, or is not a finite
# scalar number). dplyr evaluates `n` once per group, so an `n` that advances
# the RNG (`sample(1:4, 1)`) cannot be folded into one constant: the seed is
# restored and the caller declines, leaving the single evaluation to dplyr.
# Other side effects of `n` (a message, a counter) are not detected: the
# callers decline BEFORE evaluating where they cheaply can, but a rewrite that
# is declined later still runs a side-effecting `n` once more on the CPU.
top_n_const <- function(x, env, schema) {
  if (rlang::is_missing(x)) return(NULL)
  syms <- top_n_symbols(x)
  if (any(syms$vars %in% c(schema$names, top_n_mask_pronouns)) ||
      any(syms$fns %in% top_n_mask_fns)) {
    return(NULL)
  }
  q <- if (rlang::is_quosure(x)) x else rlang::new_quosure(x, env)
  res <- top_n_rng_guard(function() tryCatch(rlang::eval_tidy(q), error = function(e) NULL))
  if (res$rng) return(NULL)
  val <- res$value
  if (is.numeric(val) && !is.object(val) && length(val) == 1L && is.finite(val)) val else NULL
}

# TRUE iff `expr` is a call to dplyr's own n() with no arguments.
top_n_is_n_call <- function(expr, env) {
  if (!is.call(expr) || length(expr) != 1L) return(FALSE)
  head <- expr[[1]]
  if (is.call(head) && length(head) == 3L && as.character(head[[1]]) %in% c("::", ":::")) {
    return(identical(head[[2]], as.name("dplyr")) && identical(head[[3]], as.name("n")))
  }
  is.symbol(head) && identical(as.character(head), "n") &&
    identical(get0("n", envir = env, mode = "function"), dplyr::n)
}

# `n` as list(c =, times_n =): the constant, and whether it multiplies n().
# NULL for anything outside the allowlist.
top_n_parse_n <- function(n_arg, env, schema) {
  u <- top_n_unwrap(n_arg, env)
  const <- top_n_const(n_arg, env, schema)
  if (!is.null(const)) return(list(c = const, times_n = FALSE))
  if (top_n_is_n_call(u$expr, u$env)) return(list(c = 1, times_n = TRUE))
  if (is.call(u$expr) && length(u$expr) == 3L && identical(u$expr[[1]], as.name("*"))) {
    for (side in 2:3) {
      mine <- top_n_unwrap(u$expr[[side]], u$env)
      if (top_n_is_n_call(mine$expr, mine$env)) {
        const <- top_n_const(u$expr[[5L - side]], u$env, schema)
        if (!is.null(const)) return(list(c = const, times_n = TRUE))
      }
    }
  }
  NULL
}

# TRUE iff `head` (the call head of a dot's expression) names dplyr's own
# top_n_rank() as seen from the dot's environment.
top_n_rank_head <- function(head, env) {
  # dplyr doesn't export top_n_rank(), so only `dplyr:::top_n_rank` can name it
  if (is.call(head) && length(head) == 3L && identical(head[[1]], as.name(":::"))) {
    return(identical(head[[2]], as.name("dplyr")) && identical(head[[3]], as.name("top_n_rank")))
  }
  is.symbol(head) && identical(as.character(head), "top_n_rank") &&
    identical(get0("top_n_rank", envir = env, mode = "function"),
              get("top_n_rank", envir = asNamespace("dplyr")))
}

# The native `min_rank(...) <= ...` predicate for one top_n_rank() dot, or
# NULL to leave the dot alone.
rewrite_top_n_rank_dot <- function(dot, schema) {
  expr <- rlang::quo_get_expr(dot)
  env <- rlang::quo_get_env(dot)
  if (!is.call(expr) || length(expr) != 3L || !is.null(names(expr))) {
    return(NULL)
  }
  if (!top_n_rank_head(expr[[1]], env)) return(NULL)

  # `wt` before `n`: evaluating `n` and then declining would run it twice
  # (once here, once on the CPU).
  wt <- top_n_unwrap(expr[[3]], env)
  if (rlang::is_missing(wt$expr) || top_n_has_quosure(wt$expr)) return(NULL)
  # Parsing folds the column-free parts of `wt`. One that draws random
  # numbers (`x * rnorm(1)`) would be drawn here, again for the rewritten dot,
  # and once more when filter() parses it, where dplyr draws once (per group):
  # leave such a `wt` to dplyr, with the RNG as it was.
  parsed <- top_n_rng_guard(function() {
    tryCatch(ir_parse_expr(wt$expr, wt$env, schema), error = function(e) NULL)
  })
  if (parsed$rng) return(NULL)
  wt_ir <- parsed$value
  if (is.null(wt_ir)) return(NULL)
  wt_type <- tryCatch(ir_infer_type(wt_ir, schema), error = function(e) NULL)
  if (is.null(wt_type) || identical(unname(wt_type), "DICTIONARY32")) return(NULL)

  nn <- top_n_parse_n(expr[[2]], env, schema)
  if (is.null(nn)) return(NULL)

  bound <- if (nn$times_n) call("*", abs(nn$c), quote(dplyr::n())) else abs(nn$c)
  # dplyr's own functions, whatever the user's environment binds to these
  # names (ir_parse_expr() strips the `dplyr::`).
  rank <- if (nn$c > 0) {
    rlang::expr(dplyr::min_rank(dplyr::desc(!!wt$expr)))
  } else {
    rlang::expr(dplyr::min_rank(!!wt$expr))
  }
  new <- rlang::new_quosure(call("<=", rank, bound), wt$env)

  # A rewrite that doesn't parse (unknown column, unsupported wt) is left to
  # dplyr, which raises its own error or computes the result.
  ir <- tryCatch(ir_parse_quo(new, schema), error = function(e) NULL)
  if (is.null(ir)) return(NULL)
  new
}

# @param dots filter() quosures
# @param .data the table being filtered; its schema is read only once a dot
#   turns out to be a top_n_rank() call
# @return the dots with each eligible top_n_rank() dot replaced
# @keywords internal
rewrite_top_n_rank_dots <- function(dots, .data) {
  schema <- NULL
  for (i in seq_along(dots)) {
    expr <- rlang::quo_get_expr(dots[[i]])
    if (!is.call(expr) || !top_n_rank_head(expr[[1]], rlang::quo_get_env(dots[[i]]))) next
    schema <- schema %||% current_schema(.data)
    new <- rewrite_top_n_rank_dot(dots[[i]], schema)
    if (!is.null(new)) dots[[i]] <- new
  }
  dots
}

# ir_parse_quo() throws (rather than returning NULL) when a constant-folded
# subtree evaluates to a vector outside %in%'s RHS position
# (ir_lit_from_r()'s scalar-only rule, section 1.2 of
# scratchpad/phase1_expression_engine.md) -- exactly right for mutate(),
# where a bare vector literal is always a user error, but filter() has
# always accepted a directly-evaluable logical vector as a mask
# (`filter(rep(TRUE, n))`), which must still fall back to CPU evaluation
# rather than error at parse time. Catch specifically that "expected a
# scalar" shape and treat it like any other unsupported/opaque expression
# (NULL, triggering the fallback in filter.tbl_gpu()); anything else (e.g.
# "not found", or the `&&`/`||` guard) propagates as a real error.
#
# @param quo A quosure containing a filter expression
# @param schema The current schema (names/types) to parse against
# @return An IR node, or NULL if unsupported
# @keywords internal
parse_filter_ir <- function(quo, schema) {
  tryCatch(
    ir_parse_quo(quo, schema),
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("scalar", msg, fixed = TRUE) && grepl("length", msg, fixed = TRUE)) {
        return(NULL)
      }
      stop(e)
    }
  )
}

# Internal: CPU-eval fallback for filter expressions that aren't a
# recognized comparison shape.
#
# Materializes any pending lazy operations first (so the fallback always
# operates on a real GPU table), eval_tidy()s the quosure (no explicit data
# mask - matches the historical behavior of only supporting expressions
# that evaluate directly, e.g. `rep(TRUE, n)`, not column references),
# validates the result is logical, and applies it via gpu_filter_bool()
# (scalar/uniform case) or gpu_filter_mask() (mixed mask), same as the
# renamed-from filter_logical() did. Notifies via cuplyr_fallback_notify()
# (D4: silent by default).
#
# @param .data A tbl_gpu object
# @param quo A quosure whose evaluated result must be logical
# @param eval_result The quosure's already-evaluated value (callers that
#   evaluated it once to decide routing pass it so it isn't run twice)
# @return A filtered tbl_gpu object
# @keywords internal
filter_eval_mask <- function(.data, quo, eval_result = rlang::eval_tidy(quo)) {
  if (identical(.data$exec_mode, "lazy") && has_pending_ops(.data)) {
    .data <- compute(.data)
  }

  expr_chr <- rlang::quo_text(quo)
  eval_result <- tryCatch(eval_result, error = function(e) NULL)

  if (is.null(eval_result) || !is.logical(eval_result)) {
    stop("filter() only supports comparisons: ==, !=, >, >=, <, <=\n",
         "Or logical values: TRUE, FALSE, logical vectors\n",
         "Expression: ", expr_chr, call. = FALSE)
  }

  cuplyr_fallback_notify("filter", expr_chr)

  n_rows <- dim(.data)[1]

  if (length(.data$schema$names) == 0L) {
    # Zero-column table: no cudf column to mask, so only the R-side count
    # changes (dim() is NA when nobody recorded one).
    if (is.na(n_rows)) {
      stop("filter() needs a known row count on a tbl_gpu with no columns.",
           call. = FALSE)
    }
    if (length(eval_result) != 1L && length(eval_result) != n_rows) {
      stop("Logical vector length (", length(eval_result),
           ") must match number of rows (", n_rows, ")", call. = FALSE)
    }
    return(carry_frame_meta(new_tbl_gpu(
      ptr = .data$ptr,
      schema = .data$schema,
      groups = .data$groups,
      exec_mode = .data$exec_mode,
      nrow = sum(rep_len(eval_result, n_rows) %in% TRUE)
    ), .data))
  }

  if (length(eval_result) == 1) {
    # Single boolean: TRUE keeps all rows, FALSE keeps none
    if (isTRUE(eval_result)) {
      new_ptr <- wrap_gpu_call("filter_bool_true", gpu_filter_bool(.data$ptr, TRUE))
    } else {
      new_ptr <- wrap_gpu_call("filter_bool_false", gpu_filter_bool(.data$ptr, FALSE))
    }
  } else {
    # Logical vector: use as mask
    if (length(eval_result) != n_rows) {
      stop("Logical vector length (", length(eval_result),
           ") must match number of rows (", n_rows, ")", call. = FALSE)
    }

    # Check for all TRUE or all FALSE (optimize common cases)
    if (all(eval_result, na.rm = TRUE) && !any(is.na(eval_result))) {
      new_ptr <- wrap_gpu_call("filter_bool_all_true", gpu_filter_bool(.data$ptr, TRUE))
    } else if (!any(eval_result, na.rm = TRUE)) {
      new_ptr <- wrap_gpu_call("filter_bool_all_false", gpu_filter_bool(.data$ptr, FALSE))
    } else {
      # Mixed: apply mask
      new_ptr <- wrap_gpu_call("filter_mask", gpu_filter_mask(.data$ptr, eval_result))
    }
  }

  carry_frame_meta(new_tbl_gpu(
    ptr = new_ptr,
    schema = .data$schema,
    groups = .data$groups,
    group_drop = .data$group_drop %||% TRUE,
    exec_mode = .data$exec_mode
  ), .data)
}
