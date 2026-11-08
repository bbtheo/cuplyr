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
#' A floating `NaN` (e.g. a computed `0/0`) stays a value on the GPU, but
#' `is.na()`, the six comparisons and `coalesce()` read it as missing, as in
#' R: `NaN > 1` is `NA` (the row is dropped) and `is.na(NaN)` is `TRUE`.
#' `is.nan()` runs on the GPU too.
#'
#' ## Not yet supported
#' \itemize{
#'   \item Arbitrary R functions with no cuDF equivalent -- these fall back
#'     to CPU evaluation, see below
#' }
#'
#' ## Expressions the IR doesn't understand (CPU fallback)
#' An expression shape the IR doesn't recognize (see `R/ir.R`'s
#' `ir_parse_quo()`) is handled in one of two ways. A dot that reads no table
#' column (a pre-computed logical vector, `rep(TRUE, n)`) is evaluated once in
#' R, with no data mask: a logical scalar is a constant predicate. On an
#' ungrouped `filter()` a logical vector is a row mask: all the vectors of one
#' call are sized against the input table (a wrong size is dplyr's own error)
#' and combined with `&` with the call's other predicates, which see the same
#' input rows. On a grouped, `.by` or rowwise table, or in `filter_out()`, a
#' logical vector runs the whole call in dplyr. A dot that reads a table
#' column but doesn't parse (a rare shape given how much of
#' dplyr's filter surface the IR now covers), or that visibly calls something
#' reaching the data mask by name (`exists("x")`, `get("ok")`), runs the whole
#' call in dplyr on the CPU (see `getOption("cuplyr.fallback")` below to be
#' notified when it does). The check reads the dot's syntax, so an alias of
#' such a function or a user function calling it is not seen and has no data
#' mask when the dot names no column. A
#' lazy `.data` with pending operations runs them early, on the GPU, when a
#' logical vector needs the row count (a plan break, see
#' `getOption("cuplyr.plan_break")`).
#'
#' ## `options(cuplyr.fallback = ...)`
#' Controls whether a CPU fallback (the one described above, and every other
#' verb that runs in dplyr on the CPU) notifies the caller. One of:
#' \itemize{
#'   \item `"silent"` (default) - no notification
#'   \item `"warn"` (or `"warning"`) - emits a warning naming the verb, the
#'     expression, the reason and the size collected
#'   \item `"error"` - `stop()`s instead of falling back
#' }
#' A lazy plan that only runs early on the GPU is a plan break, reported under
#' `options(cuplyr.plan_break = )` instead. See [cuplyr-fallback].
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
  # (an argument that wasn't given is the NULL quosure, which skips an enquo())
  by_quo <- if (missing(.by) && is.null(substitute(.by))) null_quo else rlang::enquo(.by)
  filter_impl(.data, rlang::quos(..., .ignore_empty = "all"), by_quo, .preserve,
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
  by_quo <- if (missing(.by) && is.null(substitute(.by))) null_quo else rlang::enquo(.by)
  filter_impl(.data, rlang::quos(..., .ignore_empty = "all"), by_quo, .preserve,
              complement = TRUE, call = rlang::current_env())
}

# Internal: the engine behind filter() and filter_out(). `complement = TRUE`
# is filter_out(): every dot's predicate is ANDed into ONE expression P and the
# single pushed predicate is `is.na(P) | !P`.
#
# `.preserve = TRUE`, or a `.drop = FALSE` table, on a group_by() table fixes
# the input's group keys on the result when a group can have been lost (as
# slice() does, see row_slice_prepare()); every other exit drops them.
# @keywords internal
filter_impl <- function(.data, dots, by_quo, .preserve, complement, call) {
  check_unique_input_names(.data, call)
  # (F88: the reasons this call's dots record for a CPU fallback, call-scoped)
  prev <- fallback_reasons_open()
  on.exit(fallback_reasons_close(prev), add = TRUE)
  if (is_rowwise_gpu(.data)) {
    return(filter_rowwise(.data, dots, by_quo, .preserve, complement, call))
  }
  # dplyr keeps a `.drop = FALSE` table's whole group structure whatever
  # `.preserve` says, so those tables take the same route (row_slice_prepare()
  # computes a lazy input first, so its plan runs once).
  prep <- row_slice_prepare(.data, .preserve, eligible = quo_is_null_fast(by_quo))
  .data <- prep$data
  out <- filter_run(.data, dots, by_quo, .preserve, complement, call)
  row_slice_finish(out, prep)
}

# filter()/filter_out() of a rowwise table (F49): the predicates are evaluated
# once per row, which for an elementwise predicate is the vectorised filter. The
# native attempt is filter_run() with the rowwise vars protected (it treats the
# table as ungrouped), after rowwise_rewrite_dots() turned the row aggregates
# into horizontal GPU ops; anything else (`.by`, `.preserve`, a window call, a
# constant that is not a single logical, ...) runs the whole call in dplyr.
# @keywords internal
filter_rowwise <- function(.data, dots, by_quo, .preserve, complement, call) {
  cpu <- function(reason = NULL) filter_cpu_fallback(.data, dots, by_quo, .preserve, complement, reason = reason)
  if (!quo_is_null_fast(by_quo) || !identical(.preserve, FALSE)) {
    return(cpu())
  }
  # F77: a function of the user's with a native op's name runs in dplyr
  shadowed <- shadow_screen_text(dots)
  if (!is.null(shadowed)) {
    return(cpu(shadowed))
  }
  exact <- rowwise_exact_sums(.data, dots, FALSE, "filter")
  if (!is.null(exact$now)) {
    return(plan_back_to_lazy(filter_rowwise(exact$now, dots, by_quo, .preserve, complement, call)))
  }
  rw <- if (exact$rewritten) exact$rw else rowwise_rewrite_dots(.data, dots, named = FALSE)
  if (is.null(rw)) {
    return(rowwise_rerun(cpu, dots))
  }
  res <- rowwise_attempt(function() {
    filter_run(.data, rw, by_quo, .preserve, complement, call)
  })
  if (is.null(res)) rowwise_rerun(cpu, dots) else res
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
  cpu <- function(reason) filter_cpu_fallback(.data, dots, by_quo, .preserve, complement, reason = reason)
  nms <- names(dots)
  if (!is.null(nms) && any(nzchar(nms)) &&
      any(nzchar(nms) & !vapply(dots, function(q) is.logical(rlang::quo_get_expr(q)), logical(1)))) {
    return(cpu("a named input that is not a logical literal is checked by dplyr."))
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
  # (a rowwise table's groups are its rowwise vars: nothing is computed over them)
  rowwise <- is_rowwise_gpu(.data)
  verb_groups <- if (!quo_is_null_fast(by_quo)) by_cols else if (rowwise) character() else .subset2(.data, "groups")

  if (!quo_is_null_fast(by_quo) && isTRUE(.preserve)) {
    return(cpu("`.by` together with `.preserve = TRUE` is checked by dplyr."))
  }
  if (complement && length(current_schema(.data)$names) == 0L) {
    return(cpu("filter_out() of a table with no columns is evaluated by dplyr."))
  }

  # F77: a call whose head resolves to a function of the user's rather than the
  # native op (a local `abs()`) is dplyr's to run, whole, before any rewrite.
  # (A rowwise table's dots were screened by filter_rowwise(), then rewritten.)
  if (!rowwise) {
    shadowed <- shadow_screen_text(dots)
    if (!is.null(shadowed)) {
      return(filter_cpu_fallback(.data, dots, by_quo, .preserve, complement, reason = shadowed))
    }
  }

  # F75: top_n()/top_frac() with a `wt` that parsing would evaluate more often
  # than dplyr does runs whole in dplyr, before anything is parsed.
  declined <- top_n_rank_decline(dots, .data)
  if (!is.null(declined)) {
    return(filter_cpu_fallback(.data, dots, by_quo, .preserve, complement, reason = declined))
  }

  # F75: dplyr evaluates a predicate once per group, the native path folds a
  # column-free call once; only a pure call is the same either way, so a
  # grouped or `.by` call with any other runs whole in dplyr.
  if (length(verb_groups) > 0L) {
    impure <- impure_const_call_text(dots, current_schema(.data))
    if (!is.null(impure)) {
      return(filter_cpu_fallback(.data, dots, by_quo, .preserve, complement, reason = impure))
    }
  }

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
  dots <- colwise_rewrite_dots(dots, current_schema(.data))
  # (dots that mention none of the names the next screens look for pass them
  # all unchanged: dots_are_plain())
  plain <- length(dots) > 0L && dots_are_plain(dots)
  if (!plain) {
    dots <- rewrite_context_dots(dots, verb_groups)
  }
  dots <- rewrite_top_n_rank_dots(dots, .data)
  if (!plain && dots_need_fallback(dots)) {
    return(filter_across_fallback(.data, user_dots, by_quo, .preserve, complement))
  }

  # Substitution runs before expansion here so the if_any()/if_all() piece
  # check can type cur_group_id() pieces; the helper column is excluded from
  # `.cols`'s selectable set alongside the group columns.
  fallback_data <- .data
  fallback_dots <- user_dots
  cgid <- if (plain) {
    list(data = .data, dots = dots, used = FALSE)
  } else {
    substitute_cur_group_id(.data, dots, verb_groups, by = !quo_is_null_fast(by_quo),
                            rowwise = rowwise)
  }
  .data <- cgid$data
  dots <- if (plain) {
    expand_plain_dots(cgid$dots)
  } else {
    expand_across_dots(cgid$dots, current_schema(.data), c(verb_groups, cgid$helper_name), "filter")
  }
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
  schema <- current_schema(.data)
  orig_names <- schema$names
  real_orig_names <- if (isTRUE(cgid$used)) set_diff(orig_names, cgid$helper_name) else orig_names

  # First pass: parse every dot up front (no push_op()/GPU work yet) so we
  # know, before touching `.data` at all, whether any dot needs the window
  # path below. Doing this classification BEFORE the per-dot loop's
  # ir_is_const() check (further down) matters, not just for staging: a
  # bare window call like `n()` has zero column ARGUMENTS of its own
  # (ir_cols() only walks `$args`), so `ir_is_const(ir_parse_quo(quo(n() >
  # 2)))` would itself read TRUE -- without routing window calls through
  # this dedicated path first, `filter(n() > 2)` fell into the
  # constant-expression branch instead (evaluating `n() > 2` with no
  # table in scope, which either errors or silently misbehaves) rather than
  # being planned as a real window predicate. This was the crash this task
  # fixes (see the regression test): checking ir_has_window() here, before
  # any const-folding logic ever runs, routes it correctly instead.
  parsed <- vector("list", length(dots))
  inline_consts <- list()
  window_idx <- integer()
  # (F88: the user's dot each parsed dot comes from, and the cause each
  # unparsed one recorded, claimed below for the dot that falls back)
  origin <- attr(dots, "provenance")$origin %||% seq_along(dots)
  causes <- NULL

  # F82: a dot that reads no column is evaluated here, once, at its position
  # (dplyr's order), and not parsed (the parser would fold it a second time).
  # The value or the error is recorded for the checks below, and for the
  # dplyr fallbacks, which get the value inlined (`inline`). After an error
  # nothing more is evaluated: dplyr stops there.
  standalone <- vector("list", length(dots))
  evaluated_free <- !rowwise
  errored <- NA_integer_
  for (i in seq_along(dots)) {
    if (evaluated_free && filter_dot_is_column_free(dots[[i]], schema)) {
      res <- tryCatch(list(value = rlang::eval_tidy(dots[[i]], data = list())),
                      error = function(e) list(error = e))
      standalone[i] <- list(res)
      if (!is.null(res$error)) {
        # A free name bound nowhere may be a typo: the parser raises dev's
        # "object not found" for it at once, with no table copied to the host.
        # Decided only now, after the evaluation: a name the dot binds while it
        # runs (`{ v <- f(); v }`) or never reaches (`TRUE || nope`) is no typo.
        # A parse that raises nothing (a fold swallowed the error) or rejects
        # `&&`/`||` leaves the error to dplyr, as below.
        if (expr_has_unbound_symbol(rlang::quo_get_expr(dots[[i]]), rlang::quo_get_env(dots[[i]]), schema)) {
          tryCatch(parse_filter_ir(dots[[i]], schema), cuplyr_error_short_circuit = function(e) NULL)
          fallback_cause_take()
        }
        errored <- i
        break
      }
      value <- res$value
      # a plain logical scalar keeps the constant-predicate route
      parsed[i] <- list(if (is.logical(value) && length(value) == 1L && is.null(dim(value))) {
        ir_lit_from_r(value)
      })
      inline_consts[[as.character(i)]] <- res
      next
    }
    ir <- parse_filter_ir(dots[[i]], schema)
    if (is.null(ir)) {
      cause <- fallback_cause_take()
      if (!is.null(cause)) {
        causes[[as.character(i)]] <- cause
      }
    }
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

  # (grouped, `.by` and rowwise calls keep their dots as written)
  inline <- if (length(verb_groups) == 0L && !rowwise && length(inline_consts) > 0L) {
    list(consts = inline_consts, provenance = attr(dots, "provenance"))
  }

  # A window call sees one row on a rowwise table: dplyr's result, not ours.
  if (rowwise && length(window_idx) > 0) {
    rowwise_bail("a window call in a rowwise filter()")
  }

  # Phase 11, task L1: universal expression-level CPU fallback. A dot that
  # `ir_parse_quo()` couldn't parse at all (`ir` is NULL -- as opposed to a
  # genuine thrown error, which already propagated out of the loop above)
  # AND that references a real schema column somewhere in its raw
  # expression tree (`dot_references_schema_col()`) can NEVER be evaluated
  # outside the data mask (the standalone evaluation below) -- that column symbol
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
  # shape, e.g. `rep(TRUE, n)`) is untouched -- it is evaluated in R below,
  # and a vector of that kind is applied as one mask (mask_idx).
  # (every dot is judged, as a vapply() over them did, before any is acted on)
  needs_expr_fallback <- FALSE
  # (a column-free dot that errored ends the loop above: the dots after it were
  # never parsed, and dplyr stops there)
  for (i in seq_len(if (is.na(errored)) length(dots) else errored)) {
    why <- if (identical(i, errored)) {
      list(filter_free_dot_failed_reason)
    } else if (is.null(parsed[[i]]) && dot_references_schema_col(dots[[i]], orig_names)) {
      list(causes[[as.character(i)]])
    } else if (!is.null(parsed[[i]]) && ir_factor_rule(parsed[[i]], schema) != "none") {
      list("a factor in this predicate needs its labels, which dplyr compares (the GPU holds codes).")
    } else if (filter_ir_not_logical(parsed[[i]], schema, i %in% window_idx)) {
      list("this predicate is not a logical column expression; dplyr checks it.")
    }
    if (!is.null(why) && !needs_expr_fallback) {
      # (the first dot that falls back is the one the notice names)
      fallback_dot_failed(origin[[i]], why[[1L]])
      needs_expr_fallback <- TRUE
    }
  }

  if (needs_expr_fallback) {
    return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                inline = inline))
  }

  # A dot with no column reference whose IR is NULL (opaque) or constant is
  # evaluated standalone in R (below) -- no data mask. If that errors, dplyr
  # may still succeed (the call can use pick()/n()/cur_*() inside a user
  # function, valid only in the mask), so hand the WHOLE call to dplyr,
  # decided here before any predicate is pushed. Each such dot is evaluated
  # exactly once: the value is cached for the loop below. (The dots that read
  # no column were evaluated by the parse loop; only the rest are evaluated
  # here: a bare literal, a dot of a rowwise call, or one naming a mask-only
  # function. A call expr_reaches_mask() recognizes is dplyr's, unevaluated; an
  # alias or a user function making that call is not recognized and runs here.)
  # (a plain logical vector reaching the mask is combined below, see mask_idx)
  mask_idx <- integer()
  for (i in if (length(window_idx) > 0L) set_diff(seq_along(dots), window_idx) else seq_along(dots)) {
    res <- standalone[[i]]
    if (!is.null(res) || is.null(parsed[[i]]) || ir_is_const(parsed[[i]])) {
      if (is.null(res)) {
        # A dot that reaches the data mask by name (`exists("x")`, `get("ok")`)
        # has no value outside it: dplyr evaluates the call, as mutate() does.
        if (expr_reaches_mask(rlang::quo_get_expr(dots[[i]]))) {
          fallback_dot_failed(origin[[i]], mask_reaching_reason)
          return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                      inline = inline))
        }
        res <- tryCatch(list(value = rlang::eval_tidy(dots[[i]], data = list())),
                        error = function(e) list(error = e))
      }
      if (!is.null(res$error)) {
        fallback_dot_failed(origin[[i]], filter_free_dot_failed_reason)
        return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                    inline = inline))
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
        fallback_dot_failed(origin[[i]], paste0(
          "a constant predicate that is not a plain logical scalar (a matrix, a per-group vector, ",
          "a non-logical value, or any under filter_out()) is evaluated by dplyr."
        ))
        return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                    inline = inline))
      }
      standalone[i] <- list(res)
      if (!scalar_lgl) {
        mask_idx <- c(mask_idx, i)
      }
    }
  }

  # F79: dplyr evaluates every dot of one call against the same input rows and
  # checks each vector's size against them. A logical vector constant (the only
  # value left here with a size other than 1) is sized against the input now,
  # and all of them are combined with `&` into one mask: applying them where
  # they stand would size a later vector against the rows an earlier dot left.
  mask_keep <- NULL
  mask_name <- NULL
  mask_ran_plan <- FALSE
  if (length(mask_idx) > 0L) {
    if (rowwise) {
      rowwise_bail("a logical vector in a rowwise filter()")
    }
    # (the row count of a lazy table with pending steps needs them run: a plan
    # break, notified for the user's own steps. The table stays lazy, and the
    # dplyr fallbacks start from the computed table)
    if (identical(.subset2(.data, "exec_mode"), "lazy") && has_pending_ops(.data)) {
      if (has_pending_ops(fallback_data)) {
        cuplyr_plan_break_notify(
          if (complement) "filter_out" else "filter",
          "the row count is needed to size a logical vector", data = fallback_data
        )
      }
      .data <- compute(.data)
      mask_ran_plan <- TRUE
      fallback_data <- if (isTRUE(cgid$used)) plan_now_columns(.data, real_orig_names) else .data
    }
    n_in <- dim(.data)[1L]
    for (i in mask_idx) {
      if (!length(standalone[[i]]$value) %in% c(1L, n_in)) {
        fallback_dot_failed(origin[[i]], "a logical vector whose size is not that of the table is checked by dplyr.")
        return(filter_expr_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                    inline = inline))
      }
    }
    # No window dot and nothing else but constants (on a table with no columns:
    # a call with a vector and no window or cur_group_id() dot): all the dots
    # are one mask. Otherwise the native dots see the same rows through a temp
    # column holding it, below.
    mask_only <- length(window_idx) == 0L &&
      all(vapply(set_diff(seq_along(dots), mask_idx), function(i) {
        is.null(parsed[[i]]) || ir_is_const(parsed[[i]])
      }, logical(1)))
    use_idx <- if (mask_only) seq_along(dots) else mask_idx
    mask_keep <- Reduce(`&`, lapply(use_idx, function(i) standalone[[i]]$value)) %in% TRUE
    if (mask_only) {
      res <- filter_eval_mask(.data, dots[[mask_idx[[1L]]]], mask_keep)
      # (the cur_group_id() helper column does not leave with the rows)
      return(if (isTRUE(cgid$used)) {
        push_op(res, ast_select(input_node(res), real_orig_names))
      } else {
        res
      })
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
  # (set when a lazy call runs at the verb, see the integer-sum rule below)
  back_to_lazy <- FALSE
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
      return(filter_window_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                    inline = inline))
    }

    # A non-logical predicate must raise dplyr's own "`..1` must be a logical
    # vector" error (filter_out()'s cudf NOT would even accept numerics).
    # Judge each window dot by its rewritten predicate's type, before any
    # stage is pushed, and let dplyr raise its error on the CPU.
    if (any(vapply(plan$post, filter_ir_not_logical, logical(1),
                                 schema = plan$schema, is_window = FALSE))) {
      return(filter_window_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                    inline = inline))
    }

    # F42: an integer window sum() is INT32, so arithmetic on it is checked as
    # in dplyr. dplyr makes only the overflowing groups' sums doubles; the
    # other groups' sums stay integers and arithmetic on them still overflows
    # to NA. An eager call therefore reruns on the CPU when any sum overflows
    # (exact, and overflow is rare).
    eager <- identical(.subset2(.data, "exec_mode"), "eager")
    # F80: a lazy plan can't do that (its schema is fixed before the data is
    # seen), so a lazy call with an INT32 sum runs at the verb as an eager one:
    # the pending steps run now (a plan break, notified for the user's own
    # steps, not the cur_group_id() helper), the rest of this call runs eagerly
    # on the parsed plan (nothing is parsed or evaluated twice), and the result
    # goes back to lazy with nothing pending, as mutate_window()'s does. The
    # CPU rerun starts from the computed table, without the helper column. A
    # window filter is an optimizer barrier, so no pruning is lost. (A BOOL8
    # sum stays in the plan: it overflows only past 2^31 TRUE rows in a group,
    # and then comes back FLOAT64 or, nested, raises
    # `cuplyr_error_sum_overflow`; see filter_plan_window().)
    if (!eager && any(vapply(unlist(lapply(plan$stages, `[[`, "specs"), recursive = FALSE),
                             window_spec_int_sum, logical(1)))) {
      .data <- plan_run_now(.data, if (complement) "filter_out" else "filter",
                            "an integer sum() (result type depends on the data)",
                            notice_data = if (mask_ran_plan) .data else fallback_data)
      fallback_data <- if (isTRUE(cgid$used)) plan_now_columns(.data, real_orig_names) else .data
      eager <- TRUE
      back_to_lazy <- TRUE
    }
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
        res <- filter_window_fallback(fallback_data, fallback_dots, by_quo, .preserve, complement,
                                      inline = inline, reason = "an integer sum() overflowed in a window predicate; dplyr widens only the overflowing groups.")
        return(if (back_to_lazy) plan_back_to_lazy(res) else res)
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

  # F79, mixed call: R evaluated a vector here too, so the notice is the
  # mask-only route's, and it comes after the window block because a window
  # plan that falls back to dplyr has its own. The combined mask rides along as
  # one temp column, so that the one ast_filter below evaluates every native
  # predicate on the same rows (a dropped row still raises its integer
  # overflow). All TRUE: nothing to do.
  if (!is.null(mask_keep)) {
    filter_mask_notify(dots[[mask_idx[[1L]]]], n_in)
  }
  if (!is.null(mask_keep) && !all(mask_keep)) {
    mask_name <- unique_temp_name("..filtermask..", schema$names)
    .data <- push_op(.data, ast_mutate(input_node(.data), list(make_mutate_expr(
      mask_name, ir_vec(mask_keep, "BOOL8", dots[[mask_idx[[1L]]]], mask_name), schema
    ))))
    schema <- current_schema(.data)
  }

  predicates <- list()

  # filter_out(): only IR predicates are collected; they are ANDed into one P
  # and complemented once, after the loop. A constant FALSE/NA dot makes P
  # FALSE or NA, so every row is kept.
  comp_irs <- list()
  keep_all <- FALSE

  for (i in seq_along(dots)) {
    ir <- parsed[[i]]

    if (complement) {
      if (is.null(ir) || ir_is_const(ir)) {
        if (!isTRUE(standalone[[i]]$value)) keep_all <- TRUE
      } else {
        comp_irs <- c(comp_irs, list(ir))
      }
      next
    }

    if (length(mask_idx) > 0L && i %in% mask_idx) {
      # (the combined mask is one predicate, on its temp column)
      if (!is.null(mask_name) && i == mask_idx[[1L]]) {
        predicates <- c(predicates, list(make_predicate(ir_col(mask_name), schema)))
      }
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
      # predicate paths the pre-IR parser used (a vector is one combined mask, above).
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

      stop("internal error: a constant filter() predicate that is not a logical scalar", call. = FALSE)
    }

    predicates <- c(predicates, list(make_predicate(ir, schema)))
  }

  if (length(predicates) > 0) {
    .data <- push_op(.data, ast_filter(input_node(.data), predicates))
  }

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

  if (length(window_idx) > 0 || isTRUE(cgid$used) || !is.null(mask_name)) {
    # Drop every `..win*../..winarg*..` temp column introduced by the
    # window-planning block above, and/or Phase 10's `..cur_group_id..`
    # helper column (if used), restoring the table's real, user-visible
    # schema (section 1.3's "trailing select" step).
    .data <- push_op(.data, ast_select(input_node(.data), real_orig_names))
  }

  if (back_to_lazy) plan_back_to_lazy(.data) else .data
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
  # arithmetic on it, as in dplyr). An eager call reruns on the CPU on any
  # overflow, and a lazy call with an INT32 sum runs as an eager one (see the
  # caller). Only a lazy BOOL8 sum stays in the plan: its window node is pushed
  # with on_sum_overflow = "double", so a top-level sum becomes a double only
  # when it overflows (F42); the temp columns never reach the output schema. A
  # sum nested in another window call can't change type mid-plan: its spec
  # raises on overflow instead.
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
filter_cpu_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE, detail = NULL,
                                reason = detail, inline = NULL) {
  verb <- if (complement) "filter_out" else "filter"
  fn <- if (complement) dplyr::filter_out else dplyr::filter
  gpu_fallback(verb, .data, function(tbl) {
    # Inlined here, where the row count is known
    dots <- filter_inline_evaluated(dots, inline, nrow(tbl))
    # Evaluated below the global environment: dplyr attributes a deprecation
    # (a one-column matrix) to the env of its caller, and an internal frame
    # would make lifecycle treat it as indirect use and warn only once.
    # The verb is bound under its own name so dplyr's error header reads
    # `filter()` / `filter_out()`. The dots and `.by` stay splices for the
    # verb's own capture (not inlined quosures, whose raw `.data[[k]]` would be
    # read in this frame: see the splice rule after pronoun_call_name(), R/ir.R).
    env <- rlang::new_environment(
      rlang::list2(!!verb := fn, tbl = tbl, .cuplyr_dots = dots, .cuplyr_by = by_quo),
      parent = globalenv()
    )
    call <- rlang::call2(verb, quote(tbl), quote(!!!.cuplyr_dots), .by = quote(!!.cuplyr_by),
                         .preserve = .preserve)
    rlang::eval_tidy(call, env = env)
  }, reason = reason)
}

# Internal: whole-call CPU fallback for a window-bearing filter() call whose
# plan isn't (yet) fully lowerable (Phase 5, task W5).
#
# Mirrors mutate_window_fallback() (R/mutate.R): as of Phase 5 task W4,
# every window kind has C++ support (window_spec_lowerable()), so this is
# only reachable for a genuinely new, not-yet-implemented window kind added
# by some FUTURE phase -- there is currently no dplyr syntax that reaches
# it. Unlike the standalone evaluation of a column-free dot (with NO data
# mask, which can never work for a window call referencing real table
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
filter_window_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE,
                                   reason = "a window predicate has no GPU plan (its stages cannot be lowered, or it is not logical).",
                                   inline = NULL) {
  filter_cpu_fallback(.data, dots, by_quo, .preserve, complement, reason = reason, inline = inline)
}

# Internal: whole-call CPU fallback for a filter() call whose dots contain
# a deferred across()-family shape (Phase 10; dots_need_fallback(),
# R/across.R). Mirrors filter_window_fallback() exactly (re-runs the real
# dplyr::filter() call with the ORIGINAL dots, before any Phase 10
# substitution/expansion).
# @keywords internal
filter_across_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE) {
  filter_cpu_fallback(.data, dots, by_quo, .preserve, complement,
                      reason = fallback_failed_reason(dots, across_fallback_default))
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
filter_expr_fallback <- function(.data, dots, by_quo, .preserve = FALSE, complement = FALSE,
                                 reason = NULL, inline = NULL) {
  if (is.null(reason)) {
    # (the dot filter_run() judged to fail, with the reason its parse recorded)
    reason <- fallback_failed_reason(
      dots, "a predicate has a form the GPU engine does not support (or is not a logical column expression)."
    )
  }
  filter_cpu_fallback(.data, dots, by_quo, .preserve, complement, reason = reason, inline = inline)
}

# F82: TRUE iff a filter() dot is a symbol or call that reads no column and
# needs no data mask, so it can be evaluated outside one. It is the test
# mutate_constant_value() makes, plus the shapes that reach the mask without
# naming a column by a call expr_reaches_mask() recognizes (`exists("x")`,
# `get()`, ...), which are not evaluated out here; an alias of one, or a user
# function making the call, is not recognized and is. (A typo in such a dot is
# caught after it fails, in filter_run().)
# @keywords internal
filter_dot_is_column_free <- function(quo, schema) {
  # (F110) a call that provably reads a column is not column-free
  if (quo_reads_column_fast(quo, schema)) {
    return(FALSE)
  }
  expr <- rlang::quo_get_expr(quo)
  (is.symbol(expr) || is.call(expr)) && !expr_has_column_ref(expr, schema) &&
    !expr_has_mask_only_call(expr) && !expr_reaches_mask(expr)
}

# The notice's reason for a dot that reads no column and fails when evaluated
# outside the data mask.
filter_free_dot_failed_reason <- paste0(
  "a predicate that reads no column fails outside the data mask ",
  "(it may use pick(), n() or cur_*() inside a function); dplyr evaluates it."
)

# F82: the dots handed to dplyr when filter_run() sends the call there after
# evaluating its column-free dots (`inline`: values recorded by expanded
# position and the expansion's provenance). A dot passed through unchanged whose
# recorded value is a logical without dim and of vctrs size 1 or `n` is replaced
# by that value, so dplyr does not evaluate it again; it stays UNNAMED (dplyr
# rejects a named filter input). Anything else stays as written.
# @keywords internal
filter_inline_evaluated <- function(dots, inline, n) {
  if (is.null(inline)) {
    return(dots)
  }
  prov <- inline$provenance
  for (d in seq_along(dots)) {
    idx <- which(prov$origin == d)
    if (length(idx) != 1L || !identical(prov$kind[idx], "dot") ||
        !(as.character(idx) %in% names(inline$consts))) {
      next
    }
    value <- inline$consts[[as.character(idx)]]$value
    size <- NA_integer_
    if (is.logical(value) && is.null(dim(value))) {
      size <- tryCatch(vctrs::vec_size(value), error = function(e) NA_integer_)
    }
    if (!size %in% c(1L, n)) {
      next
    }
    dots[[d]] <- rlang::new_quosure(value, rlang::quo_get_env(dots[[d]]))
  }
  dots
}

# Internal: TRUE iff a (non-window, non-constant) dot's IR is not a logical
# expression; filter_out() hands it to dplyr for its own error.
# @keywords internal
filter_ir_not_logical <- function(ir, schema, is_window) {
  if (is.null(ir) || is_window || ir_is_const(ir)) {
    return(FALSE)
  }
  # a call whose type function is the constant "BOOL8" (comparisons, &, |, !,
  # is.na(), ...) is logical whether its arguments type or fail to
  if (identical(.subset2(ir, "kind"), "call") &&
        identical(ir_call_registry[[.subset2(ir, "op")]]$type, type_bool8)) {
    return(FALSE)
  }
  !identical(unname(tryCatch(ir_infer_type(ir, schema), error = function(e) "BOOL8")), "BOOL8")
}

# Internal: TRUE iff `quo`'s raw expression references at least one symbol
# matching a current schema column name, anywhere in its tree (Phase 11,
# task L1). Used only to decide, for a dot `ir_parse_quo()` couldn't parse
# at all, whether it's safe to evaluate outside the data mask
# (safe only when the expression is genuinely
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
      any(syms$fns %in% top_n_mask_fns) || expr_reaches_mask(x)) {
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

# F110: FALSE only when top_n_rank_head() is FALSE for the root head of every
# dot (top_n_rank_head() is TRUE only for the symbol `top_n_rank` or a call head,
# `dplyr:::top_n_rank`): the two top_n screens run on every filter(). A dot that
# is not a quosure takes the full test.
top_n_rank_any <- function(dots) {
  for (i in seq_along(dots)) {
    dot <- .subset2(dots, i)
    if (!inherits(dot, "quosure")) {
      return(TRUE)
    }
    if (is.call(.subset2(dot, 2L))) {
      head <- .subset2(.subset2(dot, 2L), 1L)
      if (if (is.symbol(head)) identical(head, ir_sym_top_n_rank) else is.call(head)) {
        return(TRUE)
      }
    }
  }
  FALSE
}
ir_sym_top_n_rank <- as.name("top_n_rank")

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
  # (once here, once on the CPU). (top_n_rank_decline() has already sent a
  # `wt` that must not be folded to dplyr.)
  wt <- top_n_unwrap(expr[[3]], env)
  if (rlang::is_missing(wt$expr) || top_n_has_quosure(wt$expr)) return(NULL)
  # Parsing still folds the column-free parts of `wt`. One that draws random
  # numbers (`x * rnorm(1)`) would be drawn here, again for the rewritten dot,
  # and once more when filter() parses it, where dplyr draws once (per group):
  # leave such a `wt` to dplyr, with the RNG as it was.
  # (F88: probes; the dot's own parse records any cause)
  parsed <- top_n_rng_guard(function() {
    fallback_probe(tryCatch(ir_parse_expr(wt$expr, wt$env, schema), error = function(e) NULL))
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
  ir <- fallback_probe(tryCatch(ir_parse_quo(new, schema), error = function(e) NULL))
  if (is.null(ir)) return(NULL)
  new
}

# Why the `wt` of a top_n_rank(n, wt) dot must go to dplyr whole, or NULL.
# The rewrite to `min_rank() <= n` parses `wt` (and the rewritten dot parses it
# again, as filter_run() does), and each parse folds the column-free parts of
# `wt`, so a `wt` that dplyr runs once would run several times. Three kinds of
# `wt` are therefore never parsed: one that calls a function of
# `deferred_fallback_fns` (pick, c_across, cur_group, cur_group_rows, cur_data,
# cur_data_all), one with a `.env` subscript holding a call
# (top_n_env_call_subscript()), and one with a column-free call that
# impure_const_call_text() reports, checked in that order. All three checks see
# the quosures embedded in `wt` (`{{ }}`), each with its own env. Evaluates
# nothing.
# @param wt list(expr, env) of the dot's `wt`, as top_n_unwrap() returns it
# @param schema the schema of the table being filtered
# @keywords internal
top_n_wt_decline <- function(wt, schema) {
  if (rlang::is_missing(wt$expr)) return(NULL)
  deferred <- intersect(deferred_fallback_fns, all.names(wt$expr))
  if (length(deferred) > 0L) {
    return(sprintf("`wt` calls %s, which dplyr evaluates inside the data mask",
                   paste0(deferred[[1L]], "()")))
  }
  pronoun <- top_n_env_call_subscript(wt$expr)
  if (!is.null(pronoun)) {
    return(sprintf("`wt` reads `%s`, whose subscript dplyr evaluates inside the data mask",
                   substr(paste(deparse(pronoun), collapse = " "), 1L, 60L)))
  }
  impure_const_call_text(list(rlang::new_quosure(wt$expr, wt$env)), schema, grouped = FALSE)
}

# The first `.env[[s]]` in `expr` (through embedded quosures) whose subscript
# `s` contains a call, or NULL. The parser can't use such a subscript, and
# the notice for it names the pronoun.
# `.env[s]` is not a pronoun call, so the purity walk already reaches its
# subscript; `.env$s` takes a name. A string or a bare name as the subscript is
# not reported. An empty argument (`m[, 1]`, `switch(k, a = , x)`) is skipped.
# @keywords internal
top_n_env_call_subscript <- function(expr) {
  if (rlang::is_quosure(expr)) return(top_n_env_call_subscript(rlang::quo_get_expr(expr)))
  if (!is.call(expr)) return(NULL)
  # (an argument, not a local: an empty subscript, `.env[[]]`, is then a value)
  has_call <- function(x) {
    if (rlang::is_quosure(x)) return(has_call(rlang::quo_get_expr(x)))
    is.call(x)
  }
  if (is_env_pronoun_call(expr) && identical(expr[[1L]], as.name("[[")) &&
      has_call(expr[[3L]])) {
    return(expr)
  }
  for (part in as.list(expr)) {
    if (rlang::is_missing(part)) next
    found <- top_n_env_call_subscript(part)
    if (!is.null(found)) return(found)
  }
  NULL
}

# filter_run()'s screen for top_n()/top_frac(): the detail text of the first
# top_n_rank(n, wt) dot whose `wt` must be left to dplyr (top_n_wt_decline()),
# or NULL. It runs before any dot is parsed or evaluated, and the caller then
# hands dplyr the user's dots whole.
# @param dots filter() quosures
# @param .data the table being filtered; its schema is read only once a dot
#   turns out to be a top_n_rank() call
# @keywords internal
top_n_rank_decline <- function(dots, .data) {
  if (!top_n_rank_any(dots)) {
    return(NULL)
  }
  schema <- NULL
  for (i in seq_along(dots)) {
    expr <- rlang::quo_get_expr(dots[[i]])
    env <- rlang::quo_get_env(dots[[i]])
    if (!is.call(expr) || length(expr) != 3L || !is.null(names(expr)) ||
        !top_n_rank_head(expr[[1]], env)) next
    schema <- schema %||% current_schema(.data)
    detail <- top_n_wt_decline(top_n_unwrap(expr[[3]], env), schema)
    if (!is.null(detail)) return(detail)
  }
  NULL
}

# @param dots filter() quosures
# @param .data the table being filtered; its schema is read only once a dot
#   turns out to be a top_n_rank() call
# @return the dots with each eligible top_n_rank() dot replaced
# @keywords internal
rewrite_top_n_rank_dots <- function(dots, .data) {
  if (!top_n_rank_any(dots)) {
    return(dots)
  }
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

# Internal: the notice of a filter() call that applied a logical vector, once
# per call and the same on both routes: R evaluated `quo`, the first vector,
# and no table column was collected for it.
#
# @param quo The first dot of the mask
# @param rows The input's row count
# @keywords internal
filter_mask_notify <- function(quo, rows) {
  cuplyr_fallback_notify(
    "filter", rlang::quo_text(quo),
    reason = "this predicate reads no column, so R evaluates it once.",
    rows = rows, nothing_collected = TRUE
  )
}

# Internal: apply the one combined row mask of a filter() call whose dots are
# all constants (no native predicate, no window dot; filter_run()'s mask-only
# route). A table with no columns gets here only with a logical vector and no
# window or cur_group_id() dot: scalars alone take the constant-predicate route
# (which can't run with no columns), and that helper column makes a call mixed.
#
# The caller evaluated the dots (once, with no data mask), sized every vector
# against the table, combined them with `&` and turned NA into FALSE: `keep` is
# a logical vector of the table's row count with no NA. `.data` has no pending
# steps. A table with no columns has no cudf column to mask, so only its row
# count changes. Notifies via filter_mask_notify() (D4: silent by default).
#
# @param .data A tbl_gpu object with a known row count and nothing pending
# @param quo The first dot of the mask, named in the notification
# @param keep The combined mask: logical, no NA, one value per row
# @return A filtered tbl_gpu object
# @keywords internal
filter_eval_mask <- function(.data, quo, keep) {
  filter_mask_notify(quo, max(0, dim(.data)[1], na.rm = TRUE))

  if (length(.subset2(.data, "schema")$names) == 0L) {
    return(carry_frame_meta(new_tbl_gpu(
      ptr = .subset2(.data, "ptr"),
      schema = .subset2(.data, "schema"),
      groups = .subset2(.data, "groups"),
      exec_mode = .subset2(.data, "exec_mode"),
      nrow = sum(keep)
    ), .data))
  }

  new_ptr <- if (all(keep)) {
    wrap_gpu_call("filter_bool_all_true", gpu_filter_bool(.subset2(.data, "ptr"), TRUE))
  } else if (!any(keep)) {
    wrap_gpu_call("filter_bool_all_false", gpu_filter_bool(.subset2(.data, "ptr"), FALSE))
  } else {
    wrap_gpu_call("filter_mask", gpu_filter_mask(.subset2(.data, "ptr"), keep))
  }

  carry_frame_meta(new_tbl_gpu(
    ptr = new_ptr,
    schema = .subset2(.data, "schema"),
    groups = .subset2(.data, "groups"),
    group_drop = .subset2(.data, "group_drop") %||% TRUE,
    exec_mode = .subset2(.data, "exec_mode")
  ), .data)
}

#' Drop rows with missing values
#'
#' `stats::na.omit()` for a `tbl_gpu`: keeps the rows where no column is
#' `NA` (`NaN` counts as missing, as in `is.na()`; an `addNA()` factor level is
#' a value, not a missing one). It is an ordinary native [dplyr::filter()], so
#' the table keeps its execution mode, grouping and attributes, and lazy stays
#' lazy.
#'
#' Unlike `na.omit.data.frame()`, the result has no `na.action` attribute:
#' that attribute holds the row numbers dropped, which would need an extra
#' pass over the table and a transfer to the host just to be computed.
#' @param object A `tbl_gpu`.
#' @param ... Ignored (as in `na.omit.data.frame()`).
#' @return A `tbl_gpu` without the incomplete rows.
#' @export
#' @importFrom stats na.omit
na.omit.tbl_gpu <- function(object, ...) {
  nms <- current_schema(object)$names
  if (length(nms) == 0L) {
    return(object)
  }
  # is.na() reads a NaN as missing, as na.omit() does (F55).
  preds <- lapply(nms, function(nm) rlang::expr(!is.na(!!rlang::sym(nm))))
  names(preds) <- NULL
  dplyr::filter(object, !!!preds)
}
