#' Create or modify columns in a GPU table
#'
#' Adds new columns or modifies existing columns in a GPU table using
#' expressions, similar to `dplyr::mutate()`. All computations are
#' performed on the GPU for maximum performance.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Name-value pairs of expressions. The name gives the column name
#'   (new or existing), and the value is an expression involving existing
#'   columns and/or scalar values (see "Supported expressions" below).
#' @param .by Optional on-the-fly grouping columns (tidyselect), GPU-native
#'   (Phase 5, task W6): `mutate(df, y = mean(x), .by = g)` behaves
#'   identically to `df |> group_by(g) |> mutate(y = mean(x)) |> ungroup()`
#'   -- same window-function machinery, same row-order-preservation
#'   contract, but the *result* is always ungrouped (`.by` never leaves
#'   grouping metadata behind, unlike [dplyr::group_by()]). Supplying
#'   `.by` when `.data` is already grouped (via `group_by()`) is an error,
#'   matching dplyr exactly.
#' @param .keep One of `"all"` (default), `"used"`, `"unused"`, `"none"`;
#'   controls which of `.data`'s *other* columns (besides ones `...`
#'   creates or modifies, and group columns, both of which are always kept)
#'   survive in the output -- see "`.keep` semantics" below.
#' @param .before,.after Optionally, control where new columns should
#'   appear, using the same tidyselect syntax as [relocate.tbl_gpu()].
#'   Only genuinely *new* columns are repositioned; a modified existing
#'   column stays at its original position.
#'
#' @return A `tbl_gpu` object with the new or modified columns. If a column
#'   name already exists, it is replaced. New columns are appended (subject
#'   to `.before`/`.after`), and `.data`'s other columns are kept or dropped
#'   per `.keep`.
#'
#' @details
#' ## `.keep` semantics
#' Verified empirically against dplyr 1.2.1 (dplyr's own docs understate the
#' exact rule): the columns created or modified by `...` (i.e. every dot's
#' output column, whether new or an existing name being replaced) and every
#' group column are **always** kept, regardless of `.keep`. `.keep` only
#' controls which *other* (untouched, non-group) columns of `.data` survive
#' alongside them:
#' \itemize{
#'   \item `"all"` (default): every other column.
#'   \item `"used"`: other columns that are referenced as an *input* to any
#'     of `...`'s expressions (union across all dots).
#'   \item `"unused"`: other columns NOT referenced as an input to any
#'     expression -- the complement of `"used"`.
#'   \item `"none"`: no other columns at all.
#' }
#' Column order: `.data`'s original columns keep their original relative
#' order (modified columns don't move), and genuinely new columns are
#' appended at the end (or repositioned via `.before`/`.after`); the final
#' set is then filtered down according to `.keep`, preserving that same
#' relative order. This differs from [transmute()], whose output order is
#' the literal `...` dot order (see `transmute()`'s own docs).
#'
#' ## Supported expressions
#' `mutate()` parses its arguments through the same expression IR that
#' `filter()` uses (see `R/ir.R`), lowered to `cudf::compute_column()` under
#' the hood. The IR understands:
#' \itemize{
#'   \item Arithmetic: `+`, `-`, `*`, `/`, `^`, `%%`, `%/%`, unary `-`,
#'     arbitrarily nested (`(x + y) * z - 1`), and scalar-on-either-side
#'     (`mutate(y = 10 - x)`)
#'   \item Comparisons and logic: `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`,
#'     `|`, `!`, `xor()` -- these produce a `BOOL8` column, e.g.
#'     `mutate(is_big = x > 100)`
#'   \item `is.na()`, `between(x, lo, hi)`, `near(x, y, tol = )`
#'     (`abs(x - y) < tol`, matching `dplyr::near()`'s own definition
#'     exactly), `%in%` (including a bare `NA` on the right-hand side,
#'     e.g. `x %in% NA`, matching `NA %in% NA` being `TRUE` in base R --
#'     not the same as an *empty* set, which is always `FALSE`)
#'   \item Math functions: `sqrt()`, `log()` (including `log(x, base)`),
#'     `log2()`, `log10()`, `exp()`, `abs()`, `floor()`, `ceiling()`,
#'     `sin()`, `cos()`, `tan()`, `round(x)`/`round(x, digits)` (banker's
#'     rounding, matching R)
#'   \item `if_else(cond, yes, no, missing = NULL)` -- strict-but-promoting
#'     type unification across `yes`/`no`/`missing` (int/double/logical mix
#'     freely, like dplyr's vctrs-based common type; `STRING` only unifies
#'     with `STRING`). A `NA` condition produces `NA` (or `missing`, if
#'     supplied) in the output, matching dplyr exactly (this is NOT the same
#'     as treating a `NA` condition like `FALSE`)
#'   \item `case_when(cond1 ~ val1, cond2 ~ val2, ..., .default = val)` --
#'     first-match-wins; a `NA` condition is treated as no match (falls
#'     through to the next clause, or `.default`); unmatched rows become a
#'     typed `NA` when `.default` isn't supplied; the same value-type
#'     unification rule as `if_else()` applies across every `val_i` plus
#'     `.default`. `.ptype=`/`.size=`/`.unmatched=` are not yet supported.
#'   \item `coalesce(...)` -- first non-NA value per row, left to right,
#'     across any number of columns/scalars; same value-type unification
#'     rule as `if_else()`
#'   \item `na_if(x, y)` -- `x` with `NA` wherever `x == y`; the output
#'     always keeps `x`'s own type exactly (unlike `if_else()`/`coalesce()`,
#'     `y` is never promoted into the result, only compared against)
#'   \item `case_match(.x, old_values ~ new_value, ..., .default = val)` --
#'     matches `.x` by VALUE against each clause's (possibly multi-element)
#'     left-hand side, first match wins; `NA` on a clause's left-hand side
#'     matches `.x`'s own `NA` elements (base R's `NA %in% NA` is `TRUE`);
#'     unmatched rows become `.default` (typed `NA` if not supplied), with
#'     the same value-type unification rule as `case_when()` across every
#'     `new_value` plus `.default`. Deprecated upstream in dplyr 1.2 in
#'     favor of `recode_values()`/`replace_values()`, but still supported
#'     here. `.ptype=` is not yet supported.
#'   \item `recode_values(x, old_values ~ new_value, ..., default = val)` --
#'     the non-deprecated equivalent of `case_match()` above (identical
#'     semantics, `default` instead of `.default`). The `from=`/`to=`
#'     vector-lookup-table interface and `unmatched=`/`ptype=` are not yet
#'     supported.
#'   \item `replace_values(x, old_values ~ new_value, ...)` -- like
#'     `recode_values()`, but TYPE STABLE: the output always keeps `x`'s own
#'     type (every `new_value` is cast to it, never unified/promoted), and
#'     unmatched rows keep their original `x` value (there is no
#'     `default=`). The `from=`/`to=` interface is not yet supported.
#'   \item `replace_when(x, cond1 ~ val1, cond2 ~ val2, ...)` -- like
#'     `case_when()`, but boolean-condition clauses feeding the same
#'     type-stable-to-`x` / "unmatched rows keep `x`" behavior as
#'     `replace_values()` (a `NA` condition is treated as no match, same as
#'     `case_when()`)
#'   \item `when_all(..., na_rm = FALSE)` / `when_any(..., na_rm = FALSE)`
#'     -- elementwise `&`/`|` generalized to any number of logical inputs
#'     (`when_any(x, y, z)` is `x | y | z`). `na_rm = TRUE` treats `NA` as
#'     each operator's identity element (`FALSE` for `when_any()`, `TRUE`
#'     for `when_all()`) rather than propagating it -- e.g.
#'     `when_any(NA, NA, na_rm = TRUE)` is `FALSE`, not `NA`. `size=` is not
#'     yet supported.
#'   \item Plain column copies (`mutate(y = x)`)
#'   \item A later dot in the same `mutate()` call referencing an earlier
#'     dot's output column (`mutate(a = x + y, b = a * 2)`), INCLUDING one
#'     that's itself a window function's output (`mutate(a = cumsum(x), b =
#'     lag(a))`)
#'   \item The `.data`/`.env` pronouns, and ordinary R symbols (a bare name
#'     that matches a column resolves to that column; otherwise it's
#'     evaluated as an environment variable -- "columns shadow the
#'     environment")
#'   \item Window functions, GPU-native for both ungrouped and grouped
#'     (`group_by()`) tables, arbitrarily nested with any of the above
#'     (`mutate(y = x - lag(x))`, `mutate(z = lag(cumsum(x)))`,
#'     `mutate(centered = x - mean(x))`):
#'     \itemize{
#'       \item `row_number()` (sequential position) / `row_number(x)`
#'         (ranked position, ties broken by original position) --
#'         per-group under `group_by()`
#'       \item `lag(x, n = 1, default = NA, order_by = NULL)` / `lead(...)`
#'         -- row-order-preserving shift; `order_by=` (a bare column,
#'         `desc(column)`, or `-column`) re-frames the shift to follow that
#'         ordering instead of the table's/group's current row order
#'       \item `cumsum()`/`cummax()`/`cummin()`/`cumprod()` -- matching R's
#'         own NA-propagation (`cumsum(c(1, 2, NA, 4))` is `1, 3, NA, NA`:
#'         once an NA is seen, every later cumulative value in that group is
#'         NA too)
#'       \item `cummean()` (`cumsum(x) / row_number()`), `cumall()`/
#'         `cumany()` (tri-state-encoded `cummin()`/`cummax()`), and
#'         `consecutive_id()` (single-column only; a run-length group id
#'         that increments whenever the input value changes, treating
#'         adjacent `NA`s as unchanged) -- all built from the primitives
#'         above
#'       \item `min_rank()`/`dense_rank()`/`percent_rank()`/`cume_dist()` --
#'         ties/`NA` handling matches dplyr exactly (`NA` input -> `NA`
#'         rank; `percent_rank()` is `(min_rank(x) - 1) / (n_non_NA - 1)`,
#'         `NaN` for a group with exactly one non-`NA` value, matching
#'         dplyr's own `NaN`, not `NA`; `cume_dist()` is
#'         `rank(x, ties = "max") / n_non_NA`)
#'       \item `mean()`/`sum()`/`min()`/`max()`/`n()`/`sd()`/`var()` --
#'         one aggregate broadcast to every row of its group (or the whole
#'         table, ungrouped); `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()`
#'         match R's own `na.rm = FALSE` default (ANY `NA` in the group
#'         makes the WHOLE group's result `NA`); `n()` always counts every
#'         row regardless of `NA`s
#'       \item `first(x, order_by = NULL)` / `last(x, order_by = NULL)` /
#'         `nth(x, n, order_by = NULL)` -- the first/last/`n`th row of the
#'         group in its current row order, or in `order_by`'s order when
#'         supplied; negative `n` counts from the end (`nth(x, -1)` ==
#'         `last(x)`); an out-of-range `n` gives `NA`; `default=`/`na_rm=`
#'         are not yet supported
#'       \item `ntile(x = row_number(), n)` -- desugars into `row_number()`/
#'         `sum()`, so it's native as a consequence of those being native
#'       \item `order_by(order_by, call)` / `with_order(order_by, fun, x)`
#'         -- re-frame a directly-wrapped `row_number()`/`lag()`/`lead()`/
#'         `cumsum()`/`cummax()`/`cummin()`/`cumprod()`/rank-family call's
#'         ordering (matching dplyr exactly, including that both of these
#'         genuinely error, in dplyr itself, when wrapped around an
#'         aggregate like `mean()`/`first()`/`n()` -- cuplyr matches that by
#'         rejecting the same shape as unsupported rather than silently
#'         ignoring the requested ordering)
#'     }
#'     Every one of these is evaluated within each `group_by()` group when
#'     `.data` is grouped (matching `dplyr`'s own grouped-mutate semantics:
#'     row order is always preserved in the output, unlike grouped
#'     `slice()`).
#' }
#'
#' ## Column replacement behavior
#' When the output column name matches an existing column, the existing
#' column is replaced in-place (preserving column order). For example,
#' `mutate(x = x + 1)` will modify `x` rather than creating a duplicate.
#'
#' ## Result type (type promotion)
#' The output column's GPU type is inferred from the operation and its
#' inputs (column type(s) and, for a column-scalar expression, the R
#' literal's own type), applied in this order:
#' \itemize{
#'   \item A plain column copy (`mutate(y = x)`) preserves the source
#'     column's type exactly.
#'   \item Arithmetic on `STRING`, factor (`DICTIONARY32`), or `Date`/
#'     `POSIXct` (`TIMESTAMP_*`) columns is an error: cuplyr does not
#'     support arithmetic on those types.
#'   \item `/` and `^` always promote to `FLOAT64` (matching R/dplyr:
#'     `1L / 2L` and `2L ^ 2L` are both doubles).
#'   \item `+`, `-`, `*`, `%%`, `%/%` preserve integer types: the result is
#'     `INT32` when every input column and R literal involved is integer
#'     or logical (`TRUE + TRUE` is `2L`, matching R). Note this means
#'     `INT32` arithmetic can silently wrap around on overflow (as in
#'     plain R/C++ integer arithmetic) rather than promoting to a wider
#'     type. If any input is `FLOAT64`/`FLOAT32`, the result is
#'     `FLOAT64`; otherwise if any input is `INT64`, the result is
#'     `INT64`. A bare R numeric literal (e.g. the `1` in
#'     `mutate(y = int_col + 1)`) is a *double* in R, so it promotes the
#'     result to `FLOAT64` -- write `1L` to keep an integer result.
#'   \item Comparisons, logic, `is.na()`, `between()`, `%in%` all produce
#'     `BOOL8`.
#'   \item `sqrt()`, `log()`, `exp()`, `floor()`, `ceiling()`, `sin()`,
#'     `cos()`, `tan()` always promote to `FLOAT64` (matching R, e.g.
#'     `floor(1L)` is a double) -- as does `round()`, for the same reason
#'     (`typeof(round(1L))` is also a double in R, unlike `abs()` below).
#'   \item `abs()` preserves the argument's type, except a logical column,
#'     which promotes to `INT32` (matching R: `typeof(abs(TRUE))` is
#'     `"integer"`, not `"logical"`).
#'   \item A bare `NA` (untyped) is `BOOL8` when it's an entire mutate
#'     expression by itself (`mutate(y = NA)`, matching `typeof(NA)`); as an
#'     operand inside an arithmetic expression (`mutate(y = int_col + NA)`)
#'     it's a type no-op instead, adopting the other operand's type
#'     (matching R: `1L + NA` is an integer, not a double). A typed NA
#'     literal (`NA_real_`, `NA_integer_`, ...) always keeps its own type.
#' }
#'
#' ## Not yet supported
#' \itemize{
#'   \item `order_by()`/`with_order()` wrapping a general expression (only a
#'     bare column, `desc(column)`, or `-column` is understood as the
#'     ordering argument -- `order_by(a + b, cumsum(x))` is not yet
#'     supported), or wrapping anything other than a SINGLE directly-nested
#'     window call (`order_by(y, cumsum(x) + lag(x))` is not yet supported;
#'     `order_by(y, lag(cumsum(x)))` IS supported, since `lag()` is the
#'     single directly-wrapped call there)
#'   \item Multi-column `consecutive_id(a, b, ...)` -- single-column only
#'   \item Rolling/sliding-window functions (`slider::slide_*()` and
#'     similar) -- planned for a later phase
#'   \item String manipulation (concatenation, case conversion, substr,
#'     regex, ...) -- only string *comparison* is supported
#'   \item `case_when()`'s/`case_match()`'s `.ptype=`/`.size=`/`.unmatched=`
#'     arguments, `recode_values()`'s/`replace_values()`'s `from=`/`to=`
#'     vector-lookup-table interface and `unmatched=`/`ptype=`, and
#'     `when_all()`'s/`when_any()`'s `size=` -- planned for a later phase
#' }
#' An expression shape the IR doesn't recognize at all (no window function,
#' no aggregate, not one of the arithmetic/comparison/string-comparison
#' shapes listed above) is a hard error (unlike `filter()`, which falls back
#' to CPU evaluation for some unrecognized shapes) -- a general CPU fallback
#' for `mutate()` covering every unrecognized shape (not just window/
#' aggregate ones) is planned for a later phase.
#'
#' ## Performance
#' GPU arithmetic operations are highly vectorized and can process
#' billions of elements per second. Memory bandwidth is typically
#' the limiting factor, not compute.
#'
#' @seealso
#' \code{\link{filter.tbl_gpu}} for filtering rows,
#' \code{\link{select.tbl_gpu}} for selecting columns,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr mutate
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Add a new column
#'   result <- gpu_mtcars |>
#'     mutate(kpl = mpg * 0.425) |>
#'     collect()
#'
#'   # Modify an existing column
#'   adjusted <- gpu_mtcars |>
#'     mutate(mpg = mpg + 5) |>
#'     collect()
#'
#'   # Combine two columns
#'   gpu_cars <- tbl_gpu(cars)
#'   result <- gpu_cars |>
#'     mutate(ratio = dist / speed) |>
#'     collect()
#'
#'   # Chain multiple mutations
#'   result <- gpu_mtcars |>
#'     mutate(power_weight = hp / wt) |>
#'     mutate(efficiency = mpg * power_weight) |>
#'     collect()
#' }
mutate.tbl_gpu <- function(.data, ..., .by = NULL,
                           .keep = c("all", "used", "unused", "none"),
                           .before = NULL, .after = NULL) {
  .keep <- match.arg(.keep)

  dots <- rlang::enquos(...)
  by_quo <- rlang::enquo(.by)
  before_quo <- rlang::enquo(.before)
  after_quo <- rlang::enquo(.after)

  # Phase 5, task W6: `.by=` resolution. resolve_by() (R/execute.R) errors
  # immediately if `.data` is already grouped (matching dplyr's own
  # "Can't supply `.by`..." message verbatim) -- checked up front, before
  # any dots-related early return, since dplyr raises this error even for
  # a zero-dot call (verified empirically). `verb_groups` is what actually
  # drives the window-function machinery below: `by_cols` when `.by=` was
  # supplied, `.data$groups` otherwise (identical to `group_by()`-driven
  # mutate). The result is ALWAYS ungrouped when `.by=` was used -- this
  # falls out automatically, with no special-casing needed, because
  # resolve_by() already guarantees `.data` itself is ungrouped in that
  # case, and push_op()'s propagate_groups() only ever propagates
  # `.data$groups` (never `verb_groups`) into the result.
  by_given <- !rlang::quo_is_null(by_quo)
  by_cols <- resolve_by(.data, by_quo, ".by")
  verb_groups <- if (by_given) by_cols else .data$groups

  # Narrow aggregate broadcast: `mutate(x, name = n())` / `mutate(x, name =
  # sum(expr))` -- a (per-group, if `x` is grouped; whole-table otherwise)
  # aggregate broadcast to every row, keeping all rows (unlike summarise()).
  # This narrow left-join-based path predates the general window-function
  # machinery (Phase 5 W1-W4) and is now a strict subset of it: `n()`/
  # `sum()` are themselves "agg"-kind window calls (R/ir.R's registry), so
  # `parse_mutate_dots()` already flags this exact shape's IR as
  # `has_window`, routing it through `mutate_window()` below. To avoid
  # double-pathing the same shape, this narrow branch is skipped entirely
  # when `.by=` was supplied -- `mutate_window()` already threads
  # `verb_groups` correctly for that case (see below) -- and fires only for
  # the pre-existing group_by()-or-ungrouped case, unchanged from before.
  # It was added so that `dplyr::add_tally()` and our own
  # `add_count.tbl_gpu()` (R/count.R, both of which call straight through to
  # `mutate()` with exactly this shape, see dplyr's `tally_n()`) work
  # without a CPU fallback; it also incidentally covers the same shape for
  # a plain, directly-written `mutate(x, name = n())`/`mutate(x, name =
  # sum(y))` call (grouped or not), which is otherwise a hard error (see
  # this file's "Not yet supported" roxygen section) -- a strict parity
  # improvement, not scope creep, since it's the exact same mechanism.
  # See R/count.R and `grouped_aggregate_mutate()` below for the empirical
  # verification that the grouped case's `left_join()` broadcast is safe
  # (NA-key matching, row-order preservation); the ungrouped case broadcasts
  # a single collected scalar instead (see `grouped_aggregate_mutate()`).
  if (!by_given && length(dots) == 1 &&
      identical(.keep, "all") &&
      rlang::quo_is_null(before_quo) && rlang::quo_is_null(after_quo)) {
    grouped_agg <- match_grouped_aggregate_dot(dots[[1]])
    if (!is.null(grouped_agg)) {
      dot_name <- names(dots)
      if (is.null(dot_name) || is.na(dot_name) || identical(dot_name, "")) {
        dot_name <- rlang::quo_text(dots[[1]])
      }
      return(grouped_aggregate_mutate(.data, dot_name, grouped_agg))
    }
  }

  if (!rlang::quo_is_null(before_quo) && !rlang::quo_is_null(after_quo)) {
    stop("Can't supply both `.before` and `.after`.", call. = FALSE)
  }

  if (length(dots) == 0 && identical(.keep, "all") &&
      rlang::quo_is_null(before_quo) && rlang::quo_is_null(after_quo)) {
    return(.data)
  }

  orig_schema <- current_schema(.data)
  orig_names <- orig_schema$names
  orig_groups <- verb_groups

  expressions <- parse_mutate_dots(dots, orig_schema, "mutate")

  # Phase 5, task W3: any dot containing a window-function call (row_number(),
  # lag()/lead(), cumsum()/cummax()/cummin()/cumprod(), and their R desugars
  # cummean()/cumall()/cumany()/consecutive_id()) routes the WHOLE mutate()
  # call through mutate_window() instead of the plain single-ast_mutate path
  # below -- see that function's own docs and
  # scratchpad/phase5_window_design.md section 1.3.
  if (isTRUE(attr(expressions, "has_window"))) {
    return(mutate_window(
      .data, dots, expressions, orig_schema, orig_names, orig_groups,
      .keep, before_quo, after_quo, by_quo
    ))
  }

  # One ast_mutate node carries every expression from every dot -- lowered
  # to a single gpu_mutate_expr() call by lower_mutate() (R/lower.R), and a
  # fusion target for fuse_mutates()/toposort_expressions() (R/optimizer.R)
  # across consecutive mutate() calls in lazy mode. Skipped entirely when
  # there are no expressions (e.g. `mutate(gdf, .keep = "none")`) -- an
  # ast_mutate with zero expressions would be a pure no-op anyway (see
  # R/ast.R::is_barrier()'s "empty mutate nodes are eliminated" comment on
  # the optimizer side), so there's nothing to gain by building one here.
  result <- .data
  if (length(expressions) > 0) {
    result <- push_op(.data, ast_mutate(input_node(.data), expressions))
  }

  select_node <- mutate_keep_select_node(
    result, orig_names, orig_groups, expressions, .keep, before_quo, after_quo
  )
  if (is.null(select_node)) {
    return(result)
  }

  push_op(result, select_node)
}

# Internal: window-bearing mutate() dispatch (Phase 5, task W3)
#
# Called from mutate.tbl_gpu() whenever at least one dot's parsed IR
# contains a window-function call. Implements
# scratchpad/phase5_window_design.md section 1.3's AST emission sequence
# (pre-mutate -> ast_window per stage -> the verb's own node -> trailing
# select dropping temps), generalized to interleave per BATCH (see
# window_batches()/plan_mutate_window_batches() below) rather than emitting
# every stage before a single combined final ast_mutate: a later batch's
# window-call arguments may reference an EARLIER batch's own dot output by
# name (e.g. `mutate(a = cumsum(x), b = lag(a))` -- this is exactly what
# forces "b" into its own batch, per window_batches()'s chunking rule), which
# requires "a" to already be a REAL materialized column (not just a schema
# entry) by the time "b"'s own window stage runs -- so each batch's own dots
# must be materialized via their own ast_mutate node before the NEXT batch's
# stages are pushed. Section 1.3's pseudocode is the correct picture for the
# (overwhelmingly common) single-batch case; interleaving generalizes it
# correctly for the cross-batch case without changing behavior when there is
# only one batch.
#
# Two-pass structure (plan, then emit) is mandatory, not just tidy: emitting
# AST nodes via push_op() executes immediately in eager mode, so we cannot
# discover mid-emission that some spec isn't lowerable yet and "undo" partial
# GPU work -- every spec across every batch must be validated BEFORE any
# push_op() call (see plan_mutate_window_batches()/window_spec_lowerable(),
# R/lower.R). This is the "detect at plan time... never half-lower"
# requirement from the design doc.
#
# @param .data The original tbl_gpu (BEFORE this mutate() call)
# @param dots The original (already rlang::enquos()'d) dots -- needed
#   verbatim for the CPU fallback path, which re-runs the real
#   dplyr::mutate() call
# @param expressions parse_mutate_dots()'s result: one make_mutate_expr()
#   struct per dot, built from the RAW (window-containing) per-dot IR,
#   parsed against `orig_schema` with the usual left-to-right dot-visibility
#   schema threading. Used here only as the source of per-dot raw IR (via
#   `$ir`) -- and, unchanged, handed straight to mutate_keep_select_node()
#   for `.keep`/`.before`/`.after` accounting, exactly as the non-window
#   path does: `$output_col`/`$input_cols` (dot output names / the USER's
#   own referenced columns, e.g. "x" for `cumsum(x)`) are identical whether
#   or not the dot happens to contain a window call, so no separate
#   accounting is needed for `.keep="used"` to correctly count "x" as used
#   even though the actual GPU computation reads a `..win*..` temp column,
#   never "x" directly, once decomposed
# @param orig_schema,orig_names `.data`'s schema/names before this call
# @param orig_groups The EFFECTIVE group columns to window over -- Phase 5
#   task W6's `verb_groups`: `by_cols` when `.by=` was supplied,
#   `.data$groups` otherwise (mutate.tbl_gpu() already resolved this before
#   calling here). Threaded straight into every ast_window() stage's
#   group_cols argument below, and into mutate_keep_select_node()'s
#   always-keep set (an on-the-fly `.by=` column is always kept in the
#   output, exactly like a `group_by()` column, verified empirically)
# @param keep,before_quo,after_quo As in mutate.tbl_gpu()
# @param by_quo The `.by=` quosure, as captured by mutate.tbl_gpu()
#   (`rlang::enquo(.by)`) -- only used if this call's plan turns out
#   unlowerable and we must re-run the original `dplyr::mutate()` call on
#   the CPU (mutate_window_fallback()), so that fallback re-run honors
#   `.by=` too
# @return A new tbl_gpu
# @keywords internal
mutate_window <- function(.data, dots, expressions, orig_schema, orig_names, orig_groups,
                           keep, before_quo, after_quo, by_quo) {
  exprs_raw <- stats::setNames(
    lapply(expressions, `[[`, "ir"),
    vapply(expressions, `[[`, character(1), "output_col")
  )

  plan <- tryCatch(
    plan_mutate_window_batches(exprs_raw, orig_schema),
    error = function(e) NULL
  )

  all_lowerable <- !is.null(plan) &&
    all(vapply(plan$all_specs, window_spec_lowerable, logical(1)))

  if (!all_lowerable) {
    return(mutate_window_fallback(.data, dots, keep, before_quo, after_quo, by_quo))
  }

  result <- .data
  for (batch_plan in plan$batch_plans) {
    for (stage in batch_plan$stages) {
      if (length(stage$pre) > 0) {
        result <- push_op(result, ast_mutate(input_node(result), stage$pre))
      }
      result <- push_op(result, ast_window(input_node(result), stage$specs, orig_groups))
    }
    result <- push_op(result, ast_mutate(input_node(result), batch_plan$expressions))
  }

  select_node <- mutate_keep_select_node(
    result, orig_names, orig_groups, expressions, keep, before_quo, after_quo
  )
  if (is.null(select_node)) {
    return(result)
  }

  push_op(result, select_node)
}

# Internal: pure planning pass for mutate_window() -- no push_op()/GPU work
#
# Splits `exprs` into window_batches() and, for each batch in order, drives
# decompose_window_group() (R/window.R) to a fixed point, then builds that
# batch's OWN final expressions (make_mutate_expr() per dot, from the
# batch's window-free "post" IR) threading the schema across the batch's
# dots exactly as mutate()'s own non-window loop does (parse_mutate_dots()/
# update_schema_for_expr()) -- so a later dot in the SAME batch that plainly
# references an earlier dot's output (no window call of its own, e.g.
# `mutate(a = cumsum(x), c = a * 2)`) resolves correctly.
#
# `..win<k>..`/`..winarg<k>..` temp name counters (`win_n`/`winarg_n`) are
# threaded across batches (not reset per batch) so names stay sequential
# and collision-free across the whole plan, mirroring plan_window_stages()'s
# own single-call threading.
#
# @param exprs A named list of per-dot RAW (window-containing) IR, in
#   verb-call order (see window_batches()'s own docs for the naming
#   requirement)
# @param schema `.data`'s schema before any dot of this mutate() call
# @return `list(batch_plans, all_specs)`: `batch_plans` is a list (one per
#   batch, in order) of `list(stages = <window_spec()-bearing stage list,
#   from decompose_window_group()>, expressions = <list of
#   make_mutate_expr() structs, one per dot in this batch>)`; `all_specs` is
#   the flat list of every window_spec() produced across every stage of
#   every batch, for window_spec_lowerable() validation
# @keywords internal
plan_mutate_window_batches <- function(exprs, schema) {
  batches <- window_batches(exprs)

  schema_cursor <- schema
  win_n <- 0L
  winarg_n <- 0L
  batch_plans <- vector("list", length(batches))
  all_specs <- list()

  for (bi in seq_along(batches)) {
    batch_idx <- batches[[bi]]
    batch_exprs <- exprs[batch_idx]

    decomposed <- decompose_window_group(batch_exprs, schema_cursor, win_n, winarg_n)
    win_n <- decomposed$win_n
    winarg_n <- decomposed$winarg_n

    for (stage in decomposed$stages) {
      all_specs <- c(all_specs, stage$specs)
    }

    batch_schema <- decomposed$schema
    batch_expressions <- vector("list", length(batch_idx))
    for (k in seq_along(batch_idx)) {
      nm <- names(exprs)[batch_idx[k]]
      step <- make_mutate_expr(nm, decomposed$post[[k]], batch_schema)
      batch_expressions[[k]] <- step
      batch_schema <- update_schema_for_expr(batch_schema, step)
    }

    batch_plans[[bi]] <- list(stages = decomposed$stages, expressions = batch_expressions)
    schema_cursor <- batch_schema
  }

  list(batch_plans = batch_plans, all_specs = all_specs)
}

# Internal: CPU fallback for a window-bearing mutate() call whose plan isn't
# (yet) fully lowerable. As of Phase 5 task W4, every window kind (shift/
# rank/scan/agg) has C++ support (see window_spec_lowerable(), R/lower.R),
# so this is only reachable for a genuinely NEW, not-yet-implemented window
# kind added by some FUTURE phase -- there is currently no dplyr syntax that
# reaches it (an order_by()/with_order()-wrapped agg-kind call, the other
# way a window spec used to become unlowerable, is now rejected at PARSE
# time instead -- see apply_order_override(), R/ir.R -- which is a hard
# mutate() error, not this fallback). Kept for forward-compatibility rather
# than deleted. Re-injects `.by=` (Phase 5 task W6) so the CPU re-run
# honors on-the-fly grouping too, exactly as the real call would have.
#
# @param .data The original tbl_gpu
# @param dots The original (already rlang::enquos()'d) dots
# @param keep,before_quo,after_quo As in mutate.tbl_gpu()
# @param by_quo The `.by=` quosure, as captured by mutate.tbl_gpu()
# @return A new tbl_gpu (via gpu_fallback(), which also emits the
#   cuplyr_fallback_notify() notification)
# @keywords internal
mutate_window_fallback <- function(.data, dots, keep, before_quo, after_quo, by_quo) {
  gpu_fallback("mutate", .data, function(tbl) {
    rlang::inject(dplyr::mutate(
      tbl, !!!dots, .by = !!by_quo, .keep = keep,
      .before = !!before_quo, .after = !!after_quo
    ))
  })
}

# Parse mutate()/transmute()'s `...` dots into a list of mutate expression
# structures (see make_mutate_expr(), R/ast.R). Shared by both verbs: they
# differ only in what they do with the finished `expressions` list
# afterwards (mutate.tbl_gpu() computes a `.keep`/`.before`/`.after`
# column set; transmute.tbl_gpu() always keeps just the dot outputs, in dot
# order, plus any un-dotted group columns prepended).
#
# @param dots A list of quosures, as returned by rlang::enquos(...)
#   (NOT yet auto-named -- this function names unnamed dots itself)
# @param schema The schema to parse/type expressions against (before this
#   call's dots -- i.e. .data's current schema)
# @param verb Character, the verb name for auto_name_dots()'s warning and
#   the unrecognized-expression error message
# @param warn_unnamed Logical, `TRUE` to warn on unnamed dots (mutate()'s
#   existing behavior, see auto_name_dots()), `FALSE` to silently name them
#   from their deparsed text instead -- transmute() passes `FALSE`, matching
#   real `dplyr::transmute()` (verified empirically: dplyr never warns for
#   an unnamed transmute dot, not even a bare column reference, which is
#   transmute's single most common idiom, e.g. `transmute(df, a = x + 1, y)`)
# @return A list of expression structures, one per dot. Carries an extra
#   `"has_window"` attribute (Phase 5, task W3): `TRUE` iff any dot's parsed
#   IR contains a window-function call anywhere (`ir_has_window()`) --
#   `mutate.tbl_gpu()` reads this to decide whether to route the whole call
#   through `mutate_window()` instead of pushing `expressions` straight into
#   a single `ast_mutate()` node (which would later hit `ir_bind()`'s window
#   guard at lowering time). `transmute.tbl_gpu()` ignores the attribute --
#   window-function support inside `transmute()` is not yet wired (W3 scope
#   is `mutate()` only, per `scratchpad/phase5_window_design.md` section 8's
#   W3 row).
# @keywords internal
parse_mutate_dots <- function(dots, schema, verb, warn_unnamed = TRUE) {
  dots <- if (warn_unnamed) auto_name_dots(dots, verb) else name_dots_silently(dots)

  expressions <- list()
  has_window <- FALSE

  for (i in seq_along(dots)) {
    new_name <- names(dots)[i]
    expr <- dots[[i]]

    ir <- ir_parse_quo(expr, schema)

    if (is.null(ir)) {
      stop(
        verb, "() only supports column copies, arithmetic (+, -, *, /, ^, ",
        "%%, %/%), comparisons (==, !=, <, <=, >, >=), logical operators ",
        "(&, |, !, xor), is.na(), between(), near(), %in%, sqrt()/log()/log2()/",
        "log10()/exp()/abs()/floor()/ceiling()/sin()/cos()/tan()/round(), ",
        "if_else()/case_when()/coalesce()/na_if()/case_match()/recode_values()/",
        "replace_values()/replace_when()/when_all()/when_any(), or window ",
        "functions (row_number()/lag()/lead()/cumsum()/cummax()/cummin()/",
        "cumprod()/cummean()/cumall()/cumany()/consecutive_id(), ",
        "min_rank()/dense_rank()/percent_rank()/cume_dist()/ntile(), ",
        "mean()/sum()/min()/max()/n()/sd()/var()/first()/last()/nth(), ",
        "order_by()/with_order()) ",
        "(some named arguments of these are not yet supported -- see ",
        "?mutate.tbl_gpu).\n",
        "Expression: ", rlang::quo_text(expr),
        call. = FALSE
      )
    }

    if (ir_has_window(ir)) {
      has_window <- TRUE
    }

    step <- make_mutate_expr(new_name, ir, schema)
    expressions <- c(expressions, list(step))

    # Update the running schema so subsequent dots see this dot's output.
    schema <- update_schema_for_expr(schema, step)
  }

  attr(expressions, "has_window") <- has_window
  expressions
}

# Fill in unnamed dots' names from their deparsed expression text, with no
# warning -- the silent counterpart to auto_name_dots() (R/execute.R), used
# by parse_mutate_dots(warn_unnamed = FALSE) for transmute().
# @keywords internal
name_dots_silently <- function(dots) {
  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }

  for (i in seq_along(dots)) {
    if (is.na(dot_names[i]) || dot_names[i] == "") {
      dot_names[i] <- rlang::quo_text(dots[[i]])
    }
  }

  names(dots) <- dot_names
  dots
}

# Build the ast_select node (if any) implementing mutate()'s `.keep`/
# `.before`/`.after` controls, per the empirically-verified rule (see
# mutate.tbl_gpu()'s roxygen "`.keep` semantics" section):
#
#   - Columns created or modified by any dot (`expr$output_col`, whether a
#     brand new name or an existing one being replaced), and every group
#     column, are ALWAYS kept.
#   - `.keep` controls which *other* original columns additionally survive
#     ("all" = all of them, "used" = ones referenced as an input to any
#     dot's expression, "unused" = the complement of "used", "none" =
#     none).
#   - Column order: original columns keep their original relative
#     position; genuinely new columns are appended at the end (or moved
#     via `.before`/`.after`, using the same tidyselect machinery as
#     relocate.tbl_gpu()); the final kept set is filtered down from that
#     ordered list, preserving relative order.
#
# @param result The tbl_gpu AFTER the mutate node has been pushed (or
#   .data unchanged if there were no expressions)
# @param orig_names Character vector, .data's column names BEFORE this
#   mutate() call
# @param orig_groups Character vector, .data's group columns
# @param expressions The finished expressions list (see parse_mutate_dots())
# @param keep One of "all"/"used"/"unused"/"none"
# @param before_quo,after_quo Quosures for `.before`/`.after` (already
#   checked not to be BOTH non-NULL)
# @return An ast_select node to stack on `result` via push_op(), or `NULL`
#   if no select node is needed (the "all", no `.before`/`.after` case is
#   always a pure identity and is skipped to avoid an unnecessary AST node)
# @keywords internal
mutate_keep_select_node <- function(result, orig_names, orig_groups,
                                     expressions, keep, before_quo, after_quo) {
  touched_cols <- unique(vapply(expressions, `[[`, character(1), "output_col"))
  new_cols <- touched_cols[!touched_cols %in% orig_names]
  existing_touched <- touched_cols[touched_cols %in% orig_names]
  input_cols_union <- unique(unlist(lapply(expressions, `[[`, "input_cols")))

  mode_extra <- switch(keep,
    all = orig_names,
    used = input_cols_union,
    unused = setdiff(orig_names, input_cols_union),
    none = character()
  )

  always_kept <- union(orig_groups, existing_touched)
  kept_orig_names <- orig_names[orig_names %in% union(always_kept, mode_extra)]

  full_names <- current_schema(result)$names

  if (length(new_cols) > 0 &&
      (!rlang::quo_is_null(before_quo) || !rlang::quo_is_null(after_quo))) {
    name_vec <- stats::setNames(full_names, full_names)
    loc <- tidyselect::eval_relocate(
      expr = rlang::expr(c(!!!rlang::syms(new_cols))), data = name_vec,
      before = before_quo, after = after_quo,
      before_arg = ".before", after_arg = ".after"
    )
    full_names <- full_names[loc]
  }

  final_order <- full_names[full_names %in% union(kept_orig_names, new_cols)]

  if (identical(final_order, current_schema(result)$names)) {
    return(NULL)
  }

  ast_select(input_node(result), final_order)
}

# Update the running schema after adding a mutate expression: a replacement
# keeps its existing position and just gets a new type; a new output column
# is appended. Shared by mutate.tbl_gpu()'s per-dot loop and (independently)
# by infer_schema.ast_mutate() (R/ast.R), which recomputes the same result
# from a finished `expressions` list.
#
# @param schema List with names/types
# @param expr A mutate expression structure (see make_mutate_expr())
# @return Updated schema
# @keywords internal
update_schema_for_expr <- function(schema, expr) {
  existing_idx <- match(expr$output_col, schema$names)

  if (!is.na(existing_idx)) {
    schema$types[existing_idx] <- expr$output_type
  } else {
    schema$names <- c(schema$names, expr$output_col)
    schema$types <- c(schema$types, expr$output_type)
  }

  schema
}

#' Create a new GPU table with only the specified columns
#'
#' Computes new columns (or replaces existing ones) from expressions, like
#' [mutate.tbl_gpu()], but keeps ONLY the columns named/created by `...`
#' (plus group columns), similar to `dplyr::transmute()`. `transmute()` is
#' superseded in dplyr (in favor of `mutate(.keep = "none")`) but still
#' fully supported, and its column-ordering rule is subtly different from
#' `mutate(.keep = "none")`'s -- see "Details" below.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Name-value pairs of expressions, exactly as for
#'   [mutate.tbl_gpu()]. A bare column name (`transmute(df, x)`) keeps that
#'   column.
#'
#' @return A `tbl_gpu` object with only the columns `...` names/creates,
#'   plus any group columns not otherwise mentioned.
#'
#' @details
#' Verified empirically against dplyr 1.2.1: unlike `mutate()` (whose output
#' order keeps existing columns in their original relative position and
#' appends new ones at the end, see [mutate.tbl_gpu()]'s docs),
#' `transmute()`'s output order is simply the literal `...` dot order --
#' whether a dot is a bare existing column or a brand-new expression. Group
#' columns NOT explicitly mentioned in `...` are prepended at the very
#' front (in `group_vars()` order); a group column that IS explicitly
#' mentioned in `...` takes its dot position instead of being prepended.
#'
#' @export
#' @importFrom dplyr transmute
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   result <- gpu_mtcars |>
#'     transmute(kpl = mpg * 0.425, cyl) |>
#'     collect()
#' }
transmute.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)

  schema <- current_schema(.data)
  expressions <- parse_mutate_dots(dots, schema, "transmute", warn_unnamed = FALSE)

  result <- .data
  if (length(expressions) > 0) {
    result <- push_op(.data, ast_mutate(input_node(.data), expressions))
  }

  # Order: literal dot order (whether bare-column or newly computed), with
  # any un-dotted group column prepended -- see the roxygen "Details" above
  # for how this differs from mutate(.keep = "none")'s ordering rule.
  dot_output_cols <- unique(vapply(expressions, `[[`, character(1), "output_col"))
  missing_groups <- setdiff(.data$groups, dot_output_cols)
  final_order <- c(missing_groups, dot_output_cols)

  push_op(result, ast_select(input_node(result), final_order))
}

# Internal: Detect the narrow "grouped aggregate broadcast" mutate() shape
#
# Matches a single dot's raw (unevaluated) expression against exactly two
# shapes, after stripping one leading `pkg::` qualifier from the call head
# (same convention as `decompose_agg_call()`, R/summarise.R):
#   - `n()` (zero arguments)
#   - `sum(<expr>)` or `sum(<expr>, na.rm = TRUE)` (one or two arguments;
#     a two-arg call is only matched when the second argument is exactly
#     `na.rm = TRUE` -- our own aggregation already always excludes NA, see
#     R/summarise.R's "NA handling" docs, so `na.rm = TRUE` is accepted and
#     discarded, while `na.rm = FALSE` -- a genuinely different semantic we
#     don't implement -- correctly fails to match and falls through to the
#     ordinary mutate() IR path, which will itself raise the "only supports
#     column copies, arithmetic, ..." error for a bare `sum()` call)
#
# This is deliberately narrow: no other aggregate functions (mean(), min(),
# max(), ...) and no combining an aggregate with other operations
# (`n() + 1`, `sum(x) / n()`) are recognized -- see the call site in
# mutate.tbl_gpu() for why (full grouped-mutate/window-function support is
# Phase 5 material; this exists only to let `dplyr::add_tally()`'s and
# `add_count.tbl_gpu()`'s internal `mutate(x, name := n()-or-sum(wt))` call
# succeed without a CPU fallback).
#
# @param quo A quosure (one mutate() dot)
# @return `list(kind = "n", arg = NULL)`, `list(kind = "sum", arg = <lang>)`,
#   or `NULL` if the shape doesn't match
# @keywords internal
match_grouped_aggregate_dot <- function(quo) {
  expr <- rlang::quo_get_expr(quo)
  if (!is.call(expr)) {
    return(NULL)
  }

  head <- expr[[1]]
  if (is.call(head) && length(head) == 3 && identical(head[[1]], as.name("::"))) {
    head <- head[[3]]
  }
  if (!is.symbol(head)) {
    return(NULL)
  }

  fn <- as.character(head)
  n_args <- length(expr) - 1L

  if (identical(fn, "n") && n_args == 0) {
    return(list(kind = "n", arg = NULL))
  }

  if (identical(fn, "sum") && n_args %in% c(1L, 2L)) {
    if (n_args == 2L) {
      arg_name <- names(expr)[3]
      is_na_rm_true <- (is.null(arg_name) || is.na(arg_name) || identical(arg_name, "na.rm")) &&
        isTRUE(tryCatch(eval(expr[[3]]), error = function(e) NA))
      if (!isTRUE(is_na_rm_true)) {
        return(NULL)
      }
    }
    return(list(kind = "sum", arg = expr[[2]]))
  }

  NULL
}

# Internal: Compute a grouped aggregate broadcast to every row
#
# Implements the shape `match_grouped_aggregate_dot()` recognizes: compute
# the aggregate per-group via `summarise()` (reusing the same IR-based
# aggregation sub-expression machinery, so `sum(<arbitrary-expr>)` works
# exactly as it does in summarise()), then LEFT JOIN the (one-row-per-group)
# result back onto the original (all-rows) data on the group columns to
# broadcast the aggregate to every row -- while preserving row order and
# NA-key matching:
#   - Row order: `left_join()`'s left-table row order is preserved (stable
#     sort of the join maps by `left_map`, see R/join.R/src/ops_join.cpp),
#     exactly matching mutate()'s row-order-preserving contract.
#   - NA keys: the join's null equality is unconditionally `EQUAL` (see
#     `src/ops_join.cpp`), matching `group_by()`/`summarise()`'s own
#     NA-values-form-their-own-group semantics -- an NA key in `.data`
#     matches the NA-keyed row of the summary, exactly as dplyr's own
#     grouped mutate would.
#
# If `output_name` replaces an EXISTING column (e.g. `mutate(gdf, x =
# sum(x))`), that column is dropped from the join's left side first, so the
# join doesn't produce a suffixed duplicate -- then the final column order
# is restored to the original position (replace) or appended (new column),
# matching plain mutate()'s own column-order contract.
#
# @param .data A `tbl_gpu` (grouped or not; checked by the caller)
# @param output_name Character, the output column name
# @param agg `list(kind = "n"|"sum", arg = NULL or a language object)`, as
#   returned by `match_grouped_aggregate_dot()`
# @return A new `tbl_gpu`, grouped the same as `.data`
# @keywords internal
grouped_aggregate_mutate <- function(.data, output_name, agg) {
  group_cols <- .data$groups

  agg_call <- if (identical(agg$kind, "n")) {
    quote(n())
  } else {
    rlang::call2("sum", agg$arg)
  }

  if (length(group_cols) == 0) {
    # Ungrouped: a real dplyr::mutate(ungrouped_df, name := n()) broadcasts
    # ONE whole-table aggregate to every row -- there are no keys to join
    # on. Compute it once (summarise() to a single row, then collect() the
    # one scalar value -- forcing materialization, the same one-time-collect
    # tradeoff preprocess_agg_expressions() already accepts for summarise()'s
    # own temp columns, R/summarise.R) and inject it as an ordinary literal
    # mutate() dot -- which already handles replace-vs-append column
    # ordering correctly via the normal IR path, so no extra plumbing is
    # needed for this branch (and no infinite recursion: a bare scalar
    # literal doesn't match `match_grouped_aggregate_dot()`).
    one_row <- rlang::inject(dplyr::summarise(.data, !!output_name := !!agg_call))
    scalar_value <- collect(one_row)[[output_name]][[1]]
    return(dplyr::mutate(.data, !!output_name := !!scalar_value))
  }

  schema <- current_schema(.data)
  key_summary <- rlang::inject(dplyr::summarise(.data, !!output_name := !!agg_call))

  is_replace <- output_name %in% schema$names
  join_input <- .data
  if (is_replace) {
    keep_names <- setdiff(schema$names, output_name)
    join_input <- dplyr::select(.data, dplyr::all_of(keep_names))
  }

  joined <- dplyr::left_join(join_input, key_summary, by = group_cols)

  final_order <- if (is_replace) schema$names else c(schema$names, output_name)
  result <- dplyr::select(joined, dplyr::all_of(final_order))

  new_tbl_gpu(
    ptr = result$ptr,
    schema = result$schema,
    lazy_ops = result$lazy_ops,
    groups = group_cols,
    exec_mode = result$exec_mode
  )
}
