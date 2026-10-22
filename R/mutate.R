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
#'   \item `recode(.x, old = new, ..., .default =, .missing =)` -- native only
#'     for a bare character/integer/double column `.x` and scalar literal
#'     replacements of one shared type (`dplyr::recode`, including `!!!`
#'     lookups); duplicate keys, factor `.x`, `recode_factor()`, and a result
#'     type different from `.x` without `.default` run on the CPU.
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
#'         NA too). The `cumsum()` of an integer or logical is an integer
#'         that is `NA` from the first overflow on (with R's warning), as in R
#'       \item `cummean()` (`cumsum(x) / row_number()`, accumulated in
#'         double), `cumall()`/
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
#'         row regardless of `NA`s. `sum()` of an integer or logical is an
#'         integer, as in dplyr. When a sum leaves the integer range dplyr
#'         returns a double (for the whole column, and for any expression
#'         on it): an eager call is then rerun by dplyr on the CPU, and a
#'         lazy plan, whose schema is fixed, raises
#'         `cuplyr_error_sum_overflow` at `collect()`
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
#'     or logical (`TRUE + TRUE` is `2L`, matching R). Like R, an
#'     integer `+`, `-` or `*` whose result leaves
#'     `.Machine$integer.max` is `NA` with the warning "NAs produced by
#'     integer overflow" (once per call; at `collect()` for a lazy table),
#'     and `%/%`/`%%` by `0L` are `NA`. If any input is `FLOAT64`/`FLOAT32`, the result is
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
  p <- sys.parent()
  inform_colwise_grouped(.data, "mutate", if (p > 0L) sys.function(p))
  check_unique_input_names(.data)

  dots <- rlang::enquos(...)
  # (an argument that wasn't given is the NULL quosure; skips three enquo()s.
  # missing() is also TRUE for an argument forwarded from a caller's missing
  # one, whose substitute() is that caller's symbol rather than the NULL
  # default: that one goes through enquo(), so dplyr's error stays.)
  by_quo <- if (missing(.by) && is.null(substitute(.by))) null_quo else rlang::enquo(.by)
  before_quo <- if (missing(.before) && is.null(substitute(.before))) null_quo else rlang::enquo(.before)
  after_quo <- if (missing(.after) && is.null(substitute(.after))) null_quo else rlang::enquo(.after)

  if (is_rowwise_gpu(.data)) {
    return(mutate_rowwise(.data, dots, by_quo, .keep, before_quo, after_quo))
  }
  mutate_run(.data, dots, by_quo, .keep, before_quo, after_quo)
}

# mutate() of a rowwise table (F49). Elementwise expressions are the
# vectorised mutate() (every column is one value per row), so the native path is
# the ordinary one run with the rowwise vars as protected columns instead of
# groups (mutate_run(rowwise = TRUE)), after rowwise_rewrite_dots() turned the
# row aggregates into horizontal GPU ops. The attempt is all-or-nothing
# (rowwise_attempt()): `.by` (an error in dplyr), a shape the rewrite or the
# parser does not know, a window call, anything that would fall back or fail
# natively reruns the WHOLE call in dplyr with the user's dots, so dplyr
# produces its own result or error.
# @keywords internal
mutate_rowwise <- function(.data, dots, by_quo, keep, before_quo, after_quo) {
  cpu <- function() mutate_expr_fallback(.data, dots, keep, before_quo, after_quo, by_quo)
  if (!rlang::quo_is_null(by_quo)) {
    return(cpu())
  }
  rw <- rowwise_rewrite_dots(.data, dots)
  if (is.null(rw)) {
    return(rowwise_rerun(cpu))
  }
  res <- rowwise_attempt(function() {
    mutate_run(.data, rw, by_quo, keep, before_quo, after_quo, rowwise = TRUE)
  })
  if (is.null(res)) rowwise_rerun(cpu) else rowwise_column_order(res)
}

# mutate.tbl_gpu()'s body on captured arguments. `rowwise = TRUE` (a rowwise
# `.data`, whose dots rowwise_rewrite_dots() has rewritten): the table's groups
# are the rowwise vars, which this treats as columns to keep, never as groups
# to compute over; the checks of rowwise_expressions_ok() bail out of the call.
mutate_run <- function(.data, dots, by_quo, .keep, before_quo, after_quo, rowwise = FALSE) {

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
  by_cols <- if (rowwise) character() else resolve_by(.data, by_quo, ".by")
  verb_groups <- if (by_given) by_cols else if (rowwise) character() else .subset2(.data, "groups")

  if (!rlang::quo_is_null(before_quo) && !rlang::quo_is_null(after_quo)) {
    stop("Can't supply both `.before` and `.after`.", call. = FALSE)
  }

  # Zero-column table with rows: after the arguments are validated against
  # `.data` (an invalid `.by` must still error), dplyr runs the call.
  if (length(dots) > 0L && is_zero_column_with_rows(.data)) {
    return(mutate_expr_fallback(.data, dots, .keep, before_quo, after_quo, by_quo))
  }

  if (length(dots) == 0 && identical(.keep, "all") &&
      rlang::quo_is_null(before_quo) && rlang::quo_is_null(after_quo)) {
    return(.data)
  }

  orig_schema <- current_schema(.data)
  orig_names <- orig_schema$names
  orig_groups <- if (rowwise) .subset2(.data, "groups") else verb_groups

  # Phase 10: across()/if_any()/if_all()/pick()/cur_*() -- see R/across.R's
  # own module docs for the full design. Anything genuinely unsupported
  # (pick()-as-data-frame, cur_group()/cur_group_rows()/cur_data()/
  # cur_data_all()/c_across(), a nested (non-top-level) across()/if_any()/
  # if_all(), or `across(.unpack = TRUE)`) routes the WHOLE mutate() call
  # through the CPU fallback below -- checked BEFORE cur_group_id()
  # substitution/materialization, so a call that's going to fall back
  # anyway never pays for that materialization first.
  # cur_group()$key and length(cur_group_rows()) are plain expressions; the
  # CPU fallbacks keep the dots as the user wrote them.
  user_dots <- dots
  dots <- colwise_rewrite_dots(dots, orig_schema)
  dots <- rewrite_context_dots(dots, verb_groups, sequential = TRUE, name_unnamed = TRUE)
  if (dots_need_fallback(dots)) {
    return(mutate_across_fallback(.data, user_dots, .keep, before_quo, after_quo, by_quo))
  }

  # Expand before cur_group_id() substitution, so a call that falls back
  # never pays for the substitution's GPU work and the helper column is never
  # in `.cols`'s selectable universe.
  expanded <- expand_across_dots(dots, orig_schema, verb_groups, "mutate")
  if (isTRUE(attr(expanded, "needs_fallback"))) {
    return(mutate_across_fallback(.data, user_dots, .keep, before_quo, after_quo, by_quo))
  }
  # The CPU fallbacks below re-run dplyr on the table and the ORIGINAL user
  # dots as they were BEFORE across() expansion and cur_group_id()
  # substitution. dplyr then computes cur_group_id() itself (the helper column
  # never reaches the result), does its own pre-dot evaluation for a staged
  # across() (the temps never exist there), and calls `.fns` values with
  # their own environment: an inlined body would instead read any column an
  # earlier dot creates, including ones an unnamed data-frame dot splices in.
  staged_temps <- attr(expanded, "staged_temps")
  provenance <- attr(expanded, "provenance")
  fallback_data <- .data
  fallback_dots <- user_dots
  cgid <- substitute_cur_group_id(.data, expanded, verb_groups, by = by_given, rowwise = rowwise)
  .data <- cgid$data
  if (isTRUE(cgid$used)) orig_schema <- current_schema(.data)
  dots <- cgid$dots

  # Phase 11, task L1: universal expression-level CPU fallback. Every dot is
  # parsed via the SAME ir_parse_quo()-driven parser as before -- the only
  # change is what happens when a dot's shape isn't recognized at all
  # (`ir_parse_quo()` returns NULL, as opposed to throwing a genuine parse
  # error like "column not found"): instead of parse_mutate_dots()'s old
  # hard stop() (now a defensive, internal-only fallback -- see that
  # function's `stop_on_unparseable` docs), it signals NULL back here, and
  # the WHOLE mutate() call is re-run on the CPU via mutate_expr_fallback(),
  # on the pre-substitution table with the original dots re-injected.
  # A genuine THROWN error (e.g. an undefined column/environment variable)
  # is NOT caught here -- it propagates normally, preserving every existing
  # pinned error-message test (scratchpad/phase11_design.md's "NULL-vs-throw
  # trigger rule").
  # Vector literals (`y = 1:5`) are native only for an ungrouped, no-`.by`
  # call; per-group evaluation is dplyr's job (CPU fallback).
  evaluated <- new.env(parent = emptyenv())
  # (A rowwise call evaluates each dot once per row: no vector literals, and
  # no constant is evaluated once for the CPU route to inline.)
  expressions <- parse_mutate_dots(dots, orig_schema, "mutate", stop_on_unparseable = FALSE,
                                   vector_literals = length(verb_groups) == 0 && !rowwise,
                                   ungrouped = length(verb_groups) == 0 && !rowwise,
                                   evaluated = evaluated, provenance = provenance)
  if (rowwise && !rowwise_expressions_ok(expressions)) {
    rowwise_bail("a window call or vector literal in a rowwise mutate()")
  }
  if (is.null(expressions)) {
    return(mutate_expr_fallback(fallback_data, fallback_dots, .keep, before_quo, after_quo, by_quo,
                                consts = evaluated$consts, provenance = provenance))
  }
  # A vector literal next to a window call isn't planned together: let dplyr
  # run the whole call.
  if (isTRUE(attr(expressions, "has_window")) && isTRUE(attr(expressions, "has_vec"))) {
    return(mutate_expr_fallback(fallback_data, fallback_dots, .keep, before_quo, after_quo, by_quo))
  }
  dropped <- attr(expressions, "dropped")
  if (rowwise && any(dropped %in% .subset2(.data, "groups"))) {
    rowwise_bail("a rowwise var is dropped")
  }
  check_droppable_groups(dropped, if (by_given) character() else verb_groups)
  # Floored fractional Date constants (parse_mutate_dots()); staged temps
  # aren't output columns. Warned once the native call has run (eager) or
  # been planned (lazy); a CPU fallback's upload warns by itself.
  fractional <- set_diff(attr(expressions, "fractional_dates"), staged_temps)

  # Phase 5, task W3: any dot containing a window-function call (row_number(),
  # lag()/lead(), cumsum()/cummax()/cummin()/cumprod(), and their R desugars
  # cummean()/cumall()/cumany()/consecutive_id()) routes the WHOLE mutate()
  # call through mutate_window() instead of the plain single-ast_mutate path
  # below -- see that function's own docs and
  # scratchpad/phase5_window_design.md section 1.3.
  if (isTRUE(attr(expressions, "has_window"))) {
    return(mutate_window(
      .data, fallback_dots, expressions, orig_schema, orig_names, orig_groups,
      .keep, before_quo, after_quo, by_quo, fallback_data = fallback_data,
      drop = staged_temps, dropped = dropped, fractional = fractional
    ))
  }

  # One ast_mutate node carries every expression from every dot -- lowered
  # to a single gpu_mutate_expr() call by lower_mutate() (R/lower.R), and a
  # fusion target for fuse_mutates() (R/optimizer.R)
  # across consecutive mutate() calls in lazy mode. Skipped entirely when
  # there are no expressions (e.g. `mutate(gdf, .keep = "none")`) -- an
  # ast_mutate with zero expressions would be a pure no-op anyway (see
  # R/ast.R::is_barrier()'s "empty mutate nodes are eliminated" comment on
  # the optimizer side), so there's nothing to gain by building one here.
  # The mutate alone when no select can follow it (the common `.keep = "all"`
  # call: mutate_needs_select()); otherwise an eager call runs the mutate and
  # the `.keep`/`.before`/`.after` select as one execution, push_chain().
  if (!mutate_needs_select(orig_names, expressions, .keep, before_quo, after_quo,
                           staged_temps, dropped, helper = isTRUE(cgid$used))) {
    result <- .data
    if (length(expressions) > 0) {
      result <- push_op(.data, ast_mutate(input_node(.data), expressions))
    }
  } else {
    result <- push_chain(.data, function(x) {
      result <- x
      if (length(expressions) > 0) {
        result <- push_op(x, ast_mutate(input_node(x), expressions),
                          input_schema = orig_schema, carry = FALSE)
      }
      select_node <- mutate_keep_select_node(
        result, orig_names, orig_groups, expressions, .keep, before_quo, after_quo,
        drop = staged_temps, dropped = dropped,
        result_names = if (length(expressions) > 0) .subset2(result, "schema")$names else orig_names
      )
      if (!is.null(select_node)) {
        result <- push_op(result, select_node, input_schema = .subset2(result, "schema"),
                          carry = FALSE)
      }
      result
    })
  }
  warn_fractional_dates(fractional)
  result
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
# @param .data The tbl_gpu to compute on (with cur_group_id()'s helper
#   column, if this call substituted one)
# @param dots The user's ORIGINAL dots (before across() expansion and
#   cur_group_id() substitution) -- only used by the CPU fallback path,
#   which re-runs the real dplyr::mutate() call on `fallback_data`
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
# @param fallback_data The tbl_gpu as it was before cur_group_id()
#   substitution; the CPU fallback runs on it so no helper column leaks
# @param drop Staged across() temp columns, removed from the result
# @param dropped Columns set to `NULL` by a dot (and not re-added), removed
#   from the result
# @param fractional Columns built from a floored fractional Date constant:
#   warned about only if the call stays native (the CPU fallback's upload
#   warns itself)
# @return A new tbl_gpu
# @keywords internal
mutate_window <- function(.data, dots, expressions, orig_schema, orig_names, orig_groups,
                           keep, before_quo, after_quo, by_quo, fallback_data = .data,
                           drop = character(), dropped = character(),
                           fractional = character()) {
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
    return(mutate_window_fallback(fallback_data, dots, keep, before_quo, after_quo, by_quo))
  }

  # Each push is handed the schema the previous one produced, so none of them
  # re-derives it from the pending chain.
  emit <- function(result) {
    sch <- orig_schema
    for (batch_plan in plan$batch_plans) {
      for (stage in batch_plan$stages) {
        if (length(stage$pre) > 0) {
          result <- push_op(result, ast_mutate(input_node(result), stage$pre),
                            input_schema = sch, carry = FALSE)
          sch <- .subset2(result, "schema")
        }
        result <- push_op(result, ast_window(input_node(result), stage$specs, orig_groups),
                          input_schema = sch, carry = FALSE)
        sch <- .subset2(result, "schema")
      }
      result <- push_op(result, ast_mutate(input_node(result), batch_plan$expressions),
                        input_schema = sch, carry = FALSE)
      sch <- .subset2(result, "schema")
    }

    select_node <- mutate_keep_select_node(
      result, orig_names, orig_groups, expressions, keep, before_quo, after_quo,
      drop = drop, dropped = dropped, result_names = sch$names
    )
    if (!is.null(select_node)) {
      result <- push_op(result, select_node, input_schema = sch, carry = FALSE)
    }
    result
  }

  # F42: an integer/logical window sum() is declared INT32, but dplyr returns a
  # double when a sum leaves the integer range (one overflowing group makes the
  # whole column double, and `sum(x) + 1L` follows). The eager schema is only
  # known after the fact, so such a call is rerun whole by dplyr; overflow is
  # rare, so the CPU path costs nothing in the common case. A lazy plan has a
  # fixed schema and raises `cuplyr_error_sum_overflow` at collect() instead.
  if (identical(.subset2(.data, "exec_mode"), "eager")) {
    # The CPU rerun's upload warns about fractional dates itself.
    return(retry_on_sum_overflow(
      function() {
        # The whole chain (window stages, mutates, trailing select) runs as one
        # execution and builds one table.
        result <- push_chain(.data, emit)
        warn_fractional_dates(fractional)
        result
      },
      function() mutate_window_fallback(fallback_data, dots, keep, before_quo, after_quo, by_quo)
    ))
  }
  result <- push_chain(.data, emit)
  warn_fractional_dates(fractional)
  result
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

    # F41: a window spec that IS a dot's whole value carries the dot's name, so
    # a fractional Date warning can name the output column.
    for (k in seq_along(batch_idx)) {
      post <- decomposed$post[[k]]
      if (!identical(post$kind, "col")) next
      for (si in seq_along(decomposed$stages)) {
        for (pi in seq_along(decomposed$stages[[si]]$specs)) {
          if (identical(decomposed$stages[[si]]$specs[[pi]]$output_col, post$name)) {
            decomposed$stages[[si]]$specs[[pi]]$label <- names(exprs)[batch_idx[k]]
          }
        }
      }
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
      .data = tbl, !!!dots, .by = !!by_quo, .keep = keep,
      .before = !!before_quo, .after = !!after_quo
    ))
  })
}

# Internal: whole-call CPU fallback for a mutate() call whose dots contain
# a deferred across()-family shape (Phase 10; dots_need_fallback(),
# R/across.R) -- pick()-as-data-frame, cur_group()/cur_group_rows()/
# cur_data()/cur_data_all()/c_across(), a nested (non-top-level) across()/
# if_any()/if_all(), or `across(.unpack = TRUE)`. Mirrors
# mutate_window_fallback() exactly (re-runs the real dplyr::mutate() call
# with the ORIGINAL dots, before any Phase 10 substitution/expansion).
# @keywords internal
mutate_across_fallback <- function(.data, dots, keep, before_quo, after_quo, by_quo) {
  gpu_fallback("mutate", .data, function(tbl) {
    rlang::inject(dplyr::mutate(
      .data = tbl, !!!dots, .by = !!by_quo, .keep = keep,
      .before = !!before_quo, .after = !!after_quo
    ))
  })
}

# Internal: whole-call CPU fallback for a mutate() call with at least one dot
# whose shape the expression IR doesn't recognize at all (Phase 11, task L1
# -- the "universal expression-level CPU fallback"; see
# scratchpad/phase11_design.md). Mirrors mutate_window_fallback()/
# mutate_across_fallback() exactly (same re-injection pattern). The caller
# passes the table and the ORIGINAL dots from BEFORE across() expansion and
# cur_group_id() substitution, so dplyr evaluates across() and
# cur_group_id() itself and the helper column never appears in the result.
# `consts`/`provenance` (parse_mutate_dots()'s recorded values) are inlined
# by inline_evaluated_dots() so dot values aren't computed twice.
# This is what subsumes identity()/ifelse()/scale()/lengths()/
# toupper()/local-closure mutate dots -- and, more generally, ANY
# unrecognized shape, not just window/across-family ones.
# @keywords internal
mutate_expr_fallback <- function(.data, dots, keep, before_quo, after_quo, by_quo,
                                consts = NULL, provenance = NULL) {
  gpu_fallback("mutate", .data, function(tbl) {
    # Inlined here, where the row count is known (see inline_evaluated_dots()).
    dots <- inline_evaluated_dots(dots, consts, provenance, nrow(tbl))
    rlang::inject(dplyr::mutate(
      .data = tbl, !!!dots, .by = !!by_quo, .keep = keep,
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
# @param stop_on_unparseable Logical (Phase 11, task L1). `TRUE` (default):
#   the historical behavior -- a dot whose expression `ir_parse_quo()`
#   doesn't recognize at all (returns `NULL`, as opposed to throwing a
#   genuine parse error) is a hard `stop()`. `FALSE`: instead of stopping,
#   the WHOLE FUNCTION returns `NULL` immediately (abandoning whatever
#   dots have already been parsed) the moment such a dot is found --
#   `mutate.tbl_gpu()`/`transmute.tbl_gpu()` read this `NULL` as "route the
#   whole call to the CPU fallback" (`mutate_expr_fallback()`). Either way,
#   a genuinely THROWN error from `ir_parse_quo()` (e.g. "object ... not
#   found") is never caught here -- it always propagates, which is exactly
#   the "NULL-vs-throw trigger rule" scratchpad/phase11_design.md specifies:
#   an unrecognized *shape* falls back, a genuine user error stays fast.
#   With `stop_on_unparseable = FALSE`, the old hard `stop()` becomes
#   unreachable from any real mutate()/transmute() call (both callers pass
#   `FALSE`) -- kept only as a defensive, internal-only safety net.
# @param ungrouped `TRUE` for a call with no groups and no `.by`. Only then
#   are evaluated constants recorded in `evaluated`.
# @param evaluated Optional environment. Each column-free dot (scalar, vector
#   or NULL) this function evaluates in an ungrouped call
#   (see mutate_constant_value()) is stored in `evaluated$consts`, a list
#   keyed by the dot's position (as.character(i)), as list(value). When the
#   function returns `NULL` the caller hands those values to the CPU route
#   (inline_evaluated_dots()) so dplyr doesn't evaluate the dot a second time.
# @param provenance Optional `"provenance"` attribute of expand_across_dots()
#   for `dots`: which user dot each expanded dot came from, and whether it is
#   an across()/if_any()/if_all() expansion (see the reader rule below).
#
# Evaluate-once and fractional Dates (F37). dplyr evaluates each dot once. A
# column-free dot is evaluated here (mutate_constant_value()) and becomes a
# literal, so the native path evaluates it once too. A fractional Date value
# can't be a TIMESTAMP_DAYS literal (ir_check_ts_lit()); it is built from its
# floored value instead, which is the column the CPU route would give after
# the upload floor, and its name is listed in the `"fractional_dates"`
# attribute: the caller raises `cuplyr_warning_fractional_date` once it has
# committed to the native path (warn_fractional_dates()). That only holds
# while nothing in the call sees the fraction, so a later dot that may read
# the column (expr_may_read_cols(); a bare copy `b = dd` copies the floored
# column and isn't a reader, its output is floored too) sends the call to
# the CPU, where dplyr computes it from the real value. The same goes for the
# output of a whole-value mean()/median() of a Date/POSIXct (rounded to the
# column type on the GPU, dplyr's is a double). Any expansion of a
# later user dot's across()/if_any()/if_all() counts as a reader (its `.cols`
# was resolved before this call's columns existed, so it can't be checked
# by name); expansions of the across() that floored the value don't. A
# fractional Date vector constant is floored by its upload (lower_mutate(),
# which warns itself) and follows the same reader rule. Known limit: on a
# zero-row table nothing is floored, but the native path still warns.
# Accepted limit: in a call that takes the CPU route for any reason, dplyr
# evaluates again what wasn't inlined -- column-free across() bodies (passed
# to dplyr as written, so it resolves `.cols` against the columns earlier
# dots create), constants of a grouped or `.by` call, and column-free
# sub-calls folded inside other dots (`runif(1)` in `x + runif(1)`).
# @return A list of expression structures, one per dot (or `NULL`, only
#   possible when `stop_on_unparseable = FALSE`, see above). Carries an extra
#   `"has_window"` attribute (Phase 5, task W3): `TRUE` iff any dot's parsed
#   IR contains a window-function call anywhere (`ir_has_window()`) --
#   `mutate.tbl_gpu()` reads this to decide whether to route the whole call
#   through `mutate_window()` instead of pushing `expressions` straight into
#   a single `ast_mutate()` node (which would later hit `ir_bind()`'s window
#   guard at lowering time). `transmute.tbl_gpu()` ignores the attribute --
#   window-function support inside `transmute()` is not yet wired (W3 scope
#   is `mutate()` only, per `scratchpad/phase5_window_design.md` section 8's
#   W3 row). Also carries `"fractional_dates"`: the output names whose
#   final value is a floored fractional Date constant (see above).
# @keywords internal
parse_mutate_dots <- function(dots, schema, verb, warn_unnamed = TRUE,
                               stop_on_unparseable = TRUE, vector_literals = FALSE,
                               ungrouped = FALSE, evaluated = NULL, provenance = NULL) {
  dots <- if (warn_unnamed) auto_name_dots(dots, verb) else name_dots_silently(dots)

  expressions <- list()
  has_window <- FALSE
  has_vec <- FALSE
  dropped <- character()
  # dplyr places a new column where its name first appears, even when that
  # dot is `col = NULL` (and a later dot fills it in).
  dot_order <- unique(names(dots))

  # Record the value of the column-free dot at position `j`, so the CPU route
  # inlines it instead of evaluating it again. Only for an ungrouped call
  # (dplyr evaluates a grouped call per group, so a value computed once would
  # change what the CPU route does). A vector is recorded too;
  # inline_evaluated_dots() skips one whose vec_size() can't match the data. A
  # name an earlier `col = NULL` dot removed is NULL in dplyr's mask,
  # but was read from the environment here: dplyr must evaluate such a dot,
  # and one that can see the mask without naming a column (`exists("x")`,
  # expr_reaches_mask()).
  record_const <- function(j, const, quo) {
    if (!is.null(evaluated) && ungrouped &&
        (const$kind %in% c("scalar", "vector") || (identical(const$kind, "null") && isTRUE(const$evaluated))) &&
        !any(all.vars(rlang::quo_get_expr(quo)) %in% dropped) &&
        !expr_reaches_mask(rlang::quo_get_expr(quo))) {
      evaluated$consts[[as.character(j)]] <- list(value = const$raw %||% const$value)
    }
  }

  # Names whose column holds a floored fractional Date constant, in order,
  # with the user dot each came from (`provenance`) and whether it is a
  # vector constant (floored, and warned, when lower_mutate() uploads it).
  frac <- character()
  frac_origin <- integer()
  frac_vec <- logical()
  forget_frac <- function(nm) {
    keep <- frac != nm
    frac <<- frac[keep]
    frac_origin <<- frac_origin[keep]
    frac_vec <<- frac_vec[keep]
  }

  for (i in seq_along(dots)) {
    new_name <- names(dots)[i]
    expr <- dots[[i]]
    origin <- if (is.null(provenance)) i else provenance$origin[i]
    from_across <- !is.null(provenance) && provenance$kind[i] %in% c("across", "combined")

    # A dot that may read a floored fractional Date column must see the
    # fraction: CPU route. Checked before evaluating anything of this dot.
    # An across()/if_any()/if_all() expansion resolved `.cols` against the
    # running schema, which doesn't know which names an earlier user dot
    # floored, so it may read one without naming it (dplyr would select it):
    # conservatively a reader.
    # Expansions of the across() that floored the value don't read it.
    copied <- NA_integer_
    if (length(frac) > 0) {
      e <- rlang::quo_get_expr(expr)
      if (is.symbol(e)) {
        copied <- match(as.character(e), frac)
      }
      if (is.na(copied) &&
          (expr_may_read_cols(e, frac) || (from_across && any(frac_origin != origin)))) {
        if (!stop_on_unparseable) {
          return(NULL)
        }
        stop(verb, "() can't read a fractional Date value on the GPU.\n",
             "Expression: ", rlang::quo_text(expr), call. = FALSE)
      }
    }

    const <- mutate_constant_value(expr, schema)
    record_const(i, const, expr)

    # `col = NULL` (or a constant that evaluates to NULL) drops the column:
    # later dots no longer see it, and a later dot of the same name re-adds it.
    if (identical(const$kind, "null")) {
      dropped <- set_union(dropped, new_name)
      keep_idx <- schema$names != new_name
      schema$names <- schema$names[keep_idx]
      schema$types <- schema$types[keep_idx]
      schema <- set_schema_factor_levels(schema, new_name, NULL)
      forget_frac(new_name)
      next
    }

    floored <- NULL
    floored_vec <- FALSE
    if (identical(const$kind, "scalar")) {
      # Build the literal from the value already computed: evaluating the
      # dot again (ir_parse_quo()) would run RNG/side effects twice. A
      # fractional Date is built floored (see this function's docs).
      floored <- floor_fractional_date(const$value)
      ir <- tryCatch(
        {
          lit <- ir_lit_from_r(floored %||% const$value)
          ir_check_ts_lit(lit)
          # A Period/Duration/difftime value is the CPU's (F48); the signal
          # must not reach make_mutate_expr()'s ir_infer_type().
          ir_check_timespan_lit(lit$value)
          lit
        },
        cuplyr_timestamp_needs_cpu = function(e) NULL,
        cuplyr_type_needs_cpu = function(e) NULL
      )
    } else if (identical(const$kind, "vector")) {
      if (!vector_literals || !const$supported) {
        if (!stop_on_unparseable) {
          return(NULL)
        }
        stop(verb, "() can't use a vector literal here.\n",
             "Expression: ", rlang::quo_text(expr), call. = FALSE)
      }
      ir <- ir_vec(const$value, gpu_type_from_r(const$value), expr, new_name)
      has_vec <- TRUE
      floored_vec <- inherits(const$value, "Date") && {
        v <- as.double(unclass(const$value))
        any(is.finite(v) & v != floor(v))
      }
    } else {
      # A vector constant nested inside a larger expression (`x + 1:5`) is
      # dplyr's elementwise job: CPU fallback.
      ir <- tryCatch(
        ir_parse_quo(expr, schema),
        cuplyr_vector_literal = function(e) NULL
      )
    }

    if (is.null(ir)) {
      if (!stop_on_unparseable) {
        # Phase 11, task L1: signal "route the whole call to the CPU
        # fallback" instead of erroring -- see this function's own
        # `stop_on_unparseable` docs.
        return(NULL)
      }

      # Defensive, internal-only path (Phase 11, task L1): both real
      # callers (mutate.tbl_gpu()/transmute.tbl_gpu()) pass
      # `stop_on_unparseable = FALSE`, so this stop() is unreachable from
      # any dplyr-facing call -- kept only as a safety net for any future
      # internal caller that doesn't want the fallback behavior.
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
        "mean()/sum()/min()/max()/n()/sd()/var()/median()/n_distinct()/",
        "first()/last()/nth(), ",
        "order_by()/with_order(), roll_mean()/roll_sum()/roll_min()/",
        "roll_max()/roll_sd() (data.table::frollmean()/frollsum()/frollmin()/",
        "frollmax()/frollsd() also accepted, align=\"right\" only)) ",
        "(some named arguments of these are not yet supported -- see ",
        "?mutate.tbl_gpu).\n",
        "Expression: ", rlang::quo_text(expr),
        call. = FALSE
      )
    }

    # A factor outside the places F18's rule allows (ir_factor_rule(): bare
    # copy, is.na()/n_distinct(), identical-level comparison) needs dplyr's
    # labels, warnings and errors. Same CPU route as an unparseable dot.
    if (ir_factor_rule(ir, schema) == "cpu") {
      if (!stop_on_unparseable) {
        return(NULL)
      }
      stop(
        verb, "() can't use a factor in this expression on the GPU.\n",
        "Expression: ", rlang::quo_text(expr),
        call. = FALSE
      )
    }

    if (ir_has_window(ir)) {
      has_window <- TRUE
    }

    step <- make_mutate_expr(new_name, ir, schema)

    # The GPU holds a factor as bare INT32 codes, so only a bare copy of a
    # factor column can stay native (its levels follow it, see
    # propagate_factor_levels()). Any other factor-valued expression
    # (if_else()/coalesce()/case_when()/lag() over factors) would compute on
    # the codes and lose or mislabel the levels.
    if (identical(unname(step$output_type), "DICTIONARY32") && !identical(ir$kind, "col")) {
      if (!stop_on_unparseable) {
        return(NULL)
      }
      stop(
        verb, "() can't compute a factor-valued expression on the GPU.\n",
        "Expression: ", rlang::quo_text(expr),
        call. = FALSE
      )
    }

    expressions <- c(expressions, list(step))
    dropped <- set_diff(dropped, new_name)
    copied_vec <- !is.na(copied) && frac_vec[copied]
    # F41: a native mean()/median() of a Date/POSIXct is rounded to the column
    # type (a fractional Date mean is floored, lower_window() warns), where
    # dplyr's value is a double. Same reader rule as a floored constant; the
    # warning comes from the window lowering, so it isn't in `fractional`.
    ts_mean <- identical(ir$kind, "call") && ir$op %in% c("mean", "median") &&
      unname(step$output_type) %in% c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS")
    forget_frac(new_name)
    if (!is.null(floored) || floored_vec || ts_mean || !is.na(copied)) {
      frac <- c(frac, new_name)
      frac_origin <- c(frac_origin, origin)
      frac_vec <- c(frac_vec, floored_vec || ts_mean || copied_vec)
    }

    # Update the running schema so subsequent dots see this dot's output.
    schema <- update_schema_for_expr(schema, step)
  }

  attr(expressions, "has_window") <- has_window
  attr(expressions, "has_vec") <- has_vec
  attr(expressions, "dropped") <- dropped
  attr(expressions, "dot_order") <- dot_order
  attr(expressions, "fractional_dates") <- frac[!frac_vec]
  expressions
}

# Classify a mutate()/transmute() dot that reads no column: `kind` is
# "null" (the dot is, or evaluates to, NULL), "vector" (an atomic vector of
# length != 1; `supported` says whether it can be uploaded as a column),
# "scalar" (an evaluated call/symbol with a length-1 atomic value, returned
# as `value` so the caller doesn't evaluate the dot a second time; a "null"
# also says, in `evaluated`, whether evaluating the dot was needed), or
# "none" (everything else). Evaluated in R, in the dot's environment,
# with an empty data mask so `.env` works and data-only calls (`n()`) fail
# and count as "none".
# @keywords internal
mutate_constant_value <- function(quo, schema) {
  none <- list(kind = "none")
  expr <- rlang::quo_get_expr(quo)
  if (is.null(expr)) {
    return(list(kind = "null"))
  }

  if (is.symbol(expr) || is.call(expr)) {
    if (expr_has_column_ref(expr, schema) || expr_has_mask_only_call(expr)) {
      return(none)
    }
    got <- tryCatch(
      list(value = rlang::eval_tidy(quo, data = list())),
      error = function(e) NULL
    )
    if (is.null(got)) {
      return(none)
    }
    value <- got$value
    ran <- TRUE
  } else {
    value <- expr
    ran <- FALSE
  }

  if (is.null(value)) {
    return(list(kind = "null", evaluated = ran))
  }
  if (!is.atomic(value)) {
    return(none)
  }
  if (length(value) == 1L) {
    # A bare literal (no evaluation happened) takes the ordinary parse path.
    return(if (ran) list(kind = "scalar", value = value) else none)
  }

  supported <- (is.logical(value) || is.integer(value) || is.double(value) || is.character(value)) &&
    (!is.object(value) || identical(class(value), "Date")) &&
    is.null(dim(value))
  # `value` is unnamed for ir_vec(); `raw` keeps the names a CPU-route
  # recorded constant (record_const()) must still carry (`names(v)`).
  list(kind = "vector", value = if (supported) unname(value) else value, raw = value,
       supported = supported)
}

# The CPU route must not evaluate again a column-free dot that
# parse_mutate_dots() already evaluated (RNG, counters, side effects).
# `dots` are the user's dots from before across() expansion; `consts` holds
# the values recorded per EXPANDED dot position, and `provenance` (see
# expand_across_dots()) says which original dot each expanded dot came from.
# A dot passed through unchanged ("dot") whose value was recorded is
# replaced by that value, under the name dplyr gives the original
# expression. Everything else stays as written for dplyr; in particular an
# across() dot, which dplyr resolves against the columns earlier dots
# create (its column-free bodies run again there, see parse_mutate_dots()).
# A recorded non-NULL value is inlined only if its vctrs::vec_size() is 1 or
# `n` (dplyr's own constant size rule): otherwise dplyr's size error would
# name the value ("<date>") instead of the user's expression, so the dot
# stays as written (and is evaluated again before the error). NULL is always
# inlined (dplyr doesn't size-check it). A value vec_size() errors on stays
# as written.
# @param dots list of quosures (user dots, before expansion)
# @param consts named list keyed by expanded-dot position, each list(value), or NULL
# @param provenance the `"provenance"` attribute of expand_across_dots()
# @param n row count of the data the CPU route runs on
# @keywords internal
inline_evaluated_dots <- function(dots, consts, provenance, n) {
  if (length(consts) == 0) {
    return(dots)
  }
  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }
  for (d in seq_along(dots)) {
    idx <- which(provenance$origin == d)
    if (length(idx) != 1L || !identical(provenance$kind[idx], "dot") ||
        !(as.character(idx) %in% names(consts))) {
      next
    }
    value <- consts[[as.character(idx)]]$value
    if (!is.null(value)) {
      size <- tryCatch(vctrs::vec_size(value), error = function(e) NA_integer_)
      if (!size %in% c(1L, n)) {
        next
      }
    }
    quo <- dots[[d]]
    if (is.na(dot_names[d]) || dot_names[d] == "") {
      # dplyr's own rule for an unnamed dot
      dot_names[d] <- rlang::with_options(
        "rlang:::use_as_label_infix" = FALSE,
        rlang::as_label(quo)
      )
    }
    dots[[d]] <- rlang::new_quosure(value, rlang::quo_get_env(quo))
  }
  names(dots) <- dot_names
  dots
}

# The floored value of a column-free Date constant that isn't a whole day,
# or NULL for anything else. Floored like the upload (date_to_gpu()):
# -0.5 -> -1. Non-finite or out-of-int32 values keep going to the CPU.
# @keywords internal
floor_fractional_date <- function(value) {
  if (!inherits(value, "Date") || length(value) != 1L) {
    return(NULL)
  }
  v <- as.double(unclass(value))
  if (!is.finite(v) || v == floor(v) || abs(floor(v)) > .Machine$integer.max) {
    return(NULL)
  }
  structure(floor(v), class = "Date")
}

# TRUE iff dot expression `expr` may read a column named in `cols` when
# dplyr evaluates it. Syntactic and conservative: the names as symbols
# anywhere (also under `.env$`), the `.data` pronoun, or a call that can
# reach a mask column without naming it (expr_reaches_mask()). A top-level
# across()/if_any()/if_all() dot is already expanded when this runs (its
# `.cols` resolved against the pre-call columns, so the expansion never
# names a column an earlier dot of the call creates): parse_mutate_dots()
# treats those expansions as readers through their provenance instead.
# Not detected: a user function that calls pick()/cur_data() itself.
# @keywords internal
expr_may_read_cols <- function(expr, cols) {
  any(all.vars(expr) %in% c(cols, ".data")) || expr_reaches_mask(expr)
}

# TRUE iff `expr` calls something that can reach dplyr's data mask without
# naming a column as a symbol: pick(), a nested across() and friends,
# cur_data(), get()/mget()/exists(), eval(), environment()... Such a dot's
# value depends on the mask, so it can't be computed outside it either.
# @keywords internal
expr_reaches_mask <- function(expr) {
  mask_wide <- c(
    "pick", "across", "c_across", "if_any", "if_all", "cur_data", "cur_data_all",
    "cur_group", "get", "get0", "mget", "exists", "dynGet", "eval", "evalq",
    "eval_tidy", "eval_bare", "parse", "str2lang", "str2expression",
    "environment", "parent.frame", "sys.frame", "sys.frames", "sys.function",
    "as.environment", "ls", "objects", "local", "with", "within",
    "do.call", "match.fun"
  )
  walk <- function(e) {
    if (rlang::is_missing(e) || !is.call(e)) {
      return(FALSE)
    }
    head <- call_head_name(e)
    if (!is.null(head) && head %in% mask_wide) {
      return(TRUE)
    }
    for (part in as.list(e)) {
      if (walk(part)) {
        return(TRUE)
      }
    }
    FALSE
  }
  walk(expr)
}

# TRUE iff `expr` calls a function that only works inside dplyr's data mask
# (n(), zero-arg row_number(), cur_*(), pick(), across()...). Such a dot
# can't be pre-evaluated out here: evaluation would run its column-free
# sub-calls (`runif(1)` in `runif(1) + n()`) and then fail on the mask-only
# call, and ir_parse_quo() would run them a second time. A user function
# that calls n() internally is not detected (it double-evaluates as before).
# @keywords internal
expr_has_mask_only_call <- function(expr) {
  zero_arg <- c("n", "row_number", "cur_group_id", "cur_group_rows", "cur_column",
                "cur_group", "cur_data", "cur_data_all")
  any_arg <- c("pick", "across", "c_across", "if_any", "if_all")
  if (rlang::is_quosure(expr)) {
    return(expr_has_mask_only_call(rlang::quo_get_expr(expr)))
  }
  if (rlang::is_missing(expr) || !is.call(expr)) {
    return(FALSE)
  }
  head <- call_head_name(expr)
  if (!is.null(head) && ((head %in% zero_arg && length(expr) == 1L) || head %in% any_arg)) {
    return(TRUE)
  }
  parts <- as.list(expr)[-1]
  for (i in seq_along(parts)) {
    # an empty subscript slot (`m[, 1]`) is the missing-argument sentinel:
    # reading it into a variable and forcing it errors
    if (rlang::is_missing(parts[[i]])) {
      next
    }
    if (expr_has_mask_only_call(parts[[i]])) {
      return(TRUE)
    }
  }
  FALSE
}

# dplyr refuses to drop a grouping column of a grouped table (`.by` allows it).
# @keywords internal
check_droppable_groups <- function(dropped, groups) {
  bad <- set_intersect(dropped, groups)
  if (length(bad) > 0) {
    stop("Can't drop grouping column", if (length(bad) > 1) "s", " ",
         paste0("`", bad, "`", collapse = ", "), " with NULL.", call. = FALSE)
  }
  invisible(NULL)
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
      dot_names[i] <- rlang::as_label(dots[[i]])
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
# @param drop Temp columns (staged across() expansions) the result must not
#   keep: they are not new columns, but their inputs still count as used
# @param dropped Columns a dot set to `NULL` (and no later dot re-added)
# @return An ast_select node to stack on `result` via push_op(), or `NULL`
#   if no select node is needed (the "all", no `.before`/`.after` case is
#   always a pure identity and is skipped to avoid an unnecessary AST node)
# @keywords internal
mutate_keep_select_node <- function(result, orig_names, orig_groups,
                                     expressions, keep, before_quo, after_quo,
                                     drop = character(), dropped = character(),
                                     result_names = current_schema(result)$names) {
  touched_cols <- unique(vapply(expressions, `[[`, character(1), "output_col"))
  new_cols <- touched_cols[!touched_cols %in% c(orig_names, drop, dropped)]
  # New columns sit where their name first appeared among the dots, which
  # includes a `col = NULL` dot that a later dot re-fills.
  dot_order <- attr(expressions, "dot_order")
  if (!is.null(dot_order)) {
    new_cols <- dot_order[dot_order %in% new_cols]
  }
  existing_touched <- touched_cols[touched_cols %in% orig_names]
  input_cols_union <- unique(unlist(lapply(expressions, `[[`, "input_cols")))

  mode_extra <- switch(keep,
    all = orig_names,
    used = input_cols_union,
    unused = set_diff(orig_names, input_cols_union),
    none = character()
  )

  always_kept <- set_union(orig_groups, existing_touched)
  kept_orig_names <- orig_names[orig_names %in% set_union(always_kept, mode_extra)]

  full_names <- result_names
  full_names <- c(full_names[!full_names %in% new_cols], new_cols)

  if (length(new_cols) > 0 &&
      (!rlang::quo_is_null(before_quo) || !rlang::quo_is_null(after_quo))) {
    # A column dropped by `col = NULL` is already gone for .before/.after.
    full_names <- full_names[!full_names %in% dropped]
    name_vec <- stats::setNames(full_names, full_names)
    loc <- tidyselect::eval_relocate(
      expr = rlang::expr(c(!!!rlang::syms(new_cols))), data = name_vec,
      before = before_quo, after = after_quo,
      before_arg = ".before", after_arg = ".after"
    )
    full_names <- full_names[loc]
  }

  final_order <- full_names[full_names %in% set_union(kept_orig_names, new_cols)]
  final_order <- final_order[!final_order %in% dropped]
  if (length(dropped) > 0 && length(final_order) == 0) {
    stop("Can't drop every column of a tbl_gpu.", call. = FALSE)
  }

  if (identical(final_order, result_names)) {
    return(NULL)
  }

  ast_select(input_node(result), final_order)
}

# TRUE unless mutate_keep_select_node() is certain to find nothing to select:
# `.keep = "all"` with no `.before`/`.after`, no staged temp and no dropped
# column keeps every original column where it is, and the new columns land at
# the end in the order the dots introduced them, which is the order the mutate
# node appends them in; the one exception is a `col = NULL` dot that a later dot
# re-fills (dplyr puts the column where its NAME first appeared). `helper`: the
# input carries a cur_group_id() helper column, which the select drops.
# @keywords internal
mutate_needs_select <- function(orig_names, expressions, keep, before_quo, after_quo,
                                drop, dropped, helper = FALSE) {
  if (helper || !identical(keep, "all") || length(drop) > 0L || length(dropped) > 0L ||
      !rlang::quo_is_null(before_quo) || !rlang::quo_is_null(after_quo)) {
    return(TRUE)
  }
  touched <- unique(vapply(expressions, `[[`, character(1), "output_col"))
  new_cols <- touched[!touched %in% orig_names]
  dot_order <- attr(expressions, "dot_order")
  !is.null(dot_order) && !identical(dot_order[dot_order %in% new_cols], new_cols)
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
  # Nothing to carry (no factor levels, no zones) and an output that is neither
  # a factor nor a POSIXct: only the name and type change.
  if (is.null(schema$factor_levels) && is.null(schema$tz) &&
      !(unname(expr$output_type) %in% c("TIMESTAMP_MICROSECONDS", "DICTIONARY32"))) {
    existing_idx <- match(expr$output_col, schema$names)
    if (!is.na(existing_idx)) {
      schema$types[existing_idx] <- expr$output_type
    } else {
      schema$names <- c(schema$names, expr$output_col)
      schema$types <- c(schema$types, expr$output_type)
    }
    return(schema)
  }

  # The output name's levels, read BEFORE writing (`f = f` is its own
  # source). ir_factor_rule() reads these for the dots that follow.
  schema <- set_schema_factor_levels(
    schema, expr$output_col, mutate_output_levels(schema$factor_levels, expr)
  )

  # F32: a POSIXct output takes the zone of its expression (read before the
  # output name is overwritten, `t = t + 1` is its own source); anything else
  # has no zone.
  zone <- if (identical(unname(expr$output_type), "TIMESTAMP_MICROSECONDS")) {
    ir_infer_tz(expr$ir, schema)
  } else {
    "UTC"
  }

  existing_idx <- match(expr$output_col, schema$names)

  if (!is.na(existing_idx)) {
    schema$types[existing_idx] <- expr$output_type
  } else {
    schema$names <- c(schema$names, expr$output_col)
    schema$types <- c(schema$types, expr$output_type)
  }

  set_schema_tz(schema, expr$output_col, zone)
}

# The factor levels one mutate output takes: a bare copy of a factor column
# takes its source's levels; anything else (including a copy whose source
# has no recorded levels) is not a known factor, so the output name's old
# entry must go (NULL). The one rule for update_schema_for_expr() and
# propagate_factor_levels()'s mutate branch.
# @param factor_levels The running `factor_levels` list (before this output)
# @param expr A mutate expression (output_col, output_type, ir)
# @keywords internal
mutate_output_levels <- function(factor_levels, expr) {
  if (identical(unname(expr$output_type), "DICTIONARY32") && identical(expr$ir$kind, "col")) {
    factor_levels[[expr$ir$name]]
  } else {
    NULL
  }
}

# Set (or, with NULL, drop) one column's entry in a running schema's
# `factor_levels`, leaving no empty list behind.
# @keywords internal
set_schema_factor_levels <- function(schema, name, levels) {
  fl <- schema$factor_levels %||% list()
  fl[[name]] <- levels
  schema$factor_levels <- if (length(fl) == 0L) NULL else fl
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
  p <- sys.parent()
  inform_colwise_grouped(.data, "transmute", if (p > 0L) sys.function(p))
  check_unique_input_names(.data)
  dots <- rlang::enquos(...)

  if (is_rowwise_gpu(.data)) {
    # as mutate_rowwise(): rewrite the row aggregates, attempt the native path,
    # else the whole call in dplyr
    cpu <- function() {
      gpu_fallback("transmute", .data, function(tbl) {
        rlang::inject(dplyr::transmute(.data = tbl, !!!dots))
      })
    }
    rw <- if (length(dots) > 0L || length(.subset2(.data, "groups")) > 0L) rowwise_rewrite_dots(.data, dots)
    if (is.null(rw)) {
      return(rowwise_rerun(cpu))
    }
    res <- rowwise_attempt(function() transmute_run(.data, rw, rowwise = TRUE))
    return(if (is.null(res)) rowwise_rerun(cpu) else rowwise_column_order(res))
  }
  transmute_run(.data, dots)
}

# transmute.tbl_gpu()'s body on captured dots; `rowwise = TRUE` as in
# mutate_run(): the rowwise vars are the columns kept in front, not groups.
transmute_run <- function(.data, dots, rowwise = FALSE) {
  orig_dots <- dots
  # the groups every window/cur_group/vector-literal decision is made over
  eff_groups <- if (rowwise) character() else .subset2(.data, "groups")

  if (length(dots) > 0L && is_zero_column_with_rows(.data)) {
    return(gpu_fallback("transmute", .data, function(tbl) {
      rlang::inject(dplyr::transmute(.data = tbl, !!!dots))
    }))
  }

  # No expressions on an ungrouped table: every column goes (n x 0 result).
  if (length(dots) == 0L && length(.subset2(.data, "groups")) == 0L) {
    return(zero_column_result(.data, "transmute"))
  }

  # Phase 10: across()/if_any()/if_all()/pick()/cur_*() -- see R/across.R
  # and mutate.tbl_gpu()'s own identical handling.
  user_dots <- dots
  dots <- colwise_rewrite_dots(dots, current_schema(.data))
  dots <- rewrite_context_dots(dots, eff_groups, sequential = TRUE, name_unnamed = TRUE)
  if (dots_need_fallback(dots)) {
    return(gpu_fallback("transmute", .data, function(tbl) {
      rlang::inject(dplyr::transmute(.data = tbl, !!!user_dots))
    }))
  }
  expanded <- expand_across_dots(dots, current_schema(.data), eff_groups, "transmute")
  if (isTRUE(attr(expanded, "needs_fallback"))) {
    return(gpu_fallback("transmute", .data, function(tbl) {
      rlang::inject(dplyr::transmute(.data = tbl, !!!user_dots))
    }))
  }
  # As in mutate.tbl_gpu(): the CPU fallback runs on the pre-substitution
  # table with the ORIGINAL dots, so cur_group_id()'s helper column and the
  # staged temps can't leak and `.fns` values keep their own environment.
  staged_temps <- attr(expanded, "staged_temps")
  provenance <- attr(expanded, "provenance")
  fallback_data <- .data
  fallback_dots <- orig_dots
  cgid <- substitute_cur_group_id(.data, expanded, eff_groups, rowwise = rowwise)
  .data <- cgid$data
  schema <- current_schema(.data)
  dots <- cgid$dots

  # Phase 11, task L1: same universal expression-level CPU fallback as
  # mutate.tbl_gpu() -- see mutate_expr_fallback()'s own docs for the full
  # "NULL-vs-throw trigger rule" derivation.
  evaluated <- new.env(parent = emptyenv())
  expressions <- parse_mutate_dots(dots, schema, "transmute", warn_unnamed = FALSE,
                                    stop_on_unparseable = FALSE,
                                    vector_literals = length(eff_groups) == 0 && !rowwise,
                                    ungrouped = length(eff_groups) == 0 && !rowwise,
                                    evaluated = evaluated, provenance = provenance)
  # A window call (mean()/cumsum()/lag()/..., also a timestamp mean()/median())
  # needs the staged plan of mutate_window(), which transmute() doesn't use:
  # without this the call died at lowering ("window op reached lowering").
  if (rowwise && !rowwise_expressions_ok(expressions)) {
    rowwise_bail("a window call or vector literal in a rowwise transmute()")
  }
  if (is.null(expressions) || isTRUE(attr(expressions, "has_window"))) {
    return(gpu_fallback("transmute", fallback_data, function(tbl) {
      dots_cpu <- inline_evaluated_dots(fallback_dots, evaluated$consts, provenance, nrow(tbl))
      rlang::inject(dplyr::transmute(.data = tbl, !!!dots_cpu))
    }))
  }

  dropped <- attr(expressions, "dropped")
  if (rowwise && any(dropped %in% .subset2(.data, "groups"))) {
    rowwise_bail("a rowwise var is dropped")
  }
  check_droppable_groups(dropped, eff_groups)

  # Order: literal dot order (whether bare-column or newly computed), with
  # any un-dotted group column prepended -- see the roxygen "Details" above
  # for how this differs from mutate(.keep = "none")'s ordering rule.
  dot_output_cols <- set_diff(
    attr(expressions, "dot_order"), c(staged_temps, dropped)
  )
  missing_groups <- set_diff(.subset2(.data, "groups"), dot_output_cols)
  final_order <- c(missing_groups, dot_output_cols)

  # (An eager call runs the mutate and the select as one execution, push_chain().)
  result <- push_chain(.data, function(x) {
    result <- x
    if (length(expressions) > 0) {
      result <- push_op(x, ast_mutate(input_node(x), expressions),
                        input_schema = schema, carry = FALSE)
    }
    if (length(final_order) > 0L) {
      result <- push_op(result, ast_select(input_node(result), final_order),
                        carry = FALSE)
    }
    result
  })
  if (length(final_order) == 0L) {
    return(zero_column_result(result, "transmute"))
  }
  # As in mutate.tbl_gpu(): floored fractional Date constants, warned now
  # that the call is native.
  warn_fractional_dates(set_diff(attr(expressions, "fractional_dates"), staged_temps))
  result
}

# dplyr's own (non-generic) colwise verbs and add_tally() reach the tbl_gpu
# methods through dplyr's generics; imported so they resolve inside the package
# (inform_colwise_grouped() compares against them, and the tests call them).
#' @importFrom dplyr mutate_all mutate_if mutate_at transmute_all transmute_if summarise_all add_tally
#' @name colwise-imports
#' @keywords internal
#' @noRd
NULL

# dplyr's check_grouped() for mutate_all()/mutate_if()/transmute_all()/
# transmute_if(): they inform that the grouping variables are ignored, but test
# is_grouped_df(), which is FALSE for a tbl_gpu. `caller` is the function that
# called the mutate()/transmute() generic (with UseMethod() dispatch the
# method's parent frame is the generic's caller), so only the colwise verbs
# trigger it.
inform_colwise_grouped <- function(.data, verb, caller) {
  # (dplyr's check_grouped() tests is_grouped_df(): a rowwise_df is not)
  if (length(.subset2(.data, "groups")) == 0L || is_rowwise_gpu(.data) || !is.function(caller)) {
    return(invisible())
  }
  for (suffix in c("all", "if")) {
    fn <- getExportedValue("dplyr", paste0(verb, "_", suffix))
    if (identical(caller, fn)) {
      cols <- .subset2(.data, "groups")
      msg <- paste0(if (length(cols) == 1L) "Column " else "Columns ",
                    paste0(if (length(cols) > 6L) c(paste0("`", cols[1:5], "`"), "...")
                           else paste0("`", cols, "`"), collapse = ", "))
      alt <- if (suffix == "all") {
        sprintf("Use `%s_at(df, vars(-group_cols()), myoperation)` to silence the message.", verb)
      }
      rlang::inform(c(
        sprintf("`%s_%s()` ignored the following grouping variables:", verb, suffix),
        "*" = msg,
        i = alt
      ))
    }
  }
  invisible()
}

# A zero-column table with rows: dplyr recycles a constant `mutate(b = 1)` to
# the row count, but there is no column to size a constant against (and
# push_op() refuses the input, check_zero_column_input()). mutate() and
# transmute() hand such a call to dplyr on the CPU: there is no data to move,
# and dplyr's tidyselect then sees exactly the columns the user has (none),
# so if_all() over nothing is TRUE, if_any() FALSE and an empty across() adds
# nothing. FALSE when `.data` has columns, no rows, or an unknown row count
# (the usual checks then apply).
# @keywords internal
is_zero_column_with_rows <- function(.data) {
  n <- .subset2(.data, "nrow")
  length(current_schema(.data)$names) == 0L && !is.null(n) && !is.na(n) && n > 0L
}
