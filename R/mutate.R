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
#' @param .by Optional on-the-fly grouping columns (tidyselect). Forces a
#'   CPU fallback (grouped/windowed mutate expressions aren't GPU-native
#'   yet -- see `scratchpad/todo.md` Phase 5); use [dplyr::group_by()]
#'   beforehand for the GPU-native path instead.
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
#'   \item `is.na()`, `between(x, lo, hi)`, `%in%`
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
#'   \item Plain column copies (`mutate(y = x)`)
#'   \item A later dot in the same `mutate()` call referencing an earlier
#'     dot's output column (`mutate(a = x + y, b = a * 2)`)
#'   \item The `.data`/`.env` pronouns, and ordinary R symbols (a bare name
#'     that matches a column resolves to that column; otherwise it's
#'     evaluated as an environment variable -- "columns shadow the
#'     environment")
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
#'   \item Aggregates or window functions inside a mutate expression, WITH
#'     ONE narrow exception: a dot that is exactly `n()` or `sum(<expr>)`
#'     (optionally `sum(<expr>, na.rm = TRUE)`) and nothing else -- e.g.
#'     `mutate(y = n())`, `mutate(total = sum(x))` -- broadcasts that one
#'     whole-table (or, if `x` is grouped, per-group) aggregate to every
#'     row (added to support `dplyr::add_tally()`/`add_count()`, see
#'     `R/count.R`). Anything more elaborate -- combining an aggregate with
#'     other operations (`mutate(y = x - mean(x))`), `row_number()`, other
#'     aggregate functions (`mean()`, `min()`, `max()`, ...) -- is still a
#'     hard error; full grouped-mutate/window-function support is planned
#'     for a later phase (see `scratchpad/phase1_expression_engine.md`
#'     section 6)
#'   \item String manipulation (concatenation, case conversion, substr,
#'     regex, ...) -- only string *comparison* is supported
#'   \item `case_when()`'s `.ptype=`/`.size=`/`.unmatched=` arguments, and
#'     `case_match()` -- planned for a later phase
#' }
#' An expression shape the IR doesn't recognize is currently a hard error
#' (unlike `filter()`, which falls back to CPU evaluation for some
#' unrecognized shapes) -- a CPU fallback for `mutate()` is planned for a
#' later phase.
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

  if (!rlang::quo_is_null(by_quo)) {
    # On-the-fly `.by=` grouping isn't GPU-native yet (grouped/windowed
    # mutate expressions are Phase 5 material, see scratchpad/todo.md) --
    # same scope decision slice.tbl_gpu() makes for its own `.by=`.
    return(gpu_fallback("mutate", .data, function(tbl) {
      rlang::inject(dplyr::mutate(
        tbl, !!!dots, .by = !!by_quo, .keep = .keep,
        .before = !!before_quo, .after = !!after_quo
      ))
    }))
  }

  # Narrow aggregate broadcast: `mutate(x, name = n())` / `mutate(x, name =
  # sum(expr))` -- a (per-group, if `x` is grouped; whole-table otherwise)
  # aggregate broadcast to every row, keeping all rows (unlike summarise()).
  # Full grouped-mutate/window-function support (aggregates combined with
  # other expressions, multiple aggregate dots, other aggregate functions
  # like mean()/min()/max()) is Phase 5 material and NOT handled here; this
  # fires ONLY for the single-dot, bare-`n()`-or-`sum(<expr>[, na.rm =
  # TRUE])` shape. It was added so that `dplyr::add_tally()` and our own
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
  if (length(dots) == 1 &&
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
  orig_groups <- .data$groups

  expressions <- parse_mutate_dots(dots, orig_schema, "mutate")

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
# @return A list of expression structures, one per dot
# @keywords internal
parse_mutate_dots <- function(dots, schema, verb, warn_unnamed = TRUE) {
  dots <- if (warn_unnamed) auto_name_dots(dots, verb) else name_dots_silently(dots)

  expressions <- list()

  for (i in seq_along(dots)) {
    new_name <- names(dots)[i]
    expr <- dots[[i]]

    ir <- ir_parse_quo(expr, schema)

    if (is.null(ir)) {
      stop(
        verb, "() only supports column copies, arithmetic (+, -, *, /, ^, ",
        "%%, %/%), comparisons (==, !=, <, <=, >, >=), logical operators ",
        "(&, |, !, xor), is.na(), between(), %in%, sqrt()/log()/log2()/",
        "log10()/exp()/abs()/floor()/ceiling()/sin()/cos()/tan()/round(), ",
        "and if_else()/case_when()/coalesce()/na_if().\n",
        "Expression: ", rlang::quo_text(expr),
        call. = FALSE
      )
    }

    step <- make_mutate_expr(new_name, ir, schema)
    expressions <- c(expressions, list(step))

    # Update the running schema so subsequent dots see this dot's output.
    schema <- update_schema_for_expr(schema, step)
  }

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
