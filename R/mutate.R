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
#'   \item Aggregates or window functions inside a mutate expression (e.g.
#'     `mutate(y = x - mean(x))`, `mutate(r = row_number())`) -- planned
#'     for a later phase (see `scratchpad/phase1_expression_engine.md`
#'     section 6)
#'   \item String manipulation (concatenation, case conversion, substr,
#'     regex, ...) -- only string *comparison* is supported
#'   \item `if_else()`/`case_when()`/`coalesce()` -- planned for a later
#'     phase
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
        "(&, |, !, xor), is.na(), between(), %in%, and sqrt()/log()/log2()/",
        "log10()/exp()/abs()/floor()/ceiling()/sin()/cos()/tan()/round().\n",
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
