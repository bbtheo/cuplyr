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
#'
#' @return A `tbl_gpu` object with the new or modified columns. If a column
#'   name already exists, it is replaced. New columns are appended.
#'
#' @details
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
#'     `floor(1L)` is a double). `abs()` and `round()` preserve the
#'     argument's type.
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
mutate.tbl_gpu <- function(.data, ...) {
  dots <- rlang::enquos(...)

  if (length(dots) == 0) return(.data)

  # Schema accumulates across dots within this call so a later expression can
  # reference an earlier one's output (e.g. `mutate(a = x + y, b = a * 2)`).
  # This mirrors what infer_schema.ast_mutate() will independently recompute
  # from the finished `expressions` list once push_op() calls infer_schema();
  # keeping a running local copy here is only needed to validate/type later
  # dots against the right column set, not to set `.data$schema` directly
  # (push_op() owns that per D1/D3).
  dots <- auto_name_dots(dots, "mutate")

  schema <- current_schema(.data)
  expressions <- list()

  for (i in seq_along(dots)) {
    new_name <- names(dots)[i]
    expr <- dots[[i]]

    ir <- ir_parse_quo(expr, schema)

    if (is.null(ir)) {
      stop(
        "mutate() only supports column copies, arithmetic (+, -, *, /, ^, ",
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

  # One ast_mutate node carries every expression from every dot -- lowered
  # to a single gpu_mutate_expr() call by lower_mutate() (R/lower.R), and a
  # fusion target for fuse_mutates()/toposort_expressions() (R/optimizer.R)
  # across consecutive mutate() calls in lazy mode.
  push_op(.data, ast_mutate(input_node(.data), expressions))
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
