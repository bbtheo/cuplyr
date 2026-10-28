#' Keep only distinct/unique rows of a GPU table
#'
#' Removes duplicate rows from a GPU table, similar to `dplyr::distinct()`.
#' Runs entirely on the GPU (`cudf::stable_distinct()`), keeping the first
#' occurrence of each distinct key and preserving original row order,
#' matching dplyr's contract exactly.
#'
#' @param .data A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Optional variables to use when determining uniqueness. Bare
#'   column names (`distinct(df, x, y)`) and computed expressions
#'   (`distinct(df, z = x + 1)`) are both supported, parsed through the same
#'   expression IR `mutate()` uses (see `R/ir.R`); any computed expressions
#'   are evaluated once via an internal `mutate()` step before the dedup
#'   runs. If omitted, uniqueness is determined using all columns.
#' @param .keep_all If `TRUE`, keeps all columns (in their original order,
#'   with any newly computed key columns appended at the end, matching
#'   `mutate()`'s own append/replace rule). If `FALSE` (the default), only
#'   the columns used for the uniqueness check are kept, in the order given
#'   in `...`.
#'
#' @return A `tbl_gpu` object with duplicate rows removed.
#'
#' @details
#' ## Column selection and order
#' With no `...`, every column is the dedup key and every column is kept
#' (`.keep_all` has no effect in this case -- it's already equivalent).
#' With `...`, the output's columns (when `.keep_all = FALSE`) are exactly
#' the named/deparsed `...` expressions, in the order given -- e.g.
#' `distinct(df, y, x)` returns columns `y, x`, not `x, y`.
#'
#' ## Grouped tables
#' If `.data` is grouped, the group columns are always part of the dedup key
#' and always appear in the result, even if not named in `...` -- dplyr
#' prepends any group columns missing from `...` ahead of the `...` columns
#' (a group column already present in `...` keeps its given position rather
#' than being duplicated at the front). Grouping metadata (`$groups`) is
#' preserved on the result.
#'
#' ## NA and NaN
#' Two `NA` key values are considered equal (one `NA` row survives), and two
#' (computed) `NaN` key values are likewise considered equal -- both match
#' `dplyr::distinct()`'s behavior, and are cudf's own defaults for
#' `stable_distinct()` (`null_equality::EQUAL`, `nan_equality::ALL_EQUAL`),
#' so no special-casing is needed here.
#'
#' @seealso
#' \code{\link{mutate.tbl_gpu}} for creating new columns,
#' \code{\link{group_by.tbl_gpu}} for grouping,
#' \code{\link{collect.tbl_gpu}} for retrieving results
#'
#' @export
#' @importFrom dplyr distinct
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # All columns as key, original row order preserved
#'   result <- gpu_mtcars |> distinct() |> collect()
#'
#'   # Key on specific columns, only those columns in the result
#'   result <- gpu_mtcars |> distinct(cyl, gear) |> collect()
#'
#'   # Key on specific columns, but keep every column (first row per key)
#'   result <- gpu_mtcars |> distinct(cyl, .keep_all = TRUE) |> collect()
#' }
distinct.tbl_gpu <- function(.data, ..., .keep_all = FALSE) {
  if (is_rowwise_gpu(.data)) {
    return(rowwise_cpu(.data, "distinct"))
  }
  dots <- rlang::enquos(...)
  orig_dots <- dots
  group_vars <- .subset2(.data, "groups")

  if (length(dots) == 0) {
    # dplyr: "If omitted, will use all variables" -- this is already
    # equivalent to `.keep_all = TRUE` with every column as the key (dplyr
    # itself produces byte-identical output for both), so there is nothing
    # `.keep_all` could change here.
    schema <- current_schema(.data)
    if (length(schema$names) == 0L) {
      # every row of a zero-column table is the same row
      return(zero_column_result(.data, "distinct",
                                n = min(zero_column_nrow(.data, "distinct"), 1L)))
    }
    node <- ast_distinct(input_node(.data), schema$names, keep_all = TRUE,
                         key_positions = distinct_key_positions(schema$names, schema$names))
    return(push_op(.data, node))
  }

  # NULL dots are dropped after the "no dots" check above, as in dplyr.
  dots <- dots[!vapply(dots, rlang::quo_is_null, logical(1))]
  if (length(dots) == 0) {
    # dplyr runs these through mutate(), which refuses duplicated names
    check_unique_input_names(.data)
    # Only the grouping columns are keys.
    if (length(group_vars) > 0L) {
      if (isTRUE(.keep_all)) {
        return(distinct_fallback(.data, orig_dots, .keep_all,
                                 reason = "`.keep_all = TRUE` with only the grouping columns as keys is dplyr's."))
      }
      node <- ast_distinct(input_node(.data), group_vars, keep_all = FALSE)
      return(push_op(.data, node))
    }
    if (isTRUE(.keep_all)) {
      return(distinct_fallback(.data, orig_dots, .keep_all,
                               reason = "`.keep_all = TRUE` with only NULL keys is dplyr's."))
    }
    return(zero_column_result(.data, "distinct",
                              n = min(zero_column_nrow(.data, "distinct"), 1L)))
  }

  # F77: a call whose head resolves to a function of the user's rather than the
  # native op (a local `abs()`) is dplyr's to run, whole.
  shadowed <- shadow_screen_text(dots)
  if (!is.null(shadowed)) {
    return(distinct_fallback(.data, orig_dots, .keep_all, reason = shadowed))
  }
  # (F88: the reasons the key parses record for a CPU fallback, call-scoped)
  prev <- fallback_reasons_open()
  on.exit(fallback_reasons_close(prev), add = TRUE)

  # Name dots: an unnamed bare column (`distinct(df, x)`) uses its own name;
  # an unnamed computed expression (`distinct(df, x + 1)`) uses the deparsed
  # expression as its name -- both silently, unlike mutate()'s
  # auto_name_dots(), which warns for the latter. dplyr's own distinct()
  # does not warn for this (verified empirically against dplyr 1.2.1).
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

  # Parse each dot through the same IR mutate() uses, accumulating a running
  # schema exactly like mutate.tbl_gpu() so a later dot could (in principle)
  # reference an earlier one's output. A dot that's nothing but a reference
  # to an existing column under its own name (the common case --
  # `distinct(df, x, y)`) contributes no mutate expression at all: it's
  # already present in the input schema, so there's nothing to compute.
  # Only genuine renames (`x = y`) or computed expressions (`z = x + 1`)
  # need a preceding mutate step -- this keeps the common bare-column-key
  # case a single dedup kernel with no extra mutate pass.
  schema <- current_schema(.data)
  expressions <- list()

  for (i in seq_along(dots)) {
    new_name <- dot_names[i]
    expr <- dots[[i]]

    ir <- ir_parse_quo(expr, schema)

    # Not in the expression IR (a user function, a column-free call that only
    # works inside the data mask, ...): dplyr evaluates it, or raises its own
    # error. Same for a factor outside the places F18's rule allows
    # (ir_factor_rule()): dplyr resolves it through the labels.
    if (is.null(ir) || ir_factor_rule(ir, schema) == "cpu") {
      # (F88: the parse's cause; a factor key has its own)
      fallback_dot_failed(expr, if (is.null(ir)) fallback_cause_take() else {
        "a factor key outside the places the GPU handles is resolved through its labels by dplyr."
      })
      return(distinct_fallback(
        .data, orig_dots, .keep_all,
        reason = fallback_failed_reason(orig_dots, "a key has a form the GPU engine does not support.")
      ))
    }

    is_trivial_copy <- identical(ir$kind, "col") && identical(ir$name, new_name)

    if (!is_trivial_copy) {
      step <- make_mutate_expr(new_name, ir, schema)
      expressions <- c(expressions, list(step))
      schema <- update_schema_for_expr(schema, step)
    }
  }

  # Push the key mutate as its own op (as count() does) so push_op() carries
  # factor levels onto renamed/copied factor keys in eager mode too.
  if (length(expressions) > 0) {
    .data <- push_op(.data, ast_mutate(input_node(.data), expressions))
  }

  # Group columns are always part of the key and the result; any group
  # column already named in `...` keeps its given position instead of being
  # duplicated at the front (verified against dplyr 1.2.1: `distinct(gdf, x,
  # y)` with `y` grouped returns columns `x, y` -- not `y, x` -- while
  # `distinct(gdf, x)` returns `y, x`, prepending the *missing* group column).
  key_cols <- unique(c(set_diff(group_vars, dot_names), dot_names))

  node <- ast_distinct(input_node(.data), key_cols, keep_all = .keep_all,
                       key_positions = distinct_key_positions(key_cols, schema$names))
  push_op(.data, node)
}

# Key positions for a distinct node over a table named `names`: `NULL` (key by
# name) unless the names are duplicated. Then an all-columns key is every
# position, and a bare-name key is its first match, which is the column
# dplyr's own distinct(dd, x) keys on. (A computed key went through mutate(),
# which already refused the table.)
# @keywords internal
distinct_key_positions <- function(key_cols, names) {
  if (!anyDuplicated(names)) {
    return(NULL)
  }
  if (identical(key_cols, names)) seq_along(names) else match(key_cols, names)
}

# Whole-call CPU fallback with the user's original dots.
# @keywords internal
distinct_fallback <- function(.data, dots, .keep_all, detail = NULL, reason = detail) {
  gpu_fallback("distinct", .data, function(tbl) {
    rlang::inject(dplyr::distinct(tbl, !!!dots, .keep_all = .keep_all))
  }, reason = reason)
}

#' Unique rows of a GPU table
#'
#' `unique(x)` is `distinct(x)`: duplicate rows over all columns are dropped.
#'
#' @param x A `tbl_gpu` object.
#' @param incomparables Only `FALSE`, as in `unique.data.frame()`.
#' @param ... Only `fromLast = FALSE` is supported (a GPU `distinct()` keeps
#'   the first occurrence); `fromLast = TRUE` and other arguments are errors
#'   rather than silently ignored.
#' @return A `tbl_gpu` with duplicate rows removed (groups are kept).
#' @export
unique.tbl_gpu <- function(x, incomparables = FALSE, ...) {
  if (!isFALSE(incomparables)) {
    stop("`incomparables` is not supported for tbl_gpu; use `FALSE`.", call. = FALSE)
  }
  args <- list(...)
  from_last <- args[["fromLast"]]
  args[["fromLast"]] <- NULL
  if (length(args) > 0L) {
    stop("`unique()` on a tbl_gpu only supports `fromLast = FALSE`; other arguments are not supported.",
         call. = FALSE)
  }
  if (!is.null(from_last) && !isFALSE(from_last)) {
    stop("`fromLast = TRUE` is not supported for tbl_gpu.", call. = FALSE)
  }
  dplyr::distinct(x)
}
