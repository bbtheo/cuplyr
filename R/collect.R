#' Transfer GPU table data back to R
#'
#' Copies data from GPU memory back to R as a tibble. This is typically
#' the final step in a GPU data manipulation pipeline, after filtering,
#' mutating, and selecting the data you need.
#'
#' @param x A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Additional arguments (ignored, included for compatibility).
#'
#' @return A [tibble::tibble] containing the data from the GPU table (a
#'   `grouped_df`, with the same `.drop`, when `x` is grouped, as for
#'   `dplyr::collect()` on a database table).
#'   Column types are converted back to R types:
#'   \itemize{
#'     \item FLOAT64/FLOAT32 -> numeric (double)
#'     \item INT32/INT64 -> integer or numeric
#'     \item STRING -> character
#'     \item BOOL8 -> logical (TRUE/FALSE)
#'   }
#'
#' @details
#' ## Grouped tables and `.drop`
#' The `.drop` setting of `group_by()` travels with the table through the
#' verbs that keep at least one group column (and is reset to the default when
#' none is left; `group_by()` on a grouped table keeps it unless given a new
#' `.drop`), so `group_by(f, .drop = FALSE) |> arrange(x) |> collect()` still
#' keeps empty factor-level groups. One exception for now: a `mutate()` or
#' `filter()` that uses `cur_group_id()` resets it to `TRUE`. Empty groups of a *non-factor* column
#' that a row-removing verb (`filter()`, `slice()`) emptied are not kept, unlike
#' in dplyr: see the `.drop` section of [group_by()][group_by.tbl_gpu].
#'
#' ## Memory considerations
#' Collecting transfers all data from GPU to CPU memory. For large datasets,
#' this can be slow and memory-intensive. Best practice is to:
#' \enumerate{
#'   \item Filter rows to reduce data volume
#'   \item Select only needed columns
#'   \item Then collect the results
#' }
#'
#' ## Performance
#' Data transfer between GPU and CPU is limited by PCIe bandwidth
#' (typically 16-32 GB/s). For a 1 GB dataset, expect ~50-100ms transfer time.
#'
#' @seealso
#' \code{\link{tbl_gpu}} for creating GPU tables,
#' \code{\link{filter.tbl_gpu}}, \code{\link{select.tbl_gpu}} for reducing data
#'
#' @export
#' @importFrom dplyr collect
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Process on GPU, then collect
#'   result <- gpu_mtcars |>
#'     filter(mpg > 20) |>
#'     mutate(kpl = mpg * 0.425) |>
#'     select(mpg, kpl, hp) |>
#'     collect()
#'
#'   # Result is a regular tibble
#'   class(result)
#'   print(result)
#' }
collect.tbl_gpu <- function(x, ...) {
  out <- collect_frame(x)

  # dplyr's contract (`collect.data.frame()` is the identity, dbplyr ends with
  # `grouped_df(out, intersect(op_grps(x), names(out)))`): a grouped table
  # comes back grouped, with the same `.drop`.
  groups <- intersect(x$groups, names(out))
  if (length(groups) > 0L) {
    out <- dplyr::grouped_df(out, groups, drop = isTRUE(x$group_drop %||% TRUE))
  }
  out
}

# The data of a tbl_gpu as an UNGROUPED tibble. This is what every internal
# caller that needs the rows (fallbacks, joins, sets, vctrs, `$`) wants;
# `collect()` is this plus the group structure.
collect_frame <- function(x) {
  # Handle lazy evaluation: compute first if there are pending ops
  if (is_tbl_gpu(x) && identical(x$exec_mode, "lazy") && has_pending_ops(x)) {
    x <- compute(x)
  }

  if (is.null(x$ptr)) {
    stop("Cannot collect: GPU table pointer is NULL.\n",
         "The table may have been garbage collected or never materialized.",
         call. = FALSE)
  }

  # Get final schema (might have been updated by lazy ops)
  final_schema <- x$schema

  if (length(final_schema$names) == 0L) {
    # Zero-column table: the row count lives on the R side (new_tbl_gpu()).
    # cudf would say 0, so a missing count is an error, not an empty table.
    if (is.null(x[["nrow"]])) {
      stop("Cannot collect: this tbl_gpu has no columns and its row count ",
           "is unknown.", call. = FALSE)
    }
    return(tibble::tibble(.rows = x[["nrow"]]))
  }

  df <- wrap_gpu_call("collect", gpu_collect(x$ptr, final_schema$names))
  result <- tibble::as_tibble(df)

  # Restore factor columns using stored levels
  if (!is.null(final_schema$factor_levels)) {
    for (col_name in names(final_schema$factor_levels)) {
      if (col_name %in% names(result)) {
        lvls <- final_schema$factor_levels[[col_name]]
        result[[col_name]] <- factor(lvls[result[[col_name]]], levels = lvls)
      }
    }
  }

  if (any(final_schema$types == "INT64")) {
    int64_cols <- which(final_schema$types == "INT64")
    for (col_idx in int64_cols) {
      col_name <- final_schema$names[col_idx]
      values <- result[[col_name]]
      if (is.numeric(values) && any(abs(values) > 2^53, na.rm = TRUE)) {
        warning(
          "INT64 values may lose precision when collected into R numeric vectors. ",
          "Consider converting to integer64 or avoid values > 2^53.",
          call. = FALSE
        )
        break
      }
    }
  }

  result
}

#' Coerce a GPU table to a base data frame, tibble, list or matrix
#'
#' These collect the table (running any pending lazy operations) and coerce
#' the result as the corresponding method for a tibble does: groups are dropped
#' (as for a `grouped_df`), and [as.list()] gives a plain named list of the
#' columns, so `lapply()`/`sapply()` over a `tbl_gpu` iterate the columns.
#'
#' @param x A `tbl_gpu` object.
#' @param row.names,optional Passed to [base::as.data.frame()].
#' @param ... Passed on to the tibble/data frame method.
#' @return A `data.frame`, `tibble`, `list` or `matrix`.
#' @name tbl_gpu-coerce
#' @export
as.data.frame.tbl_gpu <- function(x, row.names = NULL, optional = FALSE, ...) {
  as.data.frame(collect_frame(x), row.names = row.names, optional = optional, ...)
}

#' @rdname tbl_gpu-coerce
#' @export
#' @importFrom tibble as_tibble
as_tibble.tbl_gpu <- function(x, ...) {
  tibble::as_tibble(collect_frame(x), ...)
}

#' @rdname tbl_gpu-coerce
#' @export
as.list.tbl_gpu <- function(x, ...) {
  as.list(collect_frame(x), ...)
}

#' @rdname tbl_gpu-coerce
#' @export
as.matrix.tbl_gpu <- function(x, ...) {
  as.matrix(collect_frame(x), ...)
}
