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
#'   `dplyr::collect()` on a database table). A table uploaded from a plain
#'   `data.frame` comes back as a bare `data.frame` while it is ungrouped, and
#'   custom data-frame attributes (`attr(df, "foo")`) are kept the way dplyr
#'   keeps them through verbs.
#'   Column types are converted back to R types:
#'   \itemize{
#'     \item FLOAT64/FLOAT32 -> numeric (double)
#'     \item INT8/INT16/INT32, UINT8/UINT16 -> integer
#'     \item UINT32 -> double (exact)
#'     \item INT64/UINT64 -> `bit64::integer64` (a double with a warning when
#'       `bit64` is absent and a value is past 2^53); a UINT64 value past
#'       `INT64_MAX` is an error, never a wrap
#'     \item STRING -> character
#'     \item BOOL8 -> logical (TRUE/FALSE)
#'     \item DATE (TIMESTAMP_DAYS) -> `Date`; TIMESTAMP_SECONDS/MILLISECONDS/
#'       MICROSECONDS/NANOSECONDS -> `POSIXct` (nanoseconds finer than a
#'       microsecond are rounded, with a warning)
#'     \item DURATION_DAYS -> `difftime` in days; the other DURATION_* -> `difftime`
#'       in seconds (a warning past 2^53 ticks)
#'     \item DECIMAL32/64/128 -> double, the nearest one (a warning when it
#'       needs more than 15 significant digits; an error beyond the double range)
#'     \item a dictionary with string keys -> factor
#'   }
#'   Any other type (LIST, STRUCT, a dictionary with other keys) is an error of
#'   class `cuplyr_error_unsupported_gpu_type` naming the column; the warnings
#'   have class `cuplyr_warning_collect_precision` and name every column
#'   concerned. `print()` and `glimpse()` show such a column as a placeholder.
#'
#' @details
#' ## Grouped tables and `.drop`
#' The `.drop` setting of `group_by()` travels with the table through the
#' verbs that keep at least one group column (and is reset to the default when
#' none is left; `group_by()` on a grouped table keeps it unless given a new
#' `.drop`), so `group_by(f, .drop = FALSE) |> arrange(x) |> collect()` still
#' keeps empty factor-level groups. Empty groups of a *non-factor* column
#' that `filter()` or `slice()` emptied are not kept, unlike in dplyr, except after
#' `slice(.preserve = TRUE)`, which keeps its input's group keys: see the `.drop`
#' section of [group_by()][group_by.tbl_gpu].
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

  # A rowwise table comes back as dplyr's rowwise_df, built by dplyr itself so
  # it is identical to `dplyr::rowwise(<tibble>, vars)`, for a table from a
  # plain data.frame too (F49).
  if (is_rowwise_gpu(x)) {
    out <- with_frame_attrs(tibble::as_tibble(out), frame_attrs(x))
    return(dplyr::rowwise(out, dplyr::all_of(set_intersect(x$groups, names(out)))))
  }

  # dplyr's contract (`collect.data.frame()` is the identity, dbplyr ends with
  # `grouped_df(out, intersect(op_grps(x), names(out)))`): a grouped table
  # comes back grouped, with the same `.drop`.
  groups <- set_intersect(x$groups, names(out))
  if (length(groups) > 0L) {
    drop <- isTRUE(x$group_drop %||% TRUE)
    keys <- fixed_keys_of(x)
    out <- if (is.null(keys)) {
      dplyr::grouped_df(out, groups, drop = drop)
    } else {
      with_fixed_groups(out, keys, drop)
    }
  }
  restore_frame_meta(out, x)
}

# The data-frame attributes and the bare data.frame class of the input table
# `x` (F31) put back on a collected frame: what dplyr's data frame carries
# through every verb that keeps its class. Metadata only.
# @keywords internal
restore_frame_meta <- function(frame, x) {
  frame <- with_frame_attrs(frame, frame_attrs(x))
  # a stale flag must never strip a grouped_df's (or rowwise_df's) class
  if (isTRUE(.subset2(x, "bare_df")) && length(.subset2(x, "groups")) == 0L &&
      !is_rowwise_gpu(x)) {
    class(frame) <- "data.frame"
  }
  frame
}

# restore_frame_meta() for as_tibble()/as.data.frame(): a grouped table is
# rebuilt from its bare data there (as_tibble.grouped_df() drops attributes).
# @keywords internal
ungrouped_frame_meta <- function(frame, x) {
  if (length(.subset2(x, "groups")) > 0L || is_rowwise_gpu(x)) frame else restore_frame_meta(frame, x)
}

# The data of a tbl_gpu as an UNGROUPED tibble. This is what every internal
# caller that needs the rows (fallbacks, joins, sets, vctrs, `$`) wants;
# `collect()` is this plus the group structure.
collect_frame <- function(x, call = rlang::caller_env()) {
  # Handle lazy evaluation: compute first if there are pending ops
  if (is_tbl_gpu(x) && identical(.subset2(x, "exec_mode"), "lazy") && has_pending_ops(x)) {
    plan <- compute_plan(x)
    ptr <- plan$ptr
    final_schema <- plan$schema
  } else {
    ptr <- .subset2(x, "ptr")
    # Get final schema (might have been updated by lazy ops)
    final_schema <- .subset2(x, "schema")
  }

  if (is.null(ptr)) {
    stop("Cannot collect: GPU table pointer is NULL.\n",
         "The table may have been garbage collected or never materialized.",
         call. = FALSE)
  }

  if (length(final_schema$names) == 0L) {
    # Zero-column table: the row count lives on the R side (new_tbl_gpu()).
    # cudf would say 0, so a missing count is an error, not an empty table.
    if (is.null(x[["nrow"]])) {
      stop("Cannot collect: this tbl_gpu has no columns and its row count ",
           "is unknown.", call. = FALSE)
    }
    return(tibble::tibble(.rows = x[["nrow"]]))
  }

  df <- wrap_gpu_call("collect", gpu_collect(ptr, final_schema$names))
  df <- raise_collect_issues(df, final_schema, call = call)
  # "minimal": a relocate()/rename() result may carry duplicated names
  result <- tibble::as_tibble(df, .name_repair = "minimal")

  # Restore factor columns using stored levels
  if (!is.null(final_schema$factor_levels) && anyDuplicated(final_schema$names) > 0L) {
    # duplicated names: by position, and only the columns that are factors
    by_pos <- levels_by_position(final_schema)
    for (i in which(!vapply(by_pos, is.null, logical(1)))) {
      lvls <- by_pos[[i]]
      result[[i]] <- factor_from_codes(result[[i]], lvls)
    }
  } else if (!is.null(final_schema$factor_levels)) {
    for (col_name in names(final_schema$factor_levels)) {
      if (col_name %in% names(result)) {
        lvls <- final_schema$factor_levels[[col_name]]
        result[[col_name]] <- factor_from_codes(result[[col_name]], lvls)
      }
    }
  }

  # POSIXct columns come back from the GPU as UTC; give them their zone.
  result <- restore_tz(result, final_schema)

  # Columns that were vctrs::unspecified() at upload and went through row-only
  # verbs / filtering joins only (tbl_gpu.data.frame()). All NA by construction.
  for (nm in x[["unspecified"]]) {
    col <- result[[nm]]
    if (is.logical(col) && all(is.na(col))) {
      result[[nm]] <- vctrs::unspecified(length(col))
    }
  }

  result <- int64_columns_to_r(result, final_schema)

  result
}

# The conditions of a collect (F105), raised in this one place. The C++ collect
# marks what it could not hold with attributes on the column list:
# `cuplyr_unholdable` (1-based columns that came back NULL: a LIST, a UINT64 past
# INT64_MAX, a DECIMAL past the double range, a dictionary with non-string keys)
# is the classed error; `cuplyr_lossy` (columns whose values were rounded) is one
# classed warning naming all of them. Previews read gpu_head() directly and never
# come through here.
# @keywords internal
raise_collect_issues <- function(df, schema, call = rlang::caller_env()) {
  bad <- attr(df, "cuplyr_unholdable", exact = TRUE)
  lossy <- attr(df, "cuplyr_lossy", exact = TRUE)
  if (is.null(bad) && is.null(lossy)) {
    return(df)
  }
  if (!is.null(bad)) {
    abort_unsupported_gpu_type(names(df)[[bad[[1L]]]], unname(schema$types)[[bad[[1L]]]], call = call)
  }
  attr(df, "cuplyr_lossy") <- NULL
  cols <- paste0("`", names(df)[lossy], "`", collapse = ", ")
  warning(structure(
    class = c("cuplyr_warning_collect_precision", "warning", "condition"),
    list(message = paste0(
      "Column", if (length(lossy) > 1L) "s", " ", cols,
      " cannot be held exactly in R and ", if (length(lossy) > 1L) "were" else "was",
      " rounded to the nearest double (a double keeps 15 significant decimal digits ",
      "and whole numbers up to 2^53; POSIXct keeps microseconds)."
    ), call = NULL)
  ))
  df
}

abort_unsupported_gpu_type <- function(column, type, call = rlang::caller_env()) {
  rlang::abort(
    c(paste0("Can't collect column `", column, "` of GPU type ", type, " to R."),
      x = unholdable_reason(type)),
    class = "cuplyr_error_unsupported_gpu_type",
    call = call
  )
}

unholdable_reason <- function(type) {
  switch(type,
    UINT64 = "It holds a value past INT64_MAX (9223372036854775807), which R cannot represent without wrapping.",
    DECIMAL32 = , DECIMAL64 = , DECIMAL128 = "It holds a value beyond the range of a double.",
    DICTIONARY32 = "Only a dictionary with string keys can be collected.",
    LIST = , STRUCT = "Nested columns have no R equivalent yet.",
    "R has no equivalent of this type."
  )
}

# The R class of an INT64 column follows the schema: INT64 is integer64. The
# C++ collect hands every physical INT64 column back as integer64; without
# bit64 that class has no methods, so such a column becomes a double (exact up
# to 2^53) with one warning naming the columns that hold a larger value. A
# physical INT64 under another schema type is a producer bug; it is returned
# as a double so it is not mistaken for integer64.
# @keywords internal
int64_columns_to_r <- function(result, schema) {
  is64 <- vapply(result, inherits, logical(1), what = "integer64", USE.NAMES = FALSE)
  if (!any(is64)) {
    return(result)
  }
  types <- unname(schema$types)
  is_int64 <- types == "INT64" | types == "UINT64"  # UINT64 arrives as integer64 bits too (F105)
  keep <- is64 & is_int64 & bit64_available()
  big <- character()
  for (j in which(is64 & !keep)) {
    vals <- int64_bits_to_double(unclass(result[[j]]))
    if (is_int64[j] && isTRUE(attr(vals, "beyond_2_53"))) {
      big <- c(big, names(result)[j])
    }
    attr(vals, "beyond_2_53") <- NULL
    result[[j]] <- vals
  }
  if (length(big) > 0L) {
    warning(structure(
      class = c("cuplyr_warning_int64_precision", "warning", "condition"),
      list(message = paste0(
        "INT64/UINT64 column", if (length(big) > 1L) "s", " ",
        paste0("`", big, "`", collapse = ", "),
        " hold", if (length(big) == 1L) "s",
        " values beyond 2^53 that a double cannot store exactly. ",
        "Install bit64 to collect them as integer64."
      ), call = NULL)
    ))
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
  as.data.frame(ungrouped_frame_meta(collect_frame(x), x), row.names = row.names,
                optional = optional, ...)
}

#' @rdname tbl_gpu-coerce
#' @export
#' @importFrom tibble as_tibble
as_tibble.tbl_gpu <- function(x, ...) {
  tibble::as_tibble(ungrouped_frame_meta(collect_frame(x), x), ...)
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

# A factor from 1-based codes and a schema level vector, honoring the `ordered`
# attribute upload put on it (see tbl_gpu.data.frame()). Built from the codes,
# not by matching labels, so an explicit NA level (addNA()) keeps its own code;
# a missing or out-of-range code is a real NA.
# @keywords internal
factor_from_codes <- function(codes, lvls) {
  n <- length(lvls)
  codes <- as.integer(codes)
  codes[!is.na(codes) & (codes < 1L | codes > n)] <- NA_integer_
  structure(codes, levels = as.vector(lvls),
            class = if (isTRUE(attr(lvls, "ordered"))) c("ordered", "factor") else "factor")
}
