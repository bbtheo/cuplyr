# Print method for tbl_gpu (glimpse-style output)

#' Summary header of a tbl_gpu (pillar)
#'
#' dplyr's `tbl_sum()` for a data frame or tibble: `"A tibble"` (`"A data
#' frame"` for an ungrouped table from a plain `data.frame`) with the
#' dimensions, plus the groups of a grouped table.
#'
#' @param x A `tbl_gpu`.
#' @param ... Ignored.
#' @return A named character vector.
#' @export
#' @importFrom pillar tbl_sum
tbl_sum.tbl_gpu <- function(x, ...) {
  out <- stats::setNames(
    pillar::dim_desc(x),
    if (isTRUE(.subset2(x, "bare_df"))) "A data frame" else "A tibble"
  )
  groups <- .subset2(x, "groups")
  if (length(groups) > 0L) {
    mark <- if (identical(getOption("OutDec"), ",")) "." else ","
    n <- formatC(nrow(dplyr::group_data(x)), big.mark = mark, format = "d")
    out <- c(out, Groups = paste0(paste(groups, collapse = ", "), " [", n, "]"))
  }
  out
}

#' @export
print.tbl_gpu <- function(x, ..., width = NULL) {
  width <- width %||% getOption("width", 80)

  if (is.null(x$ptr)) {
    cat("# A GPU tibble [lazy, not materialized]\n")
    cat("# Schema: ", paste(x$schema$names, collapse = ", "), "\n")
    cat("# Operations pending: ", length(x$lazy_ops), "\n")
    return(invisible(x))
  }

  dims <- base_dim(x)
  types <- gpu_col_types(x$ptr)
  col_names <- display_names(x$schema$names)

  # Header
  cat("Rows: ", format(dims[1], big.mark = ","), "\n", sep = "")
  cat("Columns: ", dims[2], "\n", sep = "")

  if (length(x$groups) > 0) {
    cat("Groups: ", paste(x$groups, collapse = ", "), "\n")
  }

  # Get preview data (first 10 rows for value display)
  preview <- tryCatch(
    gpu_head(x$ptr, 10L, col_names),
    error = function(e) NULL
  )

  # Calculate column name width for alignment
  max_name_width <- max(nchar(col_names), 1)

  zones <- tz_by_position(x$schema)

  # Print each column
  for (i in seq_along(col_names)) {
    col_name <- col_names[i]
    col_type <- types[i]

    # Format: $ name <type> value1, value2, ...
    name_pad <- format(col_name, width = max_name_width)
    type_str <- paste0("<", col_type, ">")

    # Get preview values
    if (!is.null(preview)) {
      is_time <- inherits(preview[[i]], c("POSIXct", "Date"))
      vals <- preview_values(preview[[i]], zones[[i]])
      if (length(vals) > 0) {
        # Truncate strings for display
        if (is.character(vals) && !is_time) {
          vals <- ifelse(nchar(vals) > 20, paste0(substr(vals, 1, 17), "..."), vals)
          vals <- paste0("\"", vals, "\"")
        }
        vals[is.na(vals)] <- "NA"
        val_str <- paste(vals, collapse = ", ")

        # Truncate to fit width
        available_width <- width - max_name_width - nchar(type_str) - 6
        if (nchar(val_str) > available_width && available_width > 10) {
          val_str <- paste0(substr(val_str, 1, available_width - 3), "...")
        }
      } else {
        val_str <- ""
      }
    } else {
      val_str <- "[preview unavailable]"
    }

    cat("$ ", name_pad, " ", type_str, " ", val_str, "\n", sep = "")
  }

  invisible(x)
}

# Column names as printed: an NA name shows as "NA" (nchar(NA_character_) is
# NA and would break the width arithmetic).
display_names <- function(nms) {
  nms[is.na(nms)] <- "NA"
  nms
}
