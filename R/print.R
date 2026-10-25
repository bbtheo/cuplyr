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
  if (is_rowwise_gpu(x)) {
    # dplyr's tbl_sum.rowwise_df(): the rowwise vars, "" when there are none
    out <- c(out, Rowwise = paste(groups, collapse = ", "))
  } else if (length(groups) > 0L) {
    mark <- if (identical(getOption("OutDec"), ",")) "." else ","
    n <- formatC(count_groups(x), big.mark = mark, format = "d")
    out <- c(out, Groups = paste0(paste(groups, collapse = ", "), " [", n, "]"))
  }
  out
}

#' @export
print.tbl_gpu <- function(x, ..., width = NULL) {
  # the header wraps at the print width; the glimpse-style body follows
  # glimpse()'s rule, which ignores an infinite width option
  body_w <- body_width(width)
  width <- print_width(width)

  # A lazy table with pending operations is never run to print it: show the
  # result's schema, its grouping (metadata only) and how many plan steps
  # (AST nodes) are pending.
  if (has_pending_ops(x) || is.null(.subset2(x, "ptr"))) {
    cat("# A gibble [lazy, not materialized]\n")
    cat("# Schema: ", paste(display_names(.subset2(x, "schema")$names), collapse = ", "), "\n", sep = "")
    cat("# Operations pending: ", ast_count_ops(.subset2(x, "lazy_ops")), "\n", sep = "")
    grouping <- lazy_grouping_sum(x)
    if (length(grouping) > 0L) {
      # no group count: it would need the plan. Same layout as the eager
      # header, whose first line is dropped.
      cat(gibble_header(x, width, c("A gibble" = "", grouping))[-1L], sep = "\n")
    }
    return(invisible(x))
  }

  ptr <- .subset2(x, "ptr")
  schema <- .subset2(x, "schema")
  types <- gpu_col_types(ptr)
  col_names <- display_names(schema$names)

  cat(gibble_header(x, width), sep = "\n")

  # Get preview data (first 10 rows for value display)
  preview <- tryCatch(
    gpu_head(ptr, 10L, col_names),
    error = function(e) NULL
  )

  # Calculate column name width for alignment
  max_name_width <- max(nchar(col_names), 1)

  zones <- tz_by_position(schema)

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
        available_width <- body_w - max_name_width - nchar(type_str) - 6
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

# The group or rowwise vars of `x` as tbl_sum() entries, without a group count
# (a lazy table's plan is not run to print it).
lazy_grouping_sum <- function(x) {
  groups <- .subset2(x, "groups")
  if (is_rowwise_gpu(x)) {
    c(Rowwise = paste(groups, collapse = ", "))
  } else if (length(groups) > 0L) {
    c(Groups = paste(groups, collapse = ", "))
  } else {
    character()
  }
}

# The print width in pillar's order (pillar 1.11.1, get_width_print()): the
# `width` argument, then options pillar.width, tibble.width and dplyr.width
# (pillar plans to soft-deprecate the last two in v2.0.0), then
# getOption("width"). tbl_format_header() caps the header at
# cli::console_width() on top of that.
print_width <- function(width = NULL) {
  width %||% getOption("pillar.width") %||% getOption("tibble.width") %||%
    getOption("dplyr.width") %||% getOption("width", 80)
}

# The width of the `$ col <type> values` body: glimpse()'s rule (pillar
# 1.11.1, get_width_glimpse()). The `width` argument as given; otherwise the
# option chain of print_width(), used only if finite (an infinite
# pillar.width / tibble.width, a common setting, would let the lines run past
# the console), else getOption("width").
body_width <- function(width = NULL) {
  if (!is.null(width)) {
    return(width)
  }
  width <- print_width()
  if (is.finite(width)) width else getOption("width", 80)
}

# tibble's header lines for `sum` (default tbl_sum(x)) with the first name
# ("A tibble" / "A data frame") renamed "A gibble": pillar's own
# tbl_format_header(), so the "name:" padding, wrapping at `width` with "#   "
# continuation lines and subtle styling are exactly tibble's. `setup` carries
# the fields that method reads (tbl_sum, width; see ?pillar::tbl_format_setup).
# `width` is the resolved print width (print_width()).
gibble_header <- function(x, width, sum = tbl_sum(x)) {
  names(sum)[1] <- "A gibble"
  pillar::tbl_format_header(x, list(tbl_sum = sum, width = width))
}

# Column names as printed: an NA name shows as "NA" (nchar(NA_character_) is
# NA and would break the width arithmetic).
display_names <- function(nms) {
  nms[is.na(nms)] <- "NA"
  nms
}
