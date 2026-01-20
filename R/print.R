# Print method for tbl_gpu (glimpse-style output)

#' @export
print.tbl_gpu <- function(x, ..., width = NULL) {
  width <- width %||% getOption("width", 80)

  if (is.null(x$ptr)) {
    cat("# A GPU tibble [lazy, not materialized]\n")
    cat("# Schema: ", paste(x$schema$names, collapse = ", "), "\n")
    cat("# Operations pending: ", length(x$lazy_ops), "\n")
    return(invisible(x))
  }

  dims <- dim(x)
  types <- gpu_col_types(x$ptr)
  col_names <- x$schema$names

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

  # Print each column
  for (i in seq_along(col_names)) {
    col_name <- col_names[i]
    col_type <- types[i]

    # Format: $ name <type> value1, value2, ...
    name_pad <- format(col_name, width = max_name_width)
    type_str <- paste0("<", col_type, ">")

    # Get preview values
    if (!is.null(preview)) {
      vals <- preview[[i]]
      if (length(vals) > 0) {
        # Truncate strings for display
        if (is.character(vals)) {
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
