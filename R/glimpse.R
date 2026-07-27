# GPU-native glimpse() (Phase 3, task 4)
#
# No AST node (glimpse() is a terminal, side-effecting print method, not a
# transformation): reuses the same bounded-preview approach as
# print.tbl_gpu() (R/print.R) -- gpu_head() for a small, capped row sample
# and gpu_dim()/gpu_col_types() for the row/column counts and types --
# rather than dplyr::glimpse()'s own approach of collect()-ing the entire
# table first. This keeps glimpse() cheap even on a huge (or huge
# pending-lazy) tbl_gpu: only pending lazy ops need to run (via compute(),
# same as any other verb materializing before doing GPU work), never a full
# gpu_collect() of every row/column.
#
# Output format deliberately does not attempt to reproduce dplyr's own
# glimpse() (which formats through pillar) byte-for-byte -- only the
# structural facts (row count, column count, grouping, column names/types)
# are guaranteed to match; see glimpse.tbl_gpu()'s roxygen "Details".

#' Get a glimpse of a GPU table's structure
#'
#' Prints a compact summary of a `tbl_gpu`'s structure -- row/column counts,
#' grouping (if any), and each column's type with a preview of its first
#' few values -- similar to `dplyr::glimpse()`.
#'
#' @param x A `tbl_gpu` object created by [tbl_gpu()].
#' @param width Output width (defaults to `getOption("width")`).
#' @param ... Ignored (present for generic compatibility with
#'   `dplyr::glimpse()`).
#'
#' @return `x`, invisibly. Like `dplyr::glimpse()`, this is a
#'   side-effecting print method: it never transforms `.data`, so there is
#'   nothing to re-upload.
#'
#' @details
#' Unlike the (now-removed) Phase 2 CPU-fallback `glimpse()`, this
#' implementation never collects the whole table to R: any pending lazy
#' operations are materialized (via [compute()], exactly as any other verb
#' would need to before doing GPU work), but the actual preview data comes
#' from a small, capped row sample (like [print.tbl_gpu()]), and row/column
#' counts/types come from lightweight metadata calls -- never a full
#' `collect()`.
#'
#' Column value/type formatting does not attempt to reproduce dplyr's own
#' `glimpse()` output (which is formatted via `pillar`) byte-for-byte --
#' only the structural facts (row count, column count, grouping, column
#' names, and column types) are guaranteed to match.
#'
#' @export
#' @importFrom dplyr glimpse
#'
#' @examples
#' if (has_gpu()) {
#'   tbl_gpu(mtcars) |> glimpse()
#' }
glimpse.tbl_gpu <- function(x, width = NULL, ...) {
  width <- width %||% getOption("width", 80)

  materialized <- if (has_pending_ops(x)) compute(x) else x

  dims <- dim(materialized)
  col_names <- materialized$schema$names
  types <- gpu_col_types(materialized$ptr)

  cat("Rows: ", format(dims[1], big.mark = ","), "\n", sep = "")
  cat("Columns: ", dims[2], "\n", sep = "")

  if (length(materialized$groups) > 0) {
    n_groups <- glimpse_n_groups(materialized)
    cat("Groups: ", paste(materialized$groups, collapse = ", "),
        " [", format(n_groups, big.mark = ","), "]\n", sep = "")
  }

  preview_n <- min(100L, dims[1])
  preview <- if (preview_n > 0) {
    tryCatch(gpu_head(materialized$ptr, preview_n, col_names), error = function(e) NULL)
  } else {
    NULL
  }

  max_name_width <- max(nchar(col_names), 1)

  for (i in seq_along(col_names)) {
    col_name <- col_names[i]
    col_type <- types[i]
    name_pad <- format(col_name, width = max_name_width)
    type_str <- paste0("<", col_type, ">")

    if (!is.null(preview) && length(preview[[i]]) > 0) {
      vals <- preview[[i]]
      if (is.character(vals)) {
        vals <- paste0("\"", vals, "\"")
      }
      vals[is.na(vals)] <- "NA"
      val_str <- paste(vals, collapse = ", ")

      available_width <- width - max_name_width - nchar(type_str) - 6
      if (nchar(val_str) > available_width && available_width > 10) {
        val_str <- paste0(substr(val_str, 1, available_width - 3), "...")
      }
    } else {
      val_str <- ""
    }

    cat("$ ", name_pad, " ", type_str, " ", val_str, "\n", sep = "")
  }

  invisible(x)
}

# Number of distinct group-key combinations in a materialized (non-lazy)
# grouped tbl_gpu, for glimpse()'s "Groups: <cols> [<n>]" line. Computed
# entirely on the GPU (select() + distinct(), both GPU-native, Phase 3) plus
# a single gpu_dim() row-count read -- never a full gpu_collect().
# @keywords internal
glimpse_n_groups <- function(materialized) {
  key_tbl <- dplyr::distinct(dplyr::select(materialized, dplyr::all_of(materialized$groups)))
  if (has_pending_ops(key_tbl)) {
    key_tbl <- compute(key_tbl)
  }
  gpu_dim(key_tbl$ptr)[1]
}
