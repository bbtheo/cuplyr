# GPU-native glimpse() (Phase 3, task 4)
#
# No AST node (glimpse() is a terminal, side-effecting print method, not a
# transformation): reuses the same bounded-preview approach as
# print.tbl_gpu() (R/print.R) -- gpu_head() for a small, capped row sample
# and gpu_dim() plus the schema (schema_type_abbr()) for the counts and types --
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
#' operations are materialized (via [compute()]), but the actual preview data comes
#' from a small, capped row sample (like [print.tbl_gpu()]), and row/column
#' counts/types come from lightweight metadata calls -- never a full
#' `collect()`.
#'
#' A lazy table with pending operations runs its plan once for the preview
#' (the result is freed when `glimpse()` returns; the table itself is
#' unchanged), and one extra line starting `Lazy:` follows `Columns:`. If the
#' plan fails (an error, out of GPU memory), `glimpse()` prints `Rows: ??`, the
#' `Lazy:` line, the message and the stored schema instead of raising.
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

  # A lazy table with pending ops runs its plan once for the preview and keeps
  # nothing; if the plan fails, the failure is shown instead of raised.
  lazy_line <- NULL
  materialized <- x
  if (has_pending_ops(x)) {
    n_ops <- ast_count_ops(.subset2(x, "lazy_ops"))
    lazy_line <- paste0("Lazy: ", n_ops, " pending step", if (n_ops != 1L) "s",
                        " ran for this preview; nothing is kept\n")
    res <- tryCatch(compute(x), error = function(e) e)
    if (inherits(res, "error")) {
      glimpse_lazy_failure(x, res, lazy_line)
      return(invisible(x))
    }
    materialized <- res
  }

  dims <- dim(materialized)
  col_names <- display_names(materialized$schema$names)
  lvls <- levels_by_position(materialized$schema)
  types <- schema_type_abbr(materialized$schema, materialized[["unspecified"]], lvls)

  cat("Rows: ", format(dims[1], big.mark = ","), "\n", sep = "")
  cat("Columns: ", dims[2], "\n", sep = "")
  cat(lazy_line)

  if (is_rowwise_gpu(materialized)) {
    # one group per row: the vars are the table's header, as pillar prints it
    cat("Rowwise: ", paste(materialized$groups, collapse = ", "), "\n", sep = "")
  } else if (length(materialized$groups) > 0) {
    n_groups <- glimpse_n_groups(materialized)
    cat("Groups: ", paste(materialized$groups, collapse = ", "),
        " [", format(n_groups, big.mark = ","), "]\n", sep = "")
  }

  preview_n <- min(100L, dims[1])
  # also at zero rows: the empty preview still tells a real dictionary from an int
  preview <- tryCatch(gpu_head(materialized$ptr, preview_n, col_names), error = function(e) NULL)

  types <- preview_type_abbr(types, preview, lvls)

  max_name_width <- max(nchar(col_names), 1)
  zones <- tz_by_position(materialized$schema)
  stypes <- unname(materialized$schema$types)

  type_strs <- paste0("<", types, ">")
  val_strs <- character(length(col_names))
  for (i in seq_along(col_names)) {
    if (!is.null(preview) && is.null(preview[[i]])) {
      val_strs[i] <- unholdable_placeholder
    } else if (!is.null(preview) && length(preview[[i]]) > 0) {
      vals <- preview_strings(preview[[i]], zones[[i]], lvls[[i]], stypes[[i]])
      val_strs[i] <- paste(vals, collapse = ", ")
    }
  }

  # pillar's rule (str_trunc()): column i's values get what its "$ name <type> "
  # prefix leaves, less one column, and a longer line ends in one ellipsis
  # character. pillar hands the widths of ALL columns to the lines that are too
  # wide, in order (the first too-wide line gets the first column's width), so
  # that is what is done here: the lines then equal a tibble's.
  data_width <- width - (max_name_width + nchar(type_strs) + 4L) - 1L
  too_wide <- which(nchar(val_strs) > data_width)
  if (length(too_wide) > 0L) {
    keep <- pmax(rep_len(data_width, length(too_wide)) - 1L, 0L)
    val_strs[too_wide] <- paste0(substr(val_strs[too_wide], 1L, keep),
                                 if (cli::is_utf8_output()) "\u2026" else "~")
  }

  for (i in seq_along(col_names)) {
    name_pad <- format(col_names[i], width = max_name_width)
    cat("$ ", name_pad, " ", type_strs[i], " ", val_strs[i], "\n", sep = "")
  }

  invisible(x)
}

# glimpse() of a lazy table whose plan failed: unknown rows, the stored
# schema's columns and types, and the condition's message.
# @keywords internal
glimpse_lazy_failure <- function(x, err, lazy_line) {
  schema <- .subset2(x, "schema")
  col_names <- display_names(schema$names)
  cat("Rows: ??\n")
  cat("Columns: ", length(col_names), "\n", sep = "")
  cat(lazy_line)
  msg <- strsplit(conditionMessage(err), "\n", fixed = TRUE)[[1L]]
  cat(paste0(c("Plan failed: ", rep("  ", max(length(msg), 1L) - 1L)), msg), sep = "\n")
  grouping <- lazy_grouping_sum(x)
  if (length(grouping) > 0L) {
    cat(names(grouping), ": ", grouping, "\n", sep = "")
  }
  max_name_width <- max(nchar(col_names), 1)
  abbr <- schema_type_abbr(schema, x[["unspecified"]])
  for (i in seq_along(col_names)) {
    cat("$ ", format(col_names[i], width = max_name_width), " <",
        abbr[i], ">\n", sep = "")
  }
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
