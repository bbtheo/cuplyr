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

  # A lazy table with pending operations runs its plan once and prints the
  # result like an eager table, with "[lazy]" in the header. Nothing is kept:
  # `res` is a local, freed when print() returns, and `x` stays as it is.
  if (has_pending_ops(x)) {
    res <- tryCatch(compute(x), error = function(e) e)
    if (inherits(res, "error")) {
      print_lazy_failure(x, res, width)
    } else {
      print_gibble_body(res, width, body_w, "A gibble [lazy]")
      cat(lazy_footer(), "\n", sep = "")
    }
    return(invisible(x))
  }

  # No GPU data and nothing to run: show the schema only
  if (is.null(.subset2(x, "ptr"))) {
    cat("# A gibble [lazy, not materialized]\n")
    cat(schema_line(.subset2(x, "schema"), width, .subset2(x, "unspecified")), sep = "\n")
    cat("# Operations pending: ", ast_count_ops(.subset2(x, "lazy_ops")), "\n", sep = "")
    grouping <- lazy_grouping_sum(x)
    if (length(grouping) > 0L) {
      cat(gibble_header(x, width, c("A gibble" = "", grouping))[-1L], sep = "\n")
    }
    cat(lazy_footer(), "\n", sep = "")
    return(invisible(x))
  }

  print_gibble_body(x, width, body_w, "A gibble")
  invisible(x)
}

# A lazy table whose plan failed (an error, out of GPU memory): the header with
# unknown rows, the grouping (metadata), the schema, the plan size and the
# condition's message. print() never re-raises it.
print_lazy_failure <- function(x, err, width) {
  ncols <- length(.subset2(x, "schema")$names)
  dims <- sub("^1", "??", pillar::dim_desc(matrix(0L, 1L, ncols)))
  sum <- c(dims, lazy_grouping_sum(x))
  cat(gibble_header(x, width, sum, name = "A gibble [lazy]"), sep = "\n")
  cat(schema_line(.subset2(x, "schema"), width, .subset2(x, "unspecified")), sep = "\n")
  cat("# Operations pending: ", ast_count_ops(.subset2(x, "lazy_ops")), "\n", sep = "")
  msg <- strsplit(conditionMessage(err), "\n", fixed = TRUE)[[1L]]
  cat(paste0(c("# Plan failed: ", rep("#   ", max(length(msg), 1L) - 1L)), msg), sep = "\n")
  cat(lazy_footer(), "\n", sep = "")
}

# The one-line hint ending a lazy print: where the data and the plan are.
lazy_footer <- function() {
  pillar::style_subtle(paste0("# ", cli::symbol$info,
                              " Use collect() to get the data and explain() to see the plan"))
}

# "# Schema: a <dbl>, b <chr>" wrapped at `width`, continuation lines "#   ".
# A single entry wider than the width keeps its own line.
schema_line <- function(schema, width, unspecified = NULL) {
  entries <- paste0(display_names(schema$names), " <",
                    schema_type_abbr(schema, unspecified), ">")
  lead <- "# Schema: "
  cont <- "#   "
  lines <- character()
  cur <- lead
  first <- TRUE
  for (k in seq_along(entries)) {
    piece <- paste0(entries[[k]], if (k < length(entries)) "," else "")
    sep <- if (first) "" else " "
    if (!first && nchar(cur) + nchar(sep) + nchar(piece) > width) {
      lines <- c(lines, cur)
      cur <- paste0(cont, piece)
    } else {
      cur <- paste0(cur, sep, piece)
    }
    first <- FALSE
  }
  c(lines, cur)
}

# The label pillar shows for each column of the collected table, from the
# schema alone (no collect()): by position, so duplicated names are safe.
# `unspecified` is the table's `$unspecified` field: collect_frame() turns those
# BOOL8 columns (the first column of each name) back into vctrs::unspecified(),
# which pillar labels "???". `lvls` is levels_by_position(schema), when the
# caller already has it.
schema_type_abbr <- function(schema, unspecified = NULL, lvls = levels_by_position(schema)) {
  types <- unname(schema$types)
  out <- character(length(types))
  for (i in seq_along(types)) {
    out[[i]] <- switch(types[[i]],
      DICTIONARY32 = if (is.null(lvls[[i]])) "int"
                     else if (isTRUE(attr(lvls[[i]], "ordered"))) "ord" else "fct",
      FLOAT64 = , FLOAT32 = "dbl",
      INT64 = "int64",
      INT32 = "int",
      INT16 = "i16",
      INT8 = "i8",
      BOOL8 = "lgl",
      STRING = "chr",
      TIMESTAMP_DAYS = "date",
      if (isTRUE(startsWith(types[[i]], "TIMESTAMP_"))) "dttm" else tolower(types[[i]])
    )
  }
  if (length(unspecified) > 0L) {
    hit <- match(unspecified, schema$names)
    hit <- hit[!is.na(hit) & types[hit] %in% "BOOL8"]
    out[hit] <- "???"
  }
  out
}

# The header and the glimpse-style body of a table with GPU data and nothing
# pending, `name` being the header's first name ("A gibble").
print_gibble_body <- function(x, width, body_w, name) {
  ptr <- .subset2(x, "ptr")
  schema <- .subset2(x, "schema")
  lvls <- levels_by_position(schema)
  types <- schema_type_abbr(schema, .subset2(x, "unspecified"), lvls)
  col_names <- display_names(schema$names)

  cat(gibble_header(x, width, name = name), sep = "\n")

  # Get preview data (first 10 rows for value display)
  preview <- tryCatch(
    gpu_head(ptr, 10L, col_names),
    error = function(e) NULL
  )

  # Calculate column name width for alignment
  max_name_width <- max(nchar(col_names), 1)

  zones <- tz_by_position(schema)
  stypes <- unname(schema$types)

  # Print each column
  for (i in seq_along(col_names)) {
    col_name <- col_names[i]
    col_type <- types[i]

    # Format: $ name <type> value1, value2, ...
    name_pad <- format(col_name, width = max_name_width)
    type_str <- paste0("<", col_type, ">")

    # Get preview values
    if (!is.null(preview)) {
      vals <- preview_strings(preview[[i]], zones[[i]], lvls[[i]], stypes[[i]],
                              truncate = TRUE)
      if (length(vals) > 0) {
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
# (a lazy table's plan is not run to print it) unless fixed group keys already
# hold the groups (F71), whose count is then known.
lazy_grouping_sum <- function(x) {
  groups <- .subset2(x, "groups")
  if (is_rowwise_gpu(x)) {
    c(Rowwise = paste(groups, collapse = ", "))
  } else if (length(groups) > 0L) {
    keys <- fixed_keys_of(x)
    n <- if (!is.null(keys)) {
      mark <- if (identical(getOption("OutDec"), ",")) "." else ","
      paste0(" [", formatC(nrow(keys), big.mark = mark, format = "d"), "]")
    } else {
      ""
    }
    c(Groups = paste0(paste(groups, collapse = ", "), n))
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
# `width` is the resolved print width (print_width()). A different `name`
# ("A gibble [lazy]") only replaces the text of the first line, after the
# padding, so the other lines stay those of an eager table.
gibble_header <- function(x, width, sum = tbl_sum(x), name = "A gibble") {
  names(sum)[1] <- "A gibble"
  out <- pillar::tbl_format_header(x, list(tbl_sum = sum, width = width))
  if (!identical(name, "A gibble")) {
    out[1L] <- sub("A gibble", name, out[1L], fixed = TRUE)
  }
  out
}

# Column names as printed: an NA name shows as "NA" (nchar(NA_character_) is
# NA and would break the width arithmetic).
display_names <- function(nms) {
  nms[is.na(nms)] <- "NA"
  nms
}
