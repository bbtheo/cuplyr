# Parquet on the GPU (F85)
#
# read_parquet_gpu() decodes a parquet file with libcudf on the GPU, so the
# file's compressed bytes are what cross the PCIe bus (DECIMAL columns are the
# exception, they go through the host). A lazy read is a *scan*: an `ast_source` node whose
# `scan` field names the files, holds what the footer said at call time (the
# fingerprint) and is read by lower_scan() when the plan runs. src/io_parquet.cpp
# has the libcudf glue; every condition is raised here.

# The record of the last read, for tests (`parquet_last_read()`).
parquet_state <- new.env(parent = emptyenv())

# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------

# A classed parquet error naming the file. `kind` is one of not_found, invalid,
# unsupported_type, schema_changed, schema_mismatch, write, compression,
# unknown_column, columns.
abort_parquet <- function(kind, msg, path = NULL, bullets = NULL, call = NULL) {
  rlang::abort(
    c(msg, bullets),
    class = c(paste0("cuplyr_error_parquet_", kind), "cuplyr_error_parquet"),
    path = path, call = call
  )
}

# "'a', 'b', 'c', ...": at most three of `paths`.
parquet_files_shown <- function(paths) {
  shown <- paste0("'", utils::head(paths, 3L), "'", collapse = ", ")
  paste0(shown, if (length(paths) > 3L) ", ...")
}

# "parquet file 'a.parquet'", or for several files, which libcudf reads as one
# and does not say which of them it failed on, "one of the parquet files 'a',
# 'b'" (at most three are named).
parquet_files_label <- function(paths) {
  if (length(paths) == 1L) {
    return(sprintf("parquet file '%s'", paths))
  }
  sprintf("one of the parquet files %s", parquet_files_shown(paths))
}

# Map an error out of libcudf to a classed one. A memory error keeps its class
# for wrap_gpu_call() (retry, then `cuplyr_error_gpu_oom`), with the file added
# to the message that wrap_gpu_call() quotes.
parquet_cpp_error <- function(e, path, kind = "invalid", what = "read", pushed = character()) {
  if (inherits(e, "cuplyr_error_parquet")) {
    stop(e)
  }
  if (is_gpu_oom_error(e)) {
    e$message <- paste0(conditionMessage(e), " (", if (identical(what, "write")) "writing " else "reading ",
                        parquet_files_label(path), ")")
    stop(e)
  }
  # (files that passed our own comparison: libcudf also compares the footers'
  # schemas element by element and doesn't say which file differs)
  if (grepl("All sources must have the same schema", conditionMessage(e), fixed = TRUE)) {
    abort_parquet("schema_mismatch",
                  sprintf("Can't read the parquet files %s together.", parquet_files_shown(path)), path,
                  bullets = c(x = "Their footers describe the columns differently, as files from different writers can.",
                              i = "Read each file on its own and combine the results with `bind_rows()`."))
  }
  bullets <- c(x = conditionMessage(e))
  # the reader doesn't say whether the file or the filter it was given is at fault
  if (length(pushed) > 0L) {
    bullets <- c(bullets, i = paste0(
      "The read was given the filter ", paste(pushed, collapse = " & "),
      "; `options(cuplyr.parquet_pushdown = FALSE)` reads without it."
    ))
  }
  abort_parquet(kind, sprintf("Can't %s %s.", what, parquet_files_label(path)), path, bullets = bullets)
}

# -----------------------------------------------------------------------------
# The type map: one place (F105's collect rules)
# -----------------------------------------------------------------------------

# The type the file column has once read (the schema type), by the libcudf type
# the reader returns. A type that is not in the table stays as it is (LIST,
# STRUCT, DURATION_*): it is in the schema but reading it is an error.
parquet_type_map <- function(file_type) {
  switch(file_type,
    BOOL8 = , INT32 = , INT64 = , FLOAT64 = , STRING = , TIMESTAMP_DAYS = ,
    TIMESTAMP_MICROSECONDS = file_type,
    FLOAT32 = , UINT32 = , DECIMAL32 = , DECIMAL64 = , DECIMAL128 = "FLOAT64",
    INT8 = , INT16 = , UINT8 = , UINT16 = "INT32",
    UINT64 = "INT64",
    TIMESTAMP_SECONDS = , TIMESTAMP_MILLISECONDS = , TIMESTAMP_NANOSECONDS = "TIMESTAMP_MICROSECONDS",
    file_type
  )
}

# Can the reader's output of this type be normalised to an R-facing type? A real
# DICTIONARY32 is not in the list on purpose (N10/N14): the reader never returns
# one for the files we map, and if it ever did it would be refused here rather
# than reach a verb, a vctrs path or a preview.
parquet_type_supported <- function(file_type) {
  file_type %in% c("BOOL8", "INT32", "INT64", "FLOAT64", "STRING", "TIMESTAMP_DAYS",
                   "TIMESTAMP_MICROSECONDS", "FLOAT32", "UINT32", "DECIMAL32", "DECIMAL64",
                   "DECIMAL128", "INT8", "INT16", "UINT8", "UINT16", "UINT64",
                   "TIMESTAMP_SECONDS", "TIMESTAMP_MILLISECONDS", "TIMESTAMP_NANOSECONDS")
}

# -----------------------------------------------------------------------------
# The footer
# -----------------------------------------------------------------------------

# Footer facts and the reader's column types, file by file. A missing file, a
# directory and anything libcudf cannot open are classed errors naming the file.
parquet_probe <- function(paths) {
  lapply(paths, function(p) {
    if (!file.exists(p)) {
      abort_parquet("not_found", sprintf("Parquet file '%s' does not exist.", p), p)
    }
    if (dir.exists(p)) {
      abort_parquet("invalid", sprintf("'%s' is a directory, not a parquet file.", p), p,
                    bullets = c(i = "Partitioned directories are not supported; pass the files."))
    }
    info <- tryCatch(gpu_parquet_probe(p)[[1L]],
                     error = function(e) parquet_cpp_error(e, p, "invalid", "read"))
    if (anyDuplicated(info$names) > 0L || anyNA(info$names)) {
      abort_parquet("invalid", sprintf("Parquet file '%s' has duplicated or missing column names.", p), p)
    }
    info
  })
}

# Rows of the files from their footers alone (no column is decoded). The same
# classed errors as a probe.
parquet_nrow <- function(paths) {
  for (p in paths) {
    if (!file.exists(p)) {
      abort_parquet("not_found", sprintf("Parquet file '%s' does not exist.", p), p)
    }
  }
  tryCatch(gpu_parquet_nrow(paths),
           error = function(e) parquet_cpp_error(e, paths, "invalid", "read"))
}

# What a file's footer says that the reader does not: the factor levels, ordered
# flag and time zones, as a list(levels, tz) keyed by column name. Nothing here
# is trusted: a value that does not parse means "no information" for that
# column, and nothing in a footer is ever unserialized.
#  - files cuplyr wrote: `cuplyr.levels.<col>` (netstrings), `cuplyr.ordered.<col>`,
#    `cuplyr.tz.<col>` ("z:<zone>", or "na" for an absent tzone)
#  - files arrow wrote: the Timestamp zones of `ARROW:schema`
#  - otherwise UTC
parquet_footer_meta <- function(info) {
  kv <- info$kv
  get <- function(key) {
    i <- match(key, names(kv))
    if (is.na(i)) NULL else kv[[i]]
  }
  arrow <- NULL
  if (!is.null(raw <- get("ARROW:schema"))) {
    arrow <- parquet_arrow_tz(raw)
  }
  levels <- list()
  tz <- list()
  for (j in seq_along(info$names)) {
    nm <- info$names[[j]]
    ft <- info$types[[j]]
    if (identical(ft, "STRING") && !is.null(raw <- get(paste0("cuplyr.levels.", nm)))) {
      lv <- parquet_decode_netstrings(raw)
      if (!is.null(lv) && !anyDuplicated(lv)) {
        if (identical(get(paste0("cuplyr.ordered.", nm)), "true")) attr(lv, "ordered") <- TRUE
        levels[[nm]] <- lv
      }
    }
    if (startsWith(ft, "TIMESTAMP_") && ft != "TIMESTAMP_DAYS") {
      zone <- "UTC"
      stored <- get(paste0("cuplyr.tz.", nm))
      if (!is.null(stored) && identical(stored, "na")) {
        zone <- NA_character_
      } else if (!is.null(stored) && startsWith(stored, "z:")) {
        zone <- substring(stored, 3L)
      } else if (!is.null(arrow) && !is.na(hit <- match(nm, arrow$names))) {
        zone <- arrow$tz[[hit]]
      }
      tz[[nm]] <- zone
    }
  }
  list(levels = levels, tz = tz)
}

# The R-facing schema of the columns `cols` of a file with `info`'s footer.
parquet_schema_for <- function(cols, info, meta) {
  idx <- match(cols, info$names)
  ftypes <- info$types[idx]
  types <- vapply(ftypes, parquet_type_map, character(1), USE.NAMES = FALSE)
  lvls <- vector("list", length(cols))
  zones <- rep("UTC", length(cols))
  for (k in seq_along(cols)) {
    if (identical(ftypes[[k]], "STRING") && !is.null(l <- meta$levels[[cols[[k]]]])) {
      types[[k]] <- "DICTIONARY32"
      lvls[[k]] <- l
    }
    if (identical(types[[k]], "TIMESTAMP_MICROSECONDS")) {
      zones[[k]] <- meta$tz[[cols[[k]]]] %||% "UTC"
    }
  }
  list(
    names = cols,
    types = types,
    factor_levels = levels_from_positions(cols, lvls),
    tz = tz_from_positions(cols, types, zones)
  )
}

# A schema with the footer's columns: the fingerprint and metadata to compare a
# file against what the scan saw.
parquet_fingerprint <- function(info, meta) {
  list(names = info$names, types = info$types, meta = meta)
}

# What differs between two fingerprints, or NULL.
parquet_diff <- function(want, got) {
  if (!identical(want$names, got$names)) {
    gone <- setdiff(want$names, got$names)
    new <- setdiff(got$names, want$names)
    return(c(
      if (length(gone) > 0L) paste0("column", if (length(gone) > 1L) "s", " ",
                                    paste0("`", gone, "`", collapse = ", "), " gone"),
      if (length(new) > 0L) paste0("new column", if (length(new) > 1L) "s", " ",
                                   paste0("`", new, "`", collapse = ", ")),
      if (length(gone) == 0L && length(new) == 0L) "the columns are in a different order"
    ))
  }
  if (!identical(want$types, got$types)) {
    d <- which(want$types != got$types)[[1L]]
    return(sprintf("column `%s` was %s, now %s", want$names[[d]], want$types[[d]], got$types[[d]]))
  }
  if (!identical(want$meta, got$meta)) {
    return("the factor levels or time zones stored in the footer changed")
  }
  NULL
}

# -----------------------------------------------------------------------------
# read_parquet_gpu()
# -----------------------------------------------------------------------------

#' Read a parquet file straight onto the GPU
#'
#' `read_parquet_gpu()` decodes a parquet file on the GPU with libcudf, so the
#' file's compressed bytes are what cross the PCIe bus (DECIMAL columns are
#' converted through the host, see Types). Like [tbl_gpu()] it respects the
#' execution mode: a lazy table (the default) holds a *scan* that reads the file
#' when the plan runs, an eager one reads it now. The column names and types
#' come from the file footer at call time without reading any data.
#' `write_parquet_gpu()` writes a GPU table with libcudf's writer.
#'
#' @param path For `read_parquet_gpu()`, one path or a vector of paths, read in
#'   order. The files must have the same column names, in the same order, with
#'   the same types. libcudf also requires their footers to describe the columns
#'   identically, including parquet's type annotations, so files from different
#'   writers can be refused even when the columns agree (DuckDB annotates INT32
#'   columns and arrow does not): that is a `cuplyr_error_parquet_schema_mismatch`
#'   when the read runs. For `write_parquet_gpu()`, one path (an existing file is
#'   overwritten).
#' @param columns Optional character vector (not empty): read only these columns,
#'   in this order. A column that is not read is never decoded, so a file with a
#'   column cuplyr cannot hold is readable as long as that column is left out.
#' @param lazy `TRUE`, `FALSE`, or `NULL` for the session default, as in
#'   [tbl_gpu()].
#' @param .data A `tbl_gpu`. A lazy plan is run first (it stays lazy: the
#'   table you passed is not changed), groups are not written.
#' @param compression `"snappy"` (the default), `"zstd"`, `"lz4"` or `"none"`.
#' @param row_group_size Rows per row group, or `NULL` for libcudf's default
#'   (one million).
#'
#' @return `read_parquet_gpu()`: a `tbl_gpu`. `write_parquet_gpu()`: `.data`,
#'   invisibly.
#'
#' @details
#' ## Types
#' BOOLEAN, INT32, DOUBLE, STRING, DATE and microsecond timestamps read as
#' themselves. INT64 reads as `integer64` (exact, no downcast; with the bit64
#' package). FLOAT reads as double, INT8/INT16/UINT8/UINT16 as integer, UINT32
#' as double, UINT64 as `integer64` (an error naming the column if a value is
#' past `INT64_MAX`), DECIMAL as double (the nearest one, with a
#' `cuplyr_warning_collect_precision` warning if more than 15 digits are needed;
#' the conversion runs on the host, so a DECIMAL column makes a round trip over
#' the PCIe bus), second, millisecond and nanosecond timestamps as `POSIXct`
#' (nanoseconds finer than a microsecond are rounded, with that warning). LIST,
#' STRUCT and TIME columns are an error of class
#' `cuplyr_error_parquet_unsupported_type` naming the column and the file when
#' a plan reads the column. `columns =` always keeps it out of the read. A later
#' `select()` usually does too, but a verb that has to run the plan before the
#' `select()` (an integer `sum()` in `mutate()`, for example) reads every column,
#' so use `columns =` if you get this error.
#'
#' An INT32 or INT64 value equal to `-2147483648` or `-9223372036854775808` is
#' read as missing: R's `NA_integer_` and bit64's `NA` are those values, so R
#' (and arrow's `read_parquet()`) cannot hold them as numbers, and a filter or a
#' sum on the GPU has to agree with `collect()`.
#'
#' ## Factors
#' A factor written by `write_parquet_gpu()` comes back as the same factor
#' (levels, unused levels and `ordered`), because the levels are kept in the file
#' footer. A factor with `NA` as a level (`addNA()`) is the exception: a file
#' cannot tell that level from a missing value, so `write_parquet_gpu()` warns
#' (class `cuplyr_warning_parquet_factor_na_level`, naming the column) and the
#' column reads back as strings, with the missing value as `NA`.
#' **A factor written by `arrow::write_parquet()` reads as a string
#' column**: arrow keeps the levels in the parquet dictionary pages, which
#' libcudf does not expose. If you need the factor, convert it afterwards, e.g.
#' `mutate(f = factor(f, levels = ...))` on the collected data.
#'
#' ## Time zones
#' A `POSIXct` column written by `write_parquet_gpu()` or by arrow keeps its
#' time zone. A timestamp with no zone information in the file (no
#' `ARROW:schema` entry, e.g. written by another tool) is read as UTC instants
#' with `tzone = "UTC"`. A timestamp arrow wrote without a zone
#' (`isAdjustedToUTC = false`) is read as UTC instants and has no `tzone`
#' attribute, as in arrow.
#'
#' ## If the file changes
#' A lazy table remembers the columns and types the file had when
#' `read_parquet_gpu()` was called. The plan reads the file again every time it
#' runs. If the columns or their types differ then, the read stops with an
#' error of class `cuplyr_error_parquet_schema_changed` naming the file and the
#' difference. If only the data changed (same columns, other rows), the new data
#' is read: a lazy scan sees the file as it is when the plan runs, not as it was
#' when the table was made. The row count `nrow()` of a bare scan comes from the
#' footer at the time you ask. Reading one lazy table twice reads the file
#' twice.
#'
#' ## Errors
#' All errors have class `cuplyr_error_parquet` and a more specific one, and
#' name the file: `cuplyr_error_parquet_not_found`, `_invalid` (not a parquet
#' file, corrupt footer), `_unsupported_type`, `_schema_changed`,
#' `_schema_mismatch` (files that differ), `_unknown_column` and `_columns`
#' (`columns =` names no column or is empty), `_write` and `_compression`.
#' Several files are read by one libcudf call that doesn't say which file a
#' problem is in, so those messages say "one of the parquet files". Running out
#' of GPU memory is `cuplyr_error_gpu_oom`, with the file in the message.
#'
#' ## Writing
#' `write_parquet_gpu()` writes to a temporary file beside `path` and renames it
#' into place, so a failed write leaves an existing file as it was. It needs
#' write permission on the directory, not only on an existing file.
#'
#' ## Pushdown
#' When a lazy plan over a scan runs through the optimizer (`compute()`,
#' `collect()`), the scan reads only the columns the plan uses, and the
#' conditions of a `filter()` directly on the scan that libcudf can decide
#' exactly are given to the reader, which skips row groups from the file's
#' statistics and drops the other rows while decoding. `explain()` shows the
#' columns and the filter of the scan. A condition is pushed only when it is a
#' comparison of a column with a literal that converts to the column's file
#' type without change (`==`, `!=`, `<`, `<=`, `>`, `>=`) on an integer,
#' `integer64`, `Date` or `POSIXct` column, or `!is.na()` of an `integer` or `integer64` column;
#' the `&`s of a condition are split. Comparisons on double and string columns
#' are not pushed: libcudf 25.12 can skip a row group wrongly from a double
#' column's `NaN` statistics or a string column's deprecated min/max statistics
#' (signed byte order), which would lose rows. They are applied after the read, so a filter on doubles or
#' strings gives the same rows but skips no row groups. A factor written by
#' `arrow::write_parquet()` reads as a string (see above), so it is not pushed
#' either.
#' Everything else (`|`, `!`, `is.na()`, `%in%`, arithmetic, functions, a double
#' literal against an `integer64` column, a column that was converted on
#' reading such as a factor, a filter on a grouped table, with `.by`, or on a
#' rowwise table, a filter that comes after a `mutate()` that changes its column,
#' or after a `slice()`, `distinct()` or `summarise()`) is applied after the read,
#' as without pushdown, and a filter with anything left to apply is applied whole.
#' The answer is the same either way.
#' `options(cuplyr.parquet_pushdown = FALSE)` turns both pushdowns off.
#'
#' A file whose footer statistics are malformed (for example a fixed-width
#' statistic of the wrong length; an invalid file that no common writer
#' produces) can crash the R session when a filter is pushed into the scan. For
#' files you don't trust, `options(cuplyr.parquet_pushdown = FALSE)` avoids it.
#'
#' ## Previews
#' `print()` and `glimpse()` of a table that is a bare scan (or only `select()`s
#' and `rename()`s of one) read just the first rows of the file, which libcudf
#' decodes as whole row groups: when the row group that holds them has more than
#' 65,536 rows or 128 MB of column data, nothing is read and the schema is printed
#' instead (write files with smaller row groups, `row_group_size =`, to get a
#' preview). Any other plan over a file prints its schema and is not run.
#'
#' ## File access
#' libcudf's file layer (kvikio) is put in its compatibility mode, reading
#' through the page cache, unless the environment variable `KVIKIO_COMPAT_MODE`
#' is set before the first parquet file operation. Set it to `AUTO` or `OFF` to
#' let kvikio use GPUDirect Storage where it is installed; that loads
#' `libcufile`, which writes a `cufile.log` into the working directory.
#'
#' When it loads, cuplyr sets the environment variable
#' `LIBCUDF_USE_DEBUG_STREAM_POOL`, a libcudf debug switch that makes libcudf use
#' its default CUDA stream instead of forking new ones. After parquet reads or writes,
#' forked streams slowed later `arrange()`, `filter()` and `summarise()` calls in
#' the session by a few percent; with the switch on, reads and writes of the
#' files measured, up to 1e8 rows, were no slower. A
#' value you set before loading is left alone, but libcudf only tests whether
#' the variable exists, so any value, `0` included, turns the switch on. To turn
#' it off, call `Sys.unsetenv("LIBCUDF_USE_DEBUG_STREAM_POOL")` after loading
#' cuplyr and before the first `read_parquet_gpu()` or `write_parquet_gpu()`
#' call in the session. A lazy `read_parquet_gpu()` counts too: it runs
#' libcudf's reader when it is called. libcudf creates its stream pool once, so
#' unsetting the variable later has no effect.
#'
#' @seealso [tbl_gpu()], [collect.tbl_gpu()]
#' @export
#' @examples
#' if (has_gpu()) {
#'   path <- tempfile(fileext = ".parquet")
#'   write_parquet_gpu(tbl_gpu(mtcars), path)
#'   read_parquet_gpu(path) |>
#'     filter(mpg > 25) |>
#'     collect()
#'   unlink(path)
#' }
read_parquet_gpu <- function(path, columns = NULL, lazy = NULL) {
  if (!is.character(path) || length(path) == 0L || anyNA(path)) {
    rlang::abort("`path` must be a character vector of file paths.", class = "cuplyr_error_parquet")
  }
  if (!is.null(columns) && (!is.character(columns) || anyNA(columns) || anyDuplicated(columns))) {
    rlang::abort("`columns` must be a character vector of unique column names.",
                 class = "cuplyr_error_parquet")
  }
  if (!is.null(columns) && length(columns) == 0L) {
    # (a table without columns has no row count to carry)
    abort_parquet("columns", sprintf("`columns` is empty: nothing to read from parquet file '%s'.", path[[1L]]),
                  path[[1L]], bullets = c(i = "Leave `columns` out to read every column."))
  }
  mode <- resolve_exec_mode(lazy)
  paths <- normalizePath(path.expand(path), mustWork = FALSE)

  wrap_gpu_call("read_parquet", {
    scan <- parquet_make_scan(paths, columns)
    schema <- scan$schema
    scan$schema <- NULL
    if (identical(mode, "lazy")) {
      new_tbl_gpu(ptr = NULL, schema = schema, lazy_ops = ast_source(schema, scan = scan),
                  exec_mode = "lazy")
    } else {
      new_tbl_gpu(ptr = lower_scan(scan, schema), schema = schema, exec_mode = "eager")
    }
  })
}

# The scan record of `paths`: the footers probed, the files compared with each
# other, the requested columns checked. Returns it with its R-facing `schema`.
parquet_make_scan <- function(paths, columns) {
  infos <- parquet_probe(paths)
  metas <- lapply(infos, parquet_footer_meta)
  first <- parquet_fingerprint(infos[[1L]], metas[[1L]])
  for (k in seq_along(paths)[-1L]) {
    d <- parquet_diff(first, parquet_fingerprint(infos[[k]], metas[[k]]))
    if (!is.null(d)) {
      abort_parquet("schema_mismatch",
                    sprintf("Parquet file '%s' does not have the columns of '%s'.", paths[[k]], paths[[1L]]),
                    paths[[k]], bullets = c(x = d))
    }
  }
  if (!is.null(columns)) {
    miss <- setdiff(columns, infos[[1L]]$names)
    if (length(miss) > 0L) {
      abort_parquet("unknown_column",
                    sprintf("Parquet file '%s' has no column%s %s.", paths[[1L]],
                            if (length(miss) > 1L) "s" else "", paste0("`", miss, "`", collapse = ", ")),
                    paths[[1L]])
    }
  }
  cols <- columns %||% infos[[1L]]$names
  schema <- parquet_schema_for(cols, infos[[1L]], metas[[1L]])
  list(
    format = "parquet",
    paths = paths,
    file_names = first$names,
    file_types = first$types,
    nrow = sum(vapply(infos, function(i) i$nrow, numeric(1))),
    row_groups = sum(vapply(infos, function(i) i$row_groups, integer(1))),
    columns = columns,
    predicates = list(),
    meta = first$meta,
    schema = schema
  )
}

# Read a scan: probe the footers again, compare them with the scan's fingerprint,
# read, normalise, report. `schema` is the R-facing schema of the columns the
# read returns. `scan$limit` (a preview) reads only that many rows.
# @return the pointer to the GPU table
lower_scan <- function(scan, schema) {
  paths <- scan$paths
  infos <- parquet_probe(paths)
  want <- list(names = scan$file_names, types = scan$file_types, meta = scan$meta)
  for (k in seq_along(paths)) {
    d <- parquet_diff(want, parquet_fingerprint(infos[[k]], parquet_footer_meta(infos[[k]])))
    if (!is.null(d)) {
      abort_parquet("schema_changed",
                    sprintf("Parquet file '%s' changed after it was opened.", paths[[k]]),
                    paths[[k]], bullets = c(x = d, i = "Call `read_parquet_gpu()` again."))
    }
  }

  cols <- scan$columns %||% scan$file_names
  if (!identical(cols, schema$names)) {
    rlang::abort("Internal error: the scan's columns differ from its schema.",
                 class = "cuplyr_error_internal")
  }
  ftypes <- scan$file_types[match(cols, scan$file_names)]
  # (a preview reads what the file holds and shows a placeholder for a type R
  # can't hold, as it does for any GPU column of such a type; no verb sees it)
  bad <- which(!vapply(ftypes, parquet_type_supported, logical(1)))
  if (length(bad) > 0L && !isTRUE(scan$preview)) {
    j <- bad[[1L]]
    abort_parquet("unsupported_type",
                  sprintf("Can't read column `%s` of %s: its type (%s) has no R equivalent.",
                          cols[[j]], parquet_files_label(paths), ftypes[[j]]),
                  paths[[1L]], bullets = c(i = "Leave it out with `columns =` or `select()`."))
  }

  lvls <- levels_by_position(schema)
  limit <- scan$limit %||% -1
  atoms <- unlist(lapply(scan$predicates, `[[`, "atoms"), recursive = FALSE, use.names = FALSE)
  pushed <- vapply(scan$predicates, function(p) p$text, character(1))
  ptr <- tryCatch(
    gpu_parquet_read(paths, scan$columns, limit, unname(schema$types), lvls, atoms %||% list(), cols),
    error = function(e) parquet_cpp_error(e, paths, "invalid", "read", pushed = pushed)
  )
  parquet_raise_read_issues(ptr, cols, paths)
  rows <- attr(ptr, "rows_read", exact = TRUE)
  total <- attr(ptr, "row_groups_total", exact = TRUE)
  after <- attr(ptr, "row_groups_after_stats", exact = TRUE)
  bloom <- attr(ptr, "row_groups_after_bloom", exact = TRUE)
  for (a in c("rows_read", "row_groups_total", "row_groups_after_stats", "row_groups_after_bloom", "lossy",
              "u64_over", "dec_over", "bad_factor")) {
    attr(ptr, a) <- NULL
  }

  parquet_state$last <- list(
    paths = paths,
    columns = cols,
    predicates = pushed,
    rows_read = rows,
    row_groups_total = total,
    row_groups_after_stats = after,
    row_groups_after_bloom = bloom
  )
  ptr
}

# The conditions of a read, raised here and never in C++. A read of several
# files is one libcudf call that doesn't say which file a column's problem is in.
parquet_raise_read_issues <- function(ptr, cols, paths) {
  where <- parquet_files_label(paths)
  if (!is.null(j <- attr(ptr, "u64_over", exact = TRUE))) {
    rlang::abort(
      c(sprintf("Can't read column `%s` of %s as integer64.", cols[[j[[1L]]]], where),
        x = "It holds a UINT64 value past INT64_MAX (9223372036854775807), which R cannot represent without wrapping."),
      class = c("cuplyr_error_unsupported_gpu_type", "cuplyr_error_parquet"), path = paths[[1L]], call = NULL)
  }
  if (!is.null(j <- attr(ptr, "dec_over", exact = TRUE))) {
    rlang::abort(
      c(sprintf("Can't read column `%s` of %s as double.", cols[[j[[1L]]]], where),
        x = "It holds a DECIMAL value beyond the range of a double."),
      class = c("cuplyr_error_unsupported_gpu_type", "cuplyr_error_parquet"), path = paths[[1L]], call = NULL)
  }
  if (!is.null(j <- attr(ptr, "bad_factor", exact = TRUE))) {
    abort_parquet("invalid",
                  sprintf("Column `%s` of %s holds a value that is not one of the factor levels stored in its footer.",
                          cols[[j[[1L]]]], where), paths[[1L]])
  }
  if (!is.null(j <- attr(ptr, "lossy", exact = TRUE))) {
    warning(structure(
      class = c("cuplyr_warning_collect_precision", "warning", "condition"),
      list(message = paste0(
        "Column", if (length(j) > 1L) "s", " ", paste0("`", cols[j], "`", collapse = ", "),
        " of ", where, " cannot be held exactly in R and ", if (length(j) > 1L) "were" else "was",
        " rounded (a double keeps 15 significant decimal digits; POSIXct keeps microseconds)."
      ), call = NULL)
    ))
  }
  invisible()
}

# Internal test hook: what the last scan read asked of the reader (paths, columns,
# predicates, rows_read, row_groups_total, row_groups_after_stats,
# row_groups_after_bloom), NULL before any read since the last `reset = TRUE`. A
# footer probe is not a read.
# @keywords internal
parquet_last_read <- function(reset = FALSE) {
  out <- parquet_state$last
  if (isTRUE(reset)) {
    parquet_state$last <- NULL
    return(invisible(out))
  }
  out
}

# -----------------------------------------------------------------------------
# write_parquet_gpu()
# -----------------------------------------------------------------------------

#' @rdname read_parquet_gpu
#' @export
write_parquet_gpu <- function(.data, path, compression = "snappy", row_group_size = NULL) {
  if (!is_tbl_gpu(.data)) {
    rlang::abort("`.data` must be a tbl_gpu.", class = "cuplyr_error_parquet")
  }
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    rlang::abort("`path` must be a single file path.", class = "cuplyr_error_parquet")
  }
  if (!is.character(compression) || length(compression) != 1L ||
      !compression %in% c("snappy", "zstd", "lz4", "none")) {
    abort_parquet("compression",
                  sprintf("`compression` must be one of \"snappy\", \"zstd\", \"lz4\" or \"none\", not %s.",
                          paste0("\"", paste(compression, collapse = ", "), "\"")), path)
  }
  if (!is.null(row_group_size) &&
      !(is.numeric(row_group_size) && length(row_group_size) == 1L && !is.na(row_group_size) &&
        row_group_size >= 1 && row_group_size == round(row_group_size))) {
    rlang::abort("`row_group_size` must be a positive whole number.", class = "cuplyr_error_parquet")
  }
  path <- path.expand(path)
  if (!dir.exists(dirname(path))) {
    abort_parquet("write", sprintf("Can't write parquet file '%s': the directory '%s' does not exist.",
                                   path, dirname(path)), path)
  }
  if (dir.exists(path)) {
    abort_parquet("write", sprintf("Can't write parquet file '%s': it is a directory.", path), path)
  }
  if (file.exists(path)) {
    if (file.access(path, 2L) != 0L) {
      abort_parquet("write", sprintf("Can't write parquet file '%s': it exists and is not writable.", path), path)
    }
    # (a symbolic link is written through, not replaced)
    if (nzchar(Sys.readlink(path))) path <- normalizePath(path)
  }

  schema <- current_schema(.data)
  nms <- schema$names
  if (length(nms) == 0L) {
    abort_parquet("write", sprintf("Can't write parquet file '%s': the table has no columns.", path), path)
  }
  if (anyDuplicated(nms) > 0L || anyNA(nms)) {
    abort_parquet("write", sprintf("Can't write parquet file '%s': column names must be unique and not NA.", path),
                  path)
  }

  tmp <- NULL
  on.exit(if (!is.null(tmp)) unlink(tmp), add = TRUE)  # (an interrupt)
  wrap_gpu_call("write_parquet", {
    tbl <- if (has_pending_ops(.data) && is_lazy(.data)) compute(.data) else .data
    ptr <- tbl$ptr
    types <- unname(schema$types)
    lvls <- levels_by_position(schema)
    zones <- tz_by_position(schema)
    keys <- character()
    vals <- character()
    na_level <- character()
    for (j in seq_along(nms)) {
      if (identical(types[[j]], "DICTIONARY32") && !is.null(lv <- lvls[[j]])) {
        ptr <- gpu_decode_factor(ptr, j - 1L, as.character(lv))
        if (!anyNA(lv)) {
          keys <- c(keys, paste0("cuplyr.levels.", nms[[j]]))
          vals <- c(vals, parquet_encode_netstrings(as.character(lv)))
          if (isTRUE(attr(lv, "ordered"))) {
            keys <- c(keys, paste0("cuplyr.ordered.", nms[[j]]))
            vals <- c(vals, "true")
          }
        } else {
          na_level <- c(na_level, nms[[j]])
        }
      } else if (identical(types[[j]], "TIMESTAMP_MICROSECONDS") && (is.na(zones[[j]]) || zones[[j]] != "UTC")) {
        keys <- c(keys, paste0("cuplyr.tz.", nms[[j]]))
        vals <- c(vals, if (is.na(zones[[j]])) "na" else paste0("z:", zones[[j]]))
      }
    }
    if (length(na_level) > 0L) {
      # an NA level and a missing value are the same null in the file
      warning(structure(
        class = c("cuplyr_warning_parquet_factor_na_level", "warning", "condition"),
        list(message = paste0(
          "Factor column", if (length(na_level) > 1L) "s", " ", paste0("`", na_level, "`", collapse = ", "),
          " of the table written to '", path, "' ", if (length(na_level) > 1L) "have" else "has",
          " NA as a level. A file cannot tell that level from a missing value, so ",
          if (length(na_level) > 1L) "they" else "it", " will be read back as ",
          if (length(na_level) > 1L) "strings" else "a string", " with missing values, not as ",
          if (length(na_level) > 1L) "factors" else "a factor", "."
        ), call = NULL)
      ))
    }
    # Write beside the target and rename it into place, so a write that fails
    # (a full disk, a directory without room) leaves the file it was asked to
    # replace as it was.
    tmp <- tempfile(pattern = paste0(".", basename(path), "."), tmpdir = dirname(path), fileext = ".tmp")
    tryCatch(
      gpu_parquet_write(ptr, nms, tmp, compression, as.integer(row_group_size %||% 0L), keys, vals),
      error = function(e) {
        unlink(tmp)
        tmp <<- NULL
        parquet_cpp_error(e, path, "write", "write")
      }
    )
    if (file.exists(path)) try(Sys.chmod(tmp, file.info(path)$mode), silent = TRUE)
    if (!file.rename(tmp, path)) {
      abort_parquet("write", sprintf("Can't write parquet file '%s': the finished file could not be moved into place.",
                                     path), path)
    }
    tmp <- NULL
    NULL
  })
  invisible(.data)
}

# -----------------------------------------------------------------------------
# Plans over a scan: previews and dim()
# -----------------------------------------------------------------------------

# The scan a lazy table's plan IS (a bare scan), or NULL.
bare_scan_of <- function(x) {
  ops <- .subset2(x, "lazy_ops")
  if (inherits(ops, "ast_node") && identical(.subset2(ops, "type"), "source")) {
    .subset2(ops, "scan")
  }
}

# How a lazy table's plan relates to a parquet scan: NULL (no scan below it),
# "projection" (a bare scan, or only select()s and rename()s of one: a preview
# can read just the first rows) or "other" (anything else over a file: nothing
# is read to print it).
scan_plan_kind <- function(x) {
  node <- .subset2(x, "lazy_ops")
  if (!inherits(node, "ast_node")) {
    return(NULL)
  }
  rows_kept <- length(.subset2(x, "groups")) == 0L && !is_rowwise_gpu(x)
  plan_scan_kind(node, rows_kept)
}

plan_scan_kind <- function(node, projection_only) {
  while (!is.null(node)) {
    type <- .subset2(node, "type")
    if (identical(type, "source")) {
      if (is.null(.subset2(node, "scan"))) return(NULL)
      return(if (projection_only) "projection" else "other")
    }
    if (identical(type, "join")) {
      l <- plan_scan_kind(.subset2(node, "left"), FALSE)
      r <- plan_scan_kind(.subset2(node, "right"), FALSE)
      return(if (is.null(l) && is.null(r)) NULL else "other")
    }
    if (!type %in% c("select", "rename")) projection_only <- FALSE
    node <- .subset2(node, "input")
  }
  NULL
}

# The plan of `x` with its scan limited to `n` rows (only called on a
# "projection" plan: the leaf is reached through select/rename nodes).
with_scan_limit <- function(node, n) {
  if (identical(.subset2(node, "type"), "source")) {
    node$scan$limit <- n
    node$scan$preview <- TRUE
    return(node)
  }
  node$input <- with_scan_limit(.subset2(node, "input"), n)
  node
}

# The scan at the bottom of a "projection" plan.
scan_of_projection <- function(x) {
  node <- .subset2(x, "lazy_ops")
  while (!identical(.subset2(node, "type"), "source")) node <- .subset2(node, "input")
  .subset2(node, "scan")
}

# Rows of the file(s) behind a "projection" plan, from the footer alone.
scan_footer_nrow <- function(x) {
  parquet_nrow(scan_of_projection(x)$paths)
}

# libcudf decodes whole row groups, so a preview is only as cheap as the row
# groups that hold its first rows. A first row group of this many rows or fewer,
# and of this many bytes of column data (the footer's size of all its columns,
# uncompressed, even if the plan selects a few), is a preview-sized read; anything
# larger is not read for a preview.
scan_preview_max_rows <- 65536
scan_preview_max_bytes <- 128 * 1024^2

# The first `n` rows of a "projection" plan over a scan as a materialized table:
# the plan runs over a scan limited to `n` rows. NULL when the row groups that
# hold those rows are large (printing a table must not read a big file); errors
# propagate.
scan_preview_table <- function(x, n) {
  paths <- scan_of_projection(x)$paths
  for (p in paths) {
    if (!file.exists(p)) {
      abort_parquet("not_found", sprintf("Parquet file '%s' does not exist.", p), p)
    }
  }
  widest <- tryCatch(gpu_parquet_preview_rows(paths, n, scan_preview_max_bytes),
                     error = function(e) parquet_cpp_error(e, paths, "invalid", "read"))
  if (widest > scan_preview_max_rows) {
    return(NULL)
  }
  plan <- compute_plan(x, with_scan_limit(.subset2(x, "lazy_ops"), n))
  new_tbl_gpu(ptr = plan$ptr, schema = plan$schema, exec_mode = "eager",
              unspecified = .subset2(x, "unspecified"))
}

# "5,000 x 2" the way pillar::dim_desc() writes a table's dimensions.
dim_desc_n <- function(nrow, ncol) {
  mark <- if (identical(getOption("OutDec"), ",")) "." else ","
  paste0(formatC(nrow, big.mark = mark, format = "d"),
         if (cli::is_utf8_output()) " × " else " x ", formatC(ncol, big.mark = mark, format = "d"))
}

# "parquet[a.parquet; cols=x,y]" for explain()/show_query(): the file(s), the
# columns the scan returns and, once pushed (phase B), the filter.
scan_to_string <- function(scan, schema) {
  files <- paste(basename(scan$paths), collapse = ",")
  filt <- if (length(scan$predicates) > 0L) {
    paste0("; filter=", paste(vapply(scan$predicates, function(p) p$text %||% "?", ""), collapse = " & "))
  }
  paste0("parquet[", files, "; cols=", paste(schema$names, collapse = ","), filt, "]")
}
