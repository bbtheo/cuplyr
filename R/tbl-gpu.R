#' Create a GPU-backed data frame
#'
#' cuplyr's GPU tables, called gibbles, live in GPU memory. `gibble()` is
#' another name for `tbl_gpu()`. It transfers an R data frame to GPU memory
#' using NVIDIA's libcudf library, enabling high-performance data manipulation
#' operations. The resulting `tbl_gpu` object can be used with dplyr verbs
#' like `filter()`, `mutate()`, `select()`, and collected back to R with
#' `collect()`.
#'
#' @param data A data frame or tibble to transfer to GPU memory. Supported
#'   column types include: numeric (double), integer, character, and logical.
#' @param ... Additional arguments passed to methods (currently unused).
#' @param lazy Logical or character. Controls execution mode:
#'   \itemize{
#'     \item `TRUE` or `"lazy"` - Operations are deferred until collect()/compute()
#'     \item `FALSE` or `"eager"` - Operations execute immediately
#'     \item `NULL` (default) - Use the session option, then the environment
#'       variable, then the package default, which is lazy
#'   }
#'   The default can be changed with `options(cuplyr.exec_mode = "eager")` or
#'   the environment variable `CUPLYR_EXEC_MODE=eager`; the argument beats
#'   the option, which beats the variable. An invalid option value warns once
#'   and the default is used. All options are listed on [cuplyr-options].
#'
#' @return A `tbl_gpu` object containing:
#'   \itemize{
#'     \item `ptr` - External pointer to the GPU table
#'     \item `schema` - List with column names and types
#'     \item `lazy_ops` - AST for pending operations (lazy mode only)
#'     \item `groups` - Grouping variables
#'     \item `exec_mode` - Execution mode ("lazy" or "eager")
#'   }
#'
#' @details
#' The data is immediately copied to GPU memory when `tbl_gpu()` is called
#' (also in lazy mode; only the verbs are deferred).
#' GPU memory is automatically freed when the R object is garbage collected.
#'
#' ## Execution Modes
#'
#' In **lazy mode** (the default), operations build an AST (Abstract Syntax
#' Tree) that is optimized and executed only when `collect()` or `compute()`
#' is called.
#' This enables optimizations like:
#' \itemize{
#'   \item Projection pruning (only load needed columns)
#'   \item Mutate fusion (combine multiple mutates)
#'   \item Filter reordering (cheapest filters first)
#' }
#'
#' Errors in a verb's arguments (a missing column, a type error) are still
#' raised when the verb is called. Printing a lazy table with pending
#' operations (or calling `glimpse()` on it) runs the plan to show its first
#' rows, under a `# A gibble [lazy]:` header; the table itself is unchanged and
#' nothing is kept, so printing it twice runs the plan twice (`compute()` it
#' first to look at it repeatedly). If the plan fails (an error, out of GPU
#' memory), the print shows the schema and the message instead of raising.
#'
#' In **eager mode** (`lazy = FALSE`), each dplyr verb executes immediately on
#' the GPU. This is simple but can lead to unnecessary intermediate
#' allocations.
#'
#' For measured timings and memory use of the two modes, see
#' `vignette("performance", package = "cuplyr")`.
#'
#' A `vctrs::unspecified()` column (all `NA`, no type yet) is stored as BOOL8.
#' It collects back as `vctrs::unspecified()` only while it passes through
#' `filter()`, `arrange()`, `slice()` and `semi_join()`/`anti_join()` (as a
#' column of `x`, as in dplyr); every other verb (`select()`, `mutate()`,
#' `group_by()` with computed expressions, `slice_min()`/`slice_max()`, a
#' mutating join, `vctrs::vec_slice()`) returns it as a plain logical column.
#' An unspecified column joined against a string, `Date` or factor key, or
#' bound to a character column in `bind_rows()`, still errors.
#'
#' Column type mappings from R to GPU:
#' \itemize{
#'   \item `numeric` -> FLOAT64
#'   \item `integer` -> INT32
#'   \item `character` -> STRING
#'   \item `logical` -> BOOL8
#'   \item `Date` -> TIMESTAMP_DAYS
#'   \item `POSIXct` -> TIMESTAMP_MICROSECONDS
#' }
#'
#' ## Gibbles
#'
#' "Gibble" is the user-facing name of a `tbl_gpu`: `print()` starts
#' with `# A gibble:` and the dimensions, and `gibble()`, `as_gibble()` and
#' `is_gibble()` are aliases of `tbl_gpu()`, `as_tbl_gpu()` and
#' `is_tbl_gpu()`. The class stays `tbl_gpu`, and `tbl_sum()` still returns
#' dplyr's values.
#'
#' The CRAN package gibble (geometry decomposition, Michael Sumner) also
#' exports a function called `gibble()`. When both packages are attached, the
#' one attached last masks the other; `cuplyr::gibble()` always works.
#'
#' @seealso
#' \code{\link{collect.tbl_gpu}} to transfer data back to R,
#' \code{\link{compute.tbl_gpu}} to execute and keep on GPU,
#' \code{\link{filter.tbl_gpu}}, \code{\link{mutate.tbl_gpu}},
#' \code{\link{select.tbl_gpu}} for data manipulation
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Transfer mtcars to GPU (lazy by default)
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   print(gpu_mtcars)
#'
#'   # Operations are deferred until collect()
#'   result <- tbl_gpu(mtcars) |>
#'     filter(mpg > 20) |>
#'     mutate(kpl = mpg * 0.425) |>
#'     collect()
#'
#'   # Eager mode: each verb runs at once
#'   eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)
#'
#'   # Or make every new table eager:
#'   # options(cuplyr.exec_mode = "eager")
#' }
tbl_gpu <- function(data, ..., lazy = NULL) {
  UseMethod("tbl_gpu")
}

#' @rdname tbl_gpu
#' @export
gibble <- tbl_gpu

# With libcudf's large strings disabled (LIBCUDF_LARGE_STRINGS_ENABLED not "1"),
# a column whose characters reach the INT64-offsets threshold cannot be stored.
# Refuse it here, before df_to_gpu() allocates the columns ahead of it. With
# large strings enabled (the default) this costs one libcudf flag check.
check_string_columns_fit <- function(upload) {
  limit <- large_strings_limit_if_disabled()
  if (limit < 0) return(invisible())
  hit <- first_string_column_over(upload, limit)
  if (is.null(hit)) return(invisible())
  env <- Sys.getenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", unset = NA)
  rlang::abort(
    c(sprintf(
        "Column `%s` holds %s bytes of characters, at or above libcudf's large-strings threshold (%s bytes%s), and large strings are disabled (LIBCUDF_LARGE_STRINGS_ENABLED=\"%s\").",
        names(upload)[hit$index], format(hit$bytes, scientific = FALSE, big.mark = ""),
        format(limit, scientific = FALSE),
        if (is.na(env)) "" else paste0("; LIBCUDF_LARGE_STRINGS_THRESHOLD=\"", env, "\""),
        Sys.getenv("LIBCUDF_LARGE_STRINGS_ENABLED")),
      i = "Unset LIBCUDF_LARGE_STRINGS_ENABLED or set it to \"1\" to upload it with 64-bit string offsets."),
    class = "cuplyr_error_string_too_large", call = NULL
  )
}

#' @rdname tbl_gpu
#' @export
tbl_gpu.data.frame <- function(data, ..., lazy = NULL) {
  # A rowwise_df uploads as a rowwise table (F49): its rowwise vars become
  # `$groups` and the table is flagged `$rowwise` (see new_tbl_gpu()), so no
  # verb can mistake the vars for real groups. The data goes up ungrouped.
  orig <- data
  is_rowwise <- inherits(data, "rowwise_df")
  if (is_rowwise) {
    data <- dplyr::ungroup(data)
  }
  # `vctrs::unspecified()` columns (all NA) upload as plain logical; their
  # names ride on the table (`$unspecified`) so row-only verbs and filtering
  # joins can give the class back at collect().
  unspec <- vapply(data, inherits, logical(1), what = "vctrs_unspecified")
  unspec_names <- NULL
  if (any(unspec)) {
    for (j in which(unspec)) data[[j]] <- rep(NA, nrow(data))
    unspec_names <- set_diff(names(data)[unspec], names(data)[duplicated(names(data))])
  }
  # Raised before wrap_gpu_call() so it is not reworded as a device-memory error.
  check_supported_column_classes(data)
  # Integer-backed Date/POSIXct must reach the C++ upload as doubles.
  # Assign only where a column changes: on a grouped_df every `[[<-` regroups.
  upload <- data
  for (j in seq_along(upload)) {
    col <- upload[[j]]
    if (is.integer(col) && (inherits(col, "Date") || inherits(col, "POSIXct"))) {
      upload[[j]] <- timestamp_as_double(col)
    }
  }
  check_string_columns_fit(upload)
  ptr <- wrap_gpu_call("tbl_gpu", df_to_gpu(upload))
  # TIMESTAMP_DAYS holds whole days: the upload floors a fractional Date (the
  # day R prints) and flags the columns it touched on the pointer.
  frac <- attr(ptr, "fractional_dates")
  if (!is.null(frac)) {
    attr(ptr, "fractional_dates") <- NULL
    warn_fractional_dates(names(upload)[frac])
  }

  # Build schema with factor levels for round-trip fidelity
  factor_levels <- lapply(data, function(col) {
    if (is.factor(col)) {
      # UTF-8 like the strings on the GPU and the ones collect() returns.
      lvls <- enc2utf8(levels(col))
      # `ordered` rides on the level vector so every site that copies levels
      # carries it; collect() and the .drop expansion restore it.
      if (is.ordered(col)) attr(lvls, "ordered") <- TRUE
      lvls
    } else NULL
  })
  # By position: `data` may carry duplicated names. NULL entries are dropped
  # to keep the schema compact.
  types <- vapply(data, gpu_type_from_r, character(1))
  # POSIXct zones (F32): the instant is UTC on the GPU, the tzone is metadata.
  zones <- vapply(data, function(col) {
    if (inherits(col, "POSIXct")) tz_of_attr(col) else "UTC"
  }, character(1))
  schema <- list(
    names = names(data),
    types = types,
    factor_levels = levels_from_positions(names(data), unname(factor_levels)),
    tz = tz_from_positions(names(data), types, unname(zones))
  )

  exec_mode <- resolve_exec_mode(lazy)

  # A cudf table with no columns has no rows, so a zero-column data frame
  # keeps its row count on the R side (`$nrow`, see new_tbl_gpu()). A grouped_df
  # uploads grouped (with its `.drop`); the data frame's own attributes and a
  # bare data.frame class ride on the table (F31).
  groups <- dplyr::group_vars(orig)
  out <- new_tbl_gpu(ptr = ptr, schema = schema, exec_mode = exec_mode,
                     groups = groups,
                     group_drop = if (length(groups) > 0L && !is_rowwise) dplyr::group_by_drop_default(orig) else TRUE,
                     nrow = if (ncol(data) == 0L) nrow(data) else NULL,
                     unspecified = unspec_names,
                     bare_df = identical(class(data), "data.frame"),
                     rowwise = is_rowwise)
  with_frame_attrs(out, frame_attrs(orig))
}

# Upload for the "other side" of a two-table verb (join y, set-op y, a non-first
# bind input). dplyr ignores y's grouping there, so a rowwise_df goes up
# ungrouped (a rowwise tbl_gpu y is read for its columns only). Other data
# frames are unchanged.
# @keywords internal
upload_other_side <- function(y, lazy = FALSE) {
  if (inherits(y, "rowwise_df")) y <- dplyr::ungroup(y)
  tbl_gpu(y, lazy = lazy)
}

# Data-frame attributes (F31). dplyr's dplyr_reconstruct() copies a template's
# attributes (all but names/row.names/class) onto a verb's result. On a tbl_gpu
# they are the object's own R attributes other than names/class, so that
# `attr(gt, "foo")` works as on a data frame (attr() has no S3 dispatch, a list
# field could never answer it). `groups` is the grouped_df structure, not a
# user attribute. A bare `data.frame` class is the field `$bare_df` (see
# new_tbl_gpu()). Metadata only: no GPU work.
# @keywords internal
frame_attrs <- function(x) {
  a <- attributes(x)
  a[!names(a) %in% c("names", "row.names", "class", "groups")]
}

# `x` with exactly the attributes `attrs` added (existing ones are replaced).
# @keywords internal
with_frame_attrs <- function(x, attrs) {
  for (nm in names(attrs)) {
    attr(x, nm) <- attrs[[nm]]
  }
  x
}

# `x` without its data-frame attributes (summarise() rebuilds the frame).
# @keywords internal
strip_frame_attrs <- function(x) {
  for (nm in names(frame_attrs(x))) {
    attr(x, nm) <- NULL
  }
  x
}

# What a verb result inherits from its input `from` (a tbl_gpu, or a data
# frame for the binds): its attributes, whether it is a bare data.frame, and
# whether it is grouped.
# @keywords internal
frame_meta <- function(from) {
  # A rowwise table counts as grouped here: dplyr_reconstruct.rowwise_df()
  # rebuilds with rowwise_df(), so it hands over nothing either (F49).
  if (is_tbl_gpu(from)) {
    list(attrs = frame_attrs(from), bare = isTRUE(.subset2(from, "bare_df")),
         grouped = length(.subset2(from, "groups")) > 0L || is_rowwise_gpu(from))
  } else {
    list(attrs = frame_attrs(from), bare = identical(class(from), "data.frame"),
         grouped = dplyr::is_grouped_df(from) || inherits(from, "rowwise_df"))
  }
}

# Copy `from`'s attributes and bare class onto the verb result `out` verbatim
# (compute(), group_by(), ...: verbs that rebuild a table but whose result
# dplyr leaves as is). `bare` is dropped when `out` is grouped.
# @keywords internal
copy_frame_meta <- function(out, from, attrs = TRUE, bare = TRUE) {
  meta <- frame_meta(from)
  if (attrs) {
    out <- with_frame_attrs(out, meta$attrs)
  }
  if (bare && meta$bare && length(.subset2(out, "groups")) == 0L && !is_rowwise_gpu(out)) {
    out <- set_tbl_gpu_field(out, "bare_df", TRUE)
  }
  out
}

# dplyr's rule for a verb's result (dplyr_reconstruct() of the input): an
# UNGROUPED input hands its attributes (unless `attrs = FALSE`: summarise()
# and tally() rebuild the frame) and its bare data.frame class to the result;
# a GROUPED input hands over nothing, since dplyr_reconstruct.grouped_df()
# rebuilds with grouped_df().
# @keywords internal
carry_frame_meta <- function(out, from, attrs = TRUE) {
  # A table with only its names and class as attributes (the usual one) has no
  # attributes to hand over; all that is left is the bare data.frame class,
  # which an ungrouped input gives to an ungrouped result.
  if (is_tbl_gpu(from) && identical(names(attributes(from)), c("names", "class"))) {
    if (isTRUE(.subset2(from, "bare_df")) && length(.subset2(from, "groups")) == 0L &&
        !is_rowwise_gpu(from) && length(.subset2(out, "groups")) == 0L && !is_rowwise_gpu(out)) {
      out <- set_tbl_gpu_field(out, "bare_df", TRUE)
    }
    return(out)
  }
  if (frame_meta(from)$grouped) {
    return(out)
  }
  copy_frame_meta(out, from, attrs = attrs)
}

# `out` with `from`'s attributes and class instead of its own: for results
# assembled from pieces that may carry other tables' attributes (set operations,
# column assignment). Same rule as carry_frame_meta().
# @keywords internal
reset_frame_meta <- function(out, from) {
  carry_frame_meta(set_tbl_gpu_field(strip_frame_attrs(out), "bare_df", NULL), from)
}

# The one warning for a fractional Date column floored to whole days (a GPU
# Date holds whole days): raised by the upload (tbl_gpu(), so also every CPU
# fallback result) and by mutate()/transmute() when they build a column-free
# fractional Date constant natively (parse_mutate_dots()). `cols` names the
# columns.
# @keywords internal
warn_fractional_dates <- function(cols) {
  if (length(cols) == 0L) {
    return(invisible())
  }
  rlang::warn(
    c(paste0(
      "Fractional Date column", if (length(cols) > 1L) "s", " ",
      paste0("`", cols, "`", collapse = ", "),
      " floored to whole days: a GPU Date holds whole days only."
    )),
    class = "cuplyr_warning_fractional_date"
  )
}

# Internal: classes a column may carry and still have a GPU representation.
# Anything else (ts, difftime, hms, IDate, AsIs, POSIXlt, S4, ...) would upload
# as its bare storage type and silently change what later verbs return, so it
# is an error. integer64 keeps its own warning (gpu_type_from_r()).
# @keywords internal
check_supported_column_classes <- function(data) {
  ok <- list(c("factor"), c("ordered", "factor"), "Date", c("POSIXct", "POSIXt"), "integer64")
  bad <- character()
  for (j in seq_along(data)) {
    col <- data[[j]]
    cls <- oldClass(col)
    if (isS4(col)) {
      cls <- class(col)
    } else if (is.null(cls) || any(vapply(ok, identical, logical(1), cls))) {
      next
    } else if (!is.atomic(col) && !inherits(col, "POSIXlt")) {
      next # list / data-frame columns keep their existing errors
    }
    bad <- c(bad, sprintf("`%s` <%s>", names(data)[j], paste(cls, collapse = "/")))
  }
  if (length(bad) == 0L) {
    return(invisible())
  }
  rlang::abort(
    c(
      paste0("Can't upload column(s) with a class that has no GPU representation: ",
             paste(bad, collapse = ", "), "."),
      i = paste0("Convert them first, e.g. `unclass()` / `as.numeric()` / `as.character()`, ",
                 "or `as.POSIXct()` for POSIXlt.")
    ),
    class = "cuplyr_error_unsupported_class",
    call = NULL
  )
}

#' @rdname tbl_gpu
#' @export
tbl_gpu.tbl_gpu <- function(data, ..., lazy = NULL) {
  # If lazy is specified, potentially change execution mode
  if (!is.null(lazy)) {
    exec_mode <- resolve_exec_mode(lazy)
    if (identical(exec_mode, "eager")) {
      return(as_eager(data))
    }
    return(as_lazy(data))
  }
  data
}

# Internal constructor for tbl_gpu objects
#
# Creates the S3 structure for a GPU-backed table. This is an internal
# function and should not be called directly by users.
#
# @param ptr External pointer to cudf::table on GPU
# @param schema List with `names` (character) and `types` (character)
# @param lazy_ops AST root node for pending operations (NULL if none)
# @param groups Character vector of grouping columns
# @param exec_mode Execution mode: "lazy" or "eager"
# @param nrow Row count of a ZERO-column table (cudf cannot represent rows
#   without columns, so it lives here as `$nrow`); `NULL` otherwise. dim(),
#   collect(), print() and compute() honour it, and push_op() refuses verbs
#   on such a table when it has rows (check_zero_column_input()).
# @param fixed_keys A tibble of group keys (one column per entry of `groups`,
#   same names and order) that dplyr's `slice(.preserve = TRUE)` fixes on a
#   grouped table, or `NULL` (the default, and the only value any constructor
#   gets unless it passes it on purpose). Dropped unless it matches `groups`.
#   See `fixed_keys_of()` (R/group-by.R).
# @param unspecified Names of BOOL8 columns that were `vctrs::unspecified()`
#   at upload and are still the original column (see tbl_gpu.data.frame()).
#   Only push_op()/push_join() carry it, through row-only nodes and semi/anti
#   joins; collect_frame() restores the class.
# @param bare_df `TRUE` for a table that came from a plain `data.frame` (not a
#   tibble) and is ungrouped: it collects as a bare data.frame (F31). Only
#   present on the object when set, and dropped for a grouped table.
# @param rowwise `TRUE` for a rowwise table (F49): `groups` then holds the
#   rowwise vars (possibly none), which are NOT groups for any computation, and
#   the object carries `$rowwise = TRUE`. The field exists only on rowwise
#   tables (the conformance harness recognises an uploaded fixture by its
#   fields), see is_rowwise_gpu() in R/rowwise.R. A rowwise table is never a
#   bare data.frame.
# @return A tbl_gpu S3 object
# @keywords internal
new_tbl_gpu <- function(ptr = NULL,
                        schema = list(names = character(), types = character()),
                        lazy_ops = NULL,
                        groups = character(),
                        exec_mode = NULL,
                        group_drop = TRUE,
                        nrow = NULL,
                        fixed_keys = NULL,
                        unspecified = NULL,
                        bare_df = FALSE,
                        rowwise = FALSE) {
  if (is.list(lazy_ops) && length(lazy_ops) == 0) {
    lazy_ops <- NULL
  }

  if (is.null(exec_mode)) {
    exec_mode <- "eager"
  }
  if (!exec_mode %in% c("lazy", "eager")) {
    stop("exec_mode must be 'lazy' or 'eager'.", call. = FALSE)
  }

  check_dup_names_ungrouped(schema$names, groups)

  out <- list(
    ptr = ptr,
    schema = schema,
    lazy_ops = lazy_ops,
    groups = groups,
    exec_mode = exec_mode,
    group_drop = group_drop
  )
  if (!is.null(nrow) && length(schema$names) == 0L) {
    out$nrow <- as.integer(nrow)
  }
  if (!is.null(fixed_keys) && length(groups) > 0L && identical(names(fixed_keys), groups)) {
    out$fixed_keys <- fixed_keys
  }
  if (length(unspecified) > 0L) {
    unspecified <- set_intersect(unspecified, schema$names[schema$types == "BOOL8"])
    if (length(unspecified) > 0L) {
      out$unspecified <- unspecified
    }
  }
  if (isTRUE(rowwise)) {
    out$rowwise <- TRUE
  } else if (isTRUE(bare_df) && length(groups) == 0L) {
    out$bare_df <- TRUE
  }
  class(out) <- c("tbl_gpu", "tbl")
  out
}

#' Resolve execution mode from explicit value, option, or environment variable
#'
#' @param explicit Explicit value passed to tbl_gpu() or NULL
#' @return "lazy" or "eager"
#' @keywords internal
resolve_exec_mode <- function(explicit = NULL) {
  # Explicit parameter validation and handling (highest priority)
  if (!is.null(explicit)) {
    if (isTRUE(explicit) || identical(explicit, "lazy")) {
      return("lazy")
    } else if (isFALSE(explicit) || identical(explicit, "eager")) {
      return("eager")
    } else {
      stop("lazy= must be TRUE, FALSE, 'lazy', or 'eager'. Got: ",
           deparse(explicit), call. = FALSE)
    }
  }

  # Session option (second priority), then the environment variable (third).
  # The valid path is one getOption() and at most one Sys.getenv(); anything
  # else goes to the registry (R/options.R), which warns once about an invalid
  # value and gives the default, "lazy".
  opt <- getOption("cuplyr.exec_mode")
  if (is.null(opt)) {
    env <- Sys.getenv("CUPLYR_EXEC_MODE", unset = "")
    if (identical(env, "")) {
      return("lazy")
    }
    if (env %in% cuplyr_exec_modes) {
      return(env)
    }
  } else if (is.character(opt) && length(opt) == 1L && opt %in% cuplyr_exec_modes) {
    return(opt)
  }
  cuplyr_opt("exec_mode")
}

#' Test if an object is a GPU table
#'
#' Checks whether an object inherits from the `tbl_gpu` class.
#'
#' @param x An R object to test.
#'
#' @return `TRUE` if `x` is a `tbl_gpu` object, `FALSE` otherwise.
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_df <- tbl_gpu(mtcars)
#'   is_tbl_gpu(gpu_df)
#'   is_tbl_gpu(mtcars)
#' }
is_tbl_gpu <- function(x) {
  inherits(x, "tbl_gpu")
}

#' @rdname is_tbl_gpu
#' @export
is_gibble <- is_tbl_gpu

#' Coerce to a GPU table
#'
#' Converts a data frame or compatible object to a `tbl_gpu` object,
#' transferring data to GPU memory.
#'
#' @param x A data frame or object coercible to a data frame.
#' @param ... Additional arguments passed to `tbl_gpu()`.
#'
#' @return A `tbl_gpu` object with data stored on the GPU.
#'
#' @seealso \code{\link{tbl_gpu}} for details on GPU table creation
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_df <- as_tbl_gpu(iris)
#'   print(gpu_df)
#' }
as_tbl_gpu <- function(x, ...) {
  tbl_gpu(x, ...)
}

#' @rdname as_tbl_gpu
#' @export
as_gibble <- as_tbl_gpu

#' Get dimensions of a GPU table
#'
#' Returns the number of rows and columns in a GPU table. For a lazy table
#' with pending operations the row count is that of the RESULT, so the plan is
#' run on the GPU (`compute()`, nothing is copied to R and `x` itself is not
#' changed). `ncol()`/`NCOL()` only need the column count, so they are answered
#' from the schema without running anything.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return An integer vector of length 2: c(nrow, ncol). The row count is `NA`
#'   for a table that has no GPU data yet, for `ncol()`/`NCOL()` of a lazy
#'   table with pending operations (which ignore it), and for a zero-column
#'   table whose row count was never recorded.
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   dim(tbl_gpu(mtcars))
#' }
dim.tbl_gpu <- function(x) {
  if (length(x$schema$names) == 0L) {
    return(zero_column_dim(x))
  }
  if (is.null(x$ptr) && !has_pending_ops(x)) {
    return(c(NA_integer_, length(x$schema$names)))
  }
  if (has_pending_ops(x)) {
    # `ncol(x)` is `dim(x)[2L]` and `NCOL(x)` the same, neither is generic.
    caller <- sys.parent()
    if (caller > 0L) {
      fn <- sys.function(caller)
      if (identical(fn, base::ncol) || identical(fn, base::NCOL)) {
        return(c(NA_integer_, length(current_schema(x)$names)))
      }
    }
    return(gpu_dim(compute(x)$ptr))
  }
  gpu_dim(x$ptr)
}

# Zero-column table: cudf always reports 0 rows for it, so only the R-side
# count (`$nrow`, see new_tbl_gpu()) is trustworthy. `NA` (never 0) when
# nobody recorded one. No plan is run: the verbs that keep a zero-column
# table carry the count through push_op().
zero_column_dim <- function(x) {
  c(x[["nrow"]] %||% NA_integer_, 0L)
}

# Dimensions of the table `x$ptr` points at, WITHOUT running pending lazy
# operations (so for a lazy table with pending ops this is the size of its
# base table, not of its result). For cheap estimates (printing, memory
# accounting, join-size warnings) that must never execute a plan, which
# `dim()` does.
base_dim <- function(x) {
  if (length(x$schema$names) == 0L) {
    zero_column_dim(x)
  } else if (is.null(x$ptr)) {
    c(NA_integer_, length(x$schema$names))
  } else {
    gpu_dim(x$ptr)
  }
}

#' Dimnames of a GPU table
#'
#' Rows are unnamed; the column names are the table's current names (no plan
#' is run), so `colnames()` works.
#'
#' @param x A `tbl_gpu` object.
#' @return `list(NULL, <column names>)`.
#' @export
dimnames.tbl_gpu <- function(x) {
  list(NULL, current_schema(x)$names)
}

#' Set the dimnames of a GPU table
#'
#' Only the column names can be set (`colnames(x) <- value`); rows are
#' unnamed, so `value[[1]]` must be `NULL`.
#'
#' @param x A `tbl_gpu` object.
#' @param value `list(NULL, <column names>)`.
#' @return The modified `tbl_gpu` object.
#' @export
`dimnames<-.tbl_gpu` <- function(x, value) {
  if (!is.list(value) || length(value) != 2L) {
    stop("`value` must be a list of length 2 (row names, column names).",
         call. = FALSE)
  }
  if (length(value[[1]]) > 0L) {
    stop("A GPU table has no row names; only column names can be set.",
         call. = FALSE)
  }
  names(x) <- if (is.null(value[[2]])) character(0) else as.character(value[[2]])
  x
}

#' Get column names of a GPU table
#'
#' @param x A `tbl_gpu` object.
#' @return A character vector of column names.
#' @export
names.tbl_gpu <- function(x) {
  if (called_from_vctrs(sys.parent())) {
    # vctrs asks `names(x)` for the names of the ROWS of a vector (a data
    # frame's row names, which tbl_gpu doesn't have). Answering with the
    # column names would make `vec_slice(x, "colname")` silently pick a row
    # and mis-name `vec_c()` results.
    return(NULL)
  }
  x$schema$names
}

# TRUE if the closure that called `names()` (sys.parent of the method) lives
# in the vctrs namespace. vctrs evaluates `names(x)` in its own function's
# frame, so this is exact for vctrs internals and FALSE for user/dplyr code.
called_from_vctrs <- function(p) {
  p > 0L && identical(topenv(environment(sys.function(p))), asNamespace("vctrs"))
}

#' Set column names of a GPU table
#'
#' @param x A `tbl_gpu` object.
#' @param value A character vector of new column names. Must have the same
#'   length as the number of columns. `NA` and `""` are allowed, as in a
#'   data frame (data-masking verbs then refuse the table, like dplyr).
#' @return The modified `tbl_gpu` object.
#' @export
`names<-.tbl_gpu` <- function(x, value) {
  # vctrs sets ROW names (`vec_set_names(x, NULL)`, `vec_c(a = x)`); a GPU
  # table has none, so this is a no-op there (see `names.tbl_gpu()`).
  if (called_from_vctrs(sys.parent())) {
    return(x)
  }

  n_cols <- length(x$schema$names)

  if (!is.character(value)) {
    stop(
      "`value` must be a character vector, not ", typeof(value), ".",
      call. = FALSE
    )
  }

  if (length(value) != n_cols) {
    stop(
      "`value` must have length ", n_cols, " (number of columns), not ",
      length(value), ".",
      call. = FALSE
    )
  }

  # Like a data.frame, `NA` and "" are allowed as names; the masking verbs
  # refuse such a table (check_unique_input_names()).
  # Refuse before any state changes: a grouped table can't carry duplicates.
  check_dup_names_ungrouped(value, x$groups)

  # Pending lazy ops: the new names have to go through the AST, because
  # `current_schema()` is inferred from it and ignores the base schema's names.
  # A rename by name can't address (or produce) duplicated names, so those go
  # through a positional rename node.
  if (has_pending_ops(x)) {
    old <- current_schema(x)$names
    if (!anyDuplicated(old) && !anyDuplicated(value) && !anyNA(c(old, value)) &&
        !any(c(old, value) == "")) {
      return(dplyr::rename(x, dplyr::all_of(stats::setNames(old, value))))
    }
    changed <- is.na(old) | is.na(value) | old != value
    if (!any(changed)) {
      return(x)
    }
    return(push_op(x, ast_rename(input_node(x), old[changed], value[changed],
                                 positions = which(changed))))
  }

  # Keep the name-keyed metadata (grouping, factor levels) in step. Factor
  # levels are re-laid out by position: a by-name list keeps whatever order
  # earlier ops left it in, which stops being enough once a name repeats.
  old <- x$schema$names
  if (length(x$groups) > 0) {
    x$groups <- value[match(x$groups, old)]
  }
  if (!is.null(x$schema$factor_levels)) {
    x$schema$factor_levels <- levels_from_positions(value, levels_by_position(x$schema))
  }
  if (!is.null(x$schema$tz)) {
    x$schema$tz <- tz_from_positions(value, x$schema$types, tz_by_position(x$schema))
  }
  # The unspecified marker is name-keyed too; a name that comes out duplicated
  # can't be addressed by collect_frame(), so it loses the marker.
  if (length(x[["unspecified"]]) > 0L) {
    new_unspec <- set_diff(value[match(x[["unspecified"]], old)], value[duplicated(value)])
    x[["unspecified"]] <- if (length(new_unspec) > 0L) new_unspec
  }
  x$schema$names <- value
  x
}
