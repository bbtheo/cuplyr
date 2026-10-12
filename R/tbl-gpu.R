#' Create a GPU-backed data frame
#'
#' Transfers an R data frame to GPU memory using NVIDIA's libcudf library,
#' enabling high-performance data manipulation operations. The resulting
#' `tbl_gpu` object can be used with dplyr verbs like `filter()`, `mutate()`,
#' `select()`, and collected back to R with `collect()`.
#'
#' @param data A data frame or tibble to transfer to GPU memory. Supported
#'   column types include: numeric (double), integer, character, and logical.
#' @param ... Additional arguments passed to methods (currently unused).
#' @param lazy Logical or character. Controls execution mode:
#'   \itemize{
#'     \item `TRUE` or `"lazy"` - Operations are deferred until collect()/compute()
#'     \item `FALSE` or `"eager"` - Operations execute immediately (default)
#'     \item `NULL` - Use session option or environment variable
#'   }
#'   Can also be set via `options(cuplyr.exec_mode = "lazy")` or
#'   environment variable `CUPLYR_EXEC_MODE=lazy`.
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
#' The data is immediately copied to GPU memory when `tbl_gpu()` is called.
#' GPU memory is automatically freed when the R object is garbage collected.
#'
#' ## Execution Modes
#'
#' In **eager mode** (default), each dplyr verb executes immediately on the GPU.
#' This is simple but can lead to unnecessary intermediate allocations.
#'
#' In **lazy mode**, operations build an AST (Abstract Syntax Tree) that is
#' optimized and executed only when `collect()` or `compute()` is called.
#' This enables optimizations like:
#' \itemize{
#'   \item Projection pruning (only load needed columns)
#'   \item Mutate fusion (combine multiple mutates)
#'   \item Filter reordering (cheapest filters first)
#' }
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
#' @seealso
#' \code{\link{collect.tbl_gpu}} to transfer data back to R,
#' \code{\link{compute.tbl_gpu}} to execute and keep on GPU,
#' \code{\link{filter.tbl_gpu}}, \code{\link{mutate.tbl_gpu}},
#' \code{\link{select.tbl_gpu}} for data manipulation
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Transfer mtcars to GPU (eager mode)
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   print(gpu_mtcars)
#'
#'   # Lazy mode - operations deferred until collect()
#'   result <- tbl_gpu(mtcars, lazy = TRUE) |>
#'     filter(mpg > 20) |>
#'     mutate(kpl = mpg * 0.425) |>
#'     collect()
#'
#'   # Set lazy mode globally
#'   options(cuplyr.exec_mode = "lazy")
#' }
tbl_gpu <- function(data, ..., lazy = NULL) {
  UseMethod("tbl_gpu")
}

#' @rdname tbl_gpu
#' @export
tbl_gpu.data.frame <- function(data, ..., lazy = NULL) {
  # `vctrs::unspecified()` columns (all NA) upload as plain logical; their
  # names ride on the table (`$unspecified`) so row-only verbs and filtering
  # joins can give the class back at collect().
  unspec <- vapply(data, inherits, logical(1), what = "vctrs_unspecified")
  unspec_names <- NULL
  if (any(unspec)) {
    for (j in which(unspec)) data[[j]] <- rep(NA, nrow(data))
    unspec_names <- setdiff(names(data)[unspec], names(data)[duplicated(names(data))])
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
  ptr <- wrap_gpu_call("tbl_gpu", df_to_gpu(upload))

  # Build schema with factor levels for round-trip fidelity
  factor_levels <- lapply(data, function(col) {
    if (is.factor(col)) {
      lvls <- levels(col)
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
  # keeps its row count on the R side (`$nrow`, see new_tbl_gpu()).
  new_tbl_gpu(ptr = ptr, schema = schema, exec_mode = exec_mode,
              nrow = if (ncol(data) == 0L) nrow(data) else NULL,
              unspecified = unspec_names)
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
                        unspecified = NULL) {
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
    unspecified <- intersect(unspecified, schema$names[schema$types == "BOOL8"])
    if (length(unspecified) > 0L) {
      out$unspecified <- unspecified
    }
  }
  structure(out, class = c("tbl_gpu", "tbl"))
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

  # Session option (second priority)
  opt <- getOption("cuplyr.exec_mode")
  if (!is.null(opt) && opt %in% c("lazy", "eager")) {
    return(opt)
  }

  # Environment variable (third priority)
  env <- Sys.getenv("CUPLYR_EXEC_MODE", unset = "")
  if (env %in% c("lazy", "eager")) {
    return(env)
  }

  # Default to eager (safe during transition)
  "eager"
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
  if (is.null(x$ptr)) {
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
#'   length as the number of columns, contain no `NA` values, and no empty
#'   strings.
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

  if (anyNA(value)) {
    stop("`value` must not contain NA values.", call. = FALSE)
  }

  if (any(value == "")) {
    stop("`value` must not contain empty strings.", call. = FALSE)
  }

  # Refuse before any state changes: a grouped table can't carry duplicates.
  check_dup_names_ungrouped(value, x$groups)

  # Pending lazy ops: the new names have to go through the AST, because
  # `current_schema()` is inferred from it and ignores the base schema's names.
  # A rename by name can't address (or produce) duplicated names, so those go
  # through a positional rename node.
  if (has_pending_ops(x)) {
    old <- current_schema(x)$names
    if (!anyDuplicated(old) && !anyDuplicated(value)) {
      return(dplyr::rename(x, dplyr::all_of(stats::setNames(old, value))))
    }
    changed <- old != value
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
    new_unspec <- setdiff(value[match(x[["unspecified"]], old)], value[duplicated(value)])
    x[["unspecified"]] <- if (length(new_unspec) > 0L) new_unspec
  }
  x$schema$names <- value
  x
}
