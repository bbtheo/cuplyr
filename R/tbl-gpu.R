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
  ptr <- wrap_gpu_call("tbl_gpu", df_to_gpu(data))

  schema <- list(
    names = names(data),
    types = vapply(data, gpu_type_from_r, character(1))
  )

  exec_mode <- resolve_exec_mode(lazy)

  new_tbl_gpu(ptr = ptr, schema = schema, exec_mode = exec_mode)
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
# @return A tbl_gpu S3 object
# @keywords internal
new_tbl_gpu <- function(ptr = NULL,
                        schema = list(names = character(), types = character()),
                        lazy_ops = NULL,
                        groups = character(),
                        exec_mode = NULL) {
  if (is.list(lazy_ops) && length(lazy_ops) == 0) {
    lazy_ops <- NULL
  }

  if (is.null(exec_mode)) {
    exec_mode <- "eager"
  }
  if (!exec_mode %in% c("lazy", "eager")) {
    stop("exec_mode must be 'lazy' or 'eager'.", call. = FALSE)
  }

  structure(
    list(
      ptr = ptr,
      schema = schema,
      lazy_ops = lazy_ops,
      groups = groups,
      exec_mode = exec_mode
    ),
    class = c("tbl_gpu", "tbl")
  )
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
#' Returns the number of rows and columns in a GPU table.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return An integer vector of length 2: c(nrow, ncol).
#'   Returns `c(NA, ncol)
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'
#'
#'
#' }
dim.tbl_gpu <- function(x) {
  if (is.null(x$ptr)) {
    c(NA_integer_, length(x$schema$names))
  } else {
    gpu_dim(x$ptr)
  }
}

#' Get column names of a GPU table
#'
#' @param x A `tbl_gpu` object.
#' @return A character vector of column names.
#' @export
names.tbl_gpu <- function(x) {
  x$schema$names
}

#' Set column names of a GPU table
#'
#' @param x A `tbl_gpu` object.
#' @param value A character vector of new column names.
#' @return The modified `tbl_gpu` object.
#' @export
`names<-.tbl_gpu` <- function(x, value) {
  x$schema$names <- value
  x
}
