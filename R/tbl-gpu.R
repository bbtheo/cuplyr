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
#'
#' @return A `tbl_gpu` object containing:
#'   \itemize{
#'     \item `ptr` - External pointer to the GPU table
#'     \item `schema` - List with column names and types
#'     \item `lazy_ops` - Pending operations (for future lazy evaluation)
#'     \item `groups` - Grouping variables (for future group_by support)
#'   }
#'
#' @details
#' The data is immediately copied to GPU memory when `tbl_gpu()` is called.
#' GPU memory is automatically freed when the R object is garbage collected.
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
#' \code{\link{filter.tbl_gpu}}, \code{\link{mutate.tbl_gpu}},
#' \code{\link{select.tbl_gpu}} for data manipulation
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Transfer mtcars to GPU
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   print(gpu_mtcars)
#'
#'   # Chain operations
#'   result <- gpu_mtcars |>
#'     filter(mpg > 20) |>
#'     mutate(kpl = mpg * 0.425) |>
#'     collect()
#' }
tbl_gpu <- function(data, ...) {

  UseMethod("tbl_gpu")
}

#' @rdname tbl_gpu
#' @export
tbl_gpu.data.frame <- function(data, ...) {
  ptr <- wrap_gpu_call("tbl_gpu", df_to_gpu(data))

  schema <- list(
    names = names(data),
    types = vapply(data, gpu_type_from_r, character(1))
  )

  new_tbl_gpu(ptr = ptr, schema = schema)
}

#' @rdname tbl_gpu
#' @export
tbl_gpu.tbl_gpu <- function(data, ...) {
  data
}

# Internal constructor for tbl_gpu objects
#
# Creates the S3 structure for a GPU-backed table. This is an internal
# function and should not be called directly by users.
#
# @param ptr External pointer to cudf::table on GPU
# @param schema List with `names` (character) and `types` (character)
# @param lazy_ops List of pending operations (reserved for future use)
# @param groups Character vector of grouping columns
# @return A tbl_gpu S3 object
# @keywords internal
new_tbl_gpu <- function(ptr = NULL,
                        schema = list(names = character(), types = character()),
                        lazy_ops = list(),
                        groups = character()) {
  structure(
    list(
      ptr = ptr,
      schema = schema,
      lazy_ops = lazy_ops,
      groups = groups
    ),
    class = c("tbl_gpu", "tbl")
  )
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
    wrap_gpu_call("dim", gpu_dim(x$ptr))
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
