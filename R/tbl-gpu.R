#' Create a GPU-backed tibble
#'
#' @param data A data frame to transfer to GPU memory
#' @param ... Additional arguments (unused)
#' @return A `tbl_gpu` object
#' @export
#' @examples
#' if (interactive()) {
#'   df <- data.frame(x = 1:1000, y = rnorm(1000))
#'   gpu_df <- tbl_gpu(df)
#'   gpu_df
#' }
tbl_gpu <- function(data, ...) {
  UseMethod("tbl_gpu")
}

#' @export
tbl_gpu.data.frame <- function(data, ...) {
  # Transfer to GPU

ptr <- df_to_gpu(data)

  schema <- list(
    names = names(data),
    types = vapply(data, gpu_type_from_r, character(1))
  )

  new_tbl_gpu(ptr = ptr, schema = schema)
}

#' @export
tbl_gpu.tbl_gpu <- function(data, ...) {
  data
}

# Internal constructor
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
    class = c("tbl_gpu", "tbl_lazy", "tbl")
  )
}

#' Check if object is a tbl_gpu
#' @param x Object to test
#' @return Logical
#' @export
is_tbl_gpu <- function(x) {
  inherits(x, "tbl_gpu")
}

#' Convert to tbl_gpu
#' @param x Object to convert
#' @param ... Additional arguments
#' @return A tbl_gpu object
#' @export
as_tbl_gpu <- function(x, ...) {
  tbl_gpu(x, ...)
}

#' @export
dim.tbl_gpu <- function(x) {
  if (is.null(x$ptr)) {
    c(NA_integer_, length(x$schema$names))
  } else {
    gpu_dim(x$ptr)
  }
}

#' @export
names.tbl_gpu <- function(x) {
  x$schema$names
}

#' @export
`names<-.tbl_gpu` <- function(x, value) {
  x$schema$names <- value
  x
}
