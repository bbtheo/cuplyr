#' Transfer GPU table data back to R
#'
#' Copies data from GPU memory back to R as a tibble. This is typically
#' the final step in a GPU data manipulation pipeline, after filtering,
#' mutating, and selecting the data you need.
#'
#' @param x A `tbl_gpu` object created by [tbl_gpu()].
#' @param ... Additional arguments (ignored, included for compatibility).
#'
#' @return A [tibble::tibble] containing the data from the GPU table.
#'   Column types are converted back to R types:
#'   \itemize{
#'     \item FLOAT64/FLOAT32 -> numeric (double)
#'     \item INT32/INT64 -> integer or numeric
#'     \item STRING -> character
#'     \item BOOL8 -> logical (TRUE/FALSE)
#'   }
#'
#' @details
#' ## Memory considerations
#' Collecting transfers all data from GPU to CPU memory. For large datasets,
#' this can be slow and memory-intensive. Best practice is to:
#' \enumerate{
#'   \item Filter rows to reduce data volume
#'   \item Select only needed columns
#'   \item Then collect the results
#' }
#'
#' ## Performance
#' Data transfer between GPU and CPU is limited by PCIe bandwidth
#' (typically 16-32 GB/s). For a 1 GB dataset, expect ~50-100ms transfer time.
#'
#' @seealso
#' \code{\link{tbl_gpu}} for creating GPU tables,
#' \code{\link{filter.tbl_gpu}}, \code{\link{select.tbl_gpu}} for reducing data
#'
#' @export
#' @importFrom dplyr collect
#'
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Process on GPU, then collect
#'   result <- gpu_mtcars |>
#'     filter(mpg > 20) |>
#'     mutate(kpl = mpg * 0.425) |>
#'     select(mpg, kpl, hp) |>
#'     collect()
#'
#'   # Result is a regular tibble
#'   class(result)
#'   print(result)
#' }
collect.tbl_gpu <- function(x, ...) {
  if (is.null(x$ptr)) {
    stop("Cannot collect: GPU table pointer is NULL.\n",
         "The table may have been garbage collected or never materialized.",
         call. = FALSE)
  }

  df <- wrap_gpu_call("collect", gpu_collect(x$ptr, x$schema$names))
  result <- tibble::as_tibble(df)

  if (any(x$schema$types == "INT64")) {
    int64_cols <- which(x$schema$types == "INT64")
    for (col_idx in int64_cols) {
      col_name <- x$schema$names[col_idx]
      values <- result[[col_name]]
      if (is.numeric(values) && any(abs(values) > 2^53, na.rm = TRUE)) {
        warning(
          "INT64 values may lose precision when collected into R numeric vectors. ",
          "Consider converting to integer64 or avoid values > 2^53.",
          call. = FALSE
        )
        break
      }
    }
  }

  result
}
