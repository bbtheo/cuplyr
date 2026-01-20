# Collect verb for tbl_gpu

#' Collect GPU table back to R
#'
#' Transfers data from GPU memory back to an R tibble.
#'
#' @param x A tbl_gpu object
#' @param ... Additional arguments (ignored)
#' @return A tibble
#' @export
#' @importFrom dplyr collect
#' @examples
#' if (interactive()) {
#'   gpu_df <- tbl_gpu(mtcars)
#'   result <- gpu_df |> filter(mpg > 20) |> collect()
#' }
collect.tbl_gpu <- function(x, ...) {
  if (is.null(x$ptr)) {
    stop("Cannot collect: GPU table not materialized")
  }

  df <- gpu_collect(x$ptr, x$schema$names)
  tibble::as_tibble(df)
}
