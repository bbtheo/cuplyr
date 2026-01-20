# Select verb for tbl_gpu

#' Select columns from a GPU table
#'
#' @param .data A tbl_gpu object
#' @param ... Column selection (supports tidyselect)
#' @return A tbl_gpu with selected columns
#' @export
#' @importFrom dplyr select
#' @examples
#' if (interactive()) {
#'   gpu_df <- tbl_gpu(mtcars)
#'   gpu_df |> select(mpg, cyl, hp)
#' }
select.tbl_gpu <- function(.data, ...) {
  # tidyselect needs a named vector
  name_vec <- setNames(.data$schema$names, .data$schema$names)
  vars <- tidyselect::eval_select(rlang::expr(c(...)), name_vec)

  if (length(vars) == 0) {
    stop("select() resulted in no columns")
  }

  indices <- as.integer(vars) - 1L
  new_ptr <- gpu_select(.data$ptr, indices)

  new_schema <- list(
    names = .data$schema$names[vars],
    types = .data$schema$types[vars]
  )

  new_tbl_gpu(
    ptr = new_ptr,
    schema = new_schema,
    groups = intersect(.data$groups, new_schema$names)
  )
}
