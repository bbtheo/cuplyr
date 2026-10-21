# Loud errors for base generics that would otherwise fall through to the
# default method and operate on the tbl_gpu's internal field list (ptr,
# schema, groups, ...). No native implementations: collect() first.

unsupported_base <- function(fn, hint, call = rlang::caller_env()) {
  rlang::abort(
    paste0("`", fn, "()` is not supported on a tbl_gpu; ", hint),
    class = "cuplyr_error_unsupported_base",
    call = call
  )
}

#' @export
split.tbl_gpu <- function(x, f, drop = FALSE, ...) {
  unsupported_base("split", "use `group_split()` or `collect()` first.")
}

#' @export
with.tbl_gpu <- function(data, expr, ...) {
  unsupported_base("with", "use `collect()` (then `with()`), `pull()` or `mutate()` instead.")
}

#' @export
subset.tbl_gpu <- function(x, ...) {
  unsupported_base("subset", "use `filter()` and `select()`, or `collect()` first.")
}

#' @export
t.tbl_gpu <- function(x) {
  unsupported_base("t", "use `collect()` first.")
}

#' @export
as.environment.tbl_gpu <- function(x) {
  unsupported_base("as.environment", "use `collect()` first.")
}

# Without these, is.na()/anyNA() answer about the internal fields (all FALSE),
# so `x[is.na(x)] <- 0` would quietly do nothing.

#' @export
is.na.tbl_gpu <- function(x) {
  unsupported_base(
    "is.na",
    paste0("use `is.na()` inside `mutate()` or `filter()` (e.g. ",
           "`mutate(across(everything(), ~ coalesce(.x, 0)))` to replace missing ",
           "values), or `collect()` first.")
  )
}

#' @export
anyNA.tbl_gpu <- function(x, recursive = FALSE) {
  unsupported_base(
    "anyNA",
    "use `summarise(across(everything(), ~ any(is.na(.x))))`, or `collect()` first."
  )
}
