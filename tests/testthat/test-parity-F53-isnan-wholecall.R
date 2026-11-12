# F53 (round-3 review B7): a call with a NaN literal in a dot and is.nan() in
# that dot or a later one runs whole on the CPU (is.nan() isn't in the IR). Only the is.nan() column is compared: a NaN literal
# still reaches the device as a null (pre-existing).

test_that("verbs that parse dots through mutate() see is.nan() of an earlier NaN dot", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  th <- NaN
  pipes <- list(
    group_by = function(d) dplyr::ungroup(dplyr::group_by(d, y = x * th, z = is.nan(y))),
    count = function(d) dplyr::count(d, y = x * th, z = is.nan(y)),
    add_count = function(d) dplyr::add_count(d, y = x * th, z = is.nan(y)),
    distinct = function(d) dplyr::distinct(d, y = x * th, z = is.nan(y)),
    ns = function(d) dplyr::mutate(d, y = x * th, z = base::is.nan(y)),
    if_any = function(d) dplyr::mutate(d, y = x * th, z = dplyr::if_any(c(x, y), is.nan)),
    across = function(d) dplyr::mutate(d, y = x * th, dplyr::across(c(y), is.nan, .names = "z")),
    default = function(d) dplyr::mutate(d, y = dplyr::lag(x, default = th), z = is.nan(y))
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    want <- p(df)
    for (lazy in c(FALSE, TRUE)) {
      got <- collect(p(tbl_gpu(df, lazy = lazy)))
      expect_identical(got[["z"]], want[["z"]], info = paste(nm, "lazy =", lazy))
    }
  }
})

test_that("is.nan() computed from columns matches dplyr; dots without it stay native", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  th <- NaN
  # is.nan() with a column argument runs the whole call on the CPU: parity only
  pipes <- list(
    before = function(d) dplyr::mutate(d, z = is.nan(x / x), y = x * th),
    computed = function(d) dplyr::mutate(d, y = (x - x) / (x - x), y2 = y + 1, z = is.nan(y2))
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    want <- p(df)
    for (lazy in c(FALSE, TRUE)) {
      got <- collect(p(tbl_gpu(df, lazy = lazy)))
      expect_identical(got[["z"]], want[["z"]], info = paste(nm, "lazy =", lazy))
      expect_identical(names(got), names(want), info = paste(nm, "lazy =", lazy))
    }
  }
  # no is.nan() in the call: native
  withr::local_options(cuplyr.fallback = "error")
  p <- function(d) dplyr::mutate(d, y = x * th, y2 = y + 1)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(p(tbl_gpu(df, lazy = lazy)))
    expect_identical(names(got), names(p(df)), info = paste("lazy =", lazy))
  }
})
