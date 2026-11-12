# F53 acceptance tests added after the round-2 review (B6).
# dplyr behaviour verified by the reviewer's probe
# (scratchpad/parity/features/F53/runs/review-2/probe1.log) and by the PM with tools/r.

# ---- B6: is.nan() sees a NaN literal or a NaN-valued R variable -----------------

test_that("is.nan() is TRUE for an expression holding a NaN literal or NaN variable", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  th <- NaN
  pipes <- list(
    function(d) dplyr::filter(d, is.nan(x * th)),
    function(d) dplyr::filter(d, !is.nan(x + NaN)),
    function(d) dplyr::mutate(d,
      a = is.nan(x + NaN),
      b = is.nan(th) & x > 1,
      c = is.nan(x * th),
      e = is.nan(x)
    ),
    function(d) dplyr::summarise(d, n = sum(is.nan(x * th)), m = sum(is.nan(x))),
    function(d) dplyr::summarise(dplyr::group_by(d, g = x > 1), n = sum(is.nan(x * th)))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("is.nan() of a column-free expression folds like base R", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  th <- NaN
  k <- NA_real_
  pipes <- list(
    function(d) dplyr::mutate(d, a = is.nan(th), b = is.nan(k), c = is.nan(0/0)),
    function(d) dplyr::filter(d, is.nan(th) & x > 1),
    function(d) dplyr::filter(d, is.nan(k) | x > 2)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})
