# F53-a2: regression test for attempt 1's round-4 finding B8
# (scratchpad/parity/features/F53/reviews/round-4.md).
#
# A native is.nan() let the other dots of the same call run on the GPU, where a
# computed NaN is invisible to is.na(), compares as a number and survives
# coalesce(). On dev every call naming is.nan() ran wholly through dplyr and got
# these right. Parity only: any route is fine, a silent wrong is not.
# dplyr's answers verified by the PM with tools/r (U1 row 1: miss TRUE, nan TRUE,
# big NA, r0 0; U3: n_na 1, n_nan 1; U4 g = 1: n_na 1, n_nan 1; F1 keeps 3 rows).

f53b8_df <- function() {
  tibble::tibble(g = c(1L, 1L, 2L), x = c(0, 1, 2), y = c(0, 1, 1))
}

test_that("other dots of a call naming is.nan() see a computed NaN like dplyr", {
  skip_if_no_gpu()
  df <- f53b8_df()
  pipes <- list(
    U1 = function(d) dplyr::mutate(d,
      miss = is.na(x / y), nan = is.nan(x / y), big = x / y > 1, r0 = dplyr::coalesce(x / y, 0)
    ),
    U3 = function(d) dplyr::summarise(d, n_na = sum(is.na(x / y)), n_nan = sum(is.nan(x / y))),
    U4 = function(d) dplyr::summarise(dplyr::group_by(d, g),
      n_na = sum(is.na(x / y)), n_nan = sum(is.nan(x / y))
    ),
    F1 = function(d) dplyr::filter(d, !is.nan(x / y) | is.na(x / y))
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})
