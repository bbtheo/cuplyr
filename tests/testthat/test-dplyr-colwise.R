# Phase 11 L7 exit: VERIFICATION (not new implementation) that the
# superseded colwise family (mutate_at()/mutate_all()/mutate_if(),
# summarise_at()/summarise_all()/summarise_if(), top_n()/top_frac()) works
# on tbl_gpu now that L1's universal expression-level CPU fallback exists
# (scratchpad/phase11_design.md's own L7 scope note: "non-generic, route
# through owned verbs"). None of these are S3 generics dplyr dispatches on
# `tbl_gpu` -- they are plain functions that internally build and call
# mutate()/filter()/summarise() themselves (e.g. top_n() desugars to a
# filter() on min_rank()/rank of the ordering column), so before L1 existed
# they broke wherever their internal call hit an expression shape mutate()/
# filter() didn't understand; L1 is exactly the fix (verified directly by
# these tests, not by re-deriving how each function is implemented).

test_that("mutate_at()/mutate_all()/mutate_if() work on tbl_gpu and match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5, y = 6:10, g = c(1, 1, 2, 2, 2))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::mutate_at(tbl, dplyr::vars(x, y), ~ . * 2)
  })
  expect_same_as_dplyr(df, function(tbl) {
    dplyr::mutate_all(tbl, ~ . * 2)
  })
  expect_same_as_dplyr(df, function(tbl) {
    dplyr::mutate_if(tbl, is.numeric, ~ . + 1)
  })
})

test_that("summarise_at()/summarise_all()/summarise_if() work on tbl_gpu and match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5, y = 6:10, g = c(1, 1, 2, 2, 2))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::summarise_at(dplyr::group_by(tbl, g), dplyr::vars(x), sum)
  }, arrange_by = "g")
  expect_same_as_dplyr(df, function(tbl) {
    dplyr::summarise_all(dplyr::group_by(tbl, g), sum)
  }, arrange_by = "g")
  expect_same_as_dplyr(df, function(tbl) {
    dplyr::summarise_if(dplyr::group_by(tbl, g), is.numeric, sum)
  }, arrange_by = "g")
})

test_that("top_n()/top_frac() work on tbl_gpu and match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5, y = 6:10, g = c(1, 1, 2, 2, 2))

  expect_same_as_dplyr(df, function(tbl) dplyr::top_n(tbl, 2, x), arrange_by = "x")
  expect_same_as_dplyr(df, function(tbl) dplyr::top_frac(tbl, 0.4, x), arrange_by = "x")
  expect_same_as_dplyr(df, function(tbl) {
    dplyr::top_n(dplyr::group_by(tbl, g), 1, x)
  }, arrange_by = c("g", "x"))
})
