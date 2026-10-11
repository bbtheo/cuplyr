# F11-a2 acceptance tests: an NA_character_ scalar literal must be a null
# string on the GPU, not the text "NA" (round-4 review B6 of attempt 1).
# Root cause: build_scalar()'s STRING branch tested the whole vector with
# CharacterVector::is_na() instead of the element.
# See scratchpad/parity/features/F11-a2.md "## Design".

same_both_na <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f11na_df <- function() {
  tibble::tibble(x = 1:5, g = c(1, 1, 2, 2, 2), s = c("p", NA, "q", "r", "s"))
}

test_that("F11-a2: bare NA_character_ in mutate() is missing, not \"NA\"", {
  skip_if_no_gpu()
  df <- f11na_df()

  same_both_na(df, function(d) dplyr::mutate(d, y = NA_character_))
  same_both_na(df, function(d) dplyr::mutate(d, y = NA_character_, n = is.na(y)))
  same_both_na(df, function(d) dplyr::mutate(dplyr::group_by(d, g), y = NA_character_))
  same_both_na(df[0, ], function(d) dplyr::mutate(d, y = NA_character_))

  # the collected value is a real NA
  out <- tibble::as_tibble(dplyr::collect(dplyr::mutate(tbl_gpu(df), y = NA_character_)))
  expect_true(all(is.na(out$y)))
  expect_false(any(out$y == "NA", na.rm = TRUE))
})

test_that("F11-a2: NA_character_ as an if_else()/case_when()/coalesce() branch is missing", {
  skip_if_no_gpu()
  df <- f11na_df()

  same_both_na(df, function(d) dplyr::mutate(d, y = dplyr::if_else(x > 2, "a", NA_character_)))
  same_both_na(df, function(d) dplyr::mutate(d, y = dplyr::if_else(x > 2, NA_character_, s)))
  same_both_na(df, function(d) dplyr::mutate(
    d, y = dplyr::case_when(x == 1 ~ "one", x == 2 ~ "two", .default = NA_character_)
  ))
  same_both_na(df, function(d) dplyr::mutate(
    d, y = dplyr::case_when(x < 3 ~ NA_character_, .default = s)
  ))
  same_both_na(df, function(d) dplyr::mutate(d, y = dplyr::coalesce(s, NA_character_)))
  # downstream null semantics: is.na() and filter() see the missing values
  same_both_na(df, function(d) dplyr::filter(
    dplyr::mutate(d, y = dplyr::if_else(x > 2, "a", NA_character_)), is.na(y)
  ))
  same_both_na(df, function(d) dplyr::mutate(
    dplyr::group_by(d, g), y = dplyr::if_else(x > 2, "a", NA_character_)
  ))
})

test_that("F11-a2: lag()/lead() of a character scalar fill with NA, not \"NA\"", {
  skip_if_no_gpu()
  df <- f11na_df()
  b <- "a"

  same_both_na(df, function(d) dplyr::mutate(d, l = dplyr::lag(b), r = dplyr::lead("a")))
  same_both_na(df, function(d) dplyr::mutate(d, l = dplyr::lag(b, 2), r = dplyr::lead(b, 3)))
  same_both_na(df, function(d) dplyr::mutate(
    dplyr::group_by(d, g), l = dplyr::lag(b), r = dplyr::lead("a")
  ))
  same_both_na(df, function(d) dplyr::mutate(d, y = dplyr::if_else(x > 2, dplyr::lag(b), "z")))
  same_both_na(df, function(d) dplyr::filter(d, is.na(dplyr::lag(b))))
  same_both_na(df, function(d) dplyr::summarise(
    dplyr::mutate(d, l = dplyr::lag(b)), k = sum(is.na(l))
  ))
})

test_that("F11-a2: comparing a string column with an NA_character_ literal gives NA (round-1 B7)", {
  skip_if_no_gpu()
  df <- f11na_df()
  b <- "a"

  # F11's folded lag()/lead() of a character scalar, both operand orders
  same_both_na(df, function(d) dplyr::mutate(
    d, y = s == dplyr::lag(b), z = dplyr::lead("a") != s,
    w = dplyr::if_else(s == dplyr::lag(b), "eq", "ne")
  ))
  same_both_na(df, function(d) dplyr::filter(d, s != dplyr::lag(b)))
  same_both_na(df, function(d) dplyr::filter(dplyr::group_by(d, g), s != dplyr::lag(b)))
  # bare NA_character_ literal, every comparison operator and both sides
  same_both_na(df, function(d) dplyr::filter(d, s != NA_character_))
  same_both_na(df, function(d) dplyr::filter(dplyr::group_by(d, g), s == NA_character_))
  same_both_na(df, function(d) dplyr::mutate(
    d, a = NA_character_ == s, b2 = s < NA_character_, c = s >= NA_character_
  ))
  same_both_na(df, function(d) dplyr::filter(d, s < NA_character_ | x > 4))
  # %in% with a missing string keeps matching missing rows (no regression)
  same_both_na(df, function(d) dplyr::filter(d, s %in% NA_character_))
  same_both_na(df, function(d) dplyr::filter(d, s %in% c("p", NA_character_)))
})
