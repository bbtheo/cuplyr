# F50 acceptance tests: factor comparisons against labels on the GPU.
#
# A factor lives on the GPU as 1-based INT32 codes with its labels in
# schema$factor_levels. Comparing it with a label (`==`, `!=`, `%in%`, and the
# ordering operators for an ordered factor) is a comparison of codes once the
# label is matched against the levels, so these shapes run natively.
#
# The levels below are numeric-looking and out of label order ("10", "1",
# "2"), so a code compared with a label's text (F18's silent wrong) would give
# visibly different rows.
#
# `f50c_native()` pins the native route: cuplyr.fallback = "error" makes any
# CPU fallback an error, eager and lazy.

f50c_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1", "10", "2"),
               levels = c("10", "1", "2", "unused")),
    o = factor(c("lo", "mid", "hi", "mid", NA, "lo", "hi", "mid"),
               levels = c("lo", "mid", "hi"), ordered = TRUE),
    s = c("1", "x", "10", "2", NA, "1", "2", "2"),
    g = c(1L, 1L, 2L, 2L, 1L, 2L, 1L, 2L),
    x = c(1.5, 2, 3, 4, 5, NA, 7, 8)
  )
}

f50c_native <- function(df, pipeline, ...) {
  withr::with_options(list(cuplyr.fallback = "error"), {
    expect_same_as_dplyr(df, pipeline, ...)
    expect_same_as_dplyr_lazy(df, pipeline, ...)
  })
}

f50c_same <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# ---------------------------------------------------------------------------
# 1. == / != against a label
# ---------------------------------------------------------------------------

test_that("F50: filter(f == label) / (f != label) run on the GPU and compare labels", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::filter(d, f == "2"))
  f50c_native(df, function(d) dplyr::filter(d, f != "1"))
  f50c_native(df, function(d) dplyr::filter(d, "10" == f))
  f50c_native(df, function(d) dplyr::filter(d, o == "mid"))
  f50c_native(df, function(d) dplyr::filter(d, o != "hi"))
})

test_that("F50: mutate(f == label) gives logicals with NA for NA codes, on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::mutate(d, a = f == "2", b = f != "10", c = "1" == f,
                                             e = o == "lo"))
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(dplyr::mutate(tbl_gpu(df), a = f == "2", b = f != "10"))
  })
  expect_type(res$a, "logical")
  expect_type(res$b, "logical")
  expect_identical(res$a, c(FALSE, TRUE, FALSE, NA, TRUE, FALSE, FALSE, TRUE))
  # the factor column itself is untouched, levels and all
  expect_identical(levels(res$f), c("10", "1", "2", "unused"))
})

test_that("F50: a label that is not a level gives FALSE / TRUE (NA for NA rows), on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::mutate(d, a = f == "zz", b = f != "zz", c = "zz" == o))
  f50c_native(df, function(d) dplyr::filter(d, f != "zz"))
  f50c_native(df, function(d) dplyr::filter(d, f == "zz"))
  # an unused level is a level
  f50c_native(df, function(d) dplyr::mutate(d, a = f == "unused", b = f != "unused"))
})

test_that("F50: comparing with NA_character_ gives NA everywhere, on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::mutate(d, a = f == NA_character_, b = f != NA_character_))
  f50c_native(df, function(d) dplyr::filter(d, f == NA_character_))
})

test_that("F50: a label from the environment works like a literal, on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  lbl <- "10"
  f50c_native(df, function(d) dplyr::filter(d, f == lbl))
  f50c_native(df, function(d) dplyr::mutate(d, a = f != lbl))
  f50c_native(df, function(d) dplyr::filter(d, f == !!lbl))
})

# ---------------------------------------------------------------------------
# 2. %in% a set of labels
# ---------------------------------------------------------------------------

test_that("F50: f %in% labels runs on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  keep <- c("10", "2")
  f50c_native(df, function(d) dplyr::filter(d, f %in% c("1", "10")))
  f50c_native(df, function(d) dplyr::filter(d, f %in% keep))
  f50c_native(df, function(d) dplyr::filter(d, !(f %in% c("1"))))
  f50c_native(df, function(d) dplyr::filter(d, o %in% c("lo", "hi")))
  f50c_native(df, function(d) dplyr::mutate(d, a = f %in% c("2", "zz"), b = f %in% "unused"))
})

test_that("F50: %in% with NA in the set matches NA rows, and empty / non-level sets give FALSE", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::mutate(d, a = f %in% c("1", NA), b = f %in% NA_character_,
                                             c = f %in% character(0), e = f %in% c("zz", "q")))
  f50c_native(df, function(d) dplyr::filter(d, f %in% c(NA, "10")))
  withr::with_options(list(cuplyr.fallback = "error"), {
    res <- dplyr::collect(dplyr::mutate(tbl_gpu(df), a = f %in% c("1", NA), c = f %in% character(0)))
  })
  expect_type(res$a, "logical")
  expect_identical(res$a, c(TRUE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE))
  expect_identical(res$c, rep(FALSE, 8))
})

test_that("F50: f %in% a factor value from the environment matches on labels, on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  keepf <- factor(c("2", "q"), levels = c("q", "2"))
  f50c_native(df, function(d) dplyr::mutate(d, a = f %in% keepf))
  f50c_native(df, function(d) dplyr::filter(d, f %in% keepf))
})

# ---------------------------------------------------------------------------
# 3. Ordered factors against a label
# ---------------------------------------------------------------------------

test_that("F50: ordered factor <, <=, >, >= a label compare level positions on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::filter(d, o >= "mid"))
  f50c_native(df, function(d) dplyr::filter(d, o < "hi"))
  f50c_native(df, function(d) dplyr::mutate(d, a = o <= "lo", b = o > "mid", c = "mid" < o,
                                             e = "hi" >= o))
})

test_that("F50: ordered factor against a non-level or NA label gives NA, no warning, on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::mutate(d, a = o < "zz", b = "zz" <= o, c = o > NA_character_))
  withr::with_options(list(cuplyr.fallback = "error"), {
    expect_no_warning(res <- dplyr::collect(dplyr::mutate(tbl_gpu(df), a = o < "zz")))
  })
  expect_identical(res$a, rep(NA, 8))
})

test_that("F50: an unordered factor with < still gives dplyr's NA and warning", {
  skip_if_no_gpu()
  df <- f50c_df()
  expect_warning(
    res <- dplyr::collect(dplyr::mutate(tbl_gpu(df), a = f < "2")),
    "not meaningful for factors"
  )
  expect_identical(res$a, rep(NA, 8))
  expect_warning(
    res2 <- dplyr::collect(dplyr::filter(tbl_gpu(df), f >= "1")),
    "not meaningful for factors"
  )
  expect_equal(nrow(res2), 0L)
})

# ---------------------------------------------------------------------------
# 5. Composition: the comparison result is an ordinary logical
# ---------------------------------------------------------------------------

test_that("F50: common shapes built on label comparisons run on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::filter(d, f == "2", x > 1))
  f50c_native(df, function(d) dplyr::filter(d, f == "1" | o == "hi"))
  f50c_native(df, function(d) dplyr::mutate(d, flag = dplyr::if_else(f %in% c("1", "2"), 1L, 0L)))
  f50c_native(df, function(d) dplyr::mutate(d, k = dplyr::case_when(
    f == "10" ~ "ten",
    o >= "mid" ~ "upper",
    .default = "other"
  )))
  f50c_native(df, function(d) dplyr::mutate(d, both = f == "1" & !is.na(o) & o != "lo"))
})

test_that("F50: label comparisons inside summarise() aggregate on the GPU, grouped and not", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::summarise(d, n_a = sum(f == "2", na.rm = TRUE)))
  f50c_native(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, g),
                     n_a = sum(f == "2", na.rm = TRUE),
                     p = mean(f %in% c("1", "10"), na.rm = TRUE),
                     hi = sum(o >= "mid", na.rm = TRUE))
  }, arrange_by = "g")
  # without na.rm the NA code propagates, like dplyr
  f50c_native(df, function(d) dplyr::summarise(dplyr::group_by(d, g), n_a = sum(f == "2")),
              arrange_by = "g")
})

test_that("F50: label comparisons in grouped filter() and window mutate() run on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::filter(dplyr::group_by(d, g), f == "1"))
  f50c_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), k = sum(f == "2", na.rm = TRUE)))
  f50c_native(df, function(d) dplyr::filter(dplyr::group_by(d, g), sum(f %in% c("2")) > 1))
})

test_that("F50: across() over factor columns with a label predicate runs on the GPU", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::mutate(d, dplyr::across(c(f, o), ~ .x %in% c("1", "mid"))))
  f50c_native(df, function(d) dplyr::filter(d, dplyr::if_any(c(f, o), ~ .x == "hi" | .x == "10")))
  f50c_native(df, function(d) dplyr::summarise(d, dplyr::across(c(f, o), ~ sum(.x != "lo", na.rm = TRUE))))
})

test_that("F50: a renamed or copied factor keeps its labels through a lazy plan", {
  skip_if_no_gpu()
  df <- f50c_df()
  f50c_native(df, function(d) dplyr::filter(dplyr::mutate(d, f2 = f), f2 == "10"))
  f50c_native(df, function(d) dplyr::filter(dplyr::rename(d, ff = f), ff %in% c("1", "2")))
  f50c_native(df, function(d) dplyr::mutate(dplyr::select(d, o, x), up = o > "lo"))
  f50c_native(df, function(d) dplyr::filter(dplyr::filter(d, x > 1), f != "2"))
})

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

test_that("F50: 0-row and all-NA factors compare on the GPU", {
  skip_if_no_gpu()
  df0 <- f50c_df()[0, ]
  f50c_native(df0, function(d) dplyr::filter(d, f == "1"))
  f50c_native(df0, function(d) dplyr::mutate(d, a = f %in% c("1", "2"), b = o < "hi"))
  dfna <- tibble::tibble(f = factor(c(NA, NA, NA), levels = c("a", "b")), x = 1:3)
  f50c_native(dfna, function(d) dplyr::mutate(d, a = f == "a", b = f %in% c("a", NA), c = f != "zz"))
  f50c_native(dfna, function(d) dplyr::filter(d, f %in% NA_character_))
})

# ---------------------------------------------------------------------------
# F18's rule is kept: anything that isn't a label comparison
# ---------------------------------------------------------------------------

test_that("F50: factor against a number still compares labels (F18), never codes", {
  skip_if_no_gpu()
  df <- f50c_df()
  # code 2 is label "1"; the label "2" is code 3
  f50c_same(df, function(d) dplyr::filter(d, f == 2))
  f50c_same(df, function(d) dplyr::mutate(d, a = f == 2L, b = f != 10, c = f %in% c(1, 2)))
})

test_that("F50: factor against a factor with other levels still errors like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor(c("a", "b")), h = factor(c("a", "c")))
  expect_error(dplyr::collect(dplyr::mutate(tbl_gpu(df), y = f == h)),
               "level sets of factors are different")
})
