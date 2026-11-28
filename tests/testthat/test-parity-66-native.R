# Acceptance tests for #66, cuplyr side. The vendored dplyr tests excluded by
# #66 can't reach cuplyr (they call dplyr's non-exported compute_by() and
# join_rows(), or build their expected value / lookup table with vctrs or
# tibble functions that need a data frame). These tests check that the
# operations they are about give dplyr's result on a native tbl_gpu, eager and
# lazy, so the exclusions hide no cuplyr gap.

p66_y_like <- function(d, y) {
  if (inherits(d, "tbl_gpu")) tbl_gpu(y, lazy = identical(d$exec_mode, "lazy")) else y
}

p66_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# ---------------------------------------------------------------------------
# compute_by(): `.by` group data, in order of appearance

test_that("`.by` groups in order of first appearance, one and two keys", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 1, 2, 2, 1), v = 1:5)
  p66_both(df, function(d) dplyr::summarise(d, n = dplyr::n(), s = sum(v), .by = x))
  p66_both(df, function(d) dplyr::mutate(d, n = dplyr::n(), .by = x))

  df2 <- tibble::tibble(x = c(5, 4, 5, 5), y = c(2, 3, 1, 2))
  p66_both(df2, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c(x, y)))
  p66_both(df2, function(d) dplyr::summarise(d, n = dplyr::n(), .by = c()))
})

test_that("`.by` on zero rows and with NA keys", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(NA, 1, NA, 2), v = 1:4)
  p66_both(df, function(d) dplyr::summarise(d, n = dplyr::n(), .by = x))
  p66_both(df[0, ], function(d) dplyr::summarise(d, n = dplyr::n(), .by = x))
})

test_that("group_data() of ungrouped, grouped and rowwise tables is dplyr's", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 1, 2, 2, 1))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_identical(dplyr::group_data(g), dplyr::group_data(df), info = paste("lazy", lazy))
    expect_identical(dplyr::group_data(dplyr::group_by(g, x)),
                     dplyr::group_data(dplyr::group_by(df, x)), info = paste("lazy", lazy))
    expect_identical(dplyr::group_data(dplyr::rowwise(g)),
                     dplyr::group_data(dplyr::rowwise(df)), info = paste("lazy", lazy))
  }
})

test_that("`.by` with a grouped or rowwise table errors as in dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5)
  grouped <- "Can't supply `.by` when `.data` is a grouped data frame."
  rowwise <- "Can't supply `.by` when `.data` is a rowwise data frame."
  for (d in list(df, tbl_gpu(df), tbl_gpu(df, lazy = TRUE))) {
    expect_error(dplyr::summarise(dplyr::group_by(d, x), n = dplyr::n(), .by = x), grouped, fixed = TRUE)
    expect_error(dplyr::mutate(dplyr::group_by(d, x), n = dplyr::n(), .by = x), grouped, fixed = TRUE)
    expect_error(dplyr::summarise(dplyr::rowwise(d), n = dplyr::n(), .by = x), rowwise, fixed = TRUE)
    expect_error(dplyr::mutate(dplyr::rowwise(d), n = dplyr::n(), .by = x), rowwise, fixed = TRUE)
  }
})

# ---------------------------------------------------------------------------
# join_rows(): missing values only match under ==, >= and <=

test_that("NA keys match under ==, >= and <= and not under > or <", {
  skip_if_no_gpu()
  x <- tibble::tibble(a = c(NA, 1, NA))
  y <- tibble::tibble(b = c(NA_real_, 2))
  for (cond in c("==", ">=", "<=", ">", "<")) {
    by <- eval(parse(text = sprintf("dplyr::join_by(a %s b)", cond)))
    p <- function(d) dplyr::inner_join(d, p66_y_like(d, y), by = by, relationship = "many-to-many")
    p66_both(x, p)
    p <- function(d) dplyr::left_join(d, p66_y_like(d, y), by = by, relationship = "many-to-many")
    p66_both(x, p)
  }
})

test_that("an equality key plus an inequality key with NA", {
  skip_if_no_gpu()
  x <- tibble::tibble(x = c(1, 1), y = c(2, NA))
  y <- tibble::tibble(x = c(1, 1), y = c(3, NA))
  for (by in list(dplyr::join_by(x == x, y <= y), dplyr::join_by(x == x, y < y),
                  dplyr::join_by(x == x, y >= y))) {
    p <- function(d) dplyr::inner_join(d, p66_y_like(d, y), by = by)
    p66_both(x, p)
  }
})

test_that("na_matches = 'never' drops NA matches under every condition", {
  skip_if_no_gpu()
  x <- tibble::tibble(a = c(NA, 1))
  y <- tibble::tibble(b = c(NA, 1))
  for (cond in c("==", ">=", "<=")) {
    by <- eval(parse(text = sprintf("dplyr::join_by(a %s b)", cond)))
    p <- function(d) dplyr::inner_join(d, p66_y_like(d, y), by = by, na_matches = "never")
    p66_both(x, p)
  }
})

# ---------------------------------------------------------------------------
# overlaps(bounds =)

test_that("full_join() with overlaps() matches dplyr for every `bounds`", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x_lower = c(1, 1, 3, 4), x_upper = c(2, 3, 4, 5))
  df2 <- tibble::tibble(y_lower = 2, y_upper = 4)
  for (b in c("[]", "[)", "(]", "()")) {
    by <- eval(bquote(dplyr::join_by(overlaps(x_lower, x_upper, y_lower, y_upper, bounds = .(b)))))
    p <- function(d) dplyr::full_join(d, p66_y_like(d, df2), by = by)
    p66_both(df1, p)
    p <- function(d) dplyr::inner_join(d, p66_y_like(d, df2), by = by)
    p66_both(df1, p)
  }
})

test_that("overlaps() with an NA bound and with zero rows", {
  skip_if_no_gpu()
  df1 <- tibble::tibble(x_lower = c(1, NA, 3), x_upper = c(2, 3, NA))
  df2 <- tibble::tibble(y_lower = c(2, NA), y_upper = c(4, 5))
  by <- dplyr::join_by(overlaps(x_lower, x_upper, y_lower, y_upper))
  p66_both(df1, function(d) dplyr::left_join(d, p66_y_like(d, df2), by = by))
  p66_both(df1[0, ], function(d) dplyr::full_join(d, p66_y_like(d, df2), by = by))
})

# ---------------------------------------------------------------------------
# recode_values() with an NA entry in the lookup table

test_that("recode_values() with NA in the lookup table matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c("a", "b", "a", NA, "c"))
  from <- c("a", "b", "c", NA)
  to <- c("A", "B", "C", NA)
  p66_both(df, function(d) dplyr::mutate(d, y = dplyr::recode_values(x, from = from, to = to, unmatched = "error")))
  p66_both(df, function(d) dplyr::mutate(d, y = dplyr::recode_values(x, from = from, to = to)))
  p66_both(df[0, ], function(d) dplyr::mutate(d, y = dplyr::recode_values(x, from = from, to = to, unmatched = "error")))
})

test_that("recode_values(unmatched = 'error') errors on an NA it doesn't handle", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c("a", NA))
  p <- function(d) dplyr::collect(dplyr::mutate(d, y = dplyr::recode_values(x, from = "a", to = "A", unmatched = "error")))
  expect_error(p(df), class = "vctrs_error_combine_unmatched")
  expect_error(p(tbl_gpu(df)), class = "vctrs_error_combine_unmatched")
  expect_error(p(tbl_gpu(df, lazy = TRUE)), class = "vctrs_error_combine_unmatched")
})

test_that("a lookup table on the GPU grows by a row with bind_rows()", {
  skip_if_no_gpu()
  table <- tibble::tribble(~from, ~to, "a", "A", "b", "B")
  extra <- tibble::tibble(from = NA_character_, to = NA_character_)
  p66_both(table, function(d) dplyr::bind_rows(d, extra))
})

# ---------------------------------------------------------------------------
# summarise() strips off a subclass, as dplyr does

test_that("summarise() on a subclassed tbl_gpu drops the subclass", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    info <- paste("lazy", lazy)
    gdf <- dplyr::group_by(tbl_gpu(tibble::tibble(a = c(1, 1, 2)), lazy = lazy), a)
    df <- gdf
    class(df) <- c("p66cls", class(gdf))

    out <- dplyr::summarise(df, n = dplyr::n(), .groups = "drop")
    expect_identical(class(out), c("tbl_gpu", "tbl"), info = info)
    expect_identical(class(dplyr::collect(out)), class(tibble::tibble()), info = info)

    out <- dplyr::summarise(df, n = dplyr::n(), .groups = "keep")
    expect_identical(class(out), c("tbl_gpu", "tbl"), info = info)
    expect_identical(class(dplyr::collect(out)), class(dplyr::group_by(tibble::tibble(a = 1), a)), info = info)

    udf <- tbl_gpu(tibble::tibble(a = c(1, 1, 2)), lazy = lazy)
    class(udf) <- c("p66cls", class(udf))
    expect_identical(class(dplyr::summarise(udf, n = dplyr::n())), c("tbl_gpu", "tbl"), info = info)
    expect_identical(class(dplyr::summarise(udf, n = dplyr::n(), .by = a)), c("tbl_gpu", "tbl"), info = info)
    expect_identical(
      tibble::as_tibble(dplyr::collect(dplyr::summarise(udf, n = dplyr::n(), .by = a))),
      tibble::tibble(a = c(1, 2), n = c(2L, 1L)), info = info
    )
  }
})
