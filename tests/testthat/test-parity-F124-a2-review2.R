# F124 (attempt 2): cases from the round-2 review. On a lazy parquet plan every
# warning comes when the plan runs, in the order dplyr raises them; joins below
# the min()/max() verb stay lazy; a column-free call is evaluated as often as in
# dplyr.
# Helpers: helper-parity-F124.R, helper-parity-F85.R (pq_write()).

# `pipeline(x, y)` over pairs of sources against dplyr on the tibbles: values,
# types, the lazy schema, the warnings in order over verb + collect(); on a plan
# with a parquet table no warning at the verb and no plan break.
f124r2_check <- function(dx, dy, pipeline, info) {
  oracle <- f124_capture(pipeline(dx, dy))
  want <- oracle_normalize(oracle$value)
  px <- pq_write(dx)
  py <- pq_write(dy)
  make <- list(
    eager = function() list(tbl_gpu(dx, lazy = FALSE), tbl_gpu(dy, lazy = FALSE)),
    lazy = function() list(tbl_gpu(dx, lazy = TRUE), tbl_gpu(dy, lazy = TRUE)),
    parquet = function() list(read_parquet_gpu(px, lazy = TRUE), read_parquet_gpu(py, lazy = TRUE)),
    mixed = function() list(read_parquet_gpu(px, lazy = TRUE), tbl_gpu(dy, lazy = TRUE))
  )
  for (s in names(make)) {
    lbl <- paste0(info, " [", s, "]")
    xy <- make[[s]]()
    verb <- f124_capture(pipeline(xy[[1L]], xy[[2L]]))
    planned <- if (s != "eager") f124_col_types(tidyselect_proxy(current_schema(verb$value)))
    col <- f124_capture(collect(verb$value))
    got <- oracle_normalize(col$value)
    expect_identical(f124_col_types(got), f124_col_types(want), info = paste(lbl, "types"))
    expect_equal(got, want, info = paste(lbl, "values"))
    if (!is.null(planned)) {
      expect_identical(planned, f124_col_types(want), info = paste(lbl, "lazy schema"))
    }
    expect_identical(f124_msgs(c(verb$warnings, col$warnings)), f124_msgs(oracle$warnings),
                     info = paste(lbl, "warnings, in order"))
    expect_identical(c(verb$fallback, col$fallback), character(), info = paste(lbl, "CPU fallback"))
    if (s %in% c("parquet", "mixed")) {
      expect_length(verb$warnings, 0L)
      expect_identical(c(verb$plan_break, col$plan_break), character(), info = paste(lbl, "plan break"))
    }
  }
}

test_that("F124 review a2-2 (B13): warnings keep dplyr's order on a parquet plan", {
  skip_if_no_gpu()
  dD <- tibble::tibble(f = factor(c("a", "b"), levels = c("a", "b", "c")), a = 1:2, b = c(1, 2))
  none <- tibble::tibble(k = 1L)
  f124r2_check(dD, none, function(x, y) {
    mutate(group_by(mutate(filter(x, b > 100), m1 = max(b)), f, .drop = FALSE), m2 = max(a))
  }, "D: carried max(), then .drop = FALSE integer max()")
  f124r2_check(dD, none, function(x, y) {
    mutate(group_by(mutate(filter(x, b > 100), m1 = max(b)), f, .drop = FALSE), m2 = max(b))
  }, "D2: carried max(), then .drop = FALSE double max()")
  f124r2_check(dD, none, function(x, y) {
    filter(group_by(mutate(filter(x, b > 100), m1 = max(b)), f, .drop = FALSE), a == max(a))
  }, "D3: carried max(), then a .drop = FALSE window filter()")
  # rows left, one level empty
  f124r2_check(dD, none, function(x, y) {
    mutate(group_by(mutate(filter(x, b > 1.5), m1 = max(b)), f, .drop = FALSE), m2 = max(a))
  }, "D4: .drop = FALSE after a filter, one group with rows")
})

test_that("F124 review a2-2 (N50): joins below a min()/max() verb", {
  skip_if_no_gpu()
  dx <- tibble::tibble(k = 1:3, f = factor(c("a", "b", "c"), levels = c("a", "b", "c", "d")),
                       a = c(10L, 20L, 30L), b = c(0.5, 0.6, 0.7), wide1 = c(1, 2, 3))
  dy <- tibble::tibble(k = 2:4, c = c(100L, 200L, 300L), wide2 = c(9, 8, 7))
  dy0 <- tibble::tibble(k = 7:8, c = c(1L, 2L), wide2 = c(1, 2))
  f124r2_check(dx, dy, function(x, y) mutate(inner_join(x, y, by = "k"), m = max(a)), "J1 inner, rows")
  f124r2_check(dx, dy0, function(x, y) mutate(inner_join(x, y, by = "k"), m = max(a)), "J2 inner, no rows")
  f124r2_check(dx, dy, function(x, y) mutate(filter(left_join(x, y, by = "k"), c > 150L), m = max(a) + 1L),
               "J3 left, filter")
  f124r2_check(dx, dy, function(x, y) mutate(group_by(inner_join(x, y, by = "k"), f, .drop = FALSE), m = max(a)),
               "J4 inner, .drop = FALSE")
  f124r2_check(dx, dy0, function(x, y) mutate(semi_join(x, y, by = "k"), m = min(a)), "J5 semi, no rows")
})

test_that("F124 review a2-2 (N50): a column-free call next to an integer max() runs as often as in dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:6, b = c(0.9, 0.1, 0.8, 0.7, 0.2, 0.6), w = c(1, 2, 3, 4, 5, 6))
  path <- pq_write(df)
  calls <- 0L
  draw <- function() {
    calls <<- calls + 1L
    0.5
  }
  pl <- function(x, thr) mutate(filter(x, b > thr), r = draw(), e = a == max(a), m = max(a) + draw())
  for (thr in c(0.5, 100)) {
    calls <- 0L
    want <- suppressWarnings(pl(df, thr))
    want_calls <- calls
    for (s in c("lazy", "parquet")) {
      calls <- 0L
      src <- if (s == "lazy") tbl_gpu(df, lazy = TRUE) else read_parquet_gpu(path, lazy = TRUE)
      got <- suppressWarnings(collect(pl(src, thr)))
      expect_identical(calls, want_calls, info = paste(s, thr, "draw() calls"))
      expect_equal(oracle_normalize(got), oracle_normalize(want), info = paste(s, thr, "values"))
    }
  }
})
