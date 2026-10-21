# F48 round 2: regression tests for the review's findings.
#
# - a column-free Period/Duration/difftime constant must not leak the internal
#   `cuplyr_type_needs_cpu` routing signal (B2);
# - a Period/Duration used as a parameter (ntile()/nth() `n`, slice() index,
#   quantile() `probs`) is dplyr's error, not its raw number (B3);
# - item 3's window shapes without `na.rm`, under a non-C and a C collation.

skip_if_not_installed("lubridate")

# dplyr is imported, not attached, in the test environment
ntile <- dplyr::ntile
nth <- dplyr::nth

f48r2_df <- function() {
  data.frame(g = c(1, 1, 2, 2, 2), x = c(3, NA, 1, 2, 5),
             date = as.Date("2024-01-01") + 0:4)
}

# No `cuplyr_type_needs_cpu` (or any cuplyr-internal condition class) may reach
# the user.
f48r2_no_leak <- function(expr) {
  err <- tryCatch({ force(expr); NULL }, error = function(e) e)
  if (!is.null(err)) {
    expect_false(inherits(err, "cuplyr_type_needs_cpu"))
    expect_false(grepl("needs_cpu", conditionMessage(err), fixed = TRUE))
  }
  invisible(err)
}

test_that("a column-free difftime/Period constant that only feeds Date arithmetic matches dplyr", {
  skip_if_no_gpu()
  df <- f48r2_df()
  p1 <- function(d) mutate(d, gap = as.difftime(7, units = "days"), due = date + gap, gap = NULL)
  p2 <- function(d) mutate(d, step = lubridate::days(1), nxt = date + step, step = NULL)
  p3 <- function(d) transmute(d, due = date + as.difftime(2, units = "days"))
  for (pipe in list(p1, p2, p3)) {
    expect_same_as_dplyr(df, pipe)
    expect_same_as_dplyr_lazy(df, pipe)
  }
  # the values, not only the shape
  got <- collect(p1(tbl_gpu(df)))
  expect_equal(got$due, as.Date("2024-01-08") + 0:4)
  got <- collect(p2(tbl_gpu(df, lazy = TRUE)))
  expect_equal(got$nxt, as.Date("2024-01-02") + 0:4)
})

test_that("a column-free Duration column ends in the unsupported-class error, not a routing signal", {
  skip_if_no_gpu()
  df <- f48r2_df()
  for (lazy in c(FALSE, TRUE)) {
    gdf <- tbl_gpu(df, lazy = lazy)
    for (run in list(
      function() collect(mutate(gdf, m = lubridate::ddays(1))),
      function() collect(mutate(gdf, m = lubridate::days(1))),
      function() collect(mutate(gdf, m = as.difftime(7, units = "days"))),
      function() collect(summarise(gdf, m = lubridate::ddays(1))),
      function() collect(transmute(gdf, m = lubridate::ddays(1)))
    )) {
      err <- f48r2_no_leak(run())
      expect_s3_class(err, "cuplyr_error_unsupported_class")
    }
  }
})

test_that("ntile()/nth() with a Period/Duration `n` raise dplyr's error", {
  skip_if_no_gpu()
  df <- f48r2_df()
  f48_same_error(df, function(d) mutate(d, a = ntile(x, lubridate::dseconds(2))),
                 "Can't convert `n`")
  f48_same_error(df, function(d) mutate(d, a = nth(x, lubridate::dseconds(2))),
                 "Can't convert `n`")
  f48_same_error(df, function(d) mutate(group_by(d, g), a = nth(x, lubridate::seconds(2))),
                 "Can't convert `n`")
  f48_same_error(df, function(d) summarise(d, a = nth(x, lubridate::dseconds(2))),
                 "Can't convert `n`")
  f48_same_error(df, function(d) summarise(group_by(d, g), a = nth(x, lubridate::seconds(2))),
                 "Can't convert `n`")
})

test_that("slice() with a Period/Duration index raises dplyr's error", {
  skip_if_no_gpu()
  df <- f48r2_df()
  f48_same_error(df, function(d) slice(d, lubridate::dseconds(2)), "must be numeric")
  f48_same_error(df, function(d) slice(d, lubridate::seconds(2)), "must be numeric")
})

test_that("a plain whole-number `n` still runs natively", {
  skip_if_no_gpu()
  df <- f48r2_df()
  f48_native(df, function(d) mutate(d, a = ntile(x, 2L)))
  f48_native(df, function(d) mutate(d, a = nth(x, 2L)))
  f48_native(df, function(d) summarise(group_by(d, g), a = nth(x, 2L)))
  f48_native(df, function(d) slice(d, 2L))
})

# Item 3's window shapes without na.rm (NA propagates), data with NAs.
f48r2_coll_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 2, 2),
    t = c("B", "a", "A", "b", "c", NA),
    u = c("a", "b", "c", "A", "B", "b")
  )
}

f48r2_window_pipelines <- list(
  mutate_max = function(d) mutate(d, m = max(t)),
  mutate_min = function(d) mutate(d, m = min(t)),
  mutate_max_u = function(d) mutate(d, m = max(u)),
  mutate_min_grouped = function(d) mutate(group_by(d, g), m = min(u)),
  mutate_max_grouped = function(d) mutate(group_by(d, g), m = max(t)),
  filter_eq_max = function(d) filter(d, u == max(u)),
  filter_eq_max_grouped = function(d) filter(group_by(d, g), u == max(u)),
  filter_eq_max_na = function(d) filter(d, t == max(t)),
  filter_ge_min = function(d) filter(group_by(d, g), u >= min(u))
)

test_that("window min()/max() of a string without na.rm match dplyr under a non-C collation", {
  skip_if_no_gpu()
  skip_if_not(f48_collate_ok("en_US.UTF-8"), "en_US.UTF-8 collation not available")
  withr::local_collate("en_US.UTF-8")
  df <- f48r2_coll_df()
  for (nm in names(f48r2_window_pipelines)) {
    pipe <- f48r2_window_pipelines[[nm]]
    expect_same_as_dplyr(df, pipe)
    expect_same_as_dplyr_lazy(df, pipe)
  }
  # the comparison selects some rows and drops others under en_US ("a" < "B")
  got <- collect(f48r2_window_pipelines$filter_eq_max(tbl_gpu(df)))
  expect_identical(got$u, "c")
  got <- collect(f48r2_window_pipelines$filter_ge_min(tbl_gpu(df)))
  expect_identical(got$u, c("a", "b", "c", "A", "B", "b"))
})

test_that("window min()/max() of a string without na.rm match dplyr under a C collation", {
  skip_if_no_gpu()
  skip_if_not(f48_collate_ok("C"), "C collation not available")
  withr::local_collate("C")
  df <- f48r2_coll_df()
  for (nm in names(f48r2_window_pipelines)) {
    pipe <- f48r2_window_pipelines[[nm]]
    expect_same_as_dplyr(df, pipe)
    expect_same_as_dplyr_lazy(df, pipe)
  }
  got <- collect(f48r2_window_pipelines$filter_eq_max(tbl_gpu(df)))
  expect_identical(got$u, "c")
  # uppercase sorts before lowercase in byte order
  got <- collect(f48r2_window_pipelines$mutate_min_grouped(tbl_gpu(df)))
  expect_identical(got$m, c("a", "a", "A", "A", "A", "A"))
})
