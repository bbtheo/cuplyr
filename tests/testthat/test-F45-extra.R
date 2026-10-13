# F45 extras: edges of the running schema beyond the acceptance file.

f45x_df <- function() {
  data.frame(
    g = c(1L, 1L, 2L, 2L), x = c(1, 2, NA, 4), y = c(10L, 20L, NA, 40L),
    s = c("a", "b", NA, "d"), d = c(4, 5, 6, 7), stringsAsFactors = FALSE
  )
}

f45x_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("an unnamed expression dot is a column for a later across()", {
  skip_if_no_gpu()
  df <- f45x_df()
  suppressWarnings({
    f45x_both(df, function(d) mutate(d, x + 1, across(where(is.double), ~ .x * 2)))
  })
})

test_that("a dot with an unknown type is selected by name, not by type", {
  skip_if_no_gpu()
  df <- f45x_df()
  f45x_both(df, function(d) mutate(d, r = rev(d), across(starts_with("r"), ~ .x + 1)))
  f45x_both(df, function(d) mutate(d, r = rev(d), across(where(is.double), ~ .x + 1)))
})

test_that("a removed column that is re-created is selected again", {
  skip_if_no_gpu()
  df <- f45x_df()
  f45x_both(df, function(d) mutate(d, x = NULL, x = d * 2L, across(where(is.integer), ~ .x + 1L)))
})

test_that("if_any() after a typed overwrite", {
  skip_if_no_gpu()
  df <- f45x_df()
  f45x_both(df, function(d) mutate(d, y = y > 15L, flag = dplyr::if_all(where(is.logical))))
  f45x_both(df, function(d) mutate(d, k = x > 1, dplyr::if_any(c(k, y), ~ .x > 2)))
})

# --- round 3: where a column re-created after a NULL-valued dot sits ----------
# dplyr's mutate() mask drops a removed column and appends a re-added one at
# the end (src/mask.cpp); summarise() skips a NULL result but adds a new name
# at the end. A selection after such a re-creation must match dplyr.

test_that("a maybe-absent column set again by an unknown-valued dot sits where dplyr puts it", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  v <- NULL
  f <- function() c(5, 6)
  f45x_both(df, function(d) mutate(d, x = v, w = y * 2L, x = f(), z = 1,
                                   across(-y, ~ .x * 10, .names = "{.col}_t")))
  f45x_both(df, function(d) mutate(d, x = v, w = y * 2L, x = f(), across(last_col(), ~ .x * 10)))
})

test_that("an across() output or if_any() output re-creating a maybe-absent column", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  v <- NULL
  f45x_both(df, function(d) mutate(d, x = v, w = y * 2L, across(y, ~ .x + 1, .names = "x"),
                                   across(last_col(), ~ .x * 10)))
  f45x_both(df, function(d) mutate(d, b = v, w = y * 2L, b = dplyr::if_any(c(x, y), ~ .x > 1),
                                   across(last_col(), ~ .x + 1, .names = "{.col}_L")))
})

test_that("summarise(): a new name re-created after a NULL-valued summary", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  v <- NULL
  f45x_both(df, function(d) summarise(d, z = v, w = sum(y), z = sum(x),
                                      across(last_col(), ~ .x * 10, .names = "{.col}_L")))
  f45x_both(df, function(d) summarise(d, x = v, w = sum(y), x = sum(y),
                                      across(last_col(), ~ .x * 10, .names = "{.col}_L")))
})

test_that("a column removed with a literal NULL and re-created is appended, natively", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  f45x_both(df, function(d) mutate(d, x = NULL, w = y * 2L, x = y + 1L, across(last_col(), ~ .x * 10L)))
  f45x_both(df, function(d) mutate(d, x = NULL, w = y * 2L, x = y + 1L,
                                   across(-y, ~ .x * 10L, .names = "{.col}_t")))
})

# --- round 4: the running schema records R's type, or none -------------------
# A created column's type in the running schema must be the one R gives it,
# since a later where() selects by it. Where cuplyr's static type may differ
# (median() of an integer, max() of a logical, lubridate's year()), the type
# is NA: the column exists, a type predicate goes to dplyr.

test_that("the running schema records only types R surely gives", {
  run <- list(
    names = c("l", "i", "x", "s", "d"),
    types = c("BOOL8", "INT32", "FLOAT64", "STRING", "TIMESTAMP_DAYS")
  )
  run$base_names <- run$names
  kdt <- as.difftime(1, units = "days")
  k <- 2L
  year <- lubridate::year
  day <- lubridate::day
  type_of <- function(q) across_infer_output(q, run)
  untyped <- list(type = NA_character_, maybe = FALSE)
  expect_identical(type_of(rlang::quo(median(x)))$type, "FLOAT64")
  expect_identical(type_of(rlang::quo(median(d)))$type, "TIMESTAMP_DAYS")
  expect_identical(type_of(rlang::quo(median(i))), untyped)
  expect_identical(type_of(rlang::quo(stats::median(i))), untyped)
  expect_identical(type_of(rlang::quo(max(l))), untyped)
  expect_identical(type_of(rlang::quo(cummin(l))), untyped)
  expect_identical(type_of(rlang::quo(-l)), untyped)
  expect_identical(type_of(rlang::quo(max(i)))$type, "INT32")
  expect_identical(type_of(rlang::quo(ntile(i, 2))), untyped)
  expect_identical(type_of(rlang::quo(lag(median(i)))), untyped)
  expect_identical(type_of(rlang::quo(x + kdt)), untyped)
  expect_identical(type_of(rlang::quo(x + !!kdt)), untyped)
  expect_identical(type_of(rlang::quo(i + k))$type, "INT32")
  skip_if_not_installed("lubridate")
  expect_identical(type_of(rlang::quo(year(d))), untyped)
  expect_identical(type_of(rlang::quo(year(d) + 1L)), untyped)
  expect_identical(type_of(rlang::quo(lubridate::year(d))), untyped)
  expect_identical(type_of(rlang::quo(year(d) > 2000))$type, "BOOL8")
  expect_identical(type_of(rlang::quo(day(d)))$type, "INT32")
  expect_identical(type_of(rlang::quo(lubridate::day(d)))$type, "INT32")
})

test_that("types R surely gives keep a later where() on the GPU", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  withr::local_options(cuplyr.fallback = "error")
  year <- lubridate::year
  day <- lubridate::day
  df <- data.frame(x = c(1, 2, 3), d = as.Date(c("2024-01-01", "2024-06-15", "2024-12-31")))
  f45x_both(df, function(d) mutate(d, b = year(d) > 2023, across(where(is.logical), ~ !.x)))
  f45x_both(df, function(d) mutate(d, dd = day(d), across(where(is.integer), ~ .x + 1L)))
  f45x_both(df, function(d) mutate(d, dd = lubridate::day(d), across(where(is.integer), ~ .x + 1L)))
})

test_that("an unnamed bare dot naming a NULL-removed column: whatever the environment holds", {
  skip_if_no_gpu()
  df <- data.frame(y = c(3L, 4L), x = c(1, 2))
  v <- NULL
  x <- data.frame(a = c(5, 6), b = c(7L, 8L))
  withCallingHandlers({
    # x is the last column: a data frame in the environment is unpacked
    f45x_both(df, function(d) mutate(d, x = v, x, across(where(is.double), ~ .x * 10)))
    f45x_both(df, function(d) mutate(d, x = v, x, across(last_col(), ~ .x * 10)))
  }, warning = function(w) if (grepl("Unnamed mutate expression", conditionMessage(w))) invokeRestart("muffleWarning"))
})
