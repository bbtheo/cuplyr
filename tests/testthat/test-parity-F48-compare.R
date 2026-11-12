# F48 acceptance tests: a number compared with a string (item 6) and a
# lubridate Period/Duration (or difftime) used outside date arithmetic
# (item 4). dplyr compares a number with a string as character, and gives
# lubridate's result or error for a Period/Duration; neither may be a cuplyr
# type error, a raw cuDF error, or a silently wrong number.

# dplyr is imported, not attached, in the test environment: bind the dplyr
# functions these pipelines call unqualified (file-local).
for (fn in c("if_else", "case_when", "between", "lag", "lead", "ntile",
             "cumall", "cumany", "cummean", "across")) {
  assign(fn, getExportedValue("dplyr", fn))
}
rm(fn)

f48_cmp_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 2),
    x = c(1.5, 2, NA, 4, 10),
    i = c(1L, 5L, 3L, NA, 10L),
    l = c(TRUE, FALSE, NA, TRUE, FALSE),
    s = c("a", "2", "10", NA, "TRUE")
  )
}

# --- item 6: number vs string -------------------------------------------------

test_that("filter() comparing a number with a string literal matches dplyr", {
  skip_if_no_gpu()
  df <- f48_cmp_df()
  f48_check(df, function(d) filter(d, x != "a"))
  f48_check(df, function(d) filter(d, x == "2"))
  f48_check(df, function(d) filter(d, i == "10"))
  f48_check(df, function(d) filter(d, l == "TRUE"))
  f48_check(df, function(d) filter(group_by(d, g), i != "1"))
})

test_that("a string column compared with a number matches dplyr", {
  skip_if_no_gpu()
  df <- f48_cmp_df()
  f48_check(df, function(d) filter(d, s == 2))
  f48_check(df, function(d) filter(d, s != 10L))
  f48_check(df, function(d) filter(d, s == x))
  f48_check(df, function(d) mutate(d, m = s == i, n = l != s))
})

test_that("mutate() comparing a number with a string matches dplyr", {
  skip_if_no_gpu()
  df <- f48_cmp_df()
  f48_check(df, function(d) mutate(d, m = x == "2", n = i != "5"))
  f48_check(df, function(d) mutate(group_by(d, g), m = if_else(x == "10", "ten", "other")))
  f48_check(df, function(d) summarise(group_by(d, g), n = sum(i == "10", na.rm = TRUE)))
})

test_that("%in% mixing numbers and strings matches dplyr", {
  skip_if_no_gpu()
  df <- f48_cmp_df()
  f48_check(df, function(d) filter(d, x %in% c("2", "a")))
  f48_check(df, function(d) filter(d, i %in% c("1", "10")))
  f48_check(df, function(d) filter(d, s %in% c(2, 10)))
  f48_check(df, function(d) mutate(d, m = s %in% c(1, 2)))
})

test_that("number-vs-number and string-vs-string comparisons stay native", {
  skip_if_no_gpu()
  df <- f48_cmp_df()
  f48_native(df, function(d) filter(d, x > 2, i != 3L))
  f48_native(df, function(d) filter(d, s == "2" | s != "a"))
  f48_native(df, function(d) mutate(d, m = x %in% c(2, 10), n = s %in% c("a", "2")))
})

# --- item 4: Period/Duration/difftime values outside date arithmetic ----------

test_that("a Period added to or multiplied by a number gives dplyr's error", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(x = c(1.5, 2, 3))
  f48_same_error(df, function(d) mutate(d, m = x + lubridate::days(1)),
                 class = "dplyr:::mutate_error")
  f48_same_error(df, function(d) mutate(d, m = x * lubridate::days(1)),
                 "periods must have integer values")
  p <- lubridate::days(1)
  f48_same_error(df, function(d) mutate(d, m = x + p), class = "dplyr:::mutate_error")
})

test_that("a Period as lag()'s n gives dplyr's error", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(x = c(1.5, 2, 3))
  f48_same_error(df, function(d) mutate(d, m = lag(x, lubridate::days(1))),
                 "Can't convert `n` <Period>")
  f48_same_error(df, function(d) mutate(d, m = lead(x, lubridate::days(1))),
                 "Can't convert `n` <Period>")
})

test_that("a number compared with a Period/Duration matches dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(g = c(1, 1, 2), x = c(1.5, 2e6, 3))
  f48_check(df, function(d) filter(d, x < lubridate::days(1)))
  f48_check(df, function(d) filter(group_by(d, g), x < lubridate::ddays(1)))
  f48_check(df, function(d) mutate(d, m = x >= lubridate::hours(2)))
})

test_that("a Duration/difftime result is not returned as a plain number", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(x = c(1.5, 2, 3))
  # dplyr returns a <Duration>/<difftime> column, which has no GPU
  # representation: the CPU route raises the upload error rather than
  # dropping the class.
  pipes <- list(
    function(d) mutate(d, m = x + lubridate::ddays(1)),
    function(d) mutate(d, m = x * lubridate::dhours(1)),
    function(d) mutate(d, m = x + as.difftime(1, units = "days"))
  )
  for (pipe in pipes) {
    m <- pipe(tibble::as_tibble(df))$m
    expect_true(isS4(m) || inherits(m, "difftime"))
    for (lazy in c(FALSE, TRUE)) {
      expect_error(collect(pipe(tbl_gpu(df, lazy = lazy))),
                   class = "cuplyr_error_unsupported_class")
    }
  }
})

test_that("Date and POSIXct shifted by a Period still match dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(d = as.Date("2024-01-30") + 0:2,
                   t = as.POSIXct("2024-03-01 10:00:00", tz = "UTC") + 3600 * (0:2))
  f48_check(df, function(d) mutate(d, a = d + lubridate::days(1), b = d - lubridate::days(31)))
  f48_check(df, function(d) mutate(d, a = t + lubridate::hours(1), b = t - lubridate::ddays(1)))
})
