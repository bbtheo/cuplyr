# A Date shifted by a lubridate Period/Duration used to become a native
# date_add_days with the literal's seconds as the day count (round-2 B3).
# The acceptance tests cover mutate(); these cover filter() and a plain
# whole-number shift that must stay native.

test_that("filter() on a Date shifted by a Period/Duration matches dplyr", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- data.frame(dt = as.Date("2024-01-05") + 0:3, x = 1:4)
  p1 <- function(d) dplyr::filter(d, dt + lubridate::days(1) > as.Date("2024-01-07"))
  p2 <- function(d) dplyr::filter(d, dt - lubridate::dweeks(1) >= as.Date("2023-12-31"))
  expect_same_as_dplyr(df, p1)
  expect_same_as_dplyr_lazy(df, p1)
  expect_same_as_dplyr(df, p2)
  expect_same_as_dplyr_lazy(df, p2)
})

test_that("a Period literal is not rewritten to date_add_days; a number still is", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  schema <- list(names = "dt", types = c(dt = "TIMESTAMP_DAYS"))
  parse_or_cpu <- function(q) {
    tryCatch(ir_parse_quo(q, schema), cuplyr_timestamp_needs_cpu = function(e) "cpu")
  }
  for (q in list(rlang::quo(dt + lubridate::days(1)),
                 rlang::quo(dt - lubridate::ddays(2)),
                 rlang::quo(lubridate::weeks(1) + dt))) {
    ir <- parse_or_cpu(q)
    expect_false(is.list(ir) && ir$op %in% c("date_add_days", "date_sub_days"))
  }
  ir <- parse_or_cpu(rlang::quo(dt + 7))
  expect_identical(ir$op, "date_add_days")
  expect_identical(ir$args[[2]]$value, 7L)
})
