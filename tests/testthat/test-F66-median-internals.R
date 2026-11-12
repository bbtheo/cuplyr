# F66 round 2: details of roll_median()'s engine selection that the
# acceptance tests (test-parity-F66-median-exact.R) do not pin.

test_that("F66: an invalid cuplyr.roll_median warns on short and empty input too", {
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  withr::local_options(cuplyr.roll_median = "fast-f66b")
  expect_warning(res <- roll_median(c(1, 2), 5), "cuplyr.roll_median", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  expect_identical(res, c(NA_real_, NA_real_))
  # warned once for this value: the empty input is silent but still works
  expect_no_warning(res <- roll_median(numeric(0), 3))
  expect_identical(res, numeric(0))
  withr::local_options(cuplyr.roll_median = "fast-f66c")
  expect_warning(roll_median(numeric(0), 3), "cuplyr.roll_median", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
})

test_that("F66: roll_has_frollmedian() computes the data.table check once per session", {
  old <- roll_median_state$has_dt
  withr::defer(roll_median_state$has_dt <- old)
  roll_median_state$has_dt <- NULL
  first <- roll_has_frollmedian()
  expect_type(first, "logical")
  expect_identical(roll_median_state$has_dt, first)
  # a cached value is returned without looking at data.table again
  roll_median_state$has_dt <- !first
  expect_identical(roll_has_frollmedian(), !first)
})

test_that("F66: the note names the environment variable as well as the option", {
  old <- roll_median_state$noted
  roll_median_state$noted <- FALSE
  withr::defer(roll_median_state$noted <- old)
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = "frollmedian")
  withr::local_options(cuplyr.roll_median = NULL)
  local_mocked_bindings(roll_has_frollmedian = function() FALSE)
  cnd <- expect_message(roll_median(c(1, 2, 3), 2), class = "cuplyr_message_roll_median_exact")
  expect_match(conditionMessage(cnd), "CUPLYR_ROLL_MEDIAN", fixed = TRUE)
  expect_match(conditionMessage(cnd), "cuplyr.roll_median", fixed = TRUE)
})

test_that("F66: the opt-in's exact pass at +-Inf positions drops NAs with na.rm = TRUE", {
  skip_if_not_installed("data.table", "1.18.4")
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  withr::local_options(cuplyr.roll_median = "frollmedian")
  # w = 3 so the overflowing windows (i = 4, 5, 6) each hold an NA
  x <- c(1e308, NA, 1.5e308, 1.7e308, NA, 1.6e308)
  ref <- rep(NA_real_, length(x))
  for (i in 3:length(x)) {
    w <- x[(i - 2):i]
    w <- w[!is.na(w)]
    if (length(w)) ref[i] <- stats::median(w)
  }
  expect_true(any(is.infinite(data.table::frollmedian(x, 3, na.rm = TRUE))))
  expect_identical(roll_median(x, 3, na.rm = TRUE), ref)
  expect_true(all(is.finite(ref[!is.na(ref)])))
})

test_that("F66: the opt-in keeps data.table's value wherever it is finite", {
  skip_if_not_installed("data.table", "1.18.4")
  x <- c(0.1, 0.7, 1.5e308, 1.7e308, Inf, 0.3, NA, 0.9, 0.2, 0.4)
  for (na_rm in c(FALSE, TRUE)) {
    dt <- data.table::frollmedian(x, 2, na.rm = na_rm)
    got <- roll_median_dt(x, 2L, na_rm)
    fin <- !is.infinite(dt)
    expect_identical(got[fin], dt[fin], info = sprintf("na.rm=%s", na_rm))
    expect_identical(got[!fin], roll_median_sorted(x, 2, na_rm)[!fin],
                     info = sprintf("na.rm=%s", na_rm))
  }
})

test_that("F66: roll_median_sorted() rejects a non-flag na.rm", {
  expect_error(roll_median_sorted(c(1, 2, 3), 2, NA), "must be TRUE or FALSE", fixed = TRUE)
  expect_error(roll_median_sorted(c(1, 2, 3), 2, 1), "must be TRUE or FALSE", fixed = TRUE)
  expect_error(roll_median_sorted(c(1, 2, 3), 2, c(TRUE, TRUE)), "must be TRUE or FALSE",
               fixed = TRUE)
  expect_identical(roll_median_sorted(c(1, NA, 3), 2, TRUE), c(NA, 1, 3))
})
