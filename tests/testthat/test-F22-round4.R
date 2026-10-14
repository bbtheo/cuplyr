# F22 round 4: reframe()'s native path takes a positive allowlist of
# aggregation calls and argument shapes (R/reframe.R, "The native allowlist"),
# so it no longer inherits summarise()/IR/window divergences: window and
# aggregate calls inside an argument, string operands, NaN-prone comparisons,
# integer/logical median() and non-numeric mean() all go to dplyr.

f22r4_unname <- function(df) {
  for (nm in names(df)) df[[nm]] <- unname(df[[nm]])
  tibble::as_tibble(df)
}

# Like the acceptance file's f22_check(): eager and lazy against dplyr,
# warning presence included; `native = TRUE` forbids the CPU fallback.
f22r4_check <- function(df, pipeline, native = TRUE) {
  warned <- FALSE
  oracle <- withCallingHandlers(pipeline(tibble::as_tibble(df)), warning = function(w) {
    warned <<- TRUE
    invokeRestart("muffleWarning")
  })
  expected <- f22r4_unname(oracle)
  for (lazy in c(FALSE, TRUE)) {
    gpu_warned <- FALSE
    got <- withr::with_options(
      list(cuplyr.fallback = if (native) "error" else "silent"),
      withCallingHandlers(collect(pipeline(tbl_gpu(df, lazy = lazy))), warning = function(w) {
        gpu_warned <<- TRUE
        invokeRestart("muffleWarning")
      })
    )
    label <- if (lazy) "lazy" else "eager"
    expect_equal(tibble::as_tibble(got), expected, info = label)
    expect_identical(gpu_warned, warned, info = label)
  }
}

f22r4_df <- function() {
  data.frame(
    g = c(1, 1, 1, 2, 2, 2, 2, 3),
    x = c(1.5, NA, -3, 0, 2.25, 8, -1, 4),
    y = c(1, 0, 1, 0, 2, 1, 1, 2),
    i = c(7L, -3L, NA, 4L, 0L, 9L, 2L, 5L),
    b = c(TRUE, NA, FALSE, TRUE, FALSE, TRUE, NA, TRUE),
    s = c("a", "c", "B", "D", "e", "F", "g", "H"),
    stringsAsFactors = FALSE
  )
}

test_that("the plan takes only allowlisted aggregations and argument shapes", {
  skip_if_no_gpu()
  schema <- list(
    names = c("g", "x", "i", "b", "s", "dt", "f"),
    types = c("FLOAT64", "FLOAT64", "INT32", "BOOL8", "STRING", "TIMESTAMP_DAYS", "DICTIONARY32"),
    factor_levels = list(f = c("a", "b"))
  )
  native <- function(...) !is.null(reframe_native_plan(rlang::enquos(...), schema))
  withr::local_collate("C")

  # allowed
  expect_true(native(q = quantile(i * 2, c(0.1, 0.9)), n = n(), m = mean(x, na.rm = TRUE)))
  expect_true(native(m = median(x), s = sd(b), v = var(i), mx = max(b), r = range(dt)))
  expect_true(native(m = sum(i > 2L & b), e = sum(i %% 2 == 0), na = sum(is.na(i))))
  expect_true(native(m = mean(abs(x) + exp(i)), n = mean(-i), lo = min(i + 1L)))
  expect_true(native(m = median(i / 2L), a = base::mean(x), b2 = stats::sd(x), k = dplyr::n()))
  expect_true(native(r = range(s)))

  # aggregations off the list
  expect_false(native(m = dplyr::n_distinct(i)))
  expect_false(native(m = dplyr::first(i)))
  expect_false(native(m = any(b)))
  expect_false(native(m = mean(i) + 1))
  expect_false(native(m = stats::mean(x)))
  expect_false(native(n = n(x)))
  # argument shapes off the list: window/aggregate calls, if_else, strings
  expect_false(native(q = quantile(cummax(x), c(0, 1))))
  expect_false(native(q = quantile(x - mean(x), c(0, 1))))
  expect_false(native(m = max(dplyr::lag(x), na.rm = TRUE)))
  expect_false(native(r = range(dplyr::if_else(i > 2L, i, 0L))))
  expect_false(native(m = sum(s < "B")))
  expect_false(native(m = max(toupper(s))))
  # math the GPU gets wrong
  expect_false(native(m = mean(x ^ 2)))
  expect_false(native(m = mean(sqrt(x))))
  expect_false(native(m = mean(log(x))))
  expect_false(native(m = mean(x %% 2)))
  expect_false(native(m = sum(i %% 0 == 0)))
  expect_false(native(m = mean(-b)))
  expect_false(native(m = mean(+x)))
  expect_false(native(m = mean(x + NaN)))
  # NaN-prone operands of comparisons / logic / is.na()
  expect_false(native(m = sum(x > 0)))
  expect_false(native(m = sum(i / 2L > 1)))
  expect_false(native(m = sum(is.na(x))))
  expect_false(native(m = sum(!x)))
  # type rules
  expect_false(native(m = median(i)))
  expect_false(native(m = median(b)))
  expect_false(native(m = median(i * 2L)))
  expect_false(native(q = quantile(b, c(0, 1))))
  expect_false(native(q = quantile(i > 2L, c(0, 1))))
  expect_false(native(m = mean(s)))
  expect_false(native(m = mean(dt)))
  expect_false(native(r = range(f)))
  expect_false(native(r = range(dt + 1)))
  # call details
  expect_false(native(m = sum(i, na.rm = T)))
  expect_false(native(m = sum(i, x)))
  expect_false(native(m = mean(x, trim = 0.1)))
  expect_false(native(m = mean(.data$x)))
  v <- 2
  expect_false(native(m = mean(x + v)))
  local({
    `+` <- function(a, b) 1
    expect_false(native(m = mean(x + 1)))
  })
  local({
    mean <- function(x, ...) 1
    expect_false(native(m = mean(x)))
  })

  withr::local_collate("en_US.UTF-8")
  expect_false(native(r = range(s)))
  expect_true(native(r = range(x), m = sum(i > 2L)))
})

test_that("allowlisted computed arguments are native and match dplyr", {
  skip_if_no_gpu()
  df <- f22r4_df()
  f22r4_check(df, function(d) {
    reframe(d, m = sum(i > 2L & b, na.rm = TRUE), e = sum(i %% 2 == 0, na.rm = TRUE),
            na = sum(is.na(i)), q = quantile(x / y, c(0, 1), na.rm = TRUE), .by = g)
  })
  f22r4_check(df, function(d) {
    reframe(group_by(d, g), m = mean(abs(x) + exp(i)), n = mean(-i, na.rm = TRUE),
            lo = min(i + 1L, na.rm = TRUE), r = range(i %/% 2L, na.rm = TRUE))
  })
  f22r4_check(df, function(d) {
    reframe(d, mx = max(b, na.rm = TRUE), s = sd(b, na.rm = TRUE), v = var(i, na.rm = TRUE),
            hi = max(i > 3L, na.rm = TRUE), md = median(i / 2L, na.rm = TRUE),
            q = quantile(i * 2, c(0.25, 0.75), na.rm = TRUE), .by = g)
  })
  f22r4_check(df, function(d) {
    reframe(d, k = dplyr::n(), a = base::mean(x * 2, na.rm = TRUE), r = base::range(x, na.rm = TRUE))
  })
})

test_that("integer overflow inside an argument warns and gives NA like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), i = c(2147483647L, 1L, 3L))
  f22r4_check(df, function(d) reframe(d, s = sum(i * 2L), r = range(i), .by = g))
})

test_that("shapes off the allowlist give dplyr's result through the fallback", {
  skip_if_no_gpu()
  df <- f22r4_df()
  fb <- function(data, pipeline) f22r4_check(data, pipeline, native = FALSE)
  # a GPU NaN (0 / 0) inside comparisons, is.na() and !
  fb(df, function(d) {
    reframe(d, m = sum(x / y > 0), n = sum(is.na(x / y)), r = range(i, na.rm = TRUE), .by = g)
  })
  fb(df, function(d) {
    d |> mutate(z = x / y) |> reframe(m = sum(z > 0), q = quantile(i, c(0, 1), na.rm = TRUE), .by = g)
  })
  # integer median() keeps its type when every group is odd-sized
  odd <- data.frame(g = c(1, 1, 1, 2), i = c(4L, 1L, 9L, 6L), b = c(TRUE, FALSE, TRUE, FALSE))
  fb(odd, function(d) reframe(d, m = median(i), mb = median(b), r = range(i), .by = g))
  # window functions inside an argument
  fb(df, function(d) reframe(group_by(d, g), q = quantile(cumsum(y), c(0, 1)), m = max(x - mean(y))))
  fb(df, function(d) reframe(d, r = range(dplyr::lag(i), na.rm = TRUE), .by = g))
  # math the GPU gets wrong (R: NA ^ 0 is 1; sqrt(-3) warns)
  fb(df, function(d) reframe(d, p = sum(x ^ 0), q = range(sqrt(x), na.rm = TRUE), .by = g))
})
