# Extra cases for #13 (round 4): the summarise() check must not continue past a
# dot whose value dplyr rejects or drops, and arrange()'s bare key of a missing
# function argument.

r4_err <- function(x, pipeline) {
  tryCatch(
    suppressWarnings({
      out <- pipeline(x)
      if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
      NULL
    }),
    error = function(e) e
  )
}

# The same class vector, parent class, call and full message as dplyr, eager and lazy.
r4_same_error <- function(df, pipeline, gpu_opts = list()) {
  cpu <- r4_err(tibble::as_tibble(df), pipeline)
  expect_true(inherits(cpu, "error"))
  for (lazy in c(FALSE, TRUE)) {
    gpu <- withr::with_options(gpu_opts, r4_err(tbl_gpu(df, lazy = lazy), pipeline))
    expect_true(inherits(gpu, "error"))
    expect_identical(class(gpu), class(cpu))
    expect_identical(class(gpu$parent), class(cpu$parent))
    expect_identical(conditionMessage(gpu), conditionMessage(cpu))
    expect_identical(rlang::format_error_call(gpu$call), rlang::format_error_call(cpu$call))
  }
}

r4_df <- function() {
  tibble::tibble(x = c(2L, 1L, 3L), y = c(1.5, NA, 3), g = c("a", "a", "b"))
}

test_that("a dot whose value is no vector stops the summarise() check (dplyr rejects it first)", {
  skip_if_no_gpu()
  df <- r4_df()
  f <- function(v) v
  r4_same_error(df, function(d) summarise(d, a = f, b = sum(xx)))
  r4_same_error(df, function(d) summarise(d, a = mean, b = sum(xx)))
})

test_that("a NULL dot creates no column the summarise() check could see", {
  skip_if_no_gpu()
  df <- r4_df()
  v <- NULL
  r4_same_error(df, function(d) summarise(d, a = v, b = dplyr::first(a), c = sum(xx)))
  r4_same_error(df, function(d) summarise(d, a = v, b = n_distinct(a), c = sum(xx)))
  # (dplyr skips it, so the missing column after it is still the first error)
  r4_same_error(df, function(d) summarise(d, a = NULL, c = sum(xx)))
  r4_same_error(df, function(d) summarise(d, a = NULL, c = sum(xx)), gpu_opts = list(cuplyr.fallback = "error"))
})

test_that("arrange() by a bare missing function argument is dplyr's error", {
  skip_if_no_gpu()
  df <- r4_df()
  f <- function(d, w) arrange(d, w)
  g <- function(d, w) arrange(d, desc(w))
  r4_same_error(df, function(d) f(d))
  r4_same_error(df, function(d) g(d))
  r4_same_error(df, function(d) f(d), gpu_opts = list(cuplyr.fallback = "error"))
})

test_that("a call of a function that is not there is dplyr's error, with no CPU fallback", {
  skip_if_no_gpu()
  df <- r4_df()
  strict <- list(cuplyr.fallback = "error")
  # (a dot that reads only columns is not checked, so a typo in a function name
  # there goes the CPU route: dplyr's error, but not natively)
  r4_same_error(df, function(d) summarise(d, a = mean(y, na.rm = TRUE), m = sum(no_such_fn(x))))
  pipelines <- list(
    function(d) summarise(d, m = no_such_fn(xx)),
    function(d) summarise(d, a = mean(y, na.rm = TRUE), m = sum(no_such_fn(x) + k)),
    function(d) arrange(d, no_such_fn(x))
  )
  k <- 1
  for (p in pipelines) {
    r4_same_error(df, p)
    r4_same_error(df, p, gpu_opts = strict)
    for (lazy in c(FALSE, TRUE)) {
      fallbacks <- 0L
      withCallingHandlers(
        tryCatch(dplyr::collect(p(tbl_gpu(df, lazy = lazy))), error = function(e) NULL),
        cuplyr_cpu_fallback = function(c) fallbacks <<- fallbacks + 1L
      )
      expect_identical(fallbacks, 0L)
    }
  }
})

test_that("a missing column after first() is reported for the first() dplyr resolves", {
  skip_if_no_gpu()
  df <- r4_df()
  # (dplyr attached or not: dplyr::first() resolves either way)
  r4_same_error(df, function(d) summarise(d, c = dplyr::first(x), l = dplyr::last(g), d = sum(xx)),
                gpu_opts = list(cuplyr.fallback = "error"))
})

test_that("summarise() dots that cannot fail still compute dplyr's result", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(2L, 1L, 3L), y = c(1.5, NA, 3), b = c(TRUE, FALSE, NA), g = c("a", "a", "b"))
  k <- 2
  expect_same_as_dplyr(df, function(d) {
    summarise(d, a = mean(y, na.rm = TRUE), n = n_distinct(g), f = dplyr::first(g), l = dplyr::last(y),
              v = var(b, na.rm = TRUE), m = median(y, na.rm = TRUE), s = sd(x, na.rm = TRUE) + k)
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    summarise(group_by(d, g), a = mean(y, na.rm = TRUE), n = n_distinct(x, na.rm = TRUE), f = dplyr::first(x), s = sum(a))
  })
})
