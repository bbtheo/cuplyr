# F77: a user function that shadows the name of a native op is the function
# dplyr calls. dplyr evaluates every dot in a data mask whose parent is the
# quosure's environment, so R's ordinary lookup finds the user's `abs()` (a
# local, an enclosing function's, a global) before `base::abs()`. cuplyr must
# not map such a call to its built-in op: the verb gives dplyr's result (the
# CPU route). `pkg::fn` is resolved as written. Controls: base functions, a
# qualified call next to a shadow, a non-function variable with an op's name,
# an identical alias, and a shadow outside the quosure's scope all stay native.

f77_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L),
    x = c(-1.5, 2, NA, 4, -0.25),
    i = c(3L, -2L, 5L, 1L, NA),
    s = c("ab", "c", NA, "dddd", "Ee"),
    d = as.Date("2020-01-01") + c(0L, 400L, 800L, 1200L, 1600L)
  )
}

# eager and lazy against dplyr
f77_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# eager and lazy against dplyr, with every CPU fallback an error: the pipeline
# runs natively
f77_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# --- (a) a shadow in the test's own environment ------------------------------

test_that("F77: math shadows in mutate() are the user's functions", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  sqrt <- function(x) x + 1000
  round <- function(x, digits = 0) x * 3
  f77_both(df, function(d) mutate(d, y = abs(x)))
  f77_both(df, function(d) mutate(d, y = abs(i)))
  f77_both(df, function(d) mutate(d, y = sqrt(x) + 1))
  f77_both(df, function(d) mutate(d, y = round(x)))
  f77_both(df, function(d) transmute(d, y = abs(x) * 2, g))
})

test_that("F77: string shadows in mutate() are the user's functions", {
  skip_if_no_gpu()
  df <- f77_df()
  nchar <- function(x, ...) 99L
  toupper <- function(x) paste0("U", x)
  f77_both(df, function(d) mutate(d, y = nchar(s)))
  f77_both(df, function(d) mutate(d, y = toupper(s)))
  f77_both(df, function(d) mutate(d, y = paste0(toupper(s), "-")))
})

test_that("F77: window shadows in mutate() are the user's functions", {
  skip_if_no_gpu()
  df <- f77_df()
  lag <- function(x, ...) x * 7
  cumsum <- function(x) rev(x)
  mean <- function(x, ...) -1
  f77_both(df, function(d) mutate(d, y = lag(x)))
  f77_both(df, function(d) mutate(d, y = cumsum(i)))
  f77_both(df, function(d) mutate(d, y = x - mean(x)))
  f77_both(df, function(d) mutate(group_by(d, g), y = mean(x)))
  f77_both(df, function(d) mutate(d, y = mean(x), .by = g))
  f77_both(df, function(d) mutate(group_by(d, g), y = lag(x)))
})

test_that("F77: if_else()/case_when() shadows, and a shadow inside them", {
  skip_if_no_gpu()
  df <- f77_df()
  if_else <- function(condition, true, false, ...) 55
  case_when <- function(...) 7
  abs <- function(x) -x * 10
  f77_both(df, function(d) mutate(d, y = if_else(x > 0, 1, 2)))
  f77_both(df, function(d) mutate(d, y = case_when(x > 0 ~ 1, .default = 2)))
  f77_both(df, function(d) mutate(d, y = dplyr::case_when(abs(x) > 5 ~ "big", .default = "small")))
  f77_both(df, function(d) mutate(d, y = dplyr::if_else(abs(i) > 0, "pos", "neg")))
})

test_that("F77: shadows in filter() and filter_out() predicates", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  is.na <- function(x) rep(TRUE, length(x))
  mean <- function(x, ...) 0
  f77_both(df, function(d) filter(d, abs(x) > 3))
  f77_both(df, function(d) filter(d, !is.na(x)))
  f77_both(df, function(d) filter(d, abs(i) > 0, x < 10))
  f77_both(df, function(d) filter(group_by(d, g), x > mean(x)))
  f77_both(df, function(d) filter(d, x > mean(x), .by = g))
  f77_both(df, function(d) filter_out(d, abs(i) > 10))
})

test_that("F77: aggregate shadows in summarise()", {
  skip_if_no_gpu()
  df <- f77_df()
  mean <- function(x, ...) -1
  sum <- function(x, ...) 42
  n_distinct <- function(...) 123L
  n <- function() 99L
  f77_both(df, function(d) summarise(d, m = mean(x)))
  f77_both(df, function(d) summarise(d, m = sum(i)))
  f77_both(df, function(d) summarise(d, m = n_distinct(g)))
  f77_both(df, function(d) summarise(d, k = n()))
  f77_both(df, function(d) summarise(group_by(d, g), m = sum(i), k = n()))
  f77_both(df, function(d) summarise(d, m = mean(x), .by = g), arrange_by = "g")
  f77_both(df, function(d) summarise(group_by(d, g), m = base::max(x, na.rm = TRUE), s = sum(x)))
})

test_that("F77: a shadow inside an aggregate's argument", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  f77_both(df, function(d) summarise(d, m = base::sum(abs(i), na.rm = TRUE)))
  f77_both(df, function(d) summarise(group_by(d, g), m = base::mean(abs(x))))
  f77_both(df, function(d) summarise(d, m = base::max(abs(x), na.rm = TRUE), n = dplyr::n()))
})

test_that("F77: shadows in arrange(), slice_min()/slice_max() and top_n()", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  f77_both(df, function(d) arrange(d, abs(i)))
  f77_both(df, function(d) arrange(d, desc(abs(x))))
  f77_both(df, function(d) slice_min(d, abs(i), n = 2))
  f77_both(df, function(d) slice_max(d, abs(x), n = 2))
  f77_both(df, function(d) dplyr::top_n(d, 2, abs(i)))
})

test_that("F77: shadows in group_by(), count(), tally(), add_count() and distinct()", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  f77_both(df, function(d) summarise(group_by(d, a = abs(i)), k = dplyr::n()), arrange_by = "a")
  f77_both(df, function(d) count(d, a = abs(i)), arrange_by = "a")
  f77_both(df, function(d) count(d, g, wt = abs(i)))
  f77_both(df, function(d) tally(d, wt = abs(i)))
  f77_both(df, function(d) add_count(d, g, wt = abs(i)))
  f77_both(df, function(d) distinct(d, a = abs(i)), arrange_by = "a")
})

test_that("F77: shadows in across()/if_any()/if_all() lambdas and .fns", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  mean <- function(x, ...) -1
  f77_both(df, function(d) mutate(d, across(c(x, i), ~ abs(.x))))
  f77_both(df, function(d) mutate(d, across(c(x, i), function(v) abs(v) + 1)))
  f77_both(df, function(d) summarise(d, across(c(x, i), ~ mean(.x))))
  f77_both(df, function(d) summarise(group_by(d, g), across(c(x, i), ~ mean(.x, na.rm = TRUE))))
  f77_both(df, function(d) filter(d, if_any(c(x, i), ~ abs(.x) > 10)))
  f77_both(df, function(d) filter(d, if_all(c(x, i), ~ abs(.x) < 0)))
  f77_both(df, function(d) mutate(d, across(c(x, i), abs)))
  f77_both(df, function(d) summarise(d, across(c(x, i), list(m = mean))))
})

test_that("F77: rowwise verbs use the user's shadow", {
  skip_if_no_gpu()
  df <- f77_df()
  max <- function(...) -1
  abs <- function(x) -x * 10
  f77_both(df, function(d) mutate(rowwise(d), m = max(x, i)))
  f77_both(df, function(d) mutate(rowwise(d), m = abs(x)))
})

test_that("F77: reframe() uses the user's shadow", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  f77_both(df, function(d) reframe(d, a = abs(i)))
})

test_that("F77: an operator shadow is the user's operator", {
  skip_if_no_gpu()
  df <- f77_df()
  pipe <- function(d) {
    `+` <- function(e1, e2) base::`-`(e1, e2)
    mutate(d, y = x + i)
  }
  f77_both(df, pipe)
  pipe2 <- function(d) {
    `>` <- function(e1, e2) base::`<`(e1, e2)
    filter(d, x > 0)
  }
  f77_both(df, pipe2)
})

test_that("F77: a shadow in a column-free call next to a column", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  mean <- function(x, ...) 100
  f77_both(df, function(d) mutate(d, y = x + abs(-2)))
  f77_both(df, function(d) filter(d, x < mean(c(1, 3))))
  f77_both(df, function(d) mutate(group_by(d, g), y = x + abs(-2)))
  f77_both(df, function(d) summarise(d, m = base::sum(x + abs(1), na.rm = TRUE)))
})

# --- (b) a shadow in an enclosing function ------------------------------------

test_that("F77: a shadow defined by the function that calls the verb", {
  skip_if_no_gpu()
  df <- f77_df()
  wrapper <- function(d) {
    abs <- function(x) -x * 10
    mean <- function(x, ...) -1
    d <- mutate(d, y = abs(x))
    summarise(group_by(d, g), m = mean(y))
  }
  f77_both(df, wrapper)
})

test_that("F77: a shadow in a closure's enclosing environment", {
  skip_if_no_gpu()
  df <- f77_df()
  make_pipe <- function() {
    nchar <- function(x, ...) as.double(base::nchar(x)) + 0.5
    function(d) filter(mutate(d, n = nchar(s)), nchar(s) > 2)
  }
  f77_both(df, make_pipe())
  make_pipe2 <- function() {
    sum <- function(x, ...) 42
    inner <- function() function(d) summarise(group_by(d, g), s = sum(i))
    inner()
  }
  f77_both(df, make_pipe2())
})

# --- (c) a global shadow ------------------------------------------------------

test_that("F77: a global shadow is the user's function", {
  skip_if_no_gpu()
  df <- f77_df()
  ge <- globalenv()
  assign("abs", function(x) -x * 10, envir = ge)
  assign("mean", function(x, ...) -1, envir = ge)
  assign("nchar", function(x, ...) 99L, envir = ge)
  withr::defer(rm(list = c("abs", "mean", "nchar"), envir = ge))
  # the pipelines' environment is the global one, as for code typed at the console
  p1 <- function(d) dplyr::mutate(d, y = abs(x))
  p2 <- function(d) dplyr::summarise(dplyr::group_by(d, g), m = mean(x))
  p3 <- function(d) dplyr::filter(d, nchar(s) > 50)
  environment(p1) <- ge
  environment(p2) <- ge
  environment(p3) <- ge
  f77_both(df, p1)
  f77_both(df, p2)
  f77_both(df, p3)
})

# --- qualified calls ----------------------------------------------------------

test_that("F77: base::/dplyr::-qualified calls next to a shadow stay native", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  nchar <- function(x, ...) 99L
  lag <- function(x, ...) x * 7
  if_else <- function(condition, true, false, ...) 55
  mean <- function(x, ...) -1
  n_distinct <- function(...) 123L
  f77_native(df, function(d) mutate(d, y = base::abs(x), z = base::nchar(s)))
  f77_native(df, function(d) mutate(d, y = dplyr::lag(x)))
  f77_native(df, function(d) mutate(d, y = dplyr::if_else(x > 0, 1, 2)))
  f77_native(df, function(d) filter(d, base::abs(x) > 1))
  f77_native(df, function(d) summarise(group_by(d, g), m = base::mean(x), k = dplyr::n_distinct(i)))
})

test_that("F77: pkg::fn naming another package's function is that function", {
  skip_if_no_gpu()
  df <- f77_df()
  # stats::lag() is not dplyr::lag(): on a plain vector it returns it unchanged
  # (as.numeric(): drop the `tsp` attribute stats::lag() sets, which a GPU column cannot hold)
  f77_both(df, function(d) mutate(d, y = as.numeric(stats::lag(x))))
  f77_both(df, function(d) mutate(group_by(d, g), y = stats::lag(i)))
})

test_that("F77: data.table::second() is not lubridate::second()", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table")
  skip_if_not_installed("lubridate")
  df <- tibble::tibble(t = as.POSIXct("2020-01-01 10:20:30.5", tz = "UTC") + c(0, 3600, 1))
  f77_both(df, function(d) mutate(d, a = data.table::second(t), b = lubridate::second(t)))
})

# --- controls: dplyr recognises these by name ---------------------------------

test_that("F77: names dplyr itself resolves ignore a shadow", {
  skip_if_no_gpu()
  df <- f77_df()
  desc <- function(x) x * 0
  sum <- function(x, ...) 42
  n <- function() 99L
  # arrange() strips a top-level desc() by name
  f77_both(df, function(d) arrange(d, desc(i)))
  # count()/tally()/add_count() build base::sum() / dplyr::n() themselves
  f77_both(df, function(d) count(d, g))
  f77_both(df, function(d) count(d, g, wt = i))
  f77_both(df, function(d) tally(group_by(d, g), wt = x))
  f77_both(df, function(d) add_count(d, g, wt = i))
  f77_both(df, function(d) add_count(d, g))
})

# --- controls: no shadow in scope ---------------------------------------------

test_that("F77: base functions without a shadow stay native", {
  skip_if_no_gpu()
  df <- f77_df()
  f77_native(df, function(d) mutate(d, y = abs(x), z = sqrt(abs(x)), w = nchar(s)))
  # (dplyr is not attached under tools/test-file: a bare lag() is stats::lag())
  f77_native(df, function(d) mutate(d, y = dplyr::lag(x), z = cumsum(i)))
  f77_native(df, function(d) mutate(d, y = dplyr::if_else(x > 0, 1, 2), z = dplyr::case_when(i > 2 ~ "a", .default = "b")))
  f77_native(df, function(d) filter(d, abs(x) > 1, !is.na(i)))
  f77_native(df, function(d) summarise(d, m = mean(x, na.rm = TRUE), s = sum(i, na.rm = TRUE), k = n(), u = n_distinct(g)))
  f77_native(df, function(d) summarise(group_by(d, g), m = mean(x), k = n()))
  f77_native(df, function(d) mutate(group_by(d, g), y = x - mean(x)))
  f77_native(df, function(d) arrange(d, desc(x), i))
  f77_native(df, function(d) mutate(d, across(c(x, i), ~ abs(.x))))
  f77_native(df, function(d) count(d, g, wt = i))
})

test_that("F77: a non-function variable with an op's name is not a shadow", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- 5
  mean <- "not a function"
  nchar <- list(1)
  f77_native(df, function(d) mutate(d, y = abs(x) + abs, z = nchar(s)))
  f77_native(df, function(d) summarise(group_by(d, g), m = mean(x)))
  # a column with an op's name does not hide the function either
  df2 <- dplyr::mutate(df, abs = 7)
  f77_native(df2, function(d) mutate(d, y = abs(x) + abs))
})

test_that("F77: an identical alias of the base function stays native", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- base::abs
  mean <- base::mean
  lag <- dplyr::lag
  f77_native(df, function(d) mutate(d, y = abs(x), z = lag(i)))
  f77_native(df, function(d) summarise(group_by(d, g), m = mean(x)))
})

test_that("F77: a shadow outside the quosure's scope is not seen", {
  skip_if_no_gpu()
  df <- f77_df()
  other <- function() {
    abs <- function(x) -x * 10
    abs
  }
  shadow <- other()
  expect_false(identical(shadow, base::abs))
  f77_native(df, function(d) mutate(d, y = abs(x)))
})

test_that("F77: the check is per call: a shadow does not leak into later calls", {
  skip_if_no_gpu()
  df <- f77_df()
  with_shadow <- function(d) {
    abs <- function(x) -x * 10
    mutate(d, y = abs(x))
  }
  plain <- function(d) mutate(d, y = abs(x))
  # native, then shadowed, then native again
  f77_native(df, plain)
  f77_both(df, with_shadow)
  f77_native(df, plain)
  f77_both(df, with_shadow)
})

# --- edge cases ---------------------------------------------------------------

test_that("F77: a shadow on zero-row input", {
  skip_if_no_gpu()
  df <- f77_df()[0, ]
  abs <- function(x) -x * 10
  mean <- function(x, ...) -1
  nchar <- function(x, ...) as.double(base::nchar(x))
  f77_both(df, function(d) mutate(d, y = abs(x), z = nchar(s)))
  f77_both(df, function(d) summarise(d, m = mean(x)))
  f77_both(df, function(d) filter(d, abs(x) > 1))
})

test_that("F77: a shadow's result type is the user's", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) as.character(x)
  nchar <- function(x, ...) base::nchar(x) > 1
  round <- function(x, ...) as.integer(base::round(x))
  f77_both(df, function(d) mutate(d, y = abs(i), z = nchar(s), w = round(x)))
})

test_that("F77: a shadow's error and warning are raised", {
  skip_if_no_gpu()
  df <- f77_df()
  pipe_err <- function(d) {
    abs <- function(x) stop("boom from user abs")
    collect(mutate(d, y = abs(x)))
  }
  expect_error(pipe_err(df), "boom from user abs")
  expect_error(pipe_err(tbl_gpu(df)), "boom from user abs")
  expect_error(pipe_err(as_lazy(tbl_gpu(df))), "boom from user abs")

  pipe_warn <- function(d) {
    mean <- function(x, ...) {
      warning("user mean warns")
      -1
    }
    collect(summarise(d, m = mean(x)))
  }
  expect_warning(pipe_warn(df), "user mean warns")
  expect_warning(pipe_warn(tbl_gpu(df)), "user mean warns")
  expect_warning(pipe_warn(as_lazy(tbl_gpu(df))), "user mean warns")
})

test_that("F77: a shadow runs as often as dplyr runs it", {
  skip_if_no_gpu()
  df <- f77_df()
  counts <- integer()
  run <- function(d, verb) {
    k <- 0L
    abs <- function(x) {
      k <<- k + 1L
      -x * 10
    }
    out <- collect(verb(d, abs))
    counts <<- c(counts, k)
    out
  }
  verbs <- list(
    function(d, abs) mutate(d, y = abs(x)),
    function(d, abs) mutate(group_by(d, g), y = abs(x)),
    function(d, abs) filter(d, abs(x) > 1),
    function(d, abs) summarise(group_by(d, g), m = base::sum(abs(x)))
  )
  for (verb in verbs) {
    for (lazy in c(FALSE, TRUE)) {
      counts <- integer()
      pipeline <- function(d) run(d, verb)
      if (lazy) expect_same_as_dplyr_lazy(df, pipeline) else expect_same_as_dplyr(df, pipeline)
      expect_length(counts, 2L)
      expect_identical(counts[[2]], counts[[1]],
                       info = sprintf("abs() calls: dplyr %d, GPU %d (lazy = %s)", counts[[1]], counts[[2]], lazy))
    }
  }
})

# --- round 1 review: .fns values, with_order(), promises, failed pkg::fn ------

test_that("F77: a .fns closure or formula value calls its own environment's shadow", {
  skip_if_no_gpu()
  df <- f77_df()
  make_f <- function() {
    abs <- function(x) -x * 10
    function(v) abs(v)
  }
  f <- make_f()
  make_lam <- function() {
    abs <- function(x) -x * 10
    ~ abs(.x)
  }
  lam <- make_lam()
  make_m <- function() {
    mean <- function(x, ...) -1
    function(v) mean(v, na.rm = TRUE)
  }
  m <- make_m()
  make_p <- function() {
    abs <- function(x) -x * 10
    function(v) abs(v) > 10
  }
  p <- make_p()
  make_q <- function() {
    abs <- function(x) -x * 10
    function(v) abs(v) < 0
  }
  q <- make_q()
  f77_both(df, function(d) mutate(d, across(c(x, i), f)))
  f77_both(df, function(d) mutate(d, across(c(x, i), lam)))
  f77_both(df, function(d) mutate(d, across(c(x, i), !!f)))
  f77_both(df, function(d) mutate(d, across(c(x, i), list(a = f, b = lam))))
  f77_both(df, function(d) summarise(group_by(d, g), across(c(x, i), m)))
  f77_both(df, function(d) filter(d, if_any(c(x, i), p)))
  f77_both(df, function(d) filter(group_by(d, g), if_all(c(x, i), q)))
})

test_that("F77: a .fns closure value with no shadow stays native", {
  skip_if_no_gpu()
  df <- f77_df()
  f <- function(v) abs(v) + 1
  lam <- ~ sqrt(abs(.x))
  f77_native(df, function(d) mutate(d, across(c(x, i), f)))
  f77_native(df, function(d) mutate(d, across(c(x, i), lam)))
})

test_that("F77: with_order()'s fun is the user's function", {
  skip_if_no_gpu()
  df <- tibble::tibble(o = c(3L, 1L, 2L, 5L, 4L), x = c(1, 2, 3, 4, 5))
  cumsum <- function(x) x * 100
  f77_both(df, function(d) mutate(d, y = dplyr::with_order(o, cumsum, x)))
  f77_both(df, function(d) mutate(d, y = dplyr::order_by(o, cumsum(x))))
})

test_that("F77: with_order() over a base function stays native", {
  skip_if_no_gpu()
  df <- tibble::tibble(o = c(3L, 1L, 2L, 5L, 4L), x = c(1, 2, 3, 4, 5))
  f77_native(df, function(d) mutate(d, y = dplyr::with_order(o, cumsum, x)))
})

test_that("F77: the check forces no promise dplyr leaves alone", {
  skip_if_no_gpu()
  df <- f77_df()
  # the column `x` masks the argument: dplyr never evaluates its default
  f <- function(d, x = stop("promise x forced")) mutate(d, y = x + 1)
  f77_both(df, function(d) f(d))
  forced <- 0L
  h <- function(d, i = {
    forced <<- forced + 1L
    1L
  }) filter(d, i > 0)
  out <- collect(h(tbl_gpu(df)))
  expect_identical(forced, 0L)
  expect_equal(tibble::as_tibble(out), h(df))
  # an argument named like a column, given an undefined symbol
  s <- function(d, var) summarise(d, m = mean(var))
  df2 <- tibble::tibble(var = c(1, 2, 3))
  f77_native(df2, function(d) s(d, not_defined_anywhere))
})

test_that("F77: pkg::fn that pkg does not export is dplyr's error", {
  skip_if_no_gpu()
  df <- f77_df()
  pipe <- function(d) collect(mutate(d, y = dplyr::abs(x)))
  expect_error(pipe(df), "not an exported object")
  expect_error(pipe(tbl_gpu(df)), "not an exported object")
  expect_error(pipe(as_lazy(tbl_gpu(df))), "not an exported object")
})

test_that("F77: shadows in rowwise filter/summarise/transmute, slice() and after lazy ops", {
  skip_if_no_gpu()
  df <- f77_df()
  abs <- function(x) -x * 10
  mean <- function(x, ...) -1
  f77_both(df, function(d) filter(rowwise(d), abs(x) > 10))
  f77_both(df, function(d) summarise(rowwise(d), m = mean(c(x, i))))
  f77_both(df, function(d) transmute(rowwise(d), m = abs(i)))
  f77_both(df, function(d) slice(d, which(abs(x) > 10)))
  f77_both(df, function(d) filter(mutate(d, z = x * 2), abs(z) > 5))
  f77_both(df, function(d) summarise(group_by(mutate(d, z = i + 1L), g), m = mean(z)))
})

# --- round 2 review: function values and scoped verbs ------------------------

test_that("F77: a global shadow passed as a function value is the user's function", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, NA, 10, 2))
  ge <- globalenv()
  assign("median", function(x, ...) stats::median(x, na.rm = TRUE), envir = ge)
  assign("sd", function(x, ...) -1, envir = ge)
  assign("f77_fns", list(med = get("median", envir = ge), s = get("sd", envir = ge)), envir = ge)
  withr::defer(rm(list = c("median", "sd", "f77_fns"), envir = ge))
  pipes <- list(
    function(d) dplyr::summarise(d, dplyr::across(x, f77_fns)),
    function(d) dplyr::summarise(d, dplyr::across(x, !!median)),
    function(d) dplyr::summarise(dplyr::group_by(d, g), dplyr::across(x, f77_fns)),
    function(d) dplyr::summarise_at(d, "x", median),
    function(d) dplyr::summarise_at(dplyr::group_by(d, g), "x", sd),
    function(d) dplyr::mutate_at(d, "x", median),
    function(d) dplyr::summarise(d, dplyr::across(x, list(m = median)))
  )
  for (p in pipes) {
    environment(p) <- ge
    f77_both(df, p)
  }
})

test_that("F77: a .fns list value of canonical functions stays native", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, NA, 10, 2))
  fns <- list(m = base::mean, s = stats::sd)
  f77_native(df, function(d) summarise(group_by(d, g), across(x, fns)))
  f77_native(df, function(d) summarise(d, across(x, !!base::mean)))
})

test_that("F77: scoped verbs' lambdas use the user's shadow", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(-1.5, 2, 4), y = c(3, -2, 5))
  abs <- function(x) -x * 10
  mean <- function(x, ...) -1
  f77_both(df, function(d) dplyr::mutate_at(d, "x", ~ abs(.)))
  f77_both(df, function(d) dplyr::mutate_all(d, ~ abs(.x)))
  f77_both(df, function(d) dplyr::filter_at(d, "x", ~ abs(.x) > 10))
  f77_both(df, function(d) dplyr::filter_all(d, dplyr::any_vars(abs(.) > 10)))
  f77_both(df, function(d) dplyr::summarise_all(dplyr::select(d, x, y), ~ mean(.)))
  f77_both(df, function(d) dplyr::summarise_at(dplyr::group_by(d, g), c("x", "y"), ~ mean(.x)))
  f77_both(df, function(d) dplyr::mutate_if(d, is.double, ~ abs(.)))
  f77_both(df, function(d) dplyr::summarise_at(d, "x", function(v) mean(v)))
})

test_that("F77: scoped verbs' lambdas without a shadow stay native", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(-1.5, 2, 4), y = c(3, -2, 5))
  f77_native(df, function(d) dplyr::mutate_at(d, "x", ~ abs(.)))
  f77_native(df, function(d) dplyr::summarise_at(dplyr::group_by(d, g), c("x", "y"), ~ mean(.x)))
})
