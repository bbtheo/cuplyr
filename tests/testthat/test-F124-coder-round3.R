# F124 round 3: the call head, Date shifts by integer min()/max() amounts, the
# re-parse after a plan break (reviews/round-2.md B5's mechanism, B8, N16, N20).
# Helper: helper-parity-F124.R.

test_that("F124: the verb head is read from the frame UseMethod() dispatched", {
  f <- function(x, ...) UseMethod("f")
  helper <- function() minmax_verb_head("f", sys.parent())
  f.default <- function(x, ...) helper()
  # (a helper that is not called by the method: looked for down the stack)
  deeper <- function() minmax_verb_head("f", sys.parent())
  f.character <- function(x, ...) (function() deeper())()
  alias <- f
  expect_identical(f(1), quote(f))
  expect_identical(alias(1), quote(alias))
  expect_identical(f("a"), quote(f))
  expect_identical(alias("a"), quote(alias))
  # (the method called directly: no generic call, the default name)
  expect_identical(f.default(1), quote(f))
})

test_that("F124: only the cast min()/max() inserts is reverted in a Date shift", {
  schema <- list(names = c("a", "f"), types = c("INT32", "DICTIONARY32"))
  marked <- ir_call("max", list(ir_call("cast_float64", list(ir_col("a")), meta = list(minmax_cast = TRUE))))
  expect_identical(ir_minmax_int_shift(marked, schema), ir_call("max", list(ir_col("a"))))
  amount <- ir_call("*", list(marked, ir_lit(2L, "INT32")))
  expect_identical(ir_minmax_int_shift(amount, schema),
                   ir_call("*", list(ir_call("max", list(ir_col("a"))), ir_lit(2L, "INT32"))))
  # a user's as.numeric() is a double: not reverted
  user <- ir_call("max", list(ir_call("cast_float64", list(ir_col("a")))))
  expect_null(ir_minmax_int_shift(user, schema))
  # an amount that is a double without the cast too stays a double
  halved <- ir_call("/", list(marked, ir_lit(2L, "INT32")))
  expect_null(ir_minmax_int_shift(halved, schema))
})

test_that("F124: a Date shifted by an integer amount of min()/max() calls stays on the GPU", {
  skip_if_no_gpu()
  dd <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), a = 1:3,
                       b = c(TRUE, FALSE, TRUE), d = as.Date("2020-01-01") + 0:2)
  f124_check(dd, function(d) mutate(group_by(d, f, .drop = FALSE), e = d + max(a) * 2L))
  f124_check(dd, function(d) mutate(group_by(d, f, .drop = FALSE), e = d - (min(b) + 1L)))
  # unknown rows: the shift's calls decide no type, so the plan carries the warning
  f124_check(dd, function(d) mutate(filter(d, a > 10L), e = d + max(a) * 2L), lazy_warns_at = "collect")
})

test_that("F124: a lazy re-parse that leaves the GPU takes the CPU route and stays lazy", {
  skip_if_no_gpu()
  dd <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), a = 1:3)
  pipe <- function(d) mutate(mutate(group_by(d, f, .drop = FALSE), z = a + 1L), s = as.character(max(a)))
  expected <- suppressWarnings(pipe(dd))
  got <- f124_capture(pipe(tbl_gpu(dd, lazy = TRUE)))
  expect_identical(unclass(got$value)$exec_mode, "lazy")
  expect_length(got$warnings, 1L)
  expect_equal(tibble::as_tibble(collect(got$value)), tibble::as_tibble(expected))
})

test_that("F124: the lazy re-parse warns once for a folded call that warns", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3)
  coercion <- function(got) {
    sum(grepl("NAs introduced by coercion", f124_msgs(got$warnings), fixed = TRUE))
  }
  for (lazy in c(FALSE, TRUE)) {
    got <- f124_capture(collect(mutate(filter(tbl_gpu(df, lazy = lazy), a > 10L), m = max(a + as.integer("x")))))
    expect_identical(coercion(got), 1L, info = if (lazy) "lazy" else "eager")
    expect_type(got$value$m, "double")
  }
})

test_that("F124: a column called max next to an impure call runs no pending step", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3, max = c(1, 2, 3))
  got <- f124_capture(mutate(filter(tbl_gpu(df, lazy = TRUE), a > 10L), z = max + runif(1)))
  expect_length(got$plan_break, 0L)
  expect_identical(unclass(got$value)$exec_mode, "lazy")
})
