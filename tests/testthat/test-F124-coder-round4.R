# F124 attempt 2, round 2: what decides whether a min()/max() verb asks the data
# anything (minmax_dots_type_fixed()), and the quiet, pruned facts query.
# Helpers: helper-parity-F124.R, helper-parity-F85.R (pq_write()).

test_that("F124: a min()/max() of a double, string or timestamp column is type-fixed", {
  schema <- list(names = c("a", "b", "s", "d", "l"),
                 types = c("INT32", "FLOAT64", "STRING", "TIMESTAMP_DAYS", "BOOL8"))
  fixed <- function(...) minmax_dots_type_fixed(rlang::enquos(...), schema)
  expect_true(fixed(m = max(b)))
  expect_true(fixed(m = min(s), n = max(d)))
  expect_true(fixed(r = runif(1), m = max(b)))
  expect_true(fixed(x = a + 1L))
  expect_true(fixed(m = base::max(b)))
  # an integer or logical argument, an expression, a flag, a second argument
  expect_false(fixed(m = max(a)))
  expect_false(fixed(m = max(l)))
  expect_false(fixed(m = max(b + 1)))
  expect_false(fixed(m = max(b, na.rm = TRUE)))
  expect_false(fixed(m = max(b, a)))
  # a name no column has (an earlier dot's, a variable), or one a dot makes
  expect_false(fixed(m = max(zz)))
  expect_false(fixed(b = as.integer(b), m = max(b)))
  # inside another call
  expect_true(fixed(m = max(b) - b))
  expect_false(fixed(m = max(a) - a))
})

test_that("F124: the facts query of a parquet plan reads no unread column and warns nowhere", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, .Machine$integer.max, 5L, 7L), b = c(0.9, 0.1, 0.8, 0.7),
                       k = 1:4, v1 = c(1, 2, 3, 4))
  path <- pq_write(df)
  parquet_last_read(reset = TRUE)
  ws <- character()
  q <- withCallingHandlers(
    mutate(filter(mutate(read_parquet_gpu(path, lazy = TRUE), z = a + 1L, w = v1 * 2), b > 0.5),
           m = max(k)),
    warning = function(w) {
      ws <<- c(ws, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  expect_length(ws, 0L)
  expect_setequal(parquet_last_read()$columns, "b")
  # the option is restored: the next plan keeps its unread mutates (F99)
  expect_null(cuplyr:::.optimizer_opts$prune_unread)
  col <- f124_capture(collect(q))
  expect_length(col$warnings, 1L)
  expect_identical(col$value$m, rep(4L, 3L))
})

test_that("F124: a double min()/max() next to runif() on a parquet plan evaluates runif() once", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:6, b = c(0.9, 0.1, 0.8, 0.7, 0.2, 0.6))
  path <- pq_write(df)
  calls <- 0L
  draw <- function() {
    calls <<- calls + 1L
    0.5
  }
  q <- mutate(filter(read_parquet_gpu(path, lazy = TRUE), b > 0.5), r = draw(), m = max(b))
  expect_identical(calls, 1L)
  expect_identical(collect(q)$r, rep(0.5, 4L))
  expect_identical(calls, 1L)
})
