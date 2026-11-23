# F124 (attempt 2): cases from the round-1 review. On a lazy parquet plan the
# question "can a group be empty?" may be answered by a small query at the verb;
# that query must be invisible: the warnings of the pending steps fire once, when
# the plan runs (collect()), and it reads only what it needs.
# Helpers: helper-parity-F124.R, helper-parity-F85.R (pq_write()).

# Warnings of `pipeline` over each source against dplyr's: the same messages
# (`messages = TRUE`) or only the same count, and on parquet none at the verb.
f124r1_check <- function(df, pipeline, info, messages = TRUE) {
  oracle <- f124_capture(pipeline(tibble::as_tibble(df)))
  want <- oracle_normalize(oracle$value)
  path <- pq_write(df)
  make <- list(
    eager = function() tbl_gpu(df, lazy = FALSE),
    lazy = function() tbl_gpu(df, lazy = TRUE),
    parquet = function() read_parquet_gpu(path, lazy = TRUE)
  )
  for (s in names(make)) {
    lbl <- paste0(info, " [", s, "]")
    verb <- f124_capture(pipeline(make[[s]]()))
    col <- f124_capture(collect(verb$value))
    got <- oracle_normalize(col$value)
    expect_identical(f124_col_types(got), f124_col_types(want), info = paste(lbl, "types"))
    expect_equal(got, want, info = paste(lbl, "values"))
    ws <- c(verb$warnings, col$warnings)
    if (messages) {
      expect_identical(f124_msgs(ws), f124_msgs(oracle$warnings), info = paste(lbl, "warnings"))
    } else {
      expect_identical(length(ws), length(oracle$warnings), info = paste(lbl, "warning count"))
    }
    if (s == "parquet") {
      expect_length(verb$warnings, 0L)
      expect_identical(c(verb$plan_break, col$plan_break), character(), info = paste(lbl, "plan break"))
    }
  }
}

test_that("F124 review a2-1 (B12): pending steps that warn warn once on a parquet plan", {
  skip_if_no_gpu()
  dA <- tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), d = c(1, 2, NA, NA, 5))
  f124r1_check(dA, function(x) {
    mutate(summarise(group_by(x, g), s = max(d, na.rm = TRUE), n = n()), gap = max(n) - n)
  }, "A: a summarise() hit, then an integer max()")
  dB <- tibble::tibble(a = 1:10, b = as.double(1:10))
  f124r1_check(dB, function(x) mutate(mutate(filter(x, a > 100L), m1 = max(b)), m2 = max(a)),
               "B: a carried double max(), then an integer max()")
  dC <- tibble::tibble(a = c(1L, .Machine$integer.max, 5L), b = c(0.9, 0.1, 0.8),
                       v1 = c(1, 2, 3), v2 = c(4, 5, 6))
  # (cuplyr's own integer-overflow warning is a plain simpleWarning: count only)
  f124r1_check(dC, function(x) mutate(filter(mutate(x, z = a + 1L, w = v1 * 2), b > 0.5), m = max(a)),
               "C: a pending integer overflow", messages = FALSE)
})

test_that("F124 review a2-1 (N36): the query at the verb reads only the columns it needs", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, 2000000000L, 5L, 7L), b = c(0.9, 0.1, 0.8, 0.7),
                       k = c(10L, 20L, 30L, 40L), v1 = c(1, 2, 3, 4), v2 = c(4, 5, 6, 7))
  path <- pq_write(df)
  verb_read <- function(pipeline) {
    parquet_last_read(reset = TRUE)
    q <- suppressWarnings(pipeline(read_parquet_gpu(path, lazy = TRUE)))
    rd <- parquet_last_read()
    list(q = q, read = rd)
  }
  # a filter on b and unread mutates before an integer max(): the count reads b
  r <- verb_read(function(t) mutate(filter(mutate(t, z = a + 1L, w = v1 * 2), b > 0.5), m = max(k)))
  if (!is.null(r$read)) expect_setequal(r$read$columns, "b")
  expect_identical(collect(r$q)$m, rep(40L, 3L))
  # an integer predicate (pushed into the scan) before an integer max()
  r <- verb_read(function(t) mutate(filter(t, k > 15L), m = max(a)))
  if (!is.null(r$read)) expect_setequal(r$read$columns, "k")
  parquet_last_read(reset = TRUE)
  out <- collect(select(r$q, m))
  expect_identical(out$m, rep(2000000000L, 3L))
  expect_true(length(parquet_last_read()$predicates) > 0L)
  expect_setequal(parquet_last_read()$columns, c("a", "k"))
})

test_that("F124 review a2-1 (N37): a double min()/max() next to runif() needs nothing at the verb", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:6, b = c(0.9, 0.1, 0.8, 0.7, 0.2, 0.6))
  path <- pq_write(df)
  parquet_last_read(reset = TRUE)
  q <- mutate(filter(read_parquet_gpu(path, lazy = TRUE), b > 0.5), r = runif(1), m = max(b))
  expect_null(parquet_last_read())
  out <- collect(q)
  expect_identical(out$m, rep(0.9, 4L))
  # one evaluation of runif(1) for the whole column, as in dplyr
  expect_length(unique(out$r), 1L)
})
