# F124 attempt 2, round 3: on a plan over a parquet scan every min()/max()
# warning comes when the plan runs, in dplyr's order, each run; the facts query
# lets a once-per-session warning through and restores the optimizer flag
# however it ends.
# Helpers: helper-parity-F124.R, helper-parity-F85.R (pq_write()).

# Warnings of `pipeline` on parquet (and lazy in-memory) sources: none at the
# verb on parquet, the verb + collect() list in order equal to dplyr's, values
# and types equal.
f124c5_check <- function(dx, dy, pipeline, info, sources = c("parquet", "mixed", "lazy")) {
  oracle <- f124_capture(pipeline(dx, dy))
  want <- oracle_normalize(oracle$value)
  px <- pq_write(dx)
  py <- pq_write(dy)
  make <- list(
    parquet = function() list(read_parquet_gpu(px, lazy = TRUE), read_parquet_gpu(py, lazy = TRUE)),
    mixed = function() list(tbl_gpu(dx, lazy = TRUE), read_parquet_gpu(py, lazy = TRUE)),
    lazy = function() list(tbl_gpu(dx, lazy = TRUE), tbl_gpu(dy, lazy = TRUE))
  )
  for (s in sources) {
    lbl <- paste0(info, " [", s, "]")
    xy <- make[[s]]()
    verb <- f124_capture(pipeline(xy[[1L]], xy[[2L]]))
    col <- f124_capture(collect(verb$value))
    got <- oracle_normalize(col$value)
    expect_identical(f124_col_types(got), f124_col_types(want), info = paste(lbl, "types"))
    expect_equal(got, want, info = paste(lbl, "values"))
    expect_identical(f124_msgs(c(verb$warnings, col$warnings)), f124_msgs(oracle$warnings),
                     info = paste(lbl, "warnings, in order"))
    expect_identical(c(verb$fallback, col$fallback), character(), info = paste(lbl, "CPU fallback"))
    if (s != "lazy") {
      expect_length(verb$warnings, 0L)
      expect_identical(c(verb$plan_break, col$plan_break), character(), info = paste(lbl, "plan break"))
    }
  }
}

test_that("F124: a .drop = FALSE min()/max() on a parquet plan warns at collect() in dplyr's order", {
  skip_if_no_gpu()
  dD <- tibble::tibble(k = 1:2, f = factor(c("a", "b"), levels = c("a", "b", "c")), a = 1:2, b = c(1, 2))
  none <- tibble::tibble(k = 1L)
  # a later select() drops the column: the window node still runs, as dplyr's
  # mutate() still evaluates
  f124c5_check(dD, none, function(x, y) {
    select(mutate(group_by(mutate(filter(x, b > 100), m1 = max(b)), f, .drop = FALSE), m2 = max(a)), -m2)
  }, "dropped by a later select()", sources = c("parquet", "lazy"))
  # filter() with rows left, one level empty
  f124c5_check(dD, none, function(x, y) {
    filter(group_by(mutate(filter(x, b > 1.5), m1 = max(b)), f, .drop = FALSE), a == max(a))
  }, "window filter(), rows left", sources = c("parquet", "lazy"))
  # the parquet table on the right of a join with an in-memory one
  dy <- tibble::tibble(k = 1:2, c = c(5L, 6L))
  f124c5_check(dD, dy, function(x, y) {
    mutate(group_by(inner_join(x, y, by = "k"), f, .drop = FALSE), m = max(c))
  }, "join, parquet on the right")
})

test_that("F124: each run of a parquet plan gives its min()/max() warnings again", {
  skip_if_no_gpu()
  dD <- tibble::tibble(f = factor(c("a", "b"), levels = c("a", "b", "c")), a = 1:2, b = c(1, 2))
  path <- pq_write(dD)
  pl <- function(x) mutate(group_by(mutate(filter(x, b > 100), m1 = max(b)), f, .drop = FALSE), m2 = max(a))
  want <- f124_msgs(f124_capture(pl(dD))$warnings)
  expect_length(want, 2L)
  verb <- f124_capture(pl(read_parquet_gpu(path, lazy = TRUE)))
  expect_length(verb$warnings, 0L)
  for (run in 1:2) {
    col <- f124_capture(collect(verb$value))
    expect_identical(f124_msgs(col$warnings), want, info = paste("run", run))
  }
})

test_that("F124: the facts query lets an invalid-option warning through", {
  skip_if_no_gpu()
  forget <- function() rm(list = ls(cuplyr_opt_state, all.names = TRUE), envir = cuplyr_opt_state)
  forget()
  withr::defer(forget())
  withr::local_options(cuplyr.roll_path = "bogus")
  df <- tibble::tibble(a = 1:10, b = as.double(1:10))
  path <- pq_write(df)
  verb <- f124_capture(mutate(filter(mutate(read_parquet_gpu(path, lazy = TRUE), r = roll_mean(b, 3)),
                                     r > 100), m = max(a)))
  col <- f124_capture(collect(verb$value))
  ws <- c(verb$warnings, col$warnings)
  invalid <- vapply(ws, inherits, logical(1), "cuplyr_warning_invalid_option")
  expect_identical(sum(invalid), 1L)
  # the min()/max() warning is still the plan's
  expect_length(Filter(function(w) !inherits(w, "cuplyr_warning_invalid_option"), verb$warnings), 0L)
  expect_true(any(grepl("max(a)", f124_msgs(col$warnings), fixed = TRUE)))
  expect_identical(typeof(col$value$m), "double")
})

test_that("F124: the facts query restores the optimizer flag when it fails", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:4, b = c(0.9, 0.1, 0.8, 0.7))
  path <- tempfile(fileext = ".parquet")
  write_parquet_gpu(tbl_gpu(df, lazy = FALSE), path)
  x <- filter(read_parquet_gpu(path, lazy = TRUE), b > 0.5)
  unlink(path)
  prev <- .optimizer_opts$prune_unread
  expect_error(mutate(x, m = max(a)), class = "cuplyr_error_parquet")
  expect_identical(.optimizer_opts$prune_unread, prev)
  expect_false(isTRUE(.optimizer_opts$prune_unread))
})

test_that("F124: a filter() that keeps a parquet plan's groups stays lazy and reads only its keys", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor(c("a", "b", "b", "a"), levels = c("a", "b", "c")),
                       s = c(NA, "x", "y", "x"), a = 1:4, b = c(1, 2, 3, 4), w = c(9, 8, 7, 6))
  path <- pq_write(df)
  cases <- list(
    list(keys = "f", drop = FALSE, preserve = FALSE, read = c("f", "b")),
    list(keys = "s", drop = FALSE, preserve = FALSE, read = c("s", "b")),
    list(keys = c("f", "s"), drop = FALSE, preserve = FALSE, read = c("f", "s", "b")),
    list(keys = "s", drop = TRUE, preserve = TRUE, read = c("s", "b"))
  )
  for (cs in cases) {
    lbl <- paste(paste(cs$keys, collapse = "+"), "drop", cs$drop, "preserve", cs$preserve)
    pl <- function(x) {
      filter(group_by(filter(x, b > 1), !!!rlang::syms(cs$keys), .drop = cs$drop),
             a > 3, .preserve = cs$preserve)
    }
    want <- pl(df)
    parquet_last_read(reset = TRUE)
    got <- pl(read_parquet_gpu(path, lazy = TRUE))
    expect_true(has_pending_ops(got), info = lbl)
    expect_setequal(parquet_last_read()$columns, cs$read)
    out <- collect(got)
    expect_equal(oracle_normalize(out), oracle_normalize(want), info = lbl)
    expect_equal(dplyr::group_data(out), dplyr::group_data(want), info = paste(lbl, "group_data"))
  }
})
