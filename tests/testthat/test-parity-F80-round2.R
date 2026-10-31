# F80 round 2: constants the GPU can't hold, the integer-sum rule, plan-break
# wording of a restrictive join, and the result mode of rowwise set operations.
# (The acceptance tests are test-parity-F80-default.R and -print.R.)

f80r2_breaks <- function(expr) {
  got <- list()
  withCallingHandlers(
    expr,
    cuplyr_plan_break = function(cnd) got[[length(got) + 1L]] <<- cnd
  )
  got
}

# ---- B1: values the GPU has no type for go to dplyr at the verb -------------

test_that("an unsupported constant gives dplyr's error at the verb, in both modes", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 2), x = 1:5)
  cases <- list(
    function(d) mutate(d, out = rlang::env(a = 1)),
    function(d) transmute(d, out = rlang::env(a = 1)),
    function(d) group_by(d, e = rlang::env(a = 1)),
    function(d) mutate(d, y = mean),
    function(d) mutate(group_by(d, g), out = rlang::env(a = 1)),
    function(d) mutate(rowwise(d), out = rnorm),
    function(d) mutate(d, out = x + rlang::env(a = 1))
  )
  for (i in seq_along(cases)) {
    want <- tryCatch(cases[[i]](df), error = function(e) e)
    expect_s3_class(want, "error")
    for (lazy in c(FALSE, TRUE)) {
      got <- tryCatch(cases[[i]](tbl_gpu(df, lazy = lazy)), error = function(e) e)
      expect_s3_class(got, "error")
      expect_identical(class(got), class(want), info = paste(i, lazy))
      expect_identical(conditionMessage(got), conditionMessage(want), info = paste(i, lazy))
    }
  }
})

test_that("a POSIXlt value is dplyr's result at the verb, in both modes", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  for (lazy in c(FALSE, TRUE)) {
    out <- tibble::as_tibble(collect(mutate(tbl_gpu(df, lazy = lazy), t = as.POSIXlt("2024-01-02", tz = "UTC"))))
    expect_identical(out$x, df$x)
    expect_equal(as.POSIXct(out$t), rep(as.POSIXct("2024-01-02", tz = "UTC"), 3L), ignore_attr = TRUE)
  }
})

# ---- B5: an integer sum() is decided by the data ----------------------------

f80r2_big <- function() tibble::tibble(g = c(1, 1, 2, 2), i = c(.Machine$integer.max, 1L, 5L, 6L))

test_that("a lazy integer sum is dplyr's double or integer, whichever the data gives", {
  skip_if_no_gpu()
  big <- f80r2_big()
  small <- tibble::tibble(g = c(1, 1, 2, 2), i = 1:4)
  pipes <- list(
    function(d) mutate(d, s = sum(i)),
    function(d) mutate(group_by(d, g), s = sum(i)),
    function(d) mutate(d, s = sum(i), .by = g),
    function(d) summarise(d, s = sum(i)),
    function(d) summarise(group_by(d, g), s = sum(i)),
    function(d) summarise(d, s = sum(i), .by = g),
    function(d) transmute(d, s = sum(i) + 1L),
    function(d) count(d, g, wt = i),
    function(d) mutate(rowwise(d), s = sum(c(g, i)) + 0L),
    function(d) mutate(rowwise(d), s = sum(i, i)),
    function(d) summarise(rowwise(d), s = sum(dplyr::c_across(i)), t = sum(i, i)),
    function(d) filter(rowwise(d), sum(i, i) > 5)
  )
  for (df in list(big, small)) {
    df <- dplyr::mutate(df, g = as.integer(g))
    for (k in seq_along(pipes)) {
      want <- suppressWarnings(pipes[[k]](df))
      lz <- suppressWarnings(pipes[[k]](tbl_gpu(df, lazy = TRUE)))
      expect_true(is_lazy(lz), info = as.character(k))
      # (summarise(.by =) orders its groups with a pending arrange()/select()
      # after the step that ran)
      if (!identical(k, 6L)) expect_false(has_pending_ops(lz), info = as.character(k))
      got <- suppressWarnings(collect(lz))
      expect_identical(lapply(got, class), lapply(want, class), info = as.character(k))
      expect_equal(tibble::as_tibble(got), tibble::as_tibble(want), ignore_attr = TRUE, info = as.character(k))
    }
  }
})

test_that("a lazy integer sum runs the pending steps first, with a plan-break notice", {
  skip_if_no_gpu()
  df <- f80r2_big()
  lz <- filter(tbl_gpu(df, lazy = TRUE), g > 0)
  expect_true(has_pending_ops(lz))
  for (verb in list(
    function(d) summarise(d, s = sum(i)),
    function(d) mutate(d, s = sum(i)),
    function(d) mutate(rowwise(d), s = sum(i, i))
  )) {
    breaks <- f80r2_breaks(suppressWarnings(verb(lz)))
    expect_length(breaks, 1L)
    expect_match(breaks[[1]]$reason, "integer sum()", fixed = TRUE)
    expect_equal(breaks[[1]]$steps, 1L)
  }
  # nothing pending: nothing to announce; BOOL8 sums stay in the plan
  expect_length(f80r2_breaks(summarise(tbl_gpu(df, lazy = TRUE), s = sum(i))), 0L)
  flags <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), l = c(TRUE, FALSE, TRUE)), lazy = TRUE)
  expect_length(f80r2_breaks(summarise(filter(flags, g > 0), s = sum(l))), 0L)
  expect_true(has_pending_ops(summarise(filter(flags, g > 0), s = sum(l))))
  # ntile(x) counts its non-missing rows with an internal integer sum that can't
  # overflow, so it stays in the plan
  nt <- mutate(filter(tbl_gpu(tibble::tibble(x = c(3L, 1L, NA, 2L)), lazy = TRUE), TRUE), b = ntile(x, 2L))
  expect_true(has_pending_ops(nt))
  expect_length(f80r2_breaks(mutate(filter(tbl_gpu(df, lazy = TRUE), g > 0), b = ntile(i, 2L))), 0L)
  # eager tables are untouched
  eg <- tbl_gpu(df, lazy = FALSE)
  expect_length(f80r2_breaks(suppressWarnings(summarise(eg, s = sum(i)))), 0L)
})

# ---- N1: the notice of a join that checks its rows at the verb --------------

test_that("a restrictive join's plan-break notice does not count the join as an early step", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:3, a = c(10, 20, 30))
  y <- tibble::tibble(k = 1:3, b = c("p", "q", "r"))
  lx <- tbl_gpu(x, lazy = TRUE)
  ly <- tbl_gpu(y, lazy = TRUE)
  breaks <- f80r2_breaks(left_join(lx, ly, by = "k", relationship = "one-to-one"))
  expect_length(breaks, 1L)
  expect_match(conditionMessage(breaks[[1]]), "left_join() ran on the GPU at the verb", fixed = TRUE)
  expect_no_match(conditionMessage(breaks[[1]]), "pending step")

  breaks <- f80r2_breaks(left_join(filter(lx, a > 0), ly, by = "k", relationship = "one-to-one"))
  expect_length(breaks, 1L)
  expect_match(conditionMessage(breaks[[1]]), "with 1 pending step", fixed = TRUE)
})

# ---- N9: a set operation's result mode with a rowwise x ----------------------

test_that("a set operation with a rowwise x is lazy when either side is", {
  skip_if_no_gpu()
  x <- tibble::tibble(a = c(1, 2, 2), b = c("p", "q", "q"))
  y <- tibble::tibble(a = c(2, 3), b = c("q", "r"))
  for (op in list(dplyr::union, dplyr::union_all, dplyr::intersect, dplyr::setdiff)) {
    want <- op(dplyr::rowwise(x), y)
    for (xl in c(FALSE, TRUE)) for (yl in c(FALSE, TRUE)) {
      got <- op(dplyr::rowwise(tbl_gpu(x, lazy = xl)), tbl_gpu(y, lazy = yl))
      expect_identical(is_lazy(got), xl || yl, info = paste(xl, yl))
      expect_equal(tibble::as_tibble(collect(got)), tibble::as_tibble(want), ignore_attr = TRUE)
    }
  }
})
