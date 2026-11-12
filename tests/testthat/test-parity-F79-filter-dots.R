# F79 acceptance: one filter()/filter_out() call evaluates all its dots against
# the same input rows, as dplyr's filter_eval() does: every dot sees the table
# as it was when the call started, sizes are checked against that table (per
# group on a grouped one), and the masks are combined at the end.
#
# The two reported bugs: a logical vector constant next to a native predicate
# was applied in the wrong order, so it was sized against rows an earlier dot
# had already removed, and a later predicate only saw the rows the constant
# had left.

f79_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L),
    x = c(1L, 5L, -2L, 6L, 7L, -3L),
    y = c(1L, 1L, 1L, 1L, 1L, 1L),
    s = c("a", "b", "a", "b", "a", "b"),
    ok = c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE)
  )
}

# Not a native op: any dot that calls it runs in dplyr.
f79_user_pred <- function(v) v > 0L

# Same rows both ways, eager and lazy.
f79_same <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# dplyr's error, unchanged: same classes and same text, at the verb or at
# collect(), in both modes.
f79_same_error <- function(df, pipeline) {
  want <- tryCatch(pipeline(df), error = identity)
  expect_s3_class(want, "rlang_error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch(collect(pipeline(tbl_gpu(df, lazy = lazy))), error = identity)
    expect_s3_class(got, "rlang_error")
    expect_identical(class(got), class(want), info = paste("lazy =", lazy))
    expect_identical(conditionMessage(got), conditionMessage(want), info = paste("lazy =", lazy))
  }
}

# All the orderings of the dots in `qs`.
f79_orders <- function(n) {
  if (n == 1L) {
    return(list(1L))
  }
  out <- list()
  for (i in seq_len(n)) {
    for (rest in f79_orders(n - 1L)) {
      out <- c(out, list(c(i, setdiff(seq_len(n), i)[rest])))
    }
  }
  out
}

# Runs `verb(d, !!!qs[ord])` for every order against dplyr, eager and lazy.
f79_all_orders <- function(df, qs, verb, wrap = identity, ...) {
  for (ord in f79_orders(length(qs))) {
    f79_same(df, function(d) verb(wrap(d), !!!qs[ord]), ...)
  }
}

test_that("the two reported cases match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, 2L, -5L, 4L), b = c(1L, 1L, 1L, 1L), keep = c(TRUE, TRUE, FALSE, TRUE))
  f79_same(df, function(d) dplyr::filter(d, keep, c(TRUE, TRUE, TRUE, TRUE), a + b > 0L))

  d2 <- tibble::tibble(g = c(1L, 1L, 1L, 1L, -1L, -1L), x = c(1L, 5L, 6L, 7L, 9L, 9L))
  f79_same(d2, function(d) dplyr::filter(dplyr::filter(d, g > 0L), x > 4L, c(TRUE, FALSE, TRUE, TRUE)))
})

test_that("a constant, a native and a CPU-only dot agree with dplyr in every order", {
  skip_if_no_gpu()
  qs <- rlang::quos(c(TRUE, FALSE, TRUE, TRUE, NA, TRUE), x + y > 0L, f79_user_pred(x))
  f79_all_orders(f79_df(), qs, dplyr::filter)
  f79_all_orders(f79_df(), qs, dplyr::filter_out)
})

test_that("a constant, a native and a window dot agree with dplyr in every order", {
  skip_if_no_gpu()
  qs <- rlang::quos(c(TRUE, FALSE, TRUE, TRUE, NA, TRUE), x + y > 0L, x > mean(x))
  f79_all_orders(f79_df(), qs, dplyr::filter)
  f79_all_orders(f79_df(), qs, dplyr::filter_out)
})

test_that("a constant is sized and recycled per group on a grouped table", {
  skip_if_no_gpu()
  qs <- rlang::quos(c(TRUE, FALSE, TRUE), x + y > 0L, x == max(x))
  grouped <- function(d) dplyr::group_by(d, g)
  f79_all_orders(f79_df(), qs, dplyr::filter, wrap = grouped)
  f79_all_orders(f79_df(), qs, dplyr::filter_out, wrap = grouped)
  by_g <- function(verb) function(d, ...) verb(d, ..., .by = g)
  f79_all_orders(f79_df(), qs[1:2], by_g(dplyr::filter))
  f79_all_orders(f79_df(), qs[1:2], by_g(dplyr::filter_out))
})

test_that("group structure is kept with .preserve and .drop = FALSE", {
  skip_if_no_gpu()
  df <- f79_df()
  keys <- function(res) tibble::as_tibble(dplyr::group_keys(res))
  run <- function(d) {
    dplyr::filter(dplyr::group_by(d, g), x > 100L | g == 2L, c(TRUE, FALSE, TRUE), .preserve = TRUE)
  }
  f79_same(df, run)
  expect_equal(keys(run(tbl_gpu(df, lazy = FALSE))), keys(run(df)))
  expect_equal(keys(run(tbl_gpu(df, lazy = TRUE))), keys(run(df)))

  undrop <- function(d) {
    dplyr::filter(dplyr::group_by(d, g, .drop = FALSE), x > 100L | g == 2L, c(TRUE, FALSE, TRUE))
  }
  f79_same(df, undrop)
  expect_equal(keys(undrop(tbl_gpu(df, lazy = FALSE))), keys(undrop(df)))
})

test_that("a constant of the wrong size is dplyr's error, in any position", {
  skip_if_no_gpu()
  df <- f79_df()
  f79_same_error(df, function(d) dplyr::filter(d, c(TRUE, FALSE, TRUE), x > 0L))
  f79_same_error(df, function(d) dplyr::filter(d, x > 0L, c(TRUE, FALSE, TRUE)))
  f79_same_error(df, function(d) dplyr::filter(d, x > 0L, logical(0)))
  f79_same_error(df, function(d) dplyr::filter(d, x > 0L, c(TRUE, FALSE, TRUE), x > 1L))
  f79_same_error(df, function(d) dplyr::filter_out(d, x > 0L, c(TRUE, FALSE, TRUE)))
  f79_same_error(df, function(d) dplyr::filter(dplyr::group_by(d, g), x > 0L, c(TRUE, FALSE)))
  f79_same_error(df, function(d) dplyr::filter(d, x > 0L, c(TRUE, FALSE), .by = g))
})

test_that("every constant is sized against the input, not against what an earlier one left", {
  skip_if_no_gpu()
  df <- f79_df()
  # Applied one after the other, the second vector would fit the 2 rows left
  f79_same_error(df, function(d) dplyr::filter(d, c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE), c(TRUE, TRUE)))
  # A constant sized for the table that an inner filter() left is fine ...
  f79_same(df, function(d) dplyr::filter(dplyr::filter(d, g == 1L), x > 0L, c(TRUE, TRUE, FALSE)))
  # ... and one sized for the rows before that inner filter() is not.
  f79_same_error(df, function(d) {
    dplyr::filter(dplyr::filter(d, g == 1L), x > 0L, c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE))
  })
})

test_that("several constants, NAs and scalars combine with &", {
  skip_if_no_gpu()
  df <- f79_df()
  f79_same(df, function(d) {
    dplyr::filter(d, c(TRUE, TRUE, FALSE, TRUE, NA, TRUE), c(NA, TRUE, TRUE, TRUE, TRUE, TRUE), x > 0L)
  })
  f79_same(df, function(d) dplyr::filter(d, x > 0L, c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE), TRUE))
  f79_same(df, function(d) dplyr::filter(d, FALSE, x > 0L, c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE)))
  f79_same(df, function(d) dplyr::filter(d, NA, c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE), x > 0L))
  f79_same(df, function(d) dplyr::filter(d, c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE), FALSE))
  f79_same(df, function(d) dplyr::filter(d, rep(TRUE, 6L), x > 0L))
  f79_same(df, function(d) dplyr::filter(d, rep(FALSE, 6L), x > 0L))
  f79_same(df, function(d) dplyr::filter(d, rep(NA, 6L), x > 0L))
})

test_that("constants written as a name, a call, a splice or a named vector work", {
  skip_if_no_gpu()
  df <- f79_df()
  mask <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE)
  f79_same(df, function(d) dplyr::filter(d, x > 0L, mask))
  f79_same(df, function(d) dplyr::filter(d, mask, x > 0L))
  f79_same(df, function(d) dplyr::filter(d, x > 0L, rep(c(TRUE, FALSE), 3L)))
  f79_same(df, function(d) dplyr::filter(d, x > 0L, !!mask))
  f79_same(df, function(d) dplyr::filter(d, !!mask, x > 0L))
  f79_same(df, function(d) dplyr::filter(d, x > 0L, c(a = TRUE, b = FALSE, c = TRUE, d = TRUE, e = FALSE, f = TRUE)))
  f79_same(df, function(d) dplyr::filter(d, x > 0L, .env$mask))
})

test_that("a column named like the constant is the column", {
  skip_if_no_gpu()
  df <- f79_df()
  ok <- rep(FALSE, 6L)
  f79_same(df, function(d) dplyr::filter(d, x > 0L, ok))
})

test_that("a constant is evaluated once, and native dots see every input row", {
  skip_if_no_gpu()
  df <- f79_df()
  for (lazy in c(FALSE, TRUE)) {
    k <- 0L
    bump <- function() {
      k <<- k + 1L
      c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE)
    }
    res <- collect(dplyr::filter(tbl_gpu(df, lazy = lazy), x > 0L, bump()))
    expect_identical(k, 1L, info = paste("lazy =", lazy))
    expect_equal(res$x, df$x[df$x > 0L & bump()])
  }
})

test_that("a constant that drops a row does not hide that row's integer overflow", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(.Machine$integer.max, 1L, 2L, 3L))
  pipeline <- function(d) collect(dplyr::filter(d, c(FALSE, TRUE, TRUE, TRUE), x * 2L > 0L))
  expect_warning(want <- pipeline(df), "integer overflow")
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(got <- pipeline(tbl_gpu(df, lazy = lazy)), "integer overflow")
    expect_equal(tibble::as_tibble(got), tibble::as_tibble(want))
  }
})

test_that("a call mixing a constant and native dots collects nothing", {
  skip_if_no_gpu()
  df <- f79_df()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    collected <- character()
    res <- withCallingHandlers(
      collect(dplyr::filter(gt, x + y > 0L, c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE), s == "a" | s == "b")),
      cuplyr_cpu_fallback = function(cnd) {
        if (!grepl("Nothing was collected", conditionMessage(cnd), fixed = TRUE)) {
          collected <<- c(collected, conditionMessage(cnd))
        }
      }
    )
    expect_identical(collected, character(), info = paste("lazy =", lazy))
    expect_equal(tibble::as_tibble(res)$x, c(1L, 6L))
  }
})

test_that("a lazy result stays lazy and later verbs work through compute()", {
  skip_if_no_gpu()
  df <- f79_df()
  lazy <- dplyr::filter(tbl_gpu(df, lazy = TRUE), x > 0L, c(TRUE, TRUE, TRUE, TRUE, FALSE, TRUE))
  expect_identical(lazy$exec_mode, "lazy")
  f79_same(df, function(d) {
    d |>
      dplyr::filter(x > 0L, c(TRUE, TRUE, TRUE, TRUE, FALSE, TRUE)) |>
      dplyr::mutate(z = x * 2L) |>
      dplyr::filter(z > 2L) |>
      dplyr::compute() |>
      dplyr::select(x, z)
  })
  f79_same(df, function(d) {
    d |>
      dplyr::select(x, y) |>
      dplyr::filter(c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE), x > 0L) |>
      dplyr::compute()
  })
})

test_that("zero-row tables follow dplyr", {
  skip_if_no_gpu()
  df <- f79_df()[0L, ]
  f79_same(df, function(d) dplyr::filter(d, x > 0L, logical(0)))
  f79_same(df, function(d) dplyr::filter(d, logical(0), x > 0L))
  f79_same(df, function(d) dplyr::filter(d, x > 0L, TRUE))
  f79_same_error(df, function(d) dplyr::filter(d, x > 0L, c(TRUE, FALSE)))
})

test_that("a table with no columns uses one mask, sized against its rows", {
  skip_if_no_gpu()
  df <- tibble::new_tibble(list(), nrow = 4L)
  gt <- tbl_gpu(df, lazy = FALSE)
  res <- dplyr::filter(gt, c(TRUE, FALSE, TRUE, TRUE), c(TRUE, TRUE, FALSE, TRUE))
  expect_identical(dim(res), c(2L, 0L))
  f79_same_error(df, function(d) dplyr::filter(d, c(TRUE, FALSE, TRUE, TRUE), c(TRUE, TRUE)))
})

test_that("a rowwise table gives dplyr's result or error", {
  skip_if_no_gpu()
  df <- f79_df()
  f79_same(df, function(d) dplyr::filter(dplyr::rowwise(d), x > 0L, TRUE))
  f79_same_error(df, function(d) dplyr::filter(dplyr::rowwise(d), x > 0L, c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE)))
})

test_that("a non-logical constant next to a native dot is dplyr's error", {
  skip_if_no_gpu()
  df <- f79_df()
  f79_same_error(df, function(d) dplyr::filter(d, x > 0L, c(1, 0, 1, 1, 0, 1)))
  f79_same_error(df, function(d) dplyr::filter(d, x > 0L, c("a", "b", "a", "a", "b", "a")))
})

test_that("cur_group_id() next to a constant is evaluated on the input rows", {
  skip_if_no_gpu()
  df <- f79_df()
  f79_same(df, function(d) dplyr::filter(d, c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE), dplyr::cur_group_id() == 1L))
})

test_that("a mask-only call hands back the table without the cur_group_id() helper", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4, y = rep(2L, 4L))
  m <- c(TRUE, FALSE, TRUE, TRUE)
  res <- f79_same(df, function(d) {
    dplyr::filter(d, m, dplyr::if_all(dplyr::starts_with("zzz"), ~ dplyr::cur_group_id() > 0L))
  })
  expect_identical(names(res$gpu), c("x", "y"))
  f79_same(df, function(d) {
    dplyr::filter(d, m, dplyr::if_any(dplyr::starts_with("zzz"), ~ dplyr::cur_group_id() > 0L))
  })
})

test_that("a column named like an internal temporary does not replace the mask", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4, ..vec1.. = rep(TRUE, 4L))
  f79_same(df, function(d) dplyr::filter(d, x > 0L, c(TRUE, FALSE, TRUE, TRUE)))
  df2 <- tibble::tibble(x = 1:4, ..vec1.. = 11:14)
  f79_same(df2, function(d) dplyr::filter(d, x > 0L, c(TRUE, FALSE, TRUE, TRUE)))
  df3 <- tibble::tibble(x = 1:4, ..filtermask.. = rep(TRUE, 4L))
  f79_same(df3, function(d) dplyr::filter(d, x > 0L, c(TRUE, FALSE, TRUE, TRUE)))
})

test_that("a dot that reaches the data mask by name is evaluated by dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:6, ok = c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE))
  ok <- rep(FALSE, 6L)
  f79_same(df, function(d) dplyr::filter(d, x > 1L, exists("x")))
  f79_same(df, function(d) dplyr::filter(d, exists("x"), x > 1L))
  f79_same(df, function(d) dplyr::filter(d, x > 1L, get("ok")))
  f79_same(df, function(d) dplyr::filter(d, exists("no_such_name_f79")))
  f79_same(df, function(d) dplyr::filter_out(d, exists("x")))
})

test_that("a no-column table leaves a name-reaching dot to dplyr", {
  skip_if_no_gpu()
  df <- tibble::new_tibble(list(), nrow = 4L)
  want <- dplyr::filter(df, exists("no_such_name_f79"))
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::filter(tbl_gpu(df, lazy = lazy), exists("no_such_name_f79"))
    expect_identical(dim(got), dim(want), info = paste("lazy =", lazy))
  }
})

test_that("a constant sits next to an integer sum() window dot and typed columns", {
  skip_if_no_gpu()
  df <- f79_df()
  f79_same(df, function(d) dplyr::filter(d, c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE), sum(x) > 0L))
  f79_same(df, function(d) {
    dplyr::filter(dplyr::filter(d, g > 0L), sum(x) > 0L, c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE))
  })
  typed <- tibble::tibble(
    f = factor(c("a", "b", "a", "c")),
    d = as.Date("2024-01-01") + 0:3,
    t = as.POSIXct("2024-01-01 00:00:00", tz = "UTC") + 0:3,
    n = c(1, NA, 3, 4),
    i64 = bit64::as.integer64(c(1L, 2L, 3L, 4L))
  )
  f79_same(typed, function(d) dplyr::filter(d, !is.na(n), c(TRUE, FALSE, TRUE, FALSE)))
  f79_same(typed, function(d) dplyr::filter(d, c(TRUE, TRUE, FALSE, TRUE), f == "a" | f == "c"))
})

test_that("a name-reaching call inside a dot that names a column is evaluated by dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L, 2L), x = 1:6, flag79 = c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE))
  f79_same(df, function(d) dplyr::filter(d, x > 1L & exists("flag79")))
  f79_same(df, function(d) dplyr::filter(d, exists("flag79") & x > 1L))
  f79_same(df, function(d) dplyr::filter_out(d, x > 1L & exists("flag79")))
  f79_same(df, function(d) dplyr::filter(d, sum(x) > 0L & exists("flag79")))
  f79_same(df, function(d) dplyr::filter(dplyr::group_by(d, g), x > 1L & exists("flag79")))
  f79_same(df, function(d) dplyr::mutate(d, z = x > 1L & exists("flag79")))
  k79 <- FALSE
  d5 <- tibble::tibble(x = 1:4, k79 = c(TRUE, TRUE, FALSE, TRUE))
  f79_same(d5, function(d) dplyr::filter(d, x > 1L & get("k79")))
})

test_that("a name-reaching call runs as often as dplyr runs it", {
  skip_if_no_gpu()
  df <- f79_df()
  counts <- integer()
  run <- function(d) {
    k <- 0L
    bump <- function() {
      k <<- k + 1L
      TRUE
    }
    out <- collect(dplyr::filter(d, x > 1L, do.call(bump, list())))
    counts <<- c(counts, k)
    out
  }
  want <- run(df)
  for (lazy in c(FALSE, TRUE)) {
    got <- run(tbl_gpu(df, lazy = lazy))
    expect_equal(tibble::as_tibble(got), tibble::as_tibble(want))
  }
  expect_identical(counts, rep(counts[[1L]], 3L))
})

test_that("mask-reaching calls spelled through rlang or ::: are evaluated by dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:6, ok79 = rep(TRUE, 6L))
  f79_same(df, function(d) dplyr::filter(d, x > 1L & rlang::exec("exists", "ok79")))
  f79_same(df, function(d) dplyr::filter(d, x > 1L & rlang::exec(exists, "ok79")))
  f79_same(df, function(d) dplyr::filter(d, x > 1L & rlang::env_has(rlang::current_env(), "ok79", inherit = TRUE)))
  f79_same(df, function(d) dplyr::filter(d, base:::exists("ok79")))
  f79_same(df, function(d) dplyr::filter(d, x > 1L, base:::exists("ok79")))
})

test_that("eval() and do.call() read the mask, so they are evaluated by dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4, y = rep(10L, 4L), ok79 = c(TRUE, TRUE, FALSE, TRUE))
  y <- 0L
  k <- quote(y)
  f79_same(df, function(d) dplyr::filter(d, x > eval(k)))
  f79_same(df, function(d) dplyr::filter(d, x > 1L & do.call("exists", list("ok79"))))
})
