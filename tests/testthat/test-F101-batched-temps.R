# F101, round 3: a lazy summarise() queues its temp columns and window
# arguments and adds them in one batch. These pin what the batch must still do
# that the per-dot path did: sharing, name collisions, notices, overflow.

lag <- dplyr::lag
first <- dplyr::first
last <- dplyr::last

f101b_plan <- function(x) {
  node <- .subset2(x, "lazy_ops")
  types <- character()
  while (!is.null(node)) {
    types <- c(types, .subset2(node, "type"))
    node <- .subset2(node, "input")
  }
  types
}

f101b_nodes <- function(x, type) {
  node <- .subset2(x, "lazy_ops")
  out <- list()
  while (!is.null(node)) {
    if (identical(.subset2(node, "type"), type)) out <- c(out, list(node))
    node <- .subset2(node, "input")
  }
  out
}

f101b_df <- function() {
  tibble::tibble(
    g = c(2L, 1L, 1L, 3L, 2L, 3L, 1L, 2L),
    x = c(1.5, NA, 3, 4.25, -1, 0.5, 2, 6),
    y = c(2, 1, NA, 0.5, 3, -2, 4, 1),
    i = c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L),
    b = c(TRUE, FALSE, NA, TRUE, TRUE, FALSE, FALSE, TRUE)
  )
}

test_that("dots sharing a window argument share one column in the batch", {
  skip_if_no_gpu()
  df <- f101b_df()
  q <- tbl_gpu(df, lazy = TRUE) |> group_by(g) |>
    summarise(a = sum(lag(x) * 2, na.rm = TRUE), b = mean(lag(x) * 2, na.rm = TRUE))
  win <- f101b_nodes(q, "window")
  expect_length(win, 1L)
  expect_length(win[[1L]]$specs, 1L)
  temps <- unlist(lapply(f101b_nodes(q, "mutate"), function(n) {
    vapply(n$expressions, `[[`, character(1), "output_col")
  }))
  expect_length(grep("temp_agg", temps), 1L)
  oracle <- df |> group_by(g) |>
    summarise(a = sum(lag(x) * 2, na.rm = TRUE), b = mean(lag(x) * 2, na.rm = TRUE))
  expect_equal(tibble::as_tibble(collect(q)), tibble::as_tibble(oracle))
})

test_that("a bare window output is shared across keep_bool, with no temp", {
  skip_if_no_gpu()
  df <- f101b_df()
  q <- tbl_gpu(df, lazy = TRUE) |> group_by(g) |>
    summarise(a = sum(lag(b), na.rm = TRUE), f = first(lag(b)), l = last(lag(b)))
  win <- f101b_nodes(q, "window")
  expect_length(win, 1L)
  expect_length(win[[1L]]$specs, 1L)
  expect_length(f101b_nodes(q, "mutate"), 0L)
  oracle <- df |> group_by(g) |>
    summarise(a = sum(lag(b), na.rm = TRUE), f = first(lag(b)), l = last(lag(b)))
  expect_equal(tibble::as_tibble(collect(q)), tibble::as_tibble(oracle))
})

test_that("a non-bare window argument keeps its keep_bool per dot", {
  skip_if_no_gpu()
  df <- f101b_df()
  q <- tbl_gpu(df, lazy = TRUE) |> group_by(g) |>
    summarise(a = sum(!lag(b), na.rm = TRUE), f = first(!lag(b)))
  oracle <- df |> group_by(g) |>
    summarise(a = sum(!lag(b), na.rm = TRUE), f = first(!lag(b)))
  got <- tibble::as_tibble(collect(q))
  expect_type(got$f, "logical")
  expect_equal(got, tibble::as_tibble(oracle))
})

test_that("a user column taking a temp name does not collide across the batch", {
  skip_if_no_gpu()
  df <- f101b_df()
  df$.temp_agg_1 <- 100
  df$.temp_agg_win_2 <- 200
  df$`..agg_arg..2` <- 300
  # dot 1 becomes `.temp_agg_11`, which dot 11 asks for too
  build <- function(t) {
    t |> summarise(
      d1 = mean(x + 1, na.rm = TRUE), d2 = mean(lag(x) * 2, na.rm = TRUE),
      d3 = mean(x + 3, na.rm = TRUE), d4 = mean(x + 4, na.rm = TRUE),
      d5 = mean(x + 5, na.rm = TRUE), d6 = mean(x + 6, na.rm = TRUE),
      d7 = mean(x + 7, na.rm = TRUE), d8 = mean(x + 8, na.rm = TRUE),
      d9 = mean(x + 9, na.rm = TRUE), d10 = mean(x + 10, na.rm = TRUE),
      d11 = mean(x + 11, na.rm = TRUE),
      u1 = sum(.temp_agg_1), u2 = sum(.temp_agg_win_2), u3 = sum(`..agg_arg..2`)
    )
  }
  q <- build(tbl_gpu(df, lazy = TRUE))
  temps <- unlist(lapply(f101b_nodes(q, "mutate"), function(n) {
    vapply(n$expressions, `[[`, character(1), "output_col")
  }))
  expect_false(anyDuplicated(temps) > 0L)
  expect_equal(tibble::as_tibble(collect(q)), tibble::as_tibble(build(df)))
})

test_that("an integer window sum() computes once, with the first such dot's expression", {
  skip_if_no_gpu()
  df <- f101b_df()
  breaks <- list()
  q <- withCallingHandlers(
    tbl_gpu(df, lazy = TRUE) |> filter(i > 0L) |> group_by(g) |>
      summarise(p = mean(x * 2, na.rm = TRUE),
                w = mean(lag(x), na.rm = TRUE),
                a = mean(i * sum(i)),
                c = mean(i + sum(i))),
    cuplyr_plan_break = function(e) {
      breaks[[length(breaks) + 1L]] <<- e
      rlang::cnd_muffle(e)
    }
  )
  expect_length(breaks, 1L)
  expect_equal(breaks[[1L]]$steps, 1)
  expect_match(conditionMessage(breaks[[1L]]), "i * sum(i)", fixed = TRUE)
  expect_false("filter" %in% f101b_plan(q))
  oracle <- df |> filter(i > 0L) |> group_by(g) |>
    summarise(p = mean(x * 2, na.rm = TRUE), w = mean(lag(x), na.rm = TRUE),
              a = mean(i * sum(i)), c = mean(i + sum(i)))
  expect_equal(tibble::as_tibble(collect(q)), tibble::as_tibble(oracle))
})

test_that("an integer window sum() overflow in a batch still reruns on the CPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), i = c(.Machine$integer.max, 1L, 2L, 3L),
                       x = c(1, 2, 3, 4))
  q <- suppressWarnings(
    tbl_gpu(df, lazy = TRUE) |> group_by(g) |>
      summarise(p = mean(x * 2), a = mean(i + sum(i)), w = mean(lag(x), na.rm = TRUE))
  )
  oracle <- suppressWarnings(
    df |> group_by(g) |>
      summarise(p = mean(x * 2), a = mean(i + sum(i)), w = mean(lag(x), na.rm = TRUE))
  )
  expect_equal(tibble::as_tibble(suppressWarnings(collect(q))), tibble::as_tibble(oracle))
})

test_that("stages of different nesting depth run together", {
  skip_if_no_gpu()
  df <- f101b_df()
  build <- function(t) {
    t |> group_by(g) |>
      summarise(a = mean(lag(x), na.rm = TRUE),
                b = sum(lag(cumsum(x)), na.rm = TRUE),
                c = mean(lag(x) + lag(y, 2), na.rm = TRUE),
                d = mean(x * 2, na.rm = TRUE))
  }
  q <- build(tbl_gpu(df, lazy = TRUE))
  expect_length(f101b_nodes(q, "window"), 2L)
  expect_equal(tibble::as_tibble(collect(q)), tibble::as_tibble(build(df)))
})
