# Acceptance tests for #20: bind_cols() and the class of its result.
#
# dplyr returns "a data frame the same type as the first element of `...`"
# (dplyr_reconstruct(out, first)). With a tbl_gpu among the inputs, the GPU
# version of that rule is: the result is a tbl_gpu, it keeps the first
# input's grouping or rowwise-ness, and it collects to what dplyr returns for
# the same inputs on the CPU. When the first input is not a table (a list or
# a vector) dplyr returns vec_cbind()'s type of the data frame inputs: a bare
# data.frame when they all are, a tibble when one is a tibble.

p20b_d1 <- data.frame(a = 1:10, b = rep(1:2, each = 5))
p20b_d2 <- tibble::tibble(c = 1:10, d = rep(1:2, each = 5))

test_that("bind_cols() of tbl_gpu inputs is a tbl_gpu that collects to dplyr's result", {
  skip_if_no_gpu()
  expect_same_as_dplyr(p20b_d2, function(t) bind_cols(t, transmute(t, e = c * 2L)))
  expect_same_as_dplyr_lazy(p20b_d2, function(t) bind_cols(t, transmute(t, e = c * 2L)))
  for (lazy in c(FALSE, TRUE)) {
    g1 <- tbl_gpu(p20b_d1, lazy = lazy)
    g2 <- tbl_gpu(p20b_d2, lazy = lazy)
    out <- bind_cols(g2, g1)
    expect_s3_class(out, "tbl_gpu")
    expect_identical(class(collect(out)), class(bind_cols(p20b_d2, p20b_d1)))
    expect_equal(tibble::as_tibble(collect(out)), tibble::as_tibble(bind_cols(p20b_d2, p20b_d1)))
  }
})

test_that("bind_cols() keeps the grouping of a grouped first input (#1692)", {
  skip_if_no_gpu()
  pipeline <- function(t) bind_cols(group_by(t, d), transmute(t, e = c))
  expect_same_as_dplyr(p20b_d2, pipeline)
  expect_same_as_dplyr_lazy(p20b_d2, pipeline)
  want <- bind_cols(group_by(p20b_d2, d), p20b_d1)
  for (lazy in c(FALSE, TRUE)) {
    out <- bind_cols(group_by(tbl_gpu(p20b_d2, lazy = lazy), d), tbl_gpu(p20b_d1))
    expect_identical(group_vars(out), "d")
    got <- collect(out)
    expect_identical(class(got), class(want))
    expect_identical(lengths(dplyr::group_rows(got)), lengths(dplyr::group_rows(want)))
    expect_equal(dplyr::group_data(got), dplyr::group_data(want))
  }
})

test_that("bind_cols() keeps a rowwise first input rowwise (#1692)", {
  skip_if_no_gpu()
  want <- bind_cols(rowwise(p20b_d2), p20b_d1)
  for (lazy in c(FALSE, TRUE)) {
    out <- bind_cols(rowwise(tbl_gpu(p20b_d2, lazy = lazy)), tbl_gpu(p20b_d1))
    got <- collect(out)
    expect_identical(class(got), class(want))
    expect_equal(tibble::as_tibble(got), tibble::as_tibble(want))
  }
})

test_that("bind_cols() with a data frame, a list or a vector first and a tbl_gpu later collects to dplyr's result", {
  skip_if_no_gpu()
  d5 <- list(c = 1:10, d = rep(1:2, each = 5))
  e <- 11:20
  for (lazy in c(FALSE, TRUE)) {
    # (first, second) as plain inputs and with the second uploaded
    cases <- list(
      list(cpu = list(p20b_d1, p20b_d2), gpu = list(p20b_d1, tbl_gpu(p20b_d2, lazy = lazy))),
      list(cpu = list(p20b_d2, p20b_d1), gpu = list(p20b_d2, tbl_gpu(p20b_d1, lazy = lazy))),
      # a list first: vec_cbind()'s type of the data frames, data.frame here
      list(cpu = list(d5, p20b_d1), gpu = list(d5, tbl_gpu(p20b_d1, lazy = lazy))),
      # ... and a tibble as soon as one data frame is a tibble
      list(cpu = list(d5, p20b_d2), gpu = list(d5, tbl_gpu(p20b_d2, lazy = lazy))),
      list(cpu = list(d5, p20b_d1, p20b_d2), gpu = list(d5, tbl_gpu(p20b_d1, lazy = lazy), p20b_d2)),
      list(cpu = list(d5, p20b_d1, p20b_d2), gpu = list(d5, p20b_d1, tbl_gpu(p20b_d2, lazy = lazy))),
      # a named vector first
      list(cpu = list(e = e, p20b_d1), gpu = list(e = e, tbl_gpu(p20b_d1, lazy = lazy))),
      list(cpu = list(e = e, p20b_d2), gpu = list(e = e, tbl_gpu(p20b_d2, lazy = lazy)))
    )
    for (k in seq_along(cases)) {
      info <- paste("case", k, if (lazy) "lazy" else "eager")
      want <- suppressMessages(dplyr::bind_cols(!!!cases[[k]]$cpu))
      out <- suppressMessages(bind_cols(!!!cases[[k]]$gpu))
      expect_s3_class(out, "tbl_gpu")
      got <- collect(out)
      expect_identical(class(got), class(want), info = info)
      expect_equal(tibble::as_tibble(got), tibble::as_tibble(want), info = info)
    }
  }
})

test_that("bind_cols() of tbl_gpu uploaded from data frames collects to a data.frame (#1692)", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g1 <- tbl_gpu(p20b_d1, lazy = lazy)
    want <- suppressMessages(bind_cols(p20b_d1, p20b_d1))
    got <- collect(suppressMessages(bind_cols(g1, g1)))
    expect_identical(class(got), "data.frame")
    expect_equal(got, want)
  }
})

# collect(out) must be what dplyr::bind_cols() gives for `cpu`: class, groups
# (rowwise vars included) and data.
p20b_expect_bind_like_dplyr <- function(cpu, gpu, info) {
  want <- suppressMessages(dplyr::bind_cols(!!!cpu))
  out <- suppressMessages(bind_cols(!!!gpu))
  expect_s3_class(out, "tbl_gpu")
  got <- collect(out)
  expect_identical(class(got), class(want), info = info)
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(want), info = info)
  expect_equal(tibble::as_tibble(dplyr::ungroup(got)), tibble::as_tibble(dplyr::ungroup(want)),
               info = info)
  if (dplyr::is_grouped_df(want)) {
    expect_identical(lengths(dplyr::group_rows(got)), lengths(dplyr::group_rows(want)), info = info)
  }
}

test_that("bind_cols() with a grouped data frame first and a tbl_gpu later keeps the grouping", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    lbl <- if (lazy) "lazy" else "eager"
    p20b_expect_bind_like_dplyr(
      list(group_by(p20b_d2, d), p20b_d1),
      list(group_by(p20b_d2, d), tbl_gpu(p20b_d1, lazy = lazy)), paste("grouped df first", lbl)
    )
    # a grouping column renamed by name repair no longer groups (dplyr_reconstruct())
    p20b_expect_bind_like_dplyr(
      list(group_by(p20b_d2, d), p20b_d2),
      list(group_by(p20b_d2, d), tbl_gpu(p20b_d2, lazy = lazy)), paste("repaired group column", lbl)
    )
  }
})

test_that("bind_cols() with a vector or list first follows dplyr for grouped and rowwise inputs later", {
  skip_if_no_gpu()
  e <- 11:20
  d5 <- list(e = 1:10, f = rep(1:2, each = 5))
  for (lazy in c(FALSE, TRUE)) {
    lbl <- if (lazy) "lazy" else "eager"
    up <- function(x) tbl_gpu(x, lazy = lazy)
    # a rowwise input makes vec_cbind()'s result a rowwise_df
    p20b_expect_bind_like_dplyr(list(e = e, rowwise(p20b_d1)),
                                list(e = e, rowwise(up(p20b_d1))), paste("vector, rowwise", lbl))
    p20b_expect_bind_like_dplyr(list(e = e, rowwise(p20b_d1, a)),
                                list(e = e, rowwise(up(p20b_d1), a)), paste("vector, rowwise(a)", lbl))
    p20b_expect_bind_like_dplyr(list(d5, rowwise(p20b_d1, a)),
                                list(d5, rowwise(up(p20b_d1), a)), paste("list, rowwise(a)", lbl))
    p20b_expect_bind_like_dplyr(list(e = e, p20b_d1, rowwise(p20b_d2)),
                                list(e = e, up(p20b_d1), rowwise(up(p20b_d2))), paste("vector, df, rowwise", lbl))
    # ... a grouped one does not
    p20b_expect_bind_like_dplyr(list(e = e, group_by(p20b_d1, b)),
                                list(e = e, group_by(up(p20b_d1), b)), paste("vector, grouped", lbl))
    # a data frame first decides the class (rowwise later is dropped)
    p20b_expect_bind_like_dplyr(list(p20b_d1, rowwise(p20b_d2)),
                                list(p20b_d1, rowwise(up(p20b_d2))), paste("df, rowwise", lbl))
  }
})

test_that("bind_cols() of plain data frames is still dplyr's", {
  skip_if_no_gpu()
  d5 <- list(c = 1:10, d = rep(1:2, each = 5))
  expect_identical(class(bind_cols(p20b_d1, p20b_d1, .name_repair = "minimal")), "data.frame")
  expect_identical(bind_cols(p20b_d2, p20b_d1), dplyr::bind_cols(p20b_d2, p20b_d1))
  expect_identical(bind_cols(d5, p20b_d1), dplyr::bind_cols(d5, p20b_d1))
  expect_identical(class(bind_cols(d5, p20b_d1)), "data.frame")
})
