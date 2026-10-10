# F06 acceptance tests: joins keep x's grouping.
#
# Every dplyr join ends with dplyr_reconstruct(out, x), so the result is
# grouped by intersect(group_vars(x), names(out)) for every join type, and
# x's .drop setting is kept. y's grouping never matters. A group variable that
# gets suffixed away (a collision, or the key under keep = TRUE) drops out of
# the grouping. Verified with tools/r against dplyr 1.2.1
# (scratchpad/parity/features/F06/probe/p2.out).

f06g_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f06g_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f06g_x <- data.frame(g = c(1L, 1L, 2L, 3L), k = 1:4, v = c("a", "b", "c", "d"))
f06g_y <- data.frame(k = c(1L, 3L, 5L), w = c(10, 30, 50))

f06g_all <- list(left_join = left_join, inner_join = inner_join,
                 right_join = right_join, full_join = full_join,
                 semi_join = semi_join, anti_join = anti_join)

test_that("every join type keeps x's group_vars()", {
  skip_if_no_gpu()
  for (join in f06g_all) {
    f06g_both(f06g_x, function(d) group_by(d, g) |> join(f06g_like(d, f06g_y), by = "k"))
    f06g_both(f06g_x, function(d) group_by(d, g, v) |> join(f06g_like(d, f06g_y), by = "k"))
  }
})

test_that("grouping by the join key survives a merged-key join", {
  skip_if_no_gpu()
  for (join in f06g_all) {
    f06g_both(f06g_x, function(d) group_by(d, k) |> join(f06g_like(d, f06g_y), by = "k"))
  }
})

test_that("a group var renamed by suffixing drops out of the grouping (#2330)", {
  skip_if_no_gpu()
  x <- data.frame(x = 1:5, y = 1:5)
  y <- data.frame(x = 1:5, y = 1:5)
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06g_both(x, function(d) group_by(d, x, y) |> join(f06g_like(d, y), by = "x"))
  }
})

test_that("keep = TRUE suffixes the key, so a key group var drops out", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06g_both(f06g_x, function(d) {
      group_by(d, k, g) |> join(f06g_like(d, f06g_y), by = "k", keep = TRUE)
    })
  }
})

test_that("y's grouping is ignored", {
  skip_if_no_gpu()
  for (join in f06g_all) {
    f06g_both(f06g_x, function(d) join(d, f06g_like(d, f06g_y) |> group_by(w), by = "k"))
    f06g_both(f06g_x, function(d) {
      group_by(d, v) |> join(f06g_like(d, f06g_y) |> group_by(w), by = "k")
    })
  }
})

test_that("a grouped join with a coerced key keeps the group on the coerced column", {
  skip_if_no_gpu()
  y <- data.frame(k = c(1, 3, 5), w = c(10, 30, 50))
  for (join in f06g_all) {
    f06g_both(f06g_x, function(d) group_by(d, k) |> join(f06g_like(d, y), by = "k"))
  }
})

test_that("0-row grouped x keeps its grouping", {
  skip_if_no_gpu()
  x0 <- f06g_x[0, , drop = FALSE]
  for (join in f06g_all) {
    f06g_both(x0, function(d) group_by(d, g) |> join(f06g_like(d, f06g_y), by = "k"))
  }
})

test_that("the propagated groups drive later grouped verbs", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, right_join, full_join)) {
    f06g_both(f06g_x, function(d) {
      group_by(d, g) |>
        join(f06g_like(d, f06g_y), by = "k") |>
        summarise(n = dplyr::n(), w = sum(w, na.rm = TRUE))
    }, arrange_by = "g")  # grouped summarise() row order is a known, separate divergence
    f06g_both(f06g_x, function(d) {
      group_by(d, g) |>
        join(f06g_like(d, f06g_y), by = "k") |>
        mutate(m = max(k))
    })
  }
})

test_that("lazy grouped joins keep groups through filter/select after the join", {
  skip_if_no_gpu()
  f06g_both(f06g_x, function(d) {
    group_by(d, g) |>
      left_join(f06g_like(d, f06g_y), by = "k") |>
      filter(k > 1) |>
      select(g, k, w)
  })
})

test_that("joins respect zero-length groups (.drop = FALSE factor groups)", {
  skip_if_no_gpu()
  df1 <- data.frame(f = factor(c(1, 1, 2, 2), levels = 1:3), x = c(1, 2, 1, 4))
  df2 <- data.frame(f = factor(c(2, 2, 3, 3), levels = 1:3), y = c(1, 2, 3, 4))
  for (drop in c(TRUE, FALSE)) {
    for (nm in names(f06g_all)) {
      join <- f06g_all[[nm]]
      run <- function(x, y) {
        args <- list(group_by(x, f, .drop = drop), group_by(y, f, .drop = drop), by = "f")
        if (!nm %in% c("semi_join", "anti_join")) args$relationship <- "many-to-many"
        do.call(join, args)
      }
      oracle <- run(tibble::as_tibble(df1), tibble::as_tibble(df2))
      for (lazy in c(FALSE, TRUE)) {
        out <- run(tbl_gpu(df1, lazy = lazy), tbl_gpu(df2, lazy = lazy))
        info <- sprintf("%s .drop=%s lazy=%s", nm, drop, lazy)
        expect_identical(group_vars(out), group_vars(oracle), info = info)
        expect_identical(group_by_drop_default(out), group_by_drop_default(oracle), info = info)
        expect_equal(group_size(out), group_size(oracle), info = info)
      }
    }
  }
})

test_that("nest_join() keeps x's groups and names the list column like dplyr", {
  skip_if_no_gpu()
  ys <- tibble::as_tibble(f06g_y)
  oracle <- nest_join(group_by(tibble::as_tibble(f06g_x), g), ys, by = "k")
  for (lazy in c(FALSE, TRUE)) {
    ys <- tbl_gpu(f06g_y, lazy = lazy)
    out <- nest_join(group_by(tbl_gpu(f06g_x, lazy = lazy), g), ys, by = "k")
    expect_identical(group_vars(out), group_vars(oracle))
    expect_identical(names(out), names(oracle))
    expect_equal(
      lapply(dplyr::pull(dplyr::ungroup(tibble::as_tibble(out)), "ys"), tibble::as_tibble),
      lapply(oracle$ys, tibble::as_tibble)
    )
  }
  # An explicit `name =` still wins.
  out <- nest_join(tbl_gpu(f06g_x), tbl_gpu(f06g_y), by = "k", name = "matches")
  expect_true("matches" %in% names(out))
})

test_that("joins routed to the CPU fallback keep x's groups and .drop", {
  skip_if_no_gpu()
  # Added after round 1 (review finding N4): factor keys with different level
  # sets go to the CPU fallback; the result must still carry x's grouping and
  # .drop = FALSE, like the native path.
  df1 <- data.frame(f = factor(c(1, 1, 2, 2), levels = 1:3), x = c(1, 2, 1, 4))
  df2 <- data.frame(f = factor(c(2, 2, 3, 3), levels = 1:4), y = c(1, 2, 3, 4))
  for (nm in names(f06g_all)) {
    join <- f06g_all[[nm]]
    run <- function(x, y) {
      args <- list(group_by(x, f, .drop = FALSE), y, by = "f")
      if (!nm %in% c("semi_join", "anti_join")) args$relationship <- "many-to-many"
      do.call(join, args)
    }
    oracle <- run(tibble::as_tibble(df1), tibble::as_tibble(df2))
    for (lazy in c(FALSE, TRUE)) {
      out <- run(tbl_gpu(df1, lazy = lazy), tbl_gpu(df2, lazy = lazy))
      info <- sprintf("%s lazy=%s", nm, lazy)
      expect_identical(group_vars(out), group_vars(oracle), info = info)
      expect_identical(group_by_drop_default(out), group_by_drop_default(oracle), info = info)
      expect_equal(group_size(out), group_size(oracle), info = info)
    }
  }
})
