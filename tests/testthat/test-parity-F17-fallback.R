# F17 acceptance tests: CPU-fallback wrapper fixes.
#
# 1. Extra `...` arguments of group_modify()/group_map()/group_walk() reach
#    `.f` (regression cover; already matching dplyr on dev).
# 2. Preserved group keys (`$fixed_keys`, set by filter/slice(.preserve =
#    TRUE)) survive a verb that runs through the CPU fallback exactly when
#    dplyr's grouped_df keeps the empty groups, and are dropped when dplyr
#    drops them.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

f17_df <- function() {
  tibble::tibble(
    g = c(1, 2, 2, 3, NA, 3),
    h = c("a", "a", "b", "b", "a", "b"),
    x = c(1, NA, 3, 4, 5, 6)
  )
}

f17_inputs <- function(df) {
  list(eager = tbl_gpu(df), lazy = as_lazy(tbl_gpu(df)))
}

# An R function the GPU expression IR doesn't know, so any verb using it runs
# through the CPU fallback.
f17_twice <- function(v) v * 2

# Full grouping structure of `pipeline` on a tbl_gpu (eager and lazy) against
# dplyr on a tibble: the tbl_gpu's group metadata, the collected grouped_df's
# metadata, and the data.
f17_groups_parity <- function(df, pipeline) {
  cpu <- pipeline(tibble::as_tibble(df))
  cpu_rows <- as.list(dplyr::group_rows(cpu))
  for (mode in c("eager", "lazy")) {
    gpu <- pipeline(f17_inputs(df)[[mode]])
    expect_true(is_tbl_gpu(gpu), info = mode)
    expect_identical(dplyr::group_vars(gpu), dplyr::group_vars(cpu), info = mode)
    expect_equal(dplyr::group_keys(gpu), dplyr::group_keys(cpu),
                 info = paste(mode, "group_keys(tbl_gpu)"))
    expect_equal(as.list(dplyr::group_rows(gpu)), cpu_rows,
                 info = paste(mode, "group_rows(tbl_gpu)"))
    expect_identical(dplyr::n_groups(gpu), dplyr::n_groups(cpu),
                     info = paste(mode, "n_groups(tbl_gpu)"))
    expect_identical(dplyr::group_indices(gpu), dplyr::group_indices(cpu),
                     info = paste(mode, "group_indices(tbl_gpu)"))

    col <- collect(gpu)
    if (length(dplyr::group_vars(cpu)) > 0L) {
      expect_s3_class(col, "grouped_df")
    }
    expect_equal(dplyr::group_keys(col), dplyr::group_keys(cpu),
                 info = paste(mode, "group_keys(collect())"))
    expect_equal(as.list(dplyr::group_rows(col)), cpu_rows,
                 info = paste(mode, "group_rows(collect())"))
    expect_equal(tibble::as_tibble(dplyr::ungroup(col)),
                 tibble::as_tibble(dplyr::ungroup(cpu)),
                 info = paste(mode, "data"))
  }
}

f17_parity <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f17_keep_g <- function(d) filter(group_by(d, g), x > 3, .preserve = TRUE)

# ---------------------------------------------------------------------------
# 1. `...` forwarding (regression cover)
# ---------------------------------------------------------------------------

test_that("group_modify() forwards extra arguments to .f (#4509)", {
  skip_if_no_gpu()
  add_col <- function(.x, .y, foo) {
    .x[[foo]] <- 1
    .x
  }
  df <- tibble::tibble(A = rep(1:2, each = 3), x = 1:6)
  f17_parity(df, function(d) group_modify(group_by(d, A), add_col, "bar"))
  f17_parity(df, function(d) group_modify(group_by(d, A), add_col, foo = "bar"))
  f17_parity(df, function(d) group_modify(.data = group_by(d, A), .f = add_col, foo = "bar"))
  f17_parity(df, function(d) group_modify(d, add_col, foo = "bar"))
  # extra arguments named like the wrapper's own internals still reach .f
  set_z <- function(.x, .y, tbl, expr_fn) {
    .x$z <- tbl + expr_fn
    .x
  }
  f17_parity(df, function(d) group_modify(group_by(d, A), set_z, tbl = 1, expr_fn = 2))
  f17_parity(df, function(d) group_modify(group_by(d, A), ~ head(.x, ..3), 1L))
})

test_that("group_map() and group_walk() forward extra arguments to .f", {
  skip_if_no_gpu()
  df <- tibble::tibble(A = rep(1:2, each = 3), x = 1:6)
  f <- function(.x, .y, k, m = 0) nrow(.x) * k + m
  cpu <- dplyr::group_map(group_by(df, A), f, 10, m = 1)
  for (gpu in f17_inputs(df)) {
    expect_identical(dplyr::group_map(group_by(gpu, A), f, 10, m = 1), cpu)
    expect_identical(dplyr::group_map(group_by(gpu, A), ~ sum(.x$x) + ..3, 100L),
                     dplyr::group_map(group_by(df, A), ~ sum(.x$x) + ..3, 100L))

    seen <- integer()
    out <- withVisible(dplyr::group_walk(group_by(gpu, A), function(.x, .y, k) {
      seen <<- c(seen, nrow(.x) + k)
    }, 5L))
    expect_identical(seen, c(8L, 8L))
    expect_false(out$visible)
    expect_true(is_tbl_gpu(out$value))
    expect_identical(dplyr::group_vars(out$value), "A")
  }
})

# ---------------------------------------------------------------------------
# 2. Preserved keys through the CPU fallback
# ---------------------------------------------------------------------------

test_that("a fallback mutate() keeps the groups filter(.preserve = TRUE) emptied", {
  skip_if_no_gpu()
  df <- f17_df()
  f17_groups_parity(df, function(d) mutate(f17_keep_g(d), y = f17_twice(x)))
  f17_groups_parity(df, function(d) mutate(f17_keep_g(d), y = f17_twice(x), .keep = "none"))
  f17_groups_parity(df, function(d) mutate(f17_keep_g(d), y = cumsum(f17_twice(x))))
  f17_groups_parity(df, function(d) transmute(f17_keep_g(d), y = f17_twice(x)))
  f17_groups_parity(df, function(d) {
    filter(group_by(d, g, h), x > 3, .preserve = TRUE) |> mutate(y = f17_twice(x))
  })
})

test_that("a fallback mutate() keeps the groups slice(.preserve = TRUE) emptied", {
  skip_if_no_gpu()
  df <- f17_df()
  f17_groups_parity(df, function(d) {
    slice(group_by(d, g), 2, .preserve = TRUE) |> mutate(y = f17_twice(x))
  })
  # every group emptied
  f17_groups_parity(df, function(d) {
    slice(group_by(d, g), 5, .preserve = TRUE) |> mutate(y = f17_twice(x))
  })
})

test_that("keys kept through a fallback flow into later verbs", {
  skip_if_no_gpu()
  df <- f17_df()
  # empty groups summarise to n() = 0
  f17_parity(df, function(d) {
    mutate(f17_keep_g(d), y = f17_twice(x)) |> summarise(n = n(), s = sum(y))
  })
  # then a GPU-native column verb keeps them
  f17_groups_parity(df, function(d) {
    mutate(f17_keep_g(d), y = f17_twice(x)) |> mutate(z = x + 1) |> select(g, z)
  })
  # two fallbacks in a row
  f17_groups_parity(df, function(d) {
    mutate(f17_keep_g(d), y = f17_twice(x)) |> mutate(z = f17_twice(y))
  })
  # group_map() sees the empty groups too
  for (gpu in f17_inputs(df)) {
    expect_identical(
      dplyr::group_map(mutate(f17_keep_g(gpu), y = f17_twice(x)), ~ nrow(.x)),
      dplyr::group_map(mutate(f17_keep_g(df), y = f17_twice(x)), ~ nrow(.x))
    )
  }
})

test_that("fallback verbs that regroup drop the preserved keys, as dplyr does", {
  skip_if_no_gpu()
  df <- f17_df()
  # writing a group column regroups from the data
  f17_groups_parity(df, function(d) mutate(f17_keep_g(d), g = f17_twice(g)))
  # arrange() regroups
  f17_groups_parity(df, function(d) arrange(f17_keep_g(d), f17_twice(x)))
  # group_modify() rebuilds the groups from its output
  f17_groups_parity(df, function(d) group_modify(f17_keep_g(d), ~ .x))
  # rows_append() regroups
  extra <- tibble::tibble(g = 1, h = "z", x = 10)
  f17_groups_parity(df, function(d) rows_append(f17_keep_g(d), extra))
  # filter(.preserve = FALSE) after the fallback drops them
  f17_groups_parity(df, function(d) {
    mutate(f17_keep_g(d), y = f17_twice(x)) |> filter(x > 0)
  })
})

test_that("fallbacks without preserved keys keep their grouping unchanged", {
  skip_if_no_gpu()
  df <- f17_df()
  f17_groups_parity(df, function(d) mutate(group_by(d, g), y = f17_twice(x)))
  f17_groups_parity(df, function(d) mutate(d, y = f17_twice(x)))
  # `.drop = FALSE` factor groups: the empty levels come from the grouping,
  # not from fixed keys
  fdf <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")),
                        x = c(1, 2, 3))
  f17_groups_parity(fdf, function(d) {
    mutate(group_by(d, f, .drop = FALSE), y = f17_twice(x))
  })
  f17_groups_parity(fdf, function(d) {
    filter(group_by(d, f, .drop = FALSE), x > 2, .preserve = TRUE) |>
      mutate(y = f17_twice(x))
  })
  # (summarise() of `.drop = FALSE` empty groups is F33's, not tested here)
})

test_that("reframe() on a table with preserved keys sees the empty groups", {
  skip_if_no_gpu()
  df <- f17_df()
  f17_parity(df, function(d) reframe(mutate(f17_keep_g(d), y = f17_twice(x)), n = n()))
})
