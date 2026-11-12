# F16 acceptance tests: filter(.preserve = TRUE) and filter_out(.preserve = TRUE).
#
# On a group_by() table, `.preserve = TRUE` keeps the input's full group-keys
# table (emptied groups get `.rows = integer()`), the way dplyr's grouped_df
# does; F15's `$fixed_keys` carries them. `.preserve = FALSE` (the default,
# including the no-op `filter(TRUE)` and zero-dot `filter()`) recomputes the
# groups from the rows, dropping empty ones.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

f16p_df <- function() {
  tibble::tibble(
    g = c(1, 2, 2, 3, NA, 3),
    h = c("a", "a", "b", "b", "a", "b"),
    x = c(1, NA, 3, 4, 5, 6)
  )
}

f16p_inputs <- function(df) {
  list(eager = tbl_gpu(df, lazy = FALSE), lazy = as_lazy(tbl_gpu(df)))
}

# Compare the full grouping structure of `pipeline` run on a tbl_gpu (eager
# and lazy) with dplyr on a tibble: the tbl_gpu's group metadata methods, the
# collected grouped_df's metadata, and the data.
f16p_groups_parity <- function(df, pipeline) {
  cpu <- pipeline(tibble::as_tibble(df))
  cpu_rows <- as.list(dplyr::group_rows(cpu))
  for (mode in c("eager", "lazy")) {
    gpu <- pipeline(f16p_inputs(df)[[mode]])
    expect_true(is_tbl_gpu(gpu), info = mode)
    expect_identical(dplyr::group_vars(gpu), dplyr::group_vars(cpu), info = mode)
    expect_equal(dplyr::group_keys(gpu), dplyr::group_keys(cpu),
                 info = paste(mode, "group_keys(tbl_gpu)"))
    expect_equal(as.list(dplyr::group_rows(gpu)), cpu_rows,
                 info = paste(mode, "group_rows(tbl_gpu)"))
    expect_identical(dplyr::group_size(gpu), dplyr::group_size(cpu),
                     info = paste(mode, "group_size(tbl_gpu)"))
    expect_identical(dplyr::n_groups(gpu), dplyr::n_groups(cpu),
                     info = paste(mode, "n_groups(tbl_gpu)"))
    expect_identical(dplyr::group_indices(gpu), dplyr::group_indices(cpu),
                     info = paste(mode, "group_indices(tbl_gpu)"))

    col <- collect(gpu)
    expect_s3_class(col, "grouped_df")
    expect_equal(dplyr::group_keys(col), dplyr::group_keys(cpu),
                 info = paste(mode, "group_keys(collect())"))
    expect_equal(as.list(dplyr::group_rows(col)), cpu_rows,
                 info = paste(mode, "group_rows(collect())"))
    expect_equal(tibble::as_tibble(dplyr::ungroup(col)),
                 tibble::as_tibble(dplyr::ungroup(cpu)),
                 info = paste(mode, "data"))
  }
}

f16p_parity <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

my_gt3 <- function(v) v > 3

# ---------------------------------------------------------------------------
# .preserve = TRUE keeps emptied groups
# ---------------------------------------------------------------------------

test_that("filter(.preserve = TRUE) keeps emptied groups (single key, NA key)", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_groups_parity(df, function(d) filter(group_by(d, g), x > 3, .preserve = TRUE))
  f16p_groups_parity(df, function(d) filter(group_by(d, g), x > 100, .preserve = TRUE))
  f16p_groups_parity(df, function(d) filter(group_by(d, h), x < 4, .preserve = TRUE))
})

test_that("filter_out(.preserve = TRUE) keeps emptied groups", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g), x > 3, .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g), g == 2, .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, h), h == "a", .preserve = TRUE))
})

test_that(".preserve = TRUE with multiple keys", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_groups_parity(df, function(d) filter(group_by(d, g, h), x > 3, .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g, h), x > 3, .preserve = TRUE))
})

test_that(".preserve = TRUE on the window-predicate path", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_groups_parity(df, function(d) {
    filter(group_by(d, g), x > mean(x, na.rm = TRUE), .preserve = TRUE)
  })
  f16p_groups_parity(df, function(d) {
    dplyr::filter_out(group_by(d, g), dplyr::n() > 1, .preserve = TRUE)
  })
})

test_that(".preserve = TRUE on the CPU-fallback path", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_groups_parity(df, function(d) filter(group_by(d, g), my_gt3(x), .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g), my_gt3(x), .preserve = TRUE))
})

test_that(".preserve = TRUE with constant and empty conditions", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_groups_parity(df, function(d) filter(group_by(d, g), FALSE, .preserve = TRUE))
  f16p_groups_parity(df, function(d) filter(group_by(d, g), NA, .preserve = TRUE))
  f16p_groups_parity(df, function(d) filter(group_by(d, g), TRUE, .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g), TRUE, .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g), FALSE, .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g), .preserve = TRUE))
})

test_that("the conformance indices case: filter(FALSE) / filter_out(TRUE) on mtcars", {
  skip_if_no_gpu()
  f16p_groups_parity(mtcars, function(d) filter(group_by(d, cyl), FALSE, .preserve = TRUE))
  f16p_groups_parity(mtcars, function(d) dplyr::filter_out(group_by(d, cyl), TRUE, .preserve = TRUE))
  f16p_groups_parity(mtcars, function(d) filter(group_by(d, cyl), FALSE, .preserve = FALSE))
  f16p_groups_parity(mtcars, function(d) dplyr::filter_out(group_by(d, cyl), TRUE, .preserve = FALSE))
})

# ---------------------------------------------------------------------------
# What carries the preserved keys on, and what drops them
# ---------------------------------------------------------------------------

test_that("preserved keys survive column verbs and chained .preserve = TRUE filters", {
  skip_if_no_gpu()
  df <- f16p_df()
  pf <- function(d) filter(group_by(d, g), x > 3, .preserve = TRUE)
  f16p_groups_parity(df, function(d) mutate(pf(d), z = x * 2))
  f16p_groups_parity(df, function(d) select(pf(d), g, x))
  f16p_groups_parity(df, function(d) filter(pf(d), x > 4, .preserve = TRUE))
  f16p_groups_parity(df, function(d) dplyr::filter_out(pf(d), x > 5, .preserve = TRUE))
  # keys fixed by slice(.preserve = TRUE) are kept by a preserving filter
  f16p_groups_parity(df, function(d) {
    filter(slice(group_by(d, g), 0, .preserve = TRUE), TRUE, .preserve = TRUE)
  })
})

test_that(".preserve = FALSE drops empty groups, also after an earlier .preserve = TRUE", {
  skip_if_no_gpu()
  df <- f16p_df()
  pf <- function(d) filter(group_by(d, g), x > 3, .preserve = TRUE)
  f16p_groups_parity(df, function(d) filter(group_by(d, g), x > 3))
  f16p_groups_parity(df, function(d) dplyr::filter_out(group_by(d, g), x > 3))
  f16p_groups_parity(df, function(d) filter(pf(d), x > 0))
  f16p_groups_parity(df, function(d) dplyr::filter_out(pf(d), x > 100))
  # the no-op forms still recompute the groups
  f16p_groups_parity(df, function(d) filter(pf(d), TRUE))
  f16p_groups_parity(df, function(d) filter(pf(d)))
  f16p_groups_parity(df, function(d) dplyr::filter_out(pf(d), FALSE))
  f16p_groups_parity(df, function(d) filter(slice(group_by(d, g), 1, .preserve = TRUE), TRUE))
})

test_that("summarise() after filter(.preserve = TRUE) returns the empty groups", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_parity(df, function(d) {
    summarise(filter(group_by(d, g), x > 3, .preserve = TRUE), n = dplyr::n(), s = sum(x))
  }, arrange_by = "g", ignore_col_types = TRUE)
  f16p_parity(df, function(d) {
    summarise(dplyr::filter_out(group_by(d, g), x > 2, .preserve = TRUE), n = dplyr::n())
  }, arrange_by = "g", ignore_col_types = TRUE)
})

test_that(".preserve = TRUE is a plain filter on ungrouped input", {
  skip_if_no_gpu()
  df <- f16p_df()
  f16p_parity(df, function(d) filter(d, x > 3, .preserve = TRUE))
  f16p_parity(df, function(d) dplyr::filter_out(d, x > 3, .preserve = TRUE))
})

test_that(".drop = FALSE factor groups keep their unobserved levels either way", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    e = 1,
    f = factor(c(1, 1, 2, 2), levels = 1:3),
    g = c(1, 1, 2, 2),
    x = c(1, 2, 1, 4)
  )
  gd <- function(d) group_by(d, e, f, g, .drop = FALSE)
  f16p_groups_parity(df, function(d) filter(gd(d), f == 1))
  f16p_groups_parity(df, function(d) dplyr::filter_out(gd(d), f == 1))
  f16p_groups_parity(df, function(d) dplyr::filter_out(gd(d), f == 1, .preserve = TRUE))
  f16p_parity(df, function(d) dplyr::ungroup(dplyr::count(dplyr::filter_out(gd(d), f == 1))),
              ignore_col_types = TRUE)
})
