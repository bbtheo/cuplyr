# F15 acceptance tests: slice family parity.
#
# 1. slice(.preserve = TRUE) on a grouped table keeps the input's group keys
#    (empty groups included), the way dplyr's grouped_df does.
# 2. slice_min()/slice_max() with a data-frame order_by (tibble(a, b),
#    pick(a, b)) order lexicographically, NA last per column.
# 3. n=/prop= validation matches dplyr's check_slice_n_prop() (class,
#    message, call), eagerly, also for lazy tables.
# 4. slice_sample() on a zero-column table keeps an R-side row count.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

f15_df <- function() {
  tibble::tibble(g = c(1, 2, 2, 3, 3, 3), id = 1:6)
}

f15_inputs <- function(df) {
  list(eager = tbl_gpu(df), lazy = as_lazy(tbl_gpu(df)))
}

# Compare the full grouping structure of `pipeline` run on a tbl_gpu (eager
# and lazy) with dplyr on a tibble: the tbl_gpu's own group metadata methods,
# the collected grouped_df's metadata, and the data itself.
f15_groups_parity <- function(df, pipeline) {
  cpu <- pipeline(tibble::as_tibble(df))
  cpu_rows <- as.list(dplyr::group_rows(cpu))
  for (mode in c("eager", "lazy")) {
    gpu <- pipeline(f15_inputs(df)[[mode]])
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

# Both sides must fail with an rlang_error carrying the same message and the
# same calling verb. `gpu` and `cpu` are unevaluated (lazy promises).
f15_expect_same_error <- function(gpu, cpu) {
  c <- rlang::catch_cnd(cpu, "error")
  g <- rlang::catch_cnd(gpu, "error")
  expect_s3_class(c, "rlang_error")
  expect_s3_class(g, "rlang_error")
  if (!is.null(g) && !is.null(c)) {
    expect_identical(conditionMessage(g), conditionMessage(c))
    call_name <- function(x) if (rlang::is_call(x$call)) rlang::call_name(x$call) else NULL
    expect_identical(call_name(g), call_name(c))
  }
  invisible(g)
}

f15_parity <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# ---------------------------------------------------------------------------
# 1. slice(.preserve = TRUE) keeps the input's group keys
# ---------------------------------------------------------------------------

test_that("slice(.preserve = TRUE) keeps emptied groups (single key)", {
  skip_if_no_gpu()
  df <- f15_df()
  f15_groups_parity(df, function(d) slice(group_by(d, g), 2, 3, .preserve = TRUE))
  f15_groups_parity(df, function(d) slice(group_by(d, g), -1, .preserve = TRUE))
  f15_groups_parity(df, function(d) slice(group_by(d, g), 0, .preserve = TRUE))
})

test_that("slice(.preserve = FALSE) still drops emptied groups", {
  skip_if_no_gpu()
  df <- f15_df()
  f15_groups_parity(df, function(d) slice(group_by(d, g), 2, 3))
  f15_groups_parity(df, function(d) slice(group_by(d, g), 2, 3, .preserve = FALSE))
})

test_that("slice(.preserve = TRUE) keeps emptied groups (multiple keys, NA key)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 2, 2, 3, 3, 3), h = c(1, 1, 2, 2, 3, 3), id = 1:6)
  f15_groups_parity(df, function(d) slice(group_by(d, g, h), 2, .preserve = TRUE))

  df_na <- tibble::tibble(g = c(NA, 1, 1, NA, 2), id = 1:5)
  f15_groups_parity(df_na, function(d) slice(group_by(d, g), 2, .preserve = TRUE))
})

test_that("slice(.preserve = TRUE) keeps emptied groups on the CPU-fallback path", {
  skip_if_no_gpu()
  df <- f15_df()
  # which(id > 3) needs the data mask, so this index expression falls back.
  f15_groups_parity(df, function(d) slice(group_by(d, g), which(id > 3), .preserve = TRUE))
})

test_that("slice(.preserve = TRUE) on an ungrouped table is a plain slice", {
  skip_if_no_gpu()
  df <- f15_df()
  f15_parity(df, function(d) slice(d, 2, 3, .preserve = TRUE))
})

test_that("preserved empty groups survive column verbs and are dropped by row verbs", {
  skip_if_no_gpu()
  df <- f15_df()
  sl <- function(d) slice(group_by(d, g), 2, 3, .preserve = TRUE)
  # kept, as in dplyr
  f15_groups_parity(df, function(d) mutate(sl(d), z = id * 2L))
  f15_groups_parity(df, function(d) rename(sl(d), ID = id))
  # dropped, as in dplyr
  f15_groups_parity(df, function(d) arrange(sl(d), desc(id)))
  f15_groups_parity(df, function(d) filter(sl(d), id > 0L))
})

test_that("summarise() after slice(.preserve = TRUE) returns the empty groups", {
  skip_if_no_gpu()
  df <- f15_df()
  pipe <- function(d) {
    summarise(slice(group_by(d, g), 2, 3, .preserve = TRUE), n = n(), s = sum(id))
  }
  f15_parity(df, pipe, arrange_by = "g", ignore_col_types = TRUE)
})

# ---------------------------------------------------------------------------
# 2. slice_min()/slice_max() with a data-frame order_by
# ---------------------------------------------------------------------------

f15_ob_df <- function() {
  tibble::tibble(
    id = 1:8,
    g = c(1, 1, 1, 2, 2, 2, 2, 1),
    x = c(1, NA, 1, 2, NA, 1, 2, 1),
    y = c(NA, 1, 2, 1, NA, 2, 3, 2)
  )
}

test_that("slice_min/max(order_by = tibble(...)) order lexicographically, NA last", {
  skip_if_no_gpu()
  df <- f15_ob_df()
  f15_parity(df, function(d) slice_min(d, tibble::tibble(x, y), n = 2))
  f15_parity(df, function(d) slice_max(d, tibble::tibble(x, y), n = 2))
  f15_parity(df, function(d) slice_min(d, tibble::tibble(x, y), n = 8))
  f15_parity(df, function(d) slice_max(d, tibble::tibble(x, y), n = 8))
  f15_parity(df, function(d) slice_min(d, tibble::tibble(x, y), n = 3, with_ties = FALSE))
  f15_parity(df, function(d) slice_max(d, tibble::tibble(x, y), n = 3, with_ties = FALSE))
  f15_parity(df, function(d) slice_min(d, tibble::tibble(x, y), n = 8, na_rm = TRUE))
  f15_parity(df, function(d) slice_max(d, tibble::tibble(x, y), n = 8, na_rm = TRUE))
  f15_parity(df, function(d) slice_min(d, tibble::tibble(x, y), prop = 0.5))
  f15_parity(df, function(d) slice_max(d, tibble::tibble(x, y), n = -2))
})

test_that("slice_min/max(order_by = pick(...)) match dplyr", {
  skip_if_no_gpu()
  df <- f15_ob_df()
  f15_parity(df, function(d) slice_min(d, dplyr::pick(x, y), n = 3))
  f15_parity(df, function(d) slice_max(d, dplyr::pick(y, x), n = 3))
  # the conformance case (#6176)
  d6176 <- tibble::tibble(id = 1:4, x = 1, y = c(1, 4, 2, 3))
  f15_parity(d6176, function(d) slice_min(d, tibble::tibble(x, y), n = 1))
  f15_parity(d6176, function(d) slice_max(d, tibble::tibble(x, y), n = 1))
})

test_that("data-frame order_by works grouped, with by=, and on 0 rows", {
  skip_if_no_gpu()
  df <- f15_ob_df()
  f15_parity(df, function(d) slice_min(group_by(d, g), tibble::tibble(x, y), n = 2))
  f15_parity(df, function(d) slice_max(group_by(d, g), dplyr::pick(x, y), n = 1, with_ties = FALSE))
  f15_parity(df, function(d) slice_min(d, tibble::tibble(x, y), n = 1, by = g))
  f15_parity(df, function(d) slice_max(d, dplyr::pick(x, y), n = 2, by = g, na_rm = TRUE))
  f15_parity(df[0, ], function(d) slice_min(d, tibble::tibble(x, y), n = 1))
  f15_parity(df[0, ], function(d) slice_max(group_by(d, g), tibble::tibble(x, y), n = 1))
})

# ---------------------------------------------------------------------------
# 3. n= / prop= validation
# ---------------------------------------------------------------------------

test_that("non-whole n errors like dplyr in every slice helper (eager and lazy)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5, g = c(1, 1, 2, 2, 2))
  for (mode in c("eager", "lazy")) {
    gd <- f15_inputs(df)[[mode]]
    f15_expect_same_error(slice_head(gd, n = 1.1), slice_head(df, n = 1.1))
    f15_expect_same_error(slice_tail(gd, n = 1.1), slice_tail(df, n = 1.1))
    f15_expect_same_error(slice_min(gd, x, n = 1.5), slice_min(df, x, n = 1.5))
    f15_expect_same_error(slice_max(gd, x, n = 1.5), slice_max(df, x, n = 1.5))
    f15_expect_same_error(slice_sample(gd, n = 1.5), slice_sample(df, n = 1.5))
    f15_expect_same_error(slice_head(gd, n = -1.5), slice_head(df, n = -1.5))
    # grouped and by= inputs validate too
    f15_expect_same_error(slice_head(group_by(gd, g), n = 1.1),
                          slice_head(group_by(df, g), n = 1.1))
    f15_expect_same_error(slice_tail(gd, n = 2.5, by = g),
                          slice_tail(df, n = 2.5, by = g))
    f15_expect_same_error(slice_min(group_by(gd, g), x, n = 0.5),
                          slice_min(group_by(df, g), x, n = 0.5))
  }
})

test_that("non-whole n errors even on a 0-row table", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = integer())
  for (mode in c("eager", "lazy")) {
    gd <- f15_inputs(df)[[mode]]
    f15_expect_same_error(slice_head(gd, n = 1.1), slice_head(df, n = 1.1))
  }
})

test_that("other invalid n/prop values error like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5, g = c(1, 1, 2, 2, 2))
  for (mode in c("eager", "lazy")) {
    gd <- f15_inputs(df)[[mode]]
    f15_expect_same_error(slice_head(gd, n = NA), slice_head(df, n = NA))
    f15_expect_same_error(slice_head(gd, n = NA_integer_), slice_head(df, n = NA_integer_))
    f15_expect_same_error(slice_head(gd, n = c(1, 2)), slice_head(df, n = c(1, 2)))
    f15_expect_same_error(slice_head(gd, n = TRUE), slice_head(df, n = TRUE))
    f15_expect_same_error(slice_tail(gd, n = "a"), slice_tail(df, n = "a"))
    f15_expect_same_error(slice_tail(gd, prop = "a"), slice_tail(df, prop = "a"))
    f15_expect_same_error(slice_head(gd, prop = NA), slice_head(df, prop = NA))
    f15_expect_same_error(slice_head(gd, n = 1, prop = 0.5),
                          slice_head(df, n = 1, prop = 0.5))
    f15_expect_same_error(slice_max(gd, x, n = 1, prop = 0.5),
                          slice_max(df, x, n = 1, prop = 0.5))
    f15_expect_same_error(slice_head(group_by(gd, g), n = 1, by = x),
                          slice_head(group_by(df, g), n = 1, by = x))
  }
})

test_that("valid whole, infinite and fractional-prop sizes still match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(3L, 1L, 2L, 5L, 4L), g = c(1, 1, 2, 2, 2))
  f15_parity(df, function(d) slice_head(d, n = 2L))
  f15_parity(df, function(d) slice_head(d, n = 2))
  f15_parity(df, function(d) slice_tail(d, n = Inf))
  f15_parity(df, function(d) slice_head(d, n = -Inf))
  f15_parity(df, function(d) slice_head(d, prop = Inf))
  f15_parity(df, function(d) slice_tail(d, prop = -0.5))
  f15_parity(df, function(d) slice_head(d, prop = 0.5))
  f15_parity(df, function(d) slice_min(d, x, n = 0))
  f15_parity(df, function(d) slice_max(d, x, prop = 0.7))
  f15_parity(df, function(d) slice_head(group_by(d, g), n = 1e9))
  f15_parity(df, function(d) slice_tail(d, prop = 0.5, by = g))
})

# ---------------------------------------------------------------------------
# 4. slice_sample() on a zero-column table
# ---------------------------------------------------------------------------

f15_zero_cols <- function(lazy = FALSE) {
  x <- tbl_gpu(tibble::tibble(x = 1:5, y = c(1, 2, 3, 4, 5)), lazy = lazy)
  select(x, -everything())
}

f15_expect_zero_col_rows <- function(x, n, mode) {
  expect_true(is_tbl_gpu(x), info = mode)
  expect_identical(dim(x), c(as.integer(n), 0L), info = mode)
  out <- collect(x)
  expect_identical(dim(out), c(as.integer(n), 0L), info = mode)
}

test_that("slice_sample() on a zero-column table gives dplyr's row count", {
  skip_if_no_gpu()
  ref <- tibble::tibble(.rows = 5)
  cases <- list(
    function(d) slice_sample(d, n = 2),
    function(d) slice_sample(d, prop = 0.6),
    function(d) slice_sample(d, n = 10),
    function(d) slice_sample(d, n = 8, replace = TRUE),
    function(d) slice_sample(d, n = -1),
    function(d) slice_sample(d)
  )
  for (lazy in c(FALSE, TRUE)) {
    mode <- if (lazy) "lazy" else "eager"
    for (f in cases) {
      want <- nrow(f(ref))
      got <- f(f15_zero_cols(lazy))
      f15_expect_zero_col_rows(got, want, mode)
      expect_identical(got$exec_mode, mode)
    }
  }
})

test_that("slice_sample() on a zero-column table draws from the RNG like dplyr", {
  skip_if_no_gpu()
  ref <- tibble::tibble(.rows = 5)
  set.seed(42)
  slice_sample(ref, n = 3)
  after_cpu <- stats::runif(1)

  set.seed(42)
  slice_sample(f15_zero_cols(), n = 3)
  after_gpu <- stats::runif(1)
  expect_identical(after_gpu, after_cpu)
})

test_that("slice_sample() on a lazy zero-column table counts pending rows", {
  skip_if_no_gpu()
  x <- tbl_gpu(tibble::tibble(x = 1:6), lazy = TRUE) |>
    filter(x > 2L) |>
    select(-everything())
  f15_expect_zero_col_rows(slice_sample(x, n = 10), 4L, "lazy")
  f15_expect_zero_col_rows(slice_sample(x, prop = 0.5), 2L, "lazy")
})

# ---------------------------------------------------------------------------
# Added after round 1 review (B1-B3)
# ---------------------------------------------------------------------------

test_that("zero-dot summarise() after slice(.preserve = TRUE) returns the empty groups", {
  skip_if_no_gpu()
  df <- f15_df()
  f15_parity(df, function(d) summarise(slice(group_by(d, g), 2, 3, .preserve = TRUE)),
             arrange_by = "g")
})

test_that("cur_group_id() sees the preserved group keys", {
  skip_if_no_gpu()
  df <- f15_df()
  sl <- function(d) slice(group_by(d, g), 2, 3, .preserve = TRUE)
  f15_parity(df, function(d) mutate(sl(d), gid = dplyr::cur_group_id()))
  f15_groups_parity(df, function(d) mutate(sl(d), gid = dplyr::cur_group_id()))
  f15_parity(df, function(d) filter(sl(d), dplyr::cur_group_id() == 2L))
})

test_that("head()/tail() accept a fractional n like the base generics", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5)
  f15_parity(df, function(d) head(d, 2.5))
  f15_parity(df, function(d) tail(d, 2.5))
  f15_parity(df, function(d) head(d, -1.5))
  f15_parity(df, function(d) tail(d, -1.5))
  f15_parity(df, function(d) head(d, nrow(df) / 2))
  f15_parity(df, function(d) head(d, -0.5))
  f15_parity(df, function(d) tail(d, -0.5))
  f15_parity(df, function(d) tail(d, -0.25))
})
