# F80: printing a lazy gibble (Theo, 2026-10-27).
#
# - A lazy table with pending ops: print() runs the plan and shows exactly what
#   the eager print of the same pipeline shows, except that the first header
#   line reads "# A gibble [lazy]: ..." instead of "# A gibble: ...". The row
#   count is the real one (the whole plan ran).
# - Nothing is cached: the object is unchanged, still lazy with its plan,
#   holds no GPU memory after print() returns, and a second print runs the
#   plan again.
# - A lazy table with nothing pending prints exactly as an eager one.
# - If the plan fails (an error, out of memory), print() does not error: it
#   shows a "[lazy]" header, the schema, and the condition message.
# - glimpse() the same: the eager glimpse plus one "Lazy: ..." line.

f80p_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, NA),
    x = c(1.5, NA, 3, 4.25, -1, 2),
    s = c("a", "b", NA, "d", "e", "f"),
    dt = as.Date("2024-01-01") + 0:5
  )
}

f80p_local_default <- function(env = parent.frame()) {
  withr::local_options(cuplyr.exec_mode = NULL, .local_envir = env)
  withr::local_envvar(CUPLYR_EXEC_MODE = NA, .local_envir = env)
}

f80p_print <- function(x, ...) utils::capture.output(print(x, ...))
f80p_glimpse <- function(x, ...) utils::capture.output(dplyr::glimpse(x, ...))

# The lazy print of `pipeline` must be the eager print with "[lazy]" in the
# first line.
f80p_expect_lazy_print <- function(df, pipeline, label) {
  lz <- pipeline(tbl_gpu(df))
  expect_true(has_pending_ops(lz), info = label)
  eg <- pipeline(tbl_gpu(df, lazy = FALSE))
  out_lazy <- f80p_print(lz)
  out_eager <- f80p_print(eg)
  expect_match(out_eager[1], "^# A gibble: ", info = label)
  expected <- out_eager
  expected[1] <- sub("^# A gibble: ", "# A gibble [lazy]: ", expected[1])
  expect_identical(out_lazy, expected, info = label)
}

test_that("printing a lazy table with pending ops previews its rows like the eager print", {
  skip_if_no_gpu()
  f80p_local_default()
  df <- f80p_df()
  pipelines <- list(
    filter_mutate = function(d) d |> dplyr::filter(x > 0) |> dplyr::mutate(y = x * 2),
    grouped       = function(d) d |> dplyr::mutate(y = x + 1) |> dplyr::group_by(g),
    summarise     = function(d) d |> dplyr::group_by(g) |> dplyr::summarise(m = mean(x, na.rm = TRUE)),
    zero_rows     = function(d) d |> dplyr::filter(x > 100),
    join          = function(d) dplyr::left_join(d, dplyr::count(d, g), by = "g"),
    rowwise       = function(d) d |> dplyr::rowwise(g) |> dplyr::mutate(y = x + 1),
    arrange_slice = function(d) d |> dplyr::arrange(dplyr::desc(dt)) |> dplyr::slice_head(n = 3)
  )
  for (nm in names(pipelines)) {
    f80p_expect_lazy_print(df, pipelines[[nm]], nm)
  }
})

test_that("the lazy header carries the real row count and the group count", {
  skip_if_no_gpu()
  f80p_local_default()
  df <- f80p_df()
  out <- f80p_print(tbl_gpu(df) |> dplyr::filter(x > 0) |> dplyr::group_by(g))
  expect_match(out[1], "^# A gibble \\[lazy\\]: 4 . 4$")
  expect_true(any(grepl("^# Groups:\\s+g \\[3\\]$", out)))
})

test_that("a lazy table with nothing pending prints exactly as an eager one", {
  skip_if_no_gpu()
  f80p_local_default()
  df <- f80p_df()
  expect_identical(f80p_print(tbl_gpu(df)), f80p_print(tbl_gpu(df, lazy = FALSE)))
  expect_identical(f80p_print(dplyr::group_by(tbl_gpu(df), g)),
                   f80p_print(dplyr::group_by(tbl_gpu(df, lazy = FALSE), g)))
  done <- dplyr::compute(dplyr::filter(tbl_gpu(df), x > 0))
  expect_identical(f80p_print(done),
                   f80p_print(dplyr::filter(tbl_gpu(df, lazy = FALSE), x > 0)))
})

test_that("print() caches nothing: the table is unchanged and a second print runs the plan again", {
  skip_if_no_gpu()
  f80p_local_default()
  df <- f80p_df()
  lz <- tbl_gpu(df) |> dplyr::filter(x > 0) |> dplyr::mutate(y = x * 2)
  before <- unclass(lz)
  runs <- 0L
  real <- cuplyr:::lower_and_execute
  testthat::local_mocked_bindings(
    lower_and_execute = function(...) {
      runs <<- runs + 1L
      real(...)
    },
    .package = "cuplyr"
  )
  res <- withVisible(print(lz))
  expect_false(res$visible)
  expect_identical(unclass(res$value), before)
  n1 <- runs
  expect_gt(n1, 0L)
  # one print runs the plan exactly once: as many lowering steps as one collect()
  runs <- 0L
  collect(lz)
  expect_identical(runs, n1)
  runs <- 0L
  utils::capture.output(print(lz))
  utils::capture.output(print(lz))
  expect_identical(runs, 2L * n1)
  expect_identical(unclass(lz), before)
  expect_true(has_pending_ops(lz))
  expect_identical(.subset2(lz, "exec_mode"), "lazy")
})

test_that("print() and glimpse() of a lazy table hold no GPU memory once they return", {
  skip_if_no_gpu()
  f80p_local_default()
  n <- 1e6
  big <- tibble::tibble(a = as.double(seq_len(n)), b = rep(c(1L, 2L), n / 2))
  lz <- tbl_gpu(big) |> dplyr::filter(b == 1L) |> dplyr::mutate(c = a * 2, d = a + 1)
  gc()
  base <- gpu_alloc_used_bytes()
  utils::capture.output(print(lz))
  expect_lte(gpu_alloc_used_bytes() - base, 256 * 1024)
  utils::capture.output(dplyr::glimpse(lz))
  expect_lte(gpu_alloc_used_bytes() - base, 256 * 1024)
  expect_true(has_pending_ops(lz))
})

test_that("a plan that fails prints the schema and the error instead of erroring", {
  skip_if_no_gpu()
  f80p_local_default()
  df <- f80p_df()
  lz <- tbl_gpu(df) |> dplyr::filter(x > 0) |> dplyr::mutate(y = x * 2)
  testthat::local_mocked_bindings(
    lower_and_execute = function(...) stop("f80 plan failure"),
    .package = "cuplyr"
  )
  out <- NULL
  expect_no_error(out <- f80p_print(lz))
  expect_match(out[1], "^# A gibble \\[lazy\\]")
  for (col in c("g", "x", "s", "dt", "y")) {
    expect_true(any(grepl(paste0("\\b", col, "\\b"), out[-1])), info = col)
  }
  expect_true(any(grepl("f80 plan failure", out, fixed = TRUE)))
  out_g <- NULL
  expect_no_error(out_g <- f80p_glimpse(lz))
  expect_true(any(grepl("f80 plan failure", out_g, fixed = TRUE)))
  expect_true(any(grepl("^Columns: 5$", out_g)))
  # a grouped table keeps its groups line
  out_grp <- f80p_print(dplyr::group_by(lz, g))
  expect_true(any(grepl("^# Groups:\\s+g", out_grp)))
})

test_that("a plan that runs out of GPU memory prints instead of erroring", {
  skip_if_no_gpu()
  f80p_local_default()
  df <- f80p_df()
  lz <- tbl_gpu(df) |> dplyr::filter(x > 0) |> dplyr::mutate(y = x * 2)
  withr::defer(gpu_oom_inject(0L))
  gpu_oom_inject(100000L)
  out <- NULL
  expect_no_error(out <- f80p_print(lz))
  gpu_oom_inject(0L)
  expect_match(out[1], "^# A gibble \\[lazy\\]")
  expect_true(any(grepl("memory", out, ignore.case = TRUE)))
  # the table still works afterwards
  expect_equal(
    tibble::as_tibble(collect(lz)),
    df |> dplyr::filter(x > 0) |> dplyr::mutate(y = x * 2)
  )
})

test_that("glimpse() of a lazy table is the eager glimpse plus one Lazy line, and caches nothing", {
  skip_if_no_gpu()
  f80p_local_default()
  df <- f80p_df()
  pipelines <- list(
    filter_mutate = function(d) d |> dplyr::filter(x > 0) |> dplyr::mutate(y = x * 2),
    grouped       = function(d) d |> dplyr::mutate(y = x + 1) |> dplyr::group_by(g),
    zero_rows     = function(d) d |> dplyr::filter(x > 100)
  )
  for (nm in names(pipelines)) {
    lz <- pipelines[[nm]](tbl_gpu(df))
    out_lazy <- f80p_glimpse(lz)
    out_eager <- f80p_glimpse(pipelines[[nm]](tbl_gpu(df, lazy = FALSE)))
    is_lazy_line <- grepl("^Lazy: ", out_lazy)
    expect_identical(sum(is_lazy_line), 1L, info = nm)
    expect_identical(out_lazy[!is_lazy_line], out_eager, info = nm)
    expect_true(has_pending_ops(lz), info = nm)
  }
  # nothing pending: exactly the eager glimpse
  expect_identical(f80p_glimpse(tbl_gpu(df)), f80p_glimpse(tbl_gpu(df, lazy = FALSE)))
  # glimpse() returns its input invisibly, unchanged
  lz <- pipelines$filter_mutate(tbl_gpu(df))
  before <- unclass(lz)
  utils::capture.output(v <- dplyr::glimpse(lz))
  expect_identical(unclass(v), before)
})
