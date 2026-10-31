# F39: loud, classed errors for base generics that would otherwise see the
# tbl_gpu's internal field list, and dplyr's own error for a non-logical
# filter() condition (instead of a "device memory" message).

f39_df <- function() {
  tibble::tibble(x = c(1, 2, NA, 4), g = c("a", "b", "a", "b"))
}

f39_tables <- function() {
  df <- f39_df()
  list(
    eager = tbl_gpu(df, lazy = FALSE),
    lazy = tbl_gpu(df, lazy = TRUE),
    grouped = dplyr::group_by(tbl_gpu(df), g)
  )
}

# ---------------------------------------------------------------------------
# Base generics: split(), with(), subset(), t(), as.environment()
# ---------------------------------------------------------------------------

test_that("split() on a tbl_gpu raises a classed error pointing at group_split()", {
  skip_if_no_gpu()
  for (gt in f39_tables()) {
    err <- expect_error(split(gt, c(1, 1, 2, 2)), class = "cuplyr_error_unsupported_base")
    expect_match(conditionMessage(err), "split()", fixed = TRUE)
    expect_match(conditionMessage(err), "tbl_gpu", fixed = TRUE)
    expect_match(conditionMessage(err), "group_split()", fixed = TRUE)
    expect_match(conditionMessage(err), "collect()", fixed = TRUE)
  }
  # No warning about "data length is not a multiple of split variable".
  expect_no_warning(try(split(tbl_gpu(f39_df()), f39_df()$g), silent = TRUE))
})

test_that("with() on a tbl_gpu raises a classed error instead of 'object not found'", {
  skip_if_no_gpu()
  for (gt in f39_tables()) {
    err <- expect_error(with(gt, x + 1), class = "cuplyr_error_unsupported_base")
    expect_match(conditionMessage(err), "with()", fixed = TRUE)
    expect_match(conditionMessage(err), "collect()", fixed = TRUE)
    expect_no_match(conditionMessage(err), "not found", fixed = TRUE)
  }
})

test_that("subset() on a tbl_gpu raises a classed error pointing at filter()/select()", {
  skip_if_no_gpu()
  for (gt in f39_tables()) {
    err <- expect_error(subset(gt, x > 1), class = "cuplyr_error_unsupported_base")
    expect_match(conditionMessage(err), "subset()", fixed = TRUE)
    expect_match(conditionMessage(err), "filter()", fixed = TRUE)
  }
})

test_that("t() on a tbl_gpu raises a classed error instead of transposing the field list", {
  skip_if_no_gpu()
  for (gt in f39_tables()) {
    err <- expect_error(t(gt), class = "cuplyr_error_unsupported_base")
    expect_match(conditionMessage(err), "t()", fixed = TRUE)
    expect_match(conditionMessage(err), "collect()", fixed = TRUE)
  }
})

test_that("as.environment() on a tbl_gpu raises a classed error instead of exposing internals", {
  skip_if_no_gpu()
  for (gt in f39_tables()) {
    err <- expect_error(as.environment(gt), class = "cuplyr_error_unsupported_base")
    expect_match(conditionMessage(err), "as.environment()", fixed = TRUE)
  }
})

test_that("the guarded generics do not touch base behaviour on plain data frames", {
  skip_if_no_gpu()
  df <- f39_df()
  expect_equal(with(df, x + 1), df$x + 1)
  expect_equal(subset(df, x > 1), df[which(df$x > 1), ])
  expect_equal(split(df, df$g)$a, df[df$g == "a", ])
  expect_equal(t(df), t(as.matrix(df)))
  expect_setequal(ls(as.environment(df)), c("x", "g"))
})

# ---------------------------------------------------------------------------
# filter() with a non-logical condition: dplyr's own error
# ---------------------------------------------------------------------------

# Compare the error a pipeline raises on a tbl_gpu with dplyr's on a tibble:
# same class (rlang_error), same message (header bullets included, e.g.
# "In argument: `x + 1`." / "In group 1: ..."), same parent message.
f39_expect_same_filter_error <- function(df, pipeline, lazy = FALSE) {
  ref <- tryCatch(pipeline(df), error = identity)
  expect_s3_class(ref, "error")
  gpu_in <- tbl_gpu(df, lazy = lazy)
  got <- tryCatch(dplyr::collect(pipeline(gpu_in)), error = identity)
  expect_s3_class(got, "rlang_error")
  expect_identical(rlang::cnd_message(got), rlang::cnd_message(ref))
  expect_false(is.null(got$parent))
  expect_identical(conditionMessage(got$parent), conditionMessage(ref$parent))
  expect_no_match(rlang::cnd_message(got), "device memory", fixed = TRUE)
  invisible(got)
}

test_that("filter() with a double condition raises dplyr's logical-vector error (eager and lazy)", {
  skip_if_no_gpu()
  df <- f39_df()
  pipe <- function(d) dplyr::filter(d, x + 1)
  got <- f39_expect_same_filter_error(df, pipe)
  expect_match(conditionMessage(got$parent), "`..1` must be a logical vector, not a double vector.",
               fixed = TRUE)
  f39_expect_same_filter_error(df, pipe, lazy = TRUE)
})

test_that("filter() with a character column condition raises dplyr's error", {
  skip_if_no_gpu()
  df <- f39_df()
  pipe <- function(d) dplyr::filter(d, g)
  f39_expect_same_filter_error(df, pipe)
  f39_expect_same_filter_error(df, pipe, lazy = TRUE)
})

test_that("filter() reports the right dot index for a later non-logical condition", {
  skip_if_no_gpu()
  df <- f39_df()
  pipe <- function(d) dplyr::filter(d, x > 1, x * 2)
  got <- f39_expect_same_filter_error(df, pipe)
  expect_match(conditionMessage(got$parent), "`..2`", fixed = TRUE)
  f39_expect_same_filter_error(df, pipe, lazy = TRUE)
})

test_that("grouped filter() and .by with a non-logical condition match dplyr's error", {
  skip_if_no_gpu()
  df <- f39_df()
  f39_expect_same_filter_error(df, function(d) dplyr::filter(dplyr::group_by(d, g), x + 1))
  f39_expect_same_filter_error(df, function(d) dplyr::filter(d, x + 1, .by = g))
  f39_expect_same_filter_error(df, function(d) dplyr::filter(dplyr::group_by(d, g), x + 1),
                               lazy = TRUE)
})

test_that("filter() with an integer column expression and a numeric constant match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(i = c(1L, 0L, NA, 3L), x = c(1, 2, NA, 4))
  f39_expect_same_filter_error(df, function(d) dplyr::filter(d, i + 1L))
  f39_expect_same_filter_error(df, function(d) dplyr::filter(d, 1L))
})

test_that("filter() on an empty table with a non-logical condition matches dplyr", {
  skip_if_no_gpu()
  df <- f39_df()[0, ]
  f39_expect_same_filter_error(df, function(d) dplyr::filter(d, x + 1))
})

test_that("valid filter() calls are unaffected", {
  skip_if_no_gpu()
  df <- f39_df()
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, x > 1))
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, !is.na(x), g == "a"))
  expect_same_as_dplyr(df, function(d) dplyr::filter(dplyr::group_by(d, g), x > 1))
  expect_same_as_dplyr_lazy(df, function(d) dplyr::filter(d, x > 1))
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, c(TRUE, FALSE, TRUE, FALSE)))
})

# ---------------------------------------------------------------------------
# wrap_gpu_call(): only claim "device memory" for memory errors
# ---------------------------------------------------------------------------

test_that("wrap_gpu_call() does not blame device memory for a non-memory error", {
  skip_if_no_gpu()
  err <- expect_error(wrap_gpu_call("some_op", stop("cudf::logic_error: column type mismatch")))
  msg <- conditionMessage(err)
  expect_match(msg, "some_op", fixed = TRUE)
  expect_match(msg, "column type mismatch", fixed = TRUE)
  expect_no_match(msg, "device memory", fixed = TRUE)
})

test_that("wrap_gpu_call() still hints at device memory for an out-of-memory error", {
  skip_if_no_gpu()
  oom_msgs <- c(
    "std::bad_alloc: out_of_memory: CUDA error at: rmm/mr/device/cuda_memory_resource.hpp",
    "CUDA error: cudaErrorMemoryAllocation out of memory"
  )
  for (m in oom_msgs) {
    err <- expect_error(wrap_gpu_call("some_op", stop(m)))
    expect_match(conditionMessage(err), "device memory", fixed = TRUE)
    expect_match(conditionMessage(err), m, fixed = TRUE)
  }
})
