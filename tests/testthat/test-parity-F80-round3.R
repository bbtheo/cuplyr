# F80 round 3: lazy filter()/filter_out() with an integer window sum runs at
# the verb (review-2 B1), plan-break notices count the user's steps only (N2),
# a CPU rerun after a break doesn't run the user's plan again (N6), and the
# join size check reads memory cheaply (N1).
# (The acceptance tests are test-parity-F80-default.R and -print.R.)

f80r3_M <- .Machine$integer.max

f80r3_breaks <- function(expr) {
  got <- list()
  withCallingHandlers(
    expr,
    cuplyr_plan_break = function(cnd) got[[length(got) + 1L]] <<- cnd
  )
  got
}

f80r3_warnings <- function(expr) {
  msgs <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = msgs)
}

f80r3_plain <- function(x) {
  if (is_tbl_gpu(x)) x <- collect(x)
  tibble::as_tibble(dplyr::ungroup(x))
}

# Counts the filter nodes lowered while `expr` runs: how often a user's pending
# `filter()` step is executed.
f80r3_filter_runs <- function(expr) {
  runs <- 0L
  real <- cuplyr:::lower_and_execute
  testthat::local_mocked_bindings(
    lower_and_execute = function(ast, ...) {
      if (identical(.subset2(ast, "type"), "filter")) runs <<- runs + 1L
      real(ast, ...)
    },
    .package = "cuplyr"
  )
  value <- expr
  list(value = value, runs = runs)
}

# ---- B1: the reviewer's three inputs, on the lazy default --------------------

test_that("lazy filter() with an integer window sum gives dplyr's rows and warning", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1, 1, 2, 2), x = c(f80r3_M, 5L, 50000L, 50000L))
  e <- tibble::tibble(x = c(1L, 2L), y = c(f80r3_M, 5L))
  cases <- list(
    list(d, function(t) filter(t, x * sum(x) > 0L, .by = g)),
    list(d, function(t) filter(group_by(t, g), x * sum(x) > 0L)),
    list(e, function(t) filter(t, max(x * sum(y)) > 0L)),
    list(d, function(t) filter_out(group_by(t, g), x * sum(x) > 0L))
  )
  for (k in seq_along(cases)) {
    data <- cases[[k]][[1]]
    fn <- cases[[k]][[2]]
    cpu <- f80r3_warnings(fn(data))
    lz <- f80r3_warnings(fn(tbl_gpu(data, lazy = TRUE)))
    # the verb ran (that is where the warning is), and the result is lazy with
    # nothing pending
    expect_true(is_lazy(lz$value), info = paste("case", k))
    expect_false(has_pending_ops(lz$value), info = paste("case", k))
    expect_identical(length(lz$warnings) > 0L, length(cpu$warnings) > 0L, info = paste("case", k))
    expect_equal(f80r3_plain(lz$value), f80r3_plain(cpu$value), info = paste("case", k))
    expect_identical(dplyr::group_vars(collect(lz$value)), dplyr::group_vars(cpu$value), info = paste("case", k))
  }
})

test_that("a lazy filter() integer-sum break notices the user's steps only", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1L, 1L, 2L), x = c(f80r3_M, 1L, 5L))
  fresh <- tbl_gpu(d, lazy = TRUE)
  expect_length(f80r3_breaks(filter(fresh, sum(x) > 10L, .by = g)), 0L)
  # cur_group_id() puts a helper step on the table before the break
  expect_length(f80r3_breaks(filter(fresh, sum(x) > 10L, dplyr::cur_group_id() > 0L)), 0L)
  expect_length(f80r3_breaks(filter_out(fresh, sum(x) > 10L, .by = g)), 0L)
  pending <- filter(fresh, g > 0L)
  breaks <- f80r3_breaks(filter(pending, sum(x) > 10L, dplyr::cur_group_id() > 0L))
  expect_length(breaks, 1L)
  expect_identical(breaks[[1]]$verb, "filter")
  expect_equal(breaks[[1]]$steps, 1L)
  breaks <- f80r3_breaks(filter_out(pending, sum(x) > 10L, .by = g))
  expect_length(breaks, 1L)
  expect_identical(breaks[[1]]$verb, "filter_out")
  # a logical sum stays in the plan
  flags <- tbl_gpu(tibble::tibble(g = c(1L, 1L, 2L), l = c(TRUE, FALSE, TRUE)), lazy = TRUE)
  res <- filter(filter(flags, g > 0L), sum(l) > 0L, .by = g)
  expect_true(has_pending_ops(res))
  expect_equal(f80r3_plain(res), f80r3_plain(dplyr::filter(collect(flags), sum(l) > 0L, .by = g)))
})

test_that("a lazy filter() with an integer sum and cur_group_id() matches dplyr", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1, 1, 2, 2), x = c(f80r3_M, 5L, 50000L, 50000L))
  fns <- list(
    function(t) filter(t, x * sum(x) > 0L, dplyr::cur_group_id() > 0L, .by = g),
    function(t) filter(group_by(t, g), x * sum(x) > 0L, dplyr::cur_group_id() >= 1L),
    function(t) filter(t, sum(x) > 10L, dplyr::cur_group_id() == 1L)
  )
  for (k in seq_along(fns)) {
    cpu <- suppressWarnings(fns[[k]](d))
    lz <- suppressWarnings(fns[[k]](tbl_gpu(d, lazy = TRUE)))
    expect_true(is_lazy(lz), info = paste("case", k))
    got <- f80r3_plain(lz)
    expect_equal(got, f80r3_plain(cpu), info = paste("case", k))
    expect_false("..cur_group_id.." %in% names(got), info = paste("case", k))
  }
})

# ---- N6: a CPU rerun after the break starts from the computed table ---------

test_that("an overflow rerun after a lazy break runs the user's plan once", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1L, 1L, 2L), x = c(f80r3_M, 5L, 7L))
  pending <- filter(tbl_gpu(d, lazy = TRUE), g > 0L)
  m <- f80r3_filter_runs(suppressWarnings(
    mutate(group_by(pending, g), s = sum(x), id = dplyr::cur_group_id())
  ))
  expect_identical(m$runs, 1L)
  expect_equal(f80r3_plain(m$value),
               f80r3_plain(mutate(group_by(d, g), s = sum(x), id = dplyr::cur_group_id())))
  f <- f80r3_filter_runs(suppressWarnings(
    filter(pending, x * sum(x) > 0L, dplyr::cur_group_id() > 0L, .by = g)
  ))
  expect_identical(f$runs, 1L)
  expect_equal(f80r3_plain(f$value),
               f80r3_plain(suppressWarnings(filter(d, x * sum(x) > 0L, dplyr::cur_group_id() > 0L, .by = g))))
})

# ---- N2: summarise(.by) notices -----------------------------------------------

test_that("summarise(.by) notices count the user's steps and name the user's columns", {
  skip_if_no_gpu()
  d <- tibble::tibble(g = c(1L, 1L, 2L), i = c(f80r3_M, 1L, 5L), v = c(1, 2, 3))
  fresh <- tbl_gpu(d, lazy = TRUE)
  for (fn in list(
    function(t) summarise(t, s = sum(i), .by = g),
    function(t) summarise(t, mx = max(i), .by = g),
    function(t) summarise(t, s = sum(i * 2L), .by = g),
    function(t) summarise(t, m = mean(v * 2), .by = g)
  )) {
    expect_length(f80r3_breaks(suppressWarnings(fn(fresh))), 0L)
  }
  pending <- filter(fresh, v > 0)
  breaks <- f80r3_breaks(summarise(pending, s = sum(i), m = mean(v), .by = g))
  expect_length(breaks, 1L)
  expect_equal(breaks[[1]]$steps, 1L)
  expect_identical(breaks[[1]]$reason, "an integer sum() (result type depends on the data): s")
  breaks <- f80r3_breaks(summarise(pending, mx = max(i), m = mean(v), .by = g))
  expect_length(breaks, 1L)
  expect_equal(breaks[[1]]$steps, 1L)
  expect_match(breaks[[1]]$reason, "an integer min()/max() (result type depends on the data): mx", fixed = TRUE)
  expect_no_match(breaks[[1]]$reason, "summarise_first", fixed = TRUE)
  expect_no_match(breaks[[1]]$reason, ", m", fixed = TRUE)
  breaks <- f80r3_breaks(summarise(pending, m = mean(v * 2), .by = g))
  expect_length(breaks, 1L)
  expect_equal(breaks[[1]]$steps, 1L)
  # results are dplyr's
  expect_equal(f80r3_plain(summarise(pending, s = sum(i), mx = max(i), .by = g)),
               f80r3_plain(summarise(d, s = sum(i), mx = max(i), .by = g)))
})

# ---- N1: the join size check reads memory without the device properties -----

test_that("gpu_memory_state() reads the device's memory without gpu_info()", {
  skip_if_no_gpu()
  testthat::local_mocked_bindings(
    gpu_info = function() stop("gpu_info() must not be called"),
    .package = "cuplyr"
  )
  st <- gpu_memory_state()
  expect_true(st$available)
  expect_gt(st$total_bytes, 0)
  expect_gte(st$total_bytes, st$free_bytes)
  expect_equal(st$used_bytes, st$total_bytes - st$free_bytes)
  # a lazy join of two fresh tables still makes its size check
  x <- tbl_gpu(data.frame(id = 1:10), lazy = TRUE)
  expect_no_error(left_join(x, x, by = "id"))
})
