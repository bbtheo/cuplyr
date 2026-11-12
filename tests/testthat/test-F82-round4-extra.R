# F82 round 4: guards next to the acceptance tests for the round-3 review. A
# CPU route splices the captured dots directly, so dplyr's own checks of the
# names written in `...` (slice()'s check_dots_unnamed(), transmute()'s
# `.keep`/`.before`/`.after`, group_by_prepare()'s `add`/`.dots`, R's matching
# of count()'s `group_by(x, ..., .add = TRUE)`) run at the cuplyr method's
# entry instead. dplyr checks written names only: a name spliced with `!!!`
# reaches its capture and is a column, on both routes here too. And an
# if_any()/if_all() lambda only stays native when its calls are safe to run
# twice, a clock read included; a user function named like one is not.

r4_df <- function() {
  tibble::tibble(x = c(1, 5, 10, NA, 3), g = c(1L, 1L, 2L, 2L, 2L))
}

r4_root <- function(e) {
  while (!is.null(e$parent)) e <- e$parent
  e
}

# Same class vector and root-cause class and message as dplyr, eager and lazy.
r4_expect_same_error <- function(df, pipeline) {
  ref <- tryCatch({
    pipeline(tibble::as_tibble(df))
    NULL
  }, error = identity)
  expect_s3_class(ref, "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch({
      collect(pipeline(tbl_gpu(df, lazy = lazy)))
      NULL
    }, error = identity)
    mode <- if (lazy) "lazy" else "eager"
    expect_s3_class(got, "error")
    if (inherits(got, "error")) {
      expect_identical(class(got), class(ref), info = paste(mode, conditionMessage(got)))
      expect_identical(class(r4_root(got)), class(r4_root(ref)), info = mode)
      expect_identical(conditionMessage(r4_root(got)), conditionMessage(r4_root(ref)), info = mode)
    }
  }
}

r4_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("F82: group_by()/count()/add_count() defunct arguments name dplyr, native and CPU route", {
  skip_if_no_gpu()
  df <- r4_df()
  twice <- function(v) v * 2
  r4_expect_same_error(df, function(d) group_by(d, add = g))
  r4_expect_same_error(df, function(d) group_by(d, .dots = g))
  r4_expect_same_error(df, function(d) group_by(d, add = g, .dots = x))
  r4_expect_same_error(df, function(d) count(d, .dots = g))
  r4_expect_same_error(df, function(d) count(d, .dots = twice(g)))
  r4_expect_same_error(df, function(d) add_count(d, add = g))
  r4_expect_same_error(df, function(d) add_count(d, .dots = twice(g)))
  r4_expect_same_error(df, function(d) count(group_by(d, g), add = x))
})

test_that("F82: count()/add_count() reject a `.add` dot as dplyr's group_by() call does", {
  skip_if_no_gpu()
  df <- r4_df()
  twice <- function(v) v * 2
  r4_expect_same_error(df, function(d) count(d, .add = g))
  r4_expect_same_error(df, function(d) count(d, .add = twice(g)))
  r4_expect_same_error(df, function(d) add_count(d, .add = g))
  r4_expect_same_error(df, function(d) add_count(d, .add = twice(g)))
})

test_that("F82: a dot name spliced with !!! is a column, as in dplyr", {
  skip_if_no_gpu()
  df <- r4_df()
  twice <- function(v) v * 2
  r4_both(df, function(d) count(d, !!!list(add = rlang::quo(g))))
  r4_both(df, function(d) count(d, !!!list(add = rlang::quo(twice(g)))))
  r4_both(df, function(d) count(d, !!!list(.add = rlang::quo(g))))
  r4_both(df, function(d) add_count(d, !!!list(add = rlang::quo(twice(g)))))
  r4_both(df, function(d) summarise(group_by(d, !!!list(add = rlang::quo(g))), n = dplyr::n()))
  r4_both(df, function(d) transmute(d, !!!list(.keep = rlang::quo(x))))
  r4_both(df, function(d) transmute(d, !!!list(.before = rlang::quo(twice(x)))))
  r4_both(df, function(d) slice(d, !!!list(foo = 2)))
  r4_both(df, function(d) slice(d, !!!list(foo = rlang::quo(dplyr::n()))))
})

test_that("F82: transmute() and slice() check names before running a dot", {
  skip_if_no_gpu()
  df <- r4_df()
  calls <- 0L
  bump <- function() {
    calls <<- calls + 1L
    calls
  }
  for (lazy in c(FALSE, TRUE)) {
    calls <- 0L
    expect_error(transmute(tbl_gpu(df, lazy = lazy), y = bump(), .after = x), "`.after` argument")
    expect_error(slice(tbl_gpu(df, lazy = lazy), bump(), foo = 1), class = "rlib_error_dots_named")
    expect_error(count(tbl_gpu(df, lazy = lazy), z = bump(), add = g), class = "lifecycle_error_deprecated")
    expect_identical(calls, 0L)
  }
})

test_that("F82: an if_any() lambda with a clock read stays native; a user today() goes to dplyr", {
  skip_if_no_gpu()
  ref_day <- Sys.Date()
  df <- tibble::tibble(d1 = ref_day - c(0, 40, 100, 10), d2 = ref_day - c(50, 5, 60, NA), v = 1:4)
  ref_time <- Sys.time()
  dft <- tibble::tibble(t1 = ref_time + c(-10, 7200, -1e5, -10), t2 = ref_time + c(-9000, -5, 6e4, NA), v = 1:4)
  native <- list(
    list(df, function(d) filter(d, if_any(c(d1, d2), ~ .x >= base::Sys.Date() - 30))),
    list(dft, function(d) filter(d, if_all(c(t1, t2), ~ !is.na(.x) & .x <= Sys.time())))
  )
  if (requireNamespace("lubridate", quietly = TRUE)) {
    native <- c(native, list(list(df, function(d) filter(d, if_any(c(d1, d2), ~ .x >= lubridate::today() - 30)))))
  }
  for (case in native) {
    r4_both(case[[1]], case[[2]])
    withr::with_options(list(cuplyr.fallback = "error"), {
      expect_no_error(collect(case[[2]](tbl_gpu(case[[1]]))))
      expect_no_error(collect(case[[2]](tbl_gpu(case[[1]], lazy = TRUE))))
    })
  }
  # a function of the user's named today() is not the clock: dplyr calls it once
  # per column, and so must the GPU route
  for (lazy in c(FALSE, TRUE)) {
    k <- 0L
    today <- function() {
      k <<- k + 1L
      ref_day - 30 * k
    }
    ref <- dplyr::filter(df, if_any(c(d1, d2), ~ .x >= today()))
    n_ref <- k
    k <- 0L
    got <- collect(filter(tbl_gpu(df, lazy = lazy), if_any(c(d1, d2), ~ .x >= today())))
    expect_identical(k, n_ref)
    expect_equal(tibble::as_tibble(got), ref, ignore_attr = TRUE)
  }
})

# The acceptance test's mutate() shapes, with dplyr::if_any()/if_all(): dplyr's
# mutate() calls them as functions, which tools/test-file doesn't attach.
test_that("F82: an impure if_any()/if_all() lambda in mutate() is called as dplyr calls it", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 5, 10, NA, 3), s = c("3", "1", "0", NA, "5"),
                       g = c(5L, 4L, 3L, 2L, 1L))
  twice <- function(v) v * 2
  verbs <- list(
    function(d, bump) mutate(d, a = dplyr::if_any(c(x, g), ~ .x > bump()), w = twice(x)),
    function(d, bump) mutate(d, a = dplyr::if_all(c(x, g), ~ .x > bump())),
    function(d, bump) filter(d, if_any(c(x, s), ~ .x > bump()), twice(g) > 0)
  )
  for (verb in verbs) {
    for (lazy in c(FALSE, TRUE)) {
      counts <- integer()
      pipeline <- function(d) {
        k <- 0L
        bump <- function() {
          k <<- k + 1L
          k
        }
        out <- collect(verb(d, bump))
        counts <<- c(counts, k)
        out
      }
      if (lazy) expect_same_as_dplyr_lazy(df, pipeline) else expect_same_as_dplyr(df, pipeline)
      expect_length(counts, 2L)
      expect_identical(counts[[2]], counts[[1]],
                       info = sprintf("bump() calls: dplyr %d, GPU %d", counts[[1]], counts[[2]]))
    }
  }
})
