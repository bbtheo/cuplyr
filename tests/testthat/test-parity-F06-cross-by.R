# F06 acceptance tests: `by = character()` cross joins (gap A4).
#
# dplyr 1.2.1 treats `by = character()`, a named `character(0)` and
# `list(x = character(), y = character())` as a cross join. It emits
# lifecycle's deprecation warning (class lifecycle_warning_deprecated,
# "Using `by = character()` to perform a cross join was deprecated in dplyr
# 1.1.0.") and then runs the normal join machinery with a constant key
# (dplyr:::join_rows(cross = TRUE)). So it is NOT a plain cross_join():
# an empty side gives outer-join NA padding for left/right/full, semi/anti
# filter on whether y has rows, the default relationship is many-to-many
# (no warning), and multiple=/relationship=/unmatched= still apply.
# Verified with tools/r (scratchpad/parity/features/F06/probe/p2.out, p3.out).

f06c_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f06c_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f06c_x <- data.frame(x = 1:3, s = c("a", "b", "c"))
f06c_y <- data.frame(y = c(10, 20), t = c("u", "v"))

f06c_mutating <- list(left_join = left_join, inner_join = inner_join,
                      right_join = right_join, full_join = full_join)
f06c_all <- c(f06c_mutating, list(semi_join = semi_join, anti_join = anti_join))

f06c_cross_bys <- list(
  character(),
  stats::setNames(character(), character()),
  list(x = character(), y = character())
)

# -----------------------------------------------------------------------------
# Deprecation warning
# -----------------------------------------------------------------------------

# Collect every warning raised while evaluating `expr`.
f06c_warnings <- function(expr) {
  ws <- list()
  withCallingHandlers(expr, warning = function(w) {
    ws[[length(ws) + 1L]] <<- w
    invokeRestart("muffleWarning")
  })
  ws
}

test_that("by = character() warns exactly once, with dplyr's class and text", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "warning")
  x <- tibble::as_tibble(f06c_x)
  y <- tibble::as_tibble(f06c_y)

  for (nm in names(f06c_all)) {
    join <- f06c_all[[nm]]
    for (by in f06c_cross_bys) {
      oracle <- f06c_warnings(join(x, y, by = by))
      expect_length(oracle, 1L)

      for (lazy in c(FALSE, TRUE)) {
        # Warned by the verb call itself, not later at collect().
        ws <- f06c_warnings(out <- join(tbl_gpu(f06c_x, lazy = lazy),
                                        tbl_gpu(f06c_y, lazy = lazy), by = by))
        deprecations <- Filter(function(w) inherits(w, "lifecycle_warning_deprecated"), ws)
        expect_length(deprecations, 1L)
        expect_identical(conditionMessage(deprecations[[1]]),
                         conditionMessage(oracle[[1]]),
                         info = nm)
        expect_length(f06c_warnings(collect(out)), 0L)
      }
    }
  }
})

test_that("nest_join(by = character()) warns exactly once", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "warning")
  oracle <- f06c_warnings(nest_join(tibble::as_tibble(f06c_x), tibble::as_tibble(f06c_y),
                                    by = character()))
  ws <- f06c_warnings(nest_join(tbl_gpu(f06c_x), tbl_gpu(f06c_y), by = character()))
  deprecations <- Filter(function(w) inherits(w, "lifecycle_warning_deprecated"), ws)
  expect_length(deprecations, 1L)
  expect_identical(conditionMessage(deprecations[[1]]), conditionMessage(oracle[[1]]))
})

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

test_that("mutating joins with by = character() return the cartesian product", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  for (join in f06c_mutating) {
    for (by in f06c_cross_bys) {
      f06c_both(f06c_x, function(d) join(d, f06c_like(d, f06c_y), by = by))
    }
  }
})

test_that("by = character() suffixes every common column name", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  y <- data.frame(x = c(7L, 8L), s = c("p", "q"), z = c(TRUE, FALSE))
  for (join in f06c_mutating) {
    f06c_both(f06c_x, function(d) join(d, f06c_like(d, y), by = character()))
    f06c_both(f06c_x, function(d) {
      join(d, f06c_like(d, y), by = character(), suffix = c("_l", "_r"))
    })
    f06c_both(f06c_x, function(d) join(d, f06c_like(d, y), by = character(), keep = TRUE))
    f06c_both(f06c_x, function(d) join(d, f06c_like(d, y), by = character(), keep = FALSE))
  }
})

test_that("by = character() with a 0-row side follows outer-join semantics", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  x0 <- f06c_x[0, , drop = FALSE]
  y0 <- f06c_y[0, , drop = FALSE]
  for (join in f06c_all) {
    f06c_both(f06c_x, function(d) join(d, f06c_like(d, y0), by = character()))
    f06c_both(x0, function(d) join(d, f06c_like(d, f06c_y), by = character()))
    f06c_both(x0, function(d) join(d, f06c_like(d, y0), by = character()))
  }
})

test_that("a side emptied by an upstream lazy filter is handled at execution time", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  for (join in f06c_all) {
    f06c_both(f06c_x, function(d) {
      join(d, f06c_like(d, f06c_y) |> filter(y > 100), by = character())
    })
    f06c_both(f06c_x, function(d) {
      d |> filter(x > 100) |> join(f06c_like(d, f06c_y), by = character())
    })
    f06c_both(f06c_x, function(d) {
      d |> filter(x >= 2) |> join(f06c_like(d, f06c_y) |> filter(y < 15), by = character())
    })
  }
})

test_that("semi_join()/anti_join() with by = character() filter on whether y has rows", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  for (join in list(semi_join, anti_join)) {
    for (by in f06c_cross_bys) {
      f06c_both(f06c_x, function(d) join(d, f06c_like(d, f06c_y), by = by))
    }
  }
})

test_that("by = character() ignores na_matches and NA values", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  x <- data.frame(x = c(NA, 1, NA))
  y <- data.frame(y = c(NA_character_, "b"))
  for (join in f06c_all) {
    f06c_both(x, function(d) join(d, f06c_like(d, y), by = character(), na_matches = "never"))
  }
})

test_that("by = character() keeps x's grouping like any other join", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  for (join in f06c_all) {
    f06c_both(f06c_x, function(d) group_by(d, s) |> join(f06c_like(d, f06c_y), by = character()))
  }
})

test_that("then a verb after the cross join works lazily", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  f06c_both(f06c_x, function(d) {
    d |>
      left_join(f06c_like(d, f06c_y), by = character()) |>
      filter(y > 10) |>
      mutate(xy = x * y) |>
      select(s, xy)
  })
})

# -----------------------------------------------------------------------------
# relationship= / multiple= / unmatched= still apply
# -----------------------------------------------------------------------------

test_that("by = character() never warns about a many-to-many relationship", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  for (join in f06c_mutating) {
    for (lazy in c(FALSE, TRUE)) {
      expect_no_warning(
        join(tbl_gpu(f06c_x, lazy = lazy), tbl_gpu(f06c_y, lazy = lazy), by = character()) |>
          collect()
      )
    }
  }
})

test_that("multiple = 'first'/'last' pick y's first/last row for each x row", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  for (join in list(left_join, inner_join)) {
    for (m in c("first", "last")) {
      f06c_both(f06c_x, function(d) join(d, f06c_like(d, f06c_y), by = character(), multiple = m))
    }
  }
})

# Evaluate (and for lazy, collect) a GPU join, returning the condition.
f06c_gpu_error <- function(join, x, y, lazy, ...) {
  tryCatch({
    out <- join(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy), ...)
    collect(out)
    NULL
  }, error = identity)
}

test_that("relationship = 'one-to-one'/'many-to-one' errors like dplyr", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  for (rel in c("one-to-one", "many-to-one")) {
    oracle <- tryCatch(left_join(tibble::as_tibble(f06c_x), tibble::as_tibble(f06c_y),
                                 by = character(), relationship = rel),
                       error = identity)
    expect_s3_class(oracle, "dplyr_error_join")
    for (lazy in c(FALSE, TRUE)) {
      err <- f06c_gpu_error(left_join, f06c_x, f06c_y, lazy, by = character(),
                            relationship = rel)
      expect_s3_class(err, class(oracle)[1])
      expect_identical(conditionMessage(err), conditionMessage(oracle))
    }
  }
})

test_that("unmatched = 'error' errors only when a side is empty", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  x0 <- f06c_x[0, , drop = FALSE]
  y0 <- f06c_y[0, , drop = FALSE]
  # unmatched= only checks the rows a join would drop: left_join() checks y,
  # right_join() checks x, inner_join() checks both (verified with tools/r).
  cases <- list(
    list(join = left_join, x = x0, y = f06c_y),
    list(join = inner_join, x = f06c_x, y = y0),
    list(join = inner_join, x = x0, y = f06c_y),
    list(join = right_join, x = f06c_x, y = y0)
  )
  for (case in cases) {
    oracle <- tryCatch(case$join(tibble::as_tibble(case$x), tibble::as_tibble(case$y),
                                 by = character(), unmatched = "error"),
                       error = identity)
    expect_s3_class(oracle, "dplyr_error_join")
    for (lazy in c(FALSE, TRUE)) {
      err <- f06c_gpu_error(case$join, case$x, case$y, lazy, by = character(),
                            unmatched = "error")
      expect_s3_class(err, class(oracle)[1])
      expect_identical(conditionMessage(err), conditionMessage(oracle))
    }
  }
  # Both sides non-empty: everything matches, no error. An empty side whose
  # rows the join keeps (left_join() with empty y, right_join() with empty x)
  # is not an error either.
  f06c_both(f06c_x, function(d) {
    inner_join(d, f06c_like(d, f06c_y), by = character(), unmatched = "error")
  })
  f06c_both(f06c_x, function(d) {
    left_join(d, f06c_like(d, y0), by = character(), unmatched = "error")
  })
  f06c_both(x0, function(d) {
    right_join(d, f06c_like(d, f06c_y), by = character(), unmatched = "error")
  })
})

test_that("by = character() on the suffix-collision CPU fallback path warns once with dplyr's text", {
  skip_if_no_gpu()
  # Added after round 1 (review finding N6): suffix = c("", "") with a common
  # name routes the join to the CPU fallback; the user must still see exactly
  # dplyr's own warning, once.
  withr::local_options(lifecycle_verbosity = "warning")
  x <- data.frame(x = 1:2)
  y <- data.frame(x = 3:4)
  oracle <- f06c_warnings(left_join(tibble::as_tibble(x), tibble::as_tibble(y),
                                    by = character(), suffix = c("", "")))
  expect_length(oracle, 1L)
  for (lazy in c(FALSE, TRUE)) {
    ws <- f06c_warnings(out <- left_join(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy),
                                         by = character(), suffix = c("", "")))
    deprecations <- Filter(function(w) inherits(w, "lifecycle_warning_deprecated"), ws)
    expect_length(deprecations, 1L)
    expect_identical(conditionMessage(deprecations[[1]]), conditionMessage(oracle[[1]]))
    withr::with_options(list(lifecycle_verbosity = "quiet"), {
      expect_equal(
        tibble::as_tibble(collect(out)),
        left_join(tibble::as_tibble(x), tibble::as_tibble(y), by = character(), suffix = c("", ""))
      )
    })
  }
})
