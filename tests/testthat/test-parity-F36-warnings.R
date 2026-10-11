# F36 acceptance tests: when the advisory many-to-many warning fires.
#
# dplyr 1.2.1 R/join-rows.R::compute_join_relationship(): with the default
# `relationship = NULL` the warning is only considered for equality joins
# ("Inequality and overlap joins often generate many-to-many relationships by
# nature"): ANY non-`==` condition, including a mix of equi and non-equi
# conditions, means no warning. It is also skipped when the join is called
# indirectly (`is_direct(user_env)` is FALSE: the caller's top env is neither
# the global env nor the package under test). An explicit `relationship` is
# always checked. Verified with tools/r (scratchpad/parity/features/F36/probe/p1.out).
#
# cuplyr's lazy joins check cardinality at collect time; the direct/indirect
# decision is the one made where the join verb was called.

m2m_class <- "dplyr_warning_join_relationship_many_to_many"

f36w_x <- data.frame(id = c(1, 1, 2), v = c(1, 5, 3))
f36w_y <- data.frame(id = c(1, 1, 2), lo = c(0, 2, 0), hi = c(10, 10, 10))

f36w_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

# Same rows as dplyr, and no many-to-many warning from either side.
f36w_no_warning <- function(pipeline, arrange_by) {
  expect_no_warning(
    expect_same_as_dplyr(f36w_x, pipeline, arrange_by = arrange_by),
    class = m2m_class
  )
  expect_no_warning(
    expect_same_as_dplyr_lazy(f36w_x, pipeline, arrange_by = arrange_by),
    class = m2m_class
  )
}

test_that("inequality joins never warn many-to-many", {
  skip_if_no_gpu()
  for (join in list(inner_join, left_join, right_join, full_join)) {
    f36w_no_warning(
      function(d) join(d, f36w_like(d, f36w_y), by = dplyr::join_by(v >= lo, v <= hi)),
      arrange_by = c("id.x", "v", "id.y", "lo")
    )
    f36w_no_warning(
      function(d) join(d, f36w_like(d, f36w_y), by = dplyr::join_by(v > lo)),
      arrange_by = c("id.x", "v", "id.y", "lo")
    )
  }
})

test_that("between(), within() and overlaps() joins never warn many-to-many", {
  skip_if_no_gpu()
  x2 <- data.frame(id = c(1, 1, 2), v = c(1, 5, 3), v2 = c(2, 6, 4))
  y_like <- function(d) f36w_like(d, f36w_y)
  for (by in list(dplyr::join_by(between(v, lo, hi)),
                  dplyr::join_by(within(v, v2, lo, hi)),
                  dplyr::join_by(overlaps(v, v2, lo, hi)))) {
    expect_no_warning(
      expect_same_as_dplyr(x2, function(d) inner_join(d, y_like(d), by = by),
                           arrange_by = c("id.x", "v", "id.y", "lo")),
      class = m2m_class
    )
    expect_no_warning(
      expect_same_as_dplyr_lazy(x2, function(d) left_join(d, y_like(d), by = by),
                                arrange_by = c("id.x", "v", "id.y", "lo")),
      class = m2m_class
    )
  }
})

test_that("a mix of equi and non-equi conditions never warns", {
  skip_if_no_gpu()
  for (join in list(inner_join, left_join, full_join)) {
    f36w_no_warning(
      function(d) join(d, f36w_like(d, f36w_y), by = dplyr::join_by(id, v >= lo)),
      arrange_by = c("id", "v", "lo")
    )
  }
})

test_that("a lazy non-equi join followed by a pushed-down filter does not warn", {
  skip_if_no_gpu()
  f36w_no_warning(
    function(d) {
      d |>
        inner_join(f36w_like(d, f36w_y), by = dplyr::join_by(v >= lo)) |>
        filter(v > 1, hi > 5)
    },
    arrange_by = c("id.x", "v", "id.y", "lo")
  )
})

test_that("equi joins still warn many-to-many (eager and lazy)", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    expect_warning(
      collect(inner_join(tbl_gpu(f36w_x, lazy = lazy), tbl_gpu(f36w_y), by = "id")),
      class = m2m_class
    )
  }
})

test_that("an explicit relationship is still checked on non-equi joins", {
  skip_if_no_gpu()
  expect_error(
    inner_join(tibble::as_tibble(f36w_x), tibble::as_tibble(f36w_y),
               by = dplyr::join_by(v >= lo), relationship = "one-to-one"),
    class = "dplyr_error_join_relationship_one_to_one"
  )
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(inner_join(tbl_gpu(f36w_x, lazy = lazy), tbl_gpu(f36w_y),
                         by = dplyr::join_by(v >= lo), relationship = "one-to-one")),
      class = "dplyr_error_join_relationship_one_to_one"
    )
  }
})

test_that("indirect calls don't warn many-to-many (conformance case)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 1))
  fn <- function(df1, df2, relationship = NULL) {
    collect(left_join(df1, df2, dplyr::join_by(x), relationship = relationship))
  }
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    # direct: a function the user controls
    expect_warning(fn(g, g), class = m2m_class)

    # indirect: mimic a function from a package the user doesn't control
    ind <- fn
    rlang::fn_env(ind) <- rlang::ns_env("rlang")
    expect_no_warning(out <- ind(g, g), class = m2m_class)
    expect_equal(nrow(out), 4L)
  }
})

test_that("lazy joins decide direct/indirect where the verb was called", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 1))
  g <- tbl_gpu(df, lazy = TRUE)

  build <- function(a, b) left_join(a, b, dplyr::join_by(x))
  rlang::fn_env(build) <- rlang::ns_env("rlang")
  # built indirectly, collected directly: no warning
  expect_no_warning(collect(build(g, g)), class = m2m_class)

  collect_ind <- function(t) collect(t)
  rlang::fn_env(collect_ind) <- rlang::ns_env("rlang")
  # built directly, collected indirectly: warns
  expect_warning(collect_ind(left_join(g, g, dplyr::join_by(x))), class = m2m_class)
})
