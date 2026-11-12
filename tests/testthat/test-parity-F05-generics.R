# F05 acceptance tests: tbl_gpu methods for dplyr's show_query(), explain(),
# same_src() and auto_copy() generics, and the join verbs' `copy =` path
# going through dplyr::auto_copy() (gap analysis A14, A15).

f05_x <- function() {
  tibble::tibble(
    k = c(1L, 2L, 3L, NA),
    a = c(1.5, NA, 3, 4),
    g = c("p", "q", "p", "q")
  )
}

f05_y <- function() {
  tibble::tibble(
    k = c(3L, 1L, NA, 5L),
    b = c("x", NA, "z", "w")
  )
}

f05_lazy_pipeline <- function() {
  tbl_gpu(f05_x(), lazy = TRUE) |>
    dplyr::filter(a > 1) |>
    dplyr::select(k, a)
}

# --- show_query() -----------------------------------------------------------

test_that("cuplyr's show_query is dplyr's generic, not a masking function", {
  skip_if_no_gpu()
  expect_identical(cuplyr::show_query, dplyr::show_query)
  expect_identical(get("show_query", envir = asNamespace("cuplyr")), dplyr::show_query)
})

test_that("dplyr::show_query() dispatches on a lazy tbl_gpu and returns x invisibly", {
  skip_if_no_gpu()
  lz <- f05_lazy_pipeline()
  expect_output(out <- withVisible(dplyr::show_query(lz)), "Pending operations")
  expect_false(out$visible)
  expect_identical(out$value, lz)
  expect_output(dplyr::show_query(lz), "AST depth")
  # Nothing was executed.
  expect_true(has_pending_ops(lz))
})

test_that("dplyr::show_query() on an eager / grouped tbl_gpu", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f05_x())
  expect_output(out <- withVisible(dplyr::show_query(gt)), "No pending operations")
  expect_false(out$visible)
  expect_identical(out$value, gt)

  grp <- tbl_gpu(f05_x(), lazy = TRUE) |> dplyr::group_by(g)
  expect_output(out <- withVisible(dplyr::show_query(grp)))
  expect_identical(out$value, grp)
})

test_that("show_query() on a non-tbl_gpu falls through to dplyr's generic", {
  skip_if_no_gpu()
  expect_error(dplyr::show_query(f05_x()), "no applicable method")
})

# --- explain() --------------------------------------------------------------

test_that("explain() on a lazy tbl_gpu prints the pending AST and the optimized plan", {
  skip_if_no_gpu()
  lz <- f05_lazy_pipeline()
  expect_output(out <- withVisible(dplyr::explain(lz)), "Pending operations")
  expect_false(out$visible)
  expect_identical(out$value, lz)
  expect_output(dplyr::explain(lz), "Optimized plan")
  # explain() never executes: the ops are still pending and collect() still
  # gives dplyr's answer.
  expect_true(has_pending_ops(lz))
  expect_identical(
    tibble::as_tibble(dplyr::collect(lz)),
    f05_x() |> dplyr::filter(a > 1) |> dplyr::select(k, a)
  )
})

test_that("explain() on an eager tbl_gpu returns x invisibly", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f05_x())
  expect_output(out <- withVisible(dplyr::explain(gt)), "No pending operations")
  expect_false(out$visible)
  expect_identical(out$value, gt)
})

# --- same_src() / auto_copy() ---------------------------------------------

test_that("same_src() is TRUE only for two tbl_gpu objects", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f05_x())
  gt2 <- tbl_gpu(f05_y())
  lz <- tbl_gpu(f05_y(), lazy = TRUE)
  expect_true(dplyr::same_src(gt, gt2))
  expect_true(dplyr::same_src(gt, lz))
  expect_true(dplyr::same_src(lz, gt))
  expect_false(dplyr::same_src(gt, f05_y()))
  expect_false(dplyr::same_src(gt, as.data.frame(f05_y())))
  expect_false(dplyr::same_src(gt, NULL))
})

test_that("auto_copy() is a no-op for two tbl_gpu objects", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f05_x())
  gt2 <- tbl_gpu(f05_y())
  expect_identical(dplyr::auto_copy(gt, gt2), gt2)
  expect_identical(dplyr::auto_copy(gt, gt2, copy = TRUE), gt2)
})

test_that("auto_copy() raises dplyr's same-src error without copy = TRUE", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f05_x())
  expect_error(dplyr::auto_copy(gt, f05_y()), "must share the same src",
               class = "rlang_error")
})

test_that("auto_copy(copy = TRUE) uploads a data frame to the GPU", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f05_x())
  for (y in list(f05_y(), as.data.frame(f05_y()), f05_y()[0, ])) {
    up <- dplyr::auto_copy(gt, y, copy = TRUE)
    expect_true(is_tbl_gpu(up))
    expect_identical(tibble::as_tibble(dplyr::collect(up)), tibble::as_tibble(y))
  }
})

# --- joins: copy = FALSE errors like dplyr's auto_copy() ---------------------

test_that("GPU joins with a data-frame y and copy = FALSE raise dplyr's same-src error", {
  skip_if_no_gpu()
  y <- f05_y()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(f05_x(), lazy = lazy)
    expect_error(dplyr::left_join(gt, y, by = "k"), "must share the same src", class = "rlang_error")
    expect_error(dplyr::inner_join(gt, y, by = "k"), "must share the same src", class = "rlang_error")
    expect_error(dplyr::right_join(gt, y, by = "k"), "must share the same src", class = "rlang_error")
    expect_error(dplyr::full_join(gt, y, by = "k"), "must share the same src", class = "rlang_error")
    expect_error(dplyr::semi_join(gt, y, by = "k"), "must share the same src", class = "rlang_error")
    expect_error(dplyr::anti_join(gt, y, by = "k"), "must share the same src", class = "rlang_error")
    expect_error(dplyr::cross_join(gt, y), "must share the same src", class = "rlang_error")
  }
})

test_that("a data-frame x with a tbl_gpu y keeps dplyr's own behavior", {
  skip_if_no_gpu()
  x <- f05_x()
  gy <- tbl_gpu(f05_y())
  expect_error(dplyr::inner_join(x, gy, by = "k"), "must share the same src",
               class = "rlang_error")
  expect_identical(
    dplyr::inner_join(x, gy, by = "k", copy = TRUE),
    dplyr::inner_join(x, f05_y(), by = "k")
  )
})

# --- joins: copy = TRUE matches dplyr ----------------------------------------

test_that("mutating joins with copy = TRUE match dplyr (eager and lazy)", {
  skip_if_no_gpu()
  y <- f05_y()
  pipes <- list(
    function(t) dplyr::left_join(t, y, by = "k", copy = TRUE),
    function(t) dplyr::inner_join(t, y, by = "k", copy = TRUE),
    function(t) dplyr::right_join(t, y, by = "k", copy = TRUE),
    function(t) dplyr::full_join(t, y, by = "k", copy = TRUE)
  )
  for (p in pipes) {
    expect_same_as_dplyr(f05_x(), p)
    expect_same_as_dplyr_lazy(f05_x(), p)
  }
})

test_that("filtering and cross joins with copy = TRUE match dplyr (eager and lazy)", {
  skip_if_no_gpu()
  y <- f05_y()
  pipes <- list(
    function(t) dplyr::semi_join(t, y, by = "k", copy = TRUE),
    function(t) dplyr::anti_join(t, y, by = "k", copy = TRUE),
    function(t) dplyr::cross_join(t, y, copy = TRUE)
  )
  for (p in pipes) {
    expect_same_as_dplyr(f05_x(), p)
    expect_same_as_dplyr_lazy(f05_x(), p)
  }
})

test_that("copy = TRUE joins: grouped x, base data.frame y, zero-row y", {
  skip_if_no_gpu()
  y <- f05_y()
  y_df <- as.data.frame(y)
  y0 <- y[0, ]
  pipes <- list(
    function(t) t |> dplyr::group_by(g) |> dplyr::left_join(y, by = "k", copy = TRUE),
    function(t) dplyr::inner_join(t, y_df, by = "k", copy = TRUE),
    function(t) dplyr::left_join(t, y0, by = "k", copy = TRUE),
    function(t) dplyr::semi_join(t, y0, by = "k", copy = TRUE)
  )
  for (p in pipes) {
    expect_same_as_dplyr(f05_x(), p)
    expect_same_as_dplyr_lazy(f05_x(), p)
  }
})

# --- round-1 review additions (B1, N2) ---------------------------------------

test_that("copy = TRUE coerces a non-data-frame y with as.data.frame(), like dplyr", {
  skip_if_no_gpu()
  y_list <- list(k = c(3L, 1L), b = c("x", "y"))
  pipes <- list(
    function(t) dplyr::left_join(t, y_list, by = "k", copy = TRUE),
    function(t) dplyr::semi_join(t, y_list, by = "k", copy = TRUE)
  )
  for (p in pipes) {
    expect_same_as_dplyr(f05_x(), p)
    expect_same_as_dplyr_lazy(f05_x(), p)
  }
  up <- dplyr::auto_copy(tbl_gpu(f05_x()), y_list, copy = TRUE)
  expect_true(is_tbl_gpu(up))
  expect_identical(tibble::as_tibble(dplyr::collect(up)),
                   tibble::as_tibble(as.data.frame(y_list)))
})

test_that("explain() prints an optimized plan only when there are pending lazy ops", {
  skip_if_no_gpu()
  out <- capture.output(dplyr::explain(tbl_gpu(f05_x())))
  expect_false(any(grepl("Optimized plan", out)))
  out <- capture.output(dplyr::explain(tbl_gpu(f05_x(), lazy = TRUE)))
  expect_false(any(grepl("Optimized plan", out)))
})
