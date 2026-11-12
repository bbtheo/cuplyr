# F07 extras beyond the acceptance files: internal helper aggregations must
# not promote or warn, schema reconciliation after a data-dependent INT32
# min()/max(), and the zero-column row-count contract: group_by()/ungroup()/
# the verbs push_op() runs/bind_rows()/bind_cols()/a literal-mask filter()
# keep `$nrow`; every other verb on such a table with rows errors; collect()
# and dim() never turn a missing count into 0.

test_that("F07: summarise(.by =) on 0 rows does not warn from its internal row-position min()", {
  skip_if_no_gpu()
  df <- data.frame(g = character(), v = double(), stringsAsFactors = FALSE)

  expect_no_warning(collect(dplyr::summarise(tbl_gpu(df), s = sum(v), .by = g)))
  expect_no_warning(collect(dplyr::summarise(as_lazy(tbl_gpu(df)), s = sum(v), .by = g)))
})

test_that("F07: an INT32 min()/max() with no empty group keeps INT32 in the schema (eager and lazy)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b"), x = c(1L, NA, 3L))

  for (lazy in c(FALSE, TRUE)) {
    gpu <- tbl_gpu(df, lazy = lazy)
    res <- gpu |> dplyr::group_by(g) |> dplyr::summarise(mn = min(x, na.rm = TRUE))
    expect_identical(unname(res$schema$types), c("STRING", "INT32"))
    expect_type(collect(res)$mn, "integer")
  }
})

test_that("F07: a promoted INT32 min()/max() updates the schema to FLOAT64 (eager and lazy)", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b"), x = c(1L, NA, NA))

  for (lazy in c(FALSE, TRUE)) {
    gpu <- tbl_gpu(df, lazy = lazy)
    res <- suppressWarnings(
      gpu |> dplyr::group_by(g) |> dplyr::summarise(mx = max(x, na.rm = TRUE))
    )
    expect_identical(unname(res$schema$types), c("STRING", "FLOAT64"))
  }
})

test_that("F07: dots naming an earlier dot's output go through the CPU fallback and see the new value", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "a", "b"), x = c(1, 2, 4))

  res <- suppressMessages(
    tbl_gpu(df) |> dplyr::summarise(x = max(x), y = min(x)) |> collect()
  )
  expect_identical(res$y, 4)
})

test_that("F07: push_op() verbs on a zero-column table with rows error (filter) instead of returning a wrong row count", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(tibble::tibble(.rows = 4L))

  expect_error(dplyr::filter(gpu, FALSE), "no columns")
  # constant mutate() recycles to the row count (F44); a pure column-less
  # table can't be sized any other way, so the rest still errors.
  expect_identical(
    collect(dplyr::mutate(as_lazy(gpu), z = 1L)), tibble::tibble(z = rep(1L, 4))
  )
  expect_identical(dim(as_lazy(gpu)), c(4L, 0L))
  expect_identical(collect(compute(as_lazy(gpu))), tibble::tibble(.rows = 4L))
})

test_that("F07: a zero-column, zero-row data frame round-trips", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(tibble::tibble())

  expect_identical(dim(gpu), c(0L, 0L))
  expect_identical(collect(gpu), tibble::tibble())
})

test_that("F07: zero-column row count survives group_by(), ungroup() and a lazy table", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(tibble::tibble(.rows = 4L))

  expect_identical(dim(dplyr::ungroup(gpu)), c(4L, 0L))
  expect_identical(dim(dplyr::group_by(gpu)), c(4L, 0L))
  expect_identical(collect(dplyr::ungroup(as_lazy(gpu))), tibble::tibble(.rows = 4L))
  expect_identical(
    collect(dplyr::summarise(dplyr::group_by(as_lazy(gpu)), n = dplyr::n()))$n, 4L
  )
})

test_that("F07: collect() and dim() never invent a row count for a zero-column table", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(tibble::tibble(.rows = 4L))
  gpu$nrow <- NULL

  expect_identical(dim(gpu), c(NA_integer_, 0L))
  expect_error(collect(gpu), "row count is unknown")
  expect_error(dplyr::mutate(gpu, z = 1L), "unknown number of rows")
})

test_that("F07: bind_rows()/bind_cols()/literal-mask filter() on zero-column tables count rows right", {
  skip_if_no_gpu()
  t2 <- tbl_gpu(tibble::tibble(.rows = 2L))
  t3 <- tbl_gpu(tibble::tibble(.rows = 3L))

  expect_identical(dim(cuplyr::bind_rows(t2, t3)), c(5L, 0L))
  expect_identical(dim(cuplyr::bind_cols(t3, t3)), c(3L, 0L))
  # dplyr (vctrs) reports the size mismatch, labelled by position.
  expect_error(cuplyr::bind_cols(t2, t3), "Can't recycle `..1` (size 2) to match `..2` (size 3)", fixed = TRUE)
  expect_identical(collect(dplyr::filter(t3, c(TRUE, FALSE, NA))), tibble::tibble(.rows = 1L))
  expect_identical(collect(dplyr::filter(t3, TRUE)), tibble::tibble(.rows = 3L))

  # dplyr (F12): a zero-column table pads with NA in bind_rows() and adds no
  # columns in bind_cols(); neither is an error.
  real <- tbl_gpu(data.frame(x = 1:3))
  expect_identical(collect(cuplyr::bind_rows(real, t3))$x, c(1L, 2L, 3L, NA, NA, NA))
  expect_identical(collect(cuplyr::bind_cols(real, t3))$x, 1:3)
})

test_that("F07: dplyr_row_slice() on a zero-column table keeps the sliced row count", {
  skip_if_no_gpu()
  gpu <- tbl_gpu(tibble::tibble(.rows = 5L))
  expect_identical(dim(dplyr::dplyr_row_slice(gpu, c(2L, 4L))), c(2L, 0L))
  expect_error(dplyr::dplyr_row_slice(gpu, 9L), "out-of-range")
})

test_that("F07: .data pronoun reads of an earlier dot count as references", {
  skip_if_no_gpu()
  q <- function(...) rlang::enquos(...)
  expect_true(summarise_dots_are_sequential(q(x = max(x), y = min(.data[["x"]]))))
  expect_true(summarise_dots_are_sequential(q(x = max(x), y = min(.data$x))))
  expect_true(summarise_dots_are_sequential(q(x = max(x), y = min(abs(.data[["x"]])))))
  expect_false(summarise_dots_are_sequential(q(x = max(x), y = min(.data[["z"]]))))
  expect_false(summarise_dots_are_sequential(q(x = max(x), y = min(z[, 1]))))
})

test_that("F07: a lazy summarise forced to run by an integer min()/max() notifies like other materializations", {
  skip_if_no_gpu()
  # F88: this is a plan break (it runs on the GPU), never a CPU fallback, so
  # cuplyr.fallback = "error" does not fire and cuplyr.plan_break governs it
  withr::local_options(cuplyr.fallback = "error", cuplyr.plan_break = "warn")
  lazy <- as_lazy(tbl_gpu(data.frame(g = c("a", "b"), x = c(1L, 2L))))

  # pending ops: materialized, so the notification fires
  expect_warning(
    lazy |> dplyr::filter(x > 0L) |> dplyr::group_by(g) |> dplyr::summarise(m = min(x)),
    class = "cuplyr_warning_plan_break"
  )
  # nothing pending: nothing to materialize, no notification
  expect_no_warning(lazy |> dplyr::group_by(g) |> dplyr::summarise(m = min(x)))
})

test_that("F07: slice() family on a lazy zero-column table with pending ops computes first", {
  skip_if_no_gpu()
  lz <- dplyr::collapse(tbl_gpu(tibble::tibble(.rows = 5L), lazy = TRUE))
  expect_true(has_pending_ops(lz))
  expect_identical(dim(collect(dplyr::slice(lz, 1:2))), c(2L, 0L))
  expect_identical(dim(collect(dplyr::slice_head(lz, n = 3))), c(3L, 0L))
  expect_identical(dim(collect(dplyr::slice_tail(lz, n = -1))), c(4L, 0L))
})

test_that("F07: a user column named `nrow` does not shadow the zero-column row count", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(nrow = 1:3))
  expect_identical(dim(g), c(3L, 1L))
  expect_identical(collect(g)$nrow, 1:3)
  expect_identical(dim(dplyr::summarise(g)), c(1L, 0L))
  expect_identical(dim(collect(tbl_gpu(tibble::tibble(.rows = 4L)))), c(4L, 0L))
})

test_that("F07: `$<-`/`[[<-` on a zero-column table with rows builds every row", {
  skip_if_no_gpu()
  z5 <- tibble::tibble(.rows = 5L)
  values <- list(NA, NA_real_, as.Date("2024-01-01"), factor("u"), 1:5, 2.5, "s")
  for (lazy in c(FALSE, TRUE)) {
    for (v in values) {
      g <- tbl_gpu(z5, lazy = lazy)
      g$a <- v
      want <- z5
      want$a <- v
      expect_identical(dim(g), c(5L, 1L))
      expect_identical(tibble::as_tibble(collect(g)), want)

      g <- tbl_gpu(z5, lazy = lazy)
      g[["a"]] <- v
      expect_identical(tibble::as_tibble(collect(g)), want)
    }
    # tibble rejects a length-0 (or wrong-length) value for 5 rows.
    g <- tbl_gpu(z5, lazy = lazy)
    expect_error(g$a <- integer())
    expect_error(g[["a"]] <- 1:2)
  }
  # A count known only after pending ops (lazy zero-dot summarise: one row).
  s <- dplyr::summarise(tbl_gpu(data.frame(x = 1:3), lazy = TRUE))
  s$a <- NA
  expect_identical(tibble::as_tibble(collect(s)), tibble::tibble(a = NA))
})

test_that("F07: gpu_object_info() reports a zero-column table's row count", {
  skip_if_no_gpu()
  info <- gpu_object_info(tbl_gpu(tibble::tibble(.rows = 5L)))
  expect_identical(info$nrow, 5L)
  expect_identical(info$ncol, 0L)
})
