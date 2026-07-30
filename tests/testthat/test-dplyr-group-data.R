# Oracle parity tests for Phase 9, task 1: the grouping metadata API
# (group_data()/group_keys()/group_rows()/group_indices()/group_size()/
# n_groups()), computed group_by() columns, tbl_nongroup_vars()/
# group_by_drop_default(), and the dplyr backend extension contract
# (dplyr_row_slice()/dplyr_col_modify()/dplyr_reconstruct(), 1-d `[`).
#
# Mirrors dplyr's own installed test-group-data.R/test-generics.R
# (~/R/*/dplyr/tests/testthat/, read directly rather than assumed) adapted
# to this package's tbl_gpu/oracle conventions. See R/group-by.R for the
# implementation notes (row order, .drop scope, computed-group desugar).

# =============================================================================
# group_data(): ungrouped / grouped / multi-key / NA keys, eager + lazy
# =============================================================================

test_that("group_data() on an ungrouped table is one row holding every row", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:3)
  for (mode in c("eager", "lazy")) {
    gt <- tbl_gpu(df, lazy = identical(mode, "lazy"))
    gd <- dplyr::group_data(gt)

    expect_s3_class(gd, "tbl_df")
    expect_equal(gd, tibble::new_tibble(list(.rows = vctrs::list_of(1:3)), nrow = 1L), info = mode)
  }
})

test_that("group_data() on an empty ungrouped table is one row with .rows = integer(0)", {
  skip_if_no_gpu()

  gt <- tbl_gpu(tibble::tibble(x = integer()))
  gd <- dplyr::group_data(gt)

  expect_equal(gd, tibble::new_tibble(list(.rows = vctrs::list_of(integer())), nrow = 1L))
})

test_that("group_data() on a single-key grouped table matches dplyr (ascending, NA last)", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(2, 1, 2, NA, 1), y = 1:5)
  oracle <- dplyr::group_data(dplyr::group_by(df, x))

  for (mode in c("eager", "lazy")) {
    gt <- tbl_gpu(df, lazy = identical(mode, "lazy")) |> dplyr::group_by(x)
    expect_equal(tibble::as_tibble(dplyr::group_data(gt)), oracle, ignore_attr = TRUE, info = mode)
  }
})

test_that("group_data() on a multi-key grouped table matches dplyr", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 1, 2, 2, 1), y = c("b", "a", "a", "b", "a"), v = 1:5)
  oracle <- dplyr::group_data(dplyr::group_by(df, x, y))

  gt <- tbl_gpu(df) |> dplyr::group_by(x, y)
  expect_equal(tibble::as_tibble(dplyr::group_data(gt)), oracle, ignore_attr = TRUE)
})

test_that("group_data() shuffled-row input still assigns rows to the right group", {
  skip_if_no_gpu()

  set.seed(1)
  df <- tibble::tibble(g = sample(rep(1:5, each = 20)), x = rnorm(100))
  oracle <- dplyr::group_data(dplyr::group_by(df, g))

  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  expect_equal(tibble::as_tibble(dplyr::group_data(gt)), oracle, ignore_attr = TRUE)
})

# =============================================================================
# group_rows()/group_keys() partition group_data()
# =============================================================================

test_that("group_rows() and group_keys() partition group_data()", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:2, y = 1:2)
  gt <- tbl_gpu(df) |> dplyr::group_by(x, y)
  gd <- dplyr::group_data(gt)

  expect_equal(dplyr::group_keys(gt), gd[1:2], ignore_attr = TRUE)
  expect_equal(dplyr::group_rows(gt), gd[[3]])
})

test_that("group_keys(...)/group_indices(...) with extra args are defunct", {
  skip_if_no_gpu()

  gt <- tbl_gpu(tibble::tibble(x = 1, y = 2))

  expect_error(dplyr::group_keys(gt, x), "defunct")
  expect_error(dplyr::group_indices(gt, x), "defunct")
})

# =============================================================================
# group_indices()
# =============================================================================

test_that("group_indices() returns expected per-row group ids", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c("b", "a", "b"))
  gt <- tbl_gpu(df)
  gt_grouped <- dplyr::group_by(gt, x)

  expect_equal(dplyr::group_indices(gt), c(1, 1, 1))
  expect_equal(dplyr::group_indices(gt_grouped), dplyr::group_indices(dplyr::group_by(df, x)))
  expect_equal(dplyr::group_indices(gt_grouped), c(2, 1, 2))
})

test_that("group_indices() handles 0-row tables", {
  skip_if_no_gpu()

  gt <- tbl_gpu(data.frame(x = integer(), y = integer())) |> dplyr::group_by(x)
  expect_equal(dplyr::group_indices(gt), integer())
})

# =============================================================================
# group_size()/n_groups()
# =============================================================================

test_that("ungrouped data has 1 group, with group size = nrow()", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = rep(1:3, each = 10), y = rep(1:6, each = 5))
  gt <- tbl_gpu(df)

  expect_equal(dplyr::n_groups(gt), 1L)
  expect_equal(dplyr::group_size(gt), 30)
})

test_that("group_size()/n_groups() correct for grouped data", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = rep(1:3, each = 10), y = rep(1:6, each = 5))
  gt <- tbl_gpu(df) |> dplyr::group_by(x)

  expect_equal(dplyr::n_groups(gt), 3L)
  expect_equal(dplyr::group_size(gt), rep(10, 3))
})

test_that("n_groups() respects zero-length factor groups when .drop = FALSE", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = factor(1:3, levels = 1:4))
  gt <- tbl_gpu(df) |> dplyr::group_by(x, .drop = FALSE)

  expect_equal(dplyr::n_groups(gt), 4)
})

# =============================================================================
# .drop = FALSE: factor-level expansion (single key, mixed keys, all-factor)
# =============================================================================

test_that(".drop = FALSE expands unobserved factor levels (single factor key)", {
  skip_if_no_gpu()

  df <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c")))
  oracle <- dplyr::group_data(dplyr::group_by(df, f, .drop = FALSE))

  gt <- tbl_gpu(df) |> dplyr::group_by(f, .drop = FALSE)
  expect_equal(tibble::as_tibble(dplyr::group_data(gt)), oracle, ignore_attr = TRUE)
})

test_that(".drop = FALSE with a mixed factor/non-factor key fills the non-factor column with NA", {
  skip_if_no_gpu()

  df <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c")), n = c(1, 2, 1))
  oracle <- dplyr::group_data(dplyr::group_by(df, f, n, .drop = FALSE))

  gt <- tbl_gpu(df) |> dplyr::group_by(f, n, .drop = FALSE)
  expect_equal(tibble::as_tibble(dplyr::group_data(gt)), oracle, ignore_attr = TRUE)
})

test_that(".drop = FALSE with two factor keys expands the full cross product", {
  skip_if_no_gpu()

  df <- tibble::tibble(
    f = factor(c("a", "b"), levels = c("a", "b", "c")),
    g = factor(c("x", "x"), levels = c("x", "y"))
  )
  oracle <- dplyr::group_data(dplyr::group_by(df, f, g, .drop = FALSE))

  gt <- tbl_gpu(df) |> dplyr::group_by(f, g, .drop = FALSE)
  expect_equal(tibble::as_tibble(dplyr::group_data(gt)), oracle, ignore_attr = TRUE)
})

test_that(".drop = TRUE (default) never expands factor levels", {
  skip_if_no_gpu()

  df <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c")))
  gt <- tbl_gpu(df) |> dplyr::group_by(f)

  expect_equal(nrow(dplyr::group_data(gt)), 2L)
})

# =============================================================================
# Computed group_by() columns
# =============================================================================

test_that("group_by() with a computed expression adds the column and groups by it", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:5, y = c(10, 20, 30, 40, 50))
  oracle <- dplyr::group_by(df, g = x %/% 2)

  gt <- tbl_gpu(df) |> dplyr::group_by(g = x %/% 2)

  expect_equal(dplyr::group_vars(gt), "g")
  expect_equal(tibble::as_tibble(collect(gt)), tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that("group_by(name = existing_column) is a computed rename, keeps the original column", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:5)
  gt <- tbl_gpu(df) |> dplyr::group_by(g = x)

  expect_equal(dplyr::group_vars(gt), "g")
  expect_equal(names(collect(gt)), c("x", "g"))
})

test_that("group_by() overwriting an existing column recomputes it and groups by the new value", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:5, y = c(10, 20, 30, 40, 50))
  oracle <- dplyr::group_by(df, x = x %/% 2)

  gt <- tbl_gpu(df) |> dplyr::group_by(x = x %/% 2)

  expect_equal(dplyr::group_vars(gt), "x")
  expect_equal(names(collect(gt)), c("x", "y"))
  expect_equal(tibble::as_tibble(collect(gt)), tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that("group_by() lets a later computed dot reference an earlier one", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:5, y = c(10, 20, 30, 40, 50))
  oracle <- dplyr::group_by(df, g = x %/% 2, h = g * 10)

  gt <- tbl_gpu(df) |> dplyr::group_by(g = x %/% 2, h = g * 10)

  expect_equal(dplyr::group_vars(gt), c("g", "h"))
  expect_equal(tibble::as_tibble(collect(gt)), tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that("group_by() preserves mixed named/unnamed argument order", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:5, y = c(10, 20, 30, 40, 50))

  gt1 <- tbl_gpu(df) |> dplyr::group_by(y, g = x %/% 2)
  expect_equal(dplyr::group_vars(gt1), c("y", "g"))

  gt2 <- tbl_gpu(df) |> dplyr::group_by(g = x %/% 2, y)
  expect_equal(dplyr::group_vars(gt2), c("g", "y"))
})

test_that(".add = TRUE combines a computed group with existing groups", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:5, y = c(10, 20, 30, 40, 50))
  gt <- tbl_gpu(df) |> dplyr::group_by(x)
  gt2 <- dplyr::group_by(gt, g = y %/% 10, .add = TRUE)

  expect_equal(dplyr::group_vars(gt2), c("x", "g"))
})

test_that("group_by() with no dots and .add = FALSE (default) clears existing groups", {
  skip_if_no_gpu()

  gt <- tbl_gpu(mtcars) |> dplyr::group_by(cyl)
  expect_equal(dplyr::group_vars(dplyr::group_by(gt)), character(0))
})

test_that("group_by() with no dots and .add = TRUE keeps existing groups", {
  skip_if_no_gpu()

  gt <- tbl_gpu(mtcars) |> dplyr::group_by(cyl)
  expect_equal(dplyr::group_vars(dplyr::group_by(gt, .add = TRUE)), "cyl")
})

# =============================================================================
# tbl_nongroup_vars() / group_by_drop_default()
# =============================================================================

test_that("tbl_nongroup_vars() excludes the grouping columns", {
  skip_if_no_gpu()

  gt <- tbl_gpu(mtcars) |> dplyr::group_by(cyl, gear)
  expect_equal(dplyr::tbl_nongroup_vars(gt), setdiff(names(mtcars), c("cyl", "gear")))
})

test_that("group_by_drop_default() reflects the most recent .drop=", {
  skip_if_no_gpu()

  gt <- tbl_gpu(mtcars)
  expect_true(dplyr::group_by_drop_default(gt))

  gt2 <- dplyr::group_by(gt, cyl, .drop = FALSE)
  expect_false(dplyr::group_by_drop_default(gt2))
})

# =============================================================================
# Extension contract: dplyr_row_slice() / dplyr_col_modify() /
# dplyr_reconstruct() / 1-d `[`
# =============================================================================

test_that("dplyr_row_slice() gathers rows by position, matching dplyr", {
  skip_if_no_gpu()

  df <- data.frame(x = c(10, 20, 30, 40, 50))
  gt <- tbl_gpu(df)

  out <- dplyr::dplyr_row_slice(gt, c(3L, 1L))
  expect_equal(tibble::as_tibble(collect(out)), tibble::as_tibble(df[c(3, 1), , drop = FALSE]))
})

test_that("dplyr_row_slice() recomputes group_data() for the survivors (default .drop = TRUE)", {
  skip_if_no_gpu()

  gf <- dplyr::group_by(tbl_gpu(data.frame(g = c(1, 1, 2, 2, 3, 3))), g)

  out <- dplyr::dplyr_row_slice(gf, c(1L, 3L, 5L))
  expect_equal(dplyr::group_data(out)$.rows, vctrs::list_of(1L, 2L, 3L))

  out2 <- dplyr::dplyr_row_slice(gf, c(4L, 3L))
  expect_equal(dplyr::group_data(out2)$.rows, vctrs::list_of(c(1L, 2L)))
})

test_that("dplyr_row_slice() accepts a logical mask", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5)
  gt <- tbl_gpu(df)
  out <- dplyr::dplyr_row_slice(gt, df$x > 2)

  expect_equal(collect(out)$x, c(3L, 4L, 5L))
})

test_that("dplyr_col_modify() can add, remove, and replace columns", {
  skip_if_no_gpu()

  df <- data.frame(x = 1, y = 2)
  gt <- tbl_gpu(df)

  out1 <- dplyr::dplyr_col_modify(gt, list(y = NULL))
  expect_equal(tibble::as_tibble(collect(out1)), tibble::as_tibble(data.frame(x = 1)))

  out2 <- dplyr::dplyr_col_modify(gt, list(y = 3))
  expect_equal(tibble::as_tibble(collect(out2)), tibble::as_tibble(data.frame(x = 1, y = 3)))

  out3 <- dplyr::dplyr_col_modify(gt, list(z = 3))
  expect_equal(tibble::as_tibble(collect(out3)), tibble::as_tibble(data.frame(x = 1, y = 2, z = 3)))
})

test_that("dplyr_col_modify() applies tidyverse recycling rules", {
  skip_if_no_gpu()

  gt <- tbl_gpu(data.frame(x = 1:2))
  out <- dplyr::dplyr_col_modify(gt, list(y = 1))
  expect_equal(tibble::as_tibble(collect(out)), tibble::as_tibble(data.frame(x = 1:2, y = c(1, 1))))

  expect_error(
    dplyr::dplyr_col_modify(tbl_gpu(data.frame(x = 1:4)), list(y = 1:2)),
    class = "vctrs_error_recycle_incompatible_size"
  )
})

test_that("dplyr_col_modify() with empty cols returns input unchanged", {
  skip_if_no_gpu()

  gt <- tbl_gpu(data.frame(x = 1))
  expect_identical(dplyr::dplyr_col_modify(gt, list()), gt)
})

test_that("dplyr_reconstruct() is the identity for an already-built tbl_gpu", {
  skip_if_no_gpu()

  gt <- tbl_gpu(data.frame(x = 1))
  gt_grouped <- dplyr::group_by(tbl_gpu(data.frame(x = 1)), x)

  out <- dplyr::dplyr_reconstruct(gt, gt_grouped)
  expect_true(is_tbl_gpu(out))
})

test_that("`[.tbl_gpu` selects columns by position, name, negation, and logical mask", {
  skip_if_no_gpu()

  df <- data.frame(g = 1:3, x = c(10, 20, 30), s = letters[1:3])
  gt <- tbl_gpu(df)

  expect_equal(names(collect(gt[1])), "g")
  expect_equal(names(collect(gt["x"])), "x")
  expect_equal(names(collect(gt[-1])), c("x", "s"))
  expect_equal(names(collect(gt[c(TRUE, FALSE, TRUE)])), c("g", "s"))
})

test_that("`[.tbl_gpu` rejects 2-d indexing instead of silently misinterpreting it", {
  skip_if_no_gpu()

  gt <- tbl_gpu(data.frame(g = 1:3, x = 1:3))
  expect_error(gt[1, 2], "2-d")
})

test_that("`[.tbl_gpu` errors on an undefined column", {
  skip_if_no_gpu()

  gt <- tbl_gpu(data.frame(g = 1:3))
  expect_error(gt["nope"], "Undefined")
})

# =============================================================================
# Fallback flips: the metadata six are GPU-native; the iteration family
# (group_split/map/modify/nest/nest_by/group_trim/rowwise) stays fallback
# =============================================================================

test_that("group_data()/group_keys()/group_indices()/group_size()/n_groups() never trigger a fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gt <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = 1:3)) |> dplyr::group_by(g)

  expect_no_error(dplyr::group_data(gt))
  expect_no_error(dplyr::group_keys(gt))
  expect_no_error(dplyr::group_rows(gt))
  expect_no_error(dplyr::group_indices(gt))
  expect_no_error(dplyr::group_size(gt))
  expect_no_error(dplyr::n_groups(gt))
})

test_that("the iteration family (group_split/map/modify/nest/nest_by) still falls back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  gt <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = 1:3)) |> dplyr::group_by(g)

  expect_warning(dplyr::group_split(gt), "fell back to CPU evaluation")
  expect_warning(dplyr::group_map(gt, ~ sum(.x$x)), "fell back to CPU evaluation")
  expect_warning(dplyr::group_modify(gt, ~ dplyr::summarise(.x, s = sum(x))), "fell back to CPU evaluation")
  expect_warning(dplyr::group_nest(gt), "fell back to CPU evaluation")
  expect_warning(dplyr::nest_by(tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = 1:3)), g), "fell back to CPU evaluation")
})

test_that("rowwise() still falls back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  gt <- tbl_gpu(tibble::tibble(x = 1:3))
  expect_warning(dplyr::rowwise(gt), "fell back to CPU evaluation")
})

test_that("group_trim() still falls back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  gt <- tbl_gpu(tibble::tibble(g = c(1, 1, 2, 2, 3), x = 1:5)) |>
    dplyr::group_by(g) |>
    dplyr::filter(g != 2)

  expect_warning(dplyr::group_trim(gt), "fell back to CPU evaluation")
})
