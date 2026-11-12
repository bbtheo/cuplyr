# Extra F04 tests (the acceptance files are PM-owned): user columns that share
# a name with tbl_gpu's internal fields, the package-internal access rule,
# and assignment/grouping details the acceptance tests don't reach.

f04x_global <- function(fn) {
  environment(fn) <- globalenv()
  fn
}

f04x_df <- function() {
  data.frame(
    ptr = 1:4, schema = c("a", "b", "a", "c"), groups = c(1.5, 2.5, NA, 4),
    lazy_ops = c(TRUE, FALSE, NA, TRUE), exec_mode = 4:1,
    group_drop = c("w", "x", "y", "z"), i = c(1L, NA, 3L, 4L),
    stringsAsFactors = FALSE
  )
}

test_that("user code reads columns named like the internal fields, in every shape", {
  skip_if_no_gpu()
  df <- f04x_df()
  shapes <- list(
    eager = tbl_gpu(df, lazy = FALSE),
    lazy = as_lazy(tbl_gpu(df)),
    pending = dplyr::filter(as_lazy(tbl_gpu(df)), i > 1L | is.na(i)),
    grouped = dplyr::group_by(tbl_gpu(df), schema)
  )
  rows <- list(eager = 1:4, lazy = 1:4, pending = c(2L, 3L, 4L), grouped = 1:4)
  for (nm in names(shapes)) {
    for (col in setdiff(names(df), "i")) {
      get_dollar <- function(t) NULL
      body(get_dollar) <- call("$", quote(t), as.name(col))
      get_dollar <- f04x_global(get_dollar)
      get_dbl <- function(t) NULL
      body(get_dbl) <- call("[[", quote(t), col)
      get_dbl <- f04x_global(get_dbl)
      expect_identical(get_dollar(shapes[[nm]]), df[[col]][rows[[nm]]], info = paste(nm, col))
      expect_identical(get_dbl(shapes[[nm]]), df[[col]][rows[[nm]]], info = paste(nm, col))
    }
  }
})

test_that("package-internal code still reaches the raw fields", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  gt <- dplyr::group_by(tbl_gpu(f04x_df()), schema)
  expect_true(inherits(gt$ptr, "externalptr"))
  expect_identical(gt$groups, "schema")
  expect_identical(gt[["exec_mode"]], "eager")
  gt$lazy_ops <- NULL
  gt[["group_drop"]] <- FALSE
  expect_false(unclass(gt)$group_drop)
  expect_identical(names(gt), names(f04x_df()))
})

test_that("assigning to a field-named column from user code edits the column only", {
  skip_if_no_gpu()
  gt <- tbl_gpu(f04x_df())
  fn <- f04x_global(function(t) {
    t$groups <- 0
    t[["ptr"]] <- NULL
    t
  })
  out <- fn(gt)
  expect_s3_class(out, "tbl_gpu")
  expect_identical(unclass(out)$groups, character()) # the metadata field is untouched
  expect_identical(
    tibble::as_tibble(collect(out)),
    tibble::as_tibble(dplyr::select(transform(f04x_df(), groups = 0), -ptr))
  )
  expect_identical(dplyr::group_vars(out), character())
})

test_that("assigning a column whose type is a factor/Date round-trips, groups are kept", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b", "a"), x = 1:3, stringsAsFactors = FALSE)
  gd <- dplyr::group_by(tbl_gpu(df), g, .drop = FALSE)
  fn <- f04x_global(function(t) {
    t$f <- factor(c("u", "v", "u"))
    t$d <- as.Date("2024-01-01") + 0:2
    t
  })
  out <- fn(gd)
  expect_identical(dplyr::group_vars(out), "g")
  expect_false(dplyr::group_by_drop_default(out))
  oracle <- fn(dplyr::group_by(tibble::as_tibble(df), g, .drop = FALSE))
  expect_identical(collect(out), oracle)
})

test_that("removing the only column, and a table's last remaining column, is an error", {
  skip_if_no_gpu()
  gt <- tbl_gpu(data.frame(a = 1:2))
  fn <- f04x_global(function(t) {
    t$a <- NULL
    t
  })
  expect_error(fn(gt), "zero columns")
  expect_identical(names(gt), "a")
})

test_that("assigning replaces the only column of a one-column table", {
  skip_if_no_gpu()
  gt <- tbl_gpu(data.frame(a = 1:2))
  fn <- f04x_global(function(t) {
    t$a <- c("x", "y")
    t
  })
  expect_identical(tibble::as_tibble(collect(fn(gt))), tibble::tibble(a = c("x", "y")))
})

test_that("a lazy table with pending ops stays lazy after an assignment", {
  skip_if_no_gpu()
  lz <- dplyr::filter(as_lazy(tbl_gpu(data.frame(a = 1:5, b = 5:1))), a > 2L)
  fn <- f04x_global(function(t) {
    t$c <- 1
    t
  })
  out <- fn(lz)
  expect_identical(unclass(out)$exec_mode, "lazy")
  expect_identical(tibble::as_tibble(collect(out)), tibble::tibble(a = 3:5, b = 3:1, c = 1))
  # NULL assignment doesn't run the plan
  drop <- f04x_global(function(t) {
    t$b <- NULL
    t
  })
  d <- drop(lz)
  expect_true(has_pending_ops(d))
  expect_identical(names(d), "a")
})

test_that("dplyr_col_modify() keeps groups and exec_mode", {
  skip_if_no_gpu()
  gd <- dplyr::group_by(as_lazy(tbl_gpu(data.frame(g = c(1L, 2L), x = c(1, 2)))), g, .drop = FALSE)
  out <- dplyr::dplyr_col_modify(gd, list(y = c(5, 6), x = NULL))
  expect_identical(dplyr::group_vars(out), "g")
  expect_false(dplyr::group_by_drop_default(out))
  expect_identical(out$exec_mode, "lazy")
  expect_identical(names(out), c("g", "y"))
})

test_that("as_tibble()/as.data.frame() drop groups, collect() keeps them, internal frames are ungrouped", {
  skip_if_no_gpu()
  gd <- dplyr::group_by(tbl_gpu(data.frame(g = c("a", "b"), x = 1:2)), g)
  expect_s3_class(collect(gd), "grouped_df")
  expect_false(inherits(tibble::as_tibble(gd), "grouped_df"))
  expect_identical(class(as.data.frame(gd)), "data.frame")
  expect_false(inherits(collect_frame(gd), "grouped_df"))
  # a CPU fallback verb still sees the grouping it expects
  out <- dplyr::slice_min(gd, x, n = 1)
  expect_identical(dplyr::group_vars(out), "g")
})

test_that("ncol/NCOL on a lazy table with pending ops don't run the plan, dim does", {
  skip_if_no_gpu()
  lz <- dplyr::filter(as_lazy(tbl_gpu(data.frame(a = 1:5, b = 5:1))), a > 2L)
  n <- 0L
  local_mocked_bindings(compute = function(...) {
    n <<- n + 1L
    stop("executed")
  }, .package = "cuplyr")
  expect_identical(ncol(lz), 2L)
  expect_identical(NCOL(lz), 2L)
  expect_error(nrow(lz), "executed")
  expect_error(dim(lz), "executed")
  expect_identical(n, 2L)
})

test_that("rbind() keeps .drop of the first table; cbind() of a grouped table keeps its groups", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), x = 1:2)
  gd <- dplyr::group_by(tbl_gpu(df), g, .drop = FALSE)
  out <- rbind(gd, gd)
  expect_false(dplyr::group_by_drop_default(out))
  expect_identical(dplyr::group_vars(cbind(gd, tbl_gpu(data.frame(y = 3:4)))), "g")
})
