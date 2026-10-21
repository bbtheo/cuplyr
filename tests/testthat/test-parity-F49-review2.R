# F49 acceptance tests added after review round 2 (blocking findings B4-B5,
# scratchpad/parity/features/F49/reviews/round-2.md). dplyr/tibble behaviour
# pinned with tools/r on plain tibbles (runs/review-2/probe1.log, probe2.log).

# ---------------------------------------------------------------------------
# B4. Whole-column `[<-` on a rowwise table puts the vars in column order,
# as `[<-.rowwise_df` does (intersect(names(out), group_vars(x))).
# ---------------------------------------------------------------------------

test_that("whole-column [<- on a rowwise tbl_gpu orders the vars like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)
  for (lazy in c(FALSE, TRUE)) {
    want <- dplyr::rowwise(df, y, x)
    want["z"] <- 1L
    got <- dplyr::rowwise(tbl_gpu(df, lazy = lazy), y, x)
    got["z"] <- 1L
    expect_true(is_tbl_gpu(got))
    expect_identical(dplyr::group_vars(got), c("x", "y"), info = paste("lazy =", lazy))
    expect_identical(collect(got), want, info = paste("lazy =", lazy))
    expect_identical(
      names(collect(dplyr::summarise(got, s = x + y, .groups = "drop"))),
      names(dplyr::summarise(want, s = x + y, .groups = "drop")),
      info = paste("lazy =", lazy)
    )

    want2 <- dplyr::rowwise(df, y, x)
    want2[, "z"] <- 3:1
    got2 <- dplyr::rowwise(tbl_gpu(df, lazy = lazy), y, x)
    got2[, "z"] <- 3:1
    expect_identical(dplyr::group_vars(got2), c("x", "y"), info = paste("lazy =", lazy))
    expect_identical(collect(got2), want2, info = paste("lazy =", lazy))

    # Dropping a var by `[<-` NULL removes it from the vars, as dplyr does.
    want3 <- dplyr::rowwise(df, y, x)
    want3["y"] <- NULL
    got3 <- dplyr::rowwise(tbl_gpu(df, lazy = lazy), y, x)
    got3["y"] <- NULL
    expect_identical(dplyr::group_vars(got3), dplyr::group_vars(want3), info = paste("lazy =", lazy))
    expect_identical(collect(got3), want3, info = paste("lazy =", lazy))
  }
})

# ---------------------------------------------------------------------------
# B5. Cell assignment through a logical matrix subscript (the
# `x[is.na(x)] <- 0` idiom) must never be a silent no-op: either it errors
# loudly (F04's cell-assignment contract) or it gives tibble's result.
# ---------------------------------------------------------------------------

f49r2_cell_ok <- function(run, want, label) {
  res <- tryCatch(run(), error = function(e) e)
  if (inherits(res, "error")) {
    expect_true(TRUE)
  } else {
    expect_identical(tibble::as_tibble(collect(res)), tibble::as_tibble(want), info = label)
  }
}

test_that("x[is.na(x)] <- 0 on a tbl_gpu is never a silent no-op", {
  skip_if_no_gpu()
  # 6 columns: the same count as a tbl_gpu's internal fields, the case that
  # silently did nothing at 5107d24; 5 and 7 columns for contrast.
  df6 <- tibble::tibble(a = c(1, NA), b = c(NA, 2), c = 1:2, d = 3:4, e = 5:6, f = 7:8)
  dfs <- list(df6, df6[1:5], dplyr::mutate(df6, g = 1:2))
  for (df in dfs) {
    want <- df
    want[is.na(want)] <- 0
    for (lazy in c(FALSE, TRUE)) {
      label <- paste0(ncol(df), " cols, lazy = ", lazy)
      gt <- tbl_gpu(df, lazy = lazy)
      f49r2_cell_ok(function() { t <- gt; t[is.na(t)] <- 0; t }, want, paste("is.na idiom,", label))
      m <- is.na(df)
      f49r2_cell_ok(function() { t <- gt; t[m] <- 0; t }, want, paste("host matrix subscript,", label))
      # x itself is untouched either way.
      expect_identical(tibble::as_tibble(collect(gt)), tibble::as_tibble(df), info = label)
    }
  }

  # Rowwise table: same rule.
  rw <- dplyr::rowwise(df6, c)
  want_rw <- rw
  want_rw[is.na(want_rw)] <- 0
  grw <- dplyr::rowwise(tbl_gpu(df6), c)
  res <- tryCatch({ grw[is.na(grw)] <- 0; grw }, error = function(e) e)
  if (!inherits(res, "error")) {
    expect_identical(collect(res), want_rw)
  }
})

test_that("is.na() on a tbl_gpu does not answer about its internal fields", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, NA), b = c(NA, 2), c = 1:2, d = 3:4, e = 5:6, f = 7:8)
  gt <- tbl_gpu(df)
  res <- tryCatch(is.na(gt), error = function(e) e)
  if (inherits(res, "error")) {
    expect_true(TRUE)
  } else {
    expect_identical(res, is.na(df))
  }
})
