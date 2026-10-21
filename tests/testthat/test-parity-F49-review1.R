# F49 acceptance tests added after review round 1 (blocking findings B1-B3,
# scratchpad/parity/features/F49/reviews/round-1.md). dplyr 1.2.1 behaviour
# pinned with tools/r on plain tibbles before writing them.

f49r_col_types <- function(x) {
  vapply(x, function(col) paste(c(class(col), typeof(col)), collapse = "/"), character(1))
}

# dplyr oracle vs a native (no CPU fallback) tbl_gpu run: class, vars, column
# types, data, and NaN positions (expect_equal can't be relied on to tell NaN
# from NA).
f49r_check <- function(df, pipeline, lazy = FALSE) {
  oracle <- pipeline(tibble::as_tibble(df))
  withr::local_options(cuplyr.fallback = "error")
  got <- pipeline(tbl_gpu(df, lazy = lazy))
  expect_true(is_tbl_gpu(got))
  got <- collect(got)
  expect_identical(class(got), class(oracle))
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(oracle))
  expect_identical(f49r_col_types(got), f49r_col_types(oracle))
  expect_equal(got, oracle)
  for (nm in names(oracle)) {
    if (is.double(oracle[[nm]])) {
      expect_identical(is.nan(got[[nm]]), is.nan(oracle[[nm]]), label = paste0("is.nan(", nm, ")"))
      expect_identical(is.na(got[[nm]]), is.na(oracle[[nm]]), label = paste0("is.na(", nm, ")"))
    }
  }
  invisible(got)
}

# ---------------------------------------------------------------------------
# B2. A NaN computed on the GPU is missing for na.rm = TRUE, and min()/max()
# don't depend on operand order (R: NaN propagates, any NA trumps all NaNs)
# ---------------------------------------------------------------------------

test_that("row aggregates treat a GPU-computed NaN like dplyr, in either operand order", {
  skip_if_no_gpu()
  # row 1: r = NaN, z = 5; row 2: r = 0.5; row 3: r = NaN, z = NA; row 4: r = NA
  df <- tibble::tibble(x = c(0, 1, 0, NA), y = c(0, 2, 0, 1), z = c(5, 6, NA, 3))
  for (lazy in c(FALSE, TRUE)) {
    f49r_check(df, function(d) {
      dplyr::rowwise(d) |>
        dplyr::mutate(
          r = x / y,
          s_rm = sum(c(r, z), na.rm = TRUE),
          s_rm2 = sum(dplyr::c_across(c(z, r)), na.rm = TRUE),
          m_rm = mean(c(r, z), na.rm = TRUE),
          p_rm = prod(r, z, na.rm = TRUE),
          mn_zr = min(c(z, r)),
          mn_rz = min(c(r, z)),
          mx_zr = max(z, r),
          mx_rz = max(r, z)
        )
    }, lazy = lazy)
  }
  # rows without an all-missing pair (row 3 would make min/max(na.rm = TRUE)
  # warn and leave sum(NaN, NA) platform-dependent in R)
  df2 <- df[c(1, 2, 4), ]
  for (lazy in c(FALSE, TRUE)) {
    f49r_check(df2, function(d) {
      dplyr::rowwise(d) |>
        dplyr::mutate(
          r = x / y,
          s = sum(c(r, z)),
          m = mean(c(z, r)),
          p = prod(c(r, z)),
          mx_rz_rm = max(c(r, z), na.rm = TRUE),
          mx_zr_rm = max(z, r, na.rm = TRUE),
          mn_rz_rm = min(dplyr::c_across(c(r, z)), na.rm = TRUE),
          mn_zr_rm = min(z, r, na.rm = TRUE)
        )
    }, lazy = lazy)
  }
})

# ---------------------------------------------------------------------------
# B3. .keep = "used"/"unused" count the columns a rewritten aggregate reads
# ---------------------------------------------------------------------------

test_that(".keep = \"used\"/\"unused\" see the columns of rewritten row aggregates", {
  skip_if_no_gpu()
  df <- tibble::tibble(p = 1:2, q = 3:4, w = 5:6)
  pipes <- list(
    function(d) dplyr::mutate(dplyr::rowwise(d), k = length(dplyr::c_across(p:q)), .keep = "unused"),
    function(d) dplyr::mutate(dplyr::rowwise(d), k = length(c(p, q)), .keep = "used"),
    function(d) dplyr::mutate(dplyr::rowwise(d), k = sum(dplyr::c_across(p:q)), .keep = "unused"),
    function(d) dplyr::mutate(dplyr::rowwise(d), k = sum(dplyr::c_across(p:q)), .keep = "used"),
    function(d) dplyr::mutate(dplyr::rowwise(d), k = max(p, q), .keep = "used"),
    function(d) dplyr::mutate(dplyr::rowwise(d), k = mean(p), .keep = "unused"),
    function(d) dplyr::mutate(dplyr::rowwise(d), k = dplyr::n(), .keep = "used"),
    function(d) dplyr::mutate(dplyr::rowwise(d, w), k = sum(c(p, q)), .keep = "unused"),
    function(d) dplyr::mutate(dplyr::rowwise(d, w), k = length(dplyr::c_across(p:q)), .keep = "used")
  )
  for (lazy in c(FALSE, TRUE)) {
    for (pipe in pipes) {
      f49r_check(df, pipe, lazy = lazy)
    }
  }
})

# ---------------------------------------------------------------------------
# B1. Whole-column `[<-` (and names<-) keep a rowwise table rowwise
# ---------------------------------------------------------------------------

test_that("whole-column [<- and names<- on a rowwise tbl_gpu match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)

  want <- dplyr::rowwise(df, x)[1]
  want[, "z"] <- 3:1
  got <- dplyr::rowwise(tbl_gpu(df), x)[1]
  got[, "z"] <- 3:1
  expect_true(is_tbl_gpu(got))
  expect_identical(dplyr::group_vars(got), "x")
  expect_identical(collect(got), want)

  names(want) <- toupper(names(want))
  names(got) <- toupper(names(got))
  expect_identical(dplyr::group_vars(got), "X")
  expect_identical(collect(got), want)

  want2 <- dplyr::rowwise(df, x)
  want2["w"] <- 7:9
  want2[, "y"] <- c(0L, 0L, 1L)
  got2 <- dplyr::rowwise(tbl_gpu(df), x)
  got2["w"] <- 7:9
  got2[, "y"] <- c(0L, 0L, 1L)
  expect_identical(collect(got2), want2)
})

test_that("whole-column [<- works on a plain tbl_gpu too", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)
  want <- df
  want[, "z"] <- 3:1
  want["y"] <- c(1.5, 2.5, 3.5)
  got <- tbl_gpu(df)
  got[, "z"] <- 3:1
  got["y"] <- c(1.5, 2.5, 3.5)
  expect_true(is_tbl_gpu(got))
  expect_identical(collect(got), want)
})
