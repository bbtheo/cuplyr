# F71 round 2: the shared rule for fixed group keys of a row slice
# (row_slice_prepare()/row_slice_finish()/row_slice_keys(), R/slice.R) and the
# keys-only capture (slice_group_keys(), gpu_group_keys()). Keys are fixed only
# when a group can have been lost; the group structure always equals dplyr's.

r2_df <- data.frame(
  g = c("a", "a", "b", "c", "d", NA, "b", "d"),
  h = c(1L, 1L, 2L, NA, 2L, 3L, 3L, 1L),
  f = factor(c("x", "x", "y", "z", "x", NA, "y", "y"), levels = c("x", "y", "z", "w")),
  f2 = factor(c("u", "v", "u", "v", "u", "v", "u", "u"), levels = c("u", "v", "t")),
  d = as.Date("2024-01-01") + c(0, 0, 1, 2, 3, 3, 4, 4),
  x = as.double(1:8),
  stringsAsFactors = FALSE
)

r2_keys <- function(x) fixed_keys_of(x)
r2_as <- function(d, ydf) if (is_tbl_gpu(d)) tbl_gpu(ydf) else tibble::as_tibble(ydf)
r2_fell_back <- function(expr) {
  fell <- FALSE
  withCallingHandlers(expr, cuplyr_cpu_fallback = function(cnd) fell <<- TRUE)
  fell
}

test_that("slice_group_keys() equals the keys of compute_group_data()", {
  skip_if_no_gpu()
  shapes <- list(
    list("g"), list("g", drop = FALSE), list("h", drop = FALSE),
    list("f", drop = FALSE), list("f", drop = TRUE), list("f", "f2", drop = FALSE),
    list("g", "f", drop = FALSE), list("f", "g", "h", drop = FALSE),
    list("d", drop = FALSE), list("h", "d")
  )
  for (shape in shapes) {
    drop <- if (!is.null(shape$drop)) shape$drop else TRUE
    cols <- unlist(shape[names(shape) == ""])
    gt <- dplyr::group_by(tbl_gpu(r2_df), dplyr::across(dplyr::all_of(cols)), .drop = drop)
    want <- slice_group_keys_of(compute_group_data(gt))
    expect_identical(slice_group_keys(gt), want,
                     info = paste(cols, collapse = ","), ignore_attr = FALSE)
    lz <- dplyr::group_by(dplyr::mutate(tbl_gpu(r2_df, lazy = TRUE), x = x + 1),
                          dplyr::across(dplyr::all_of(cols)), .drop = drop)
    expect_identical(slice_group_keys(lz), want, info = paste("lazy", paste(cols, collapse = ",")))
  }
  zero <- dplyr::group_by(tbl_gpu(r2_df[0, ]), f, .drop = FALSE)
  expect_identical(slice_group_keys(zero), slice_group_keys_of(compute_group_data(zero)))
})

test_that("gpu_null_counts() counts the NAs of each column", {
  skip_if_no_gpu()
  gt <- tbl_gpu(r2_df)
  expect_identical(gpu_null_counts(gt$ptr, 0:5), c(1L, 1L, 1L, 0L, 0L, 0L))
})

test_that("eager: no group lost means no fixed keys (and summarise() stays native)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  yall <- data.frame(g = c("a", "b", "c", "d", NA))
  gt <- dplyr::group_by(tbl_gpu(r2_df), g, .drop = FALSE)
  semi <- dplyr::semi_join(gt, tbl_gpu(yall), by = "g")
  expect_null(r2_keys(semi))
  expect_false(r2_fell_back(collect(dplyr::summarise(semi, n = dplyr::n()))))
  # a lost group fixes them
  lost <- dplyr::semi_join(gt, tbl_gpu(data.frame(g = "a")), by = "g")
  expect_false(is.null(r2_keys(lost)))
  expect_identical(nrow(r2_keys(lost)), 5L)
  # slices
  expect_null(r2_keys(dplyr::slice_head(gt, n = 1)))
  expect_false(is.null(r2_keys(dplyr::slice(gt, 2))))
  # and the values are dplyr's whichever way
  for (p in list(
    function(d) dplyr::semi_join(dplyr::group_by(d, g, .drop = FALSE), r2_as(d, yall), by = "g") |>
      dplyr::summarise(n = dplyr::n(), s = sum(x)),
    function(d) dplyr::group_by(d, g, .drop = FALSE) |> dplyr::slice_head(n = 1) |> dplyr::count(),
    function(d) dplyr::group_by(d, g, .drop = FALSE) |> dplyr::slice(2) |> dplyr::count()
  )) {
    expect_same_as_dplyr(r2_df, p)
    expect_same_as_dplyr_lazy(r2_df, p)
  }
})

test_that("factor-only keys: none while no NA, kept when an NA group can be lost", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  noNA <- r2_df[!is.na(r2_df$f), ]
  yf <- data.frame(f = factor("x", levels = c("x", "y", "z", "w")))
  for (lazy in c(FALSE, TRUE)) {
    gt <- dplyr::group_by(tbl_gpu(noNA, lazy = lazy), f, f2, .drop = FALSE)
    out <- dplyr::slice(gt, 1)
    expect_null(r2_keys(out))
    out <- dplyr::semi_join(dplyr::group_by(tbl_gpu(noNA, lazy = lazy), f, .drop = FALSE),
                            tbl_gpu(yf), by = "f")
    expect_null(r2_keys(out))
    # with an NA in a key the NA group can go
    gna <- dplyr::group_by(tbl_gpu(r2_df, lazy = lazy), f, .drop = FALSE)
    expect_false(is.null(r2_keys(dplyr::semi_join(gna, tbl_gpu(yf), by = "f"))))
  }
  # .drop = TRUE (with preserve) has no level expansion to rely on
  gt <- dplyr::group_by(tbl_gpu(noNA), f)
  expect_null(r2_keys(dplyr::slice(gt, 1:3, .preserve = TRUE)))
  kept <- dplyr::filter(gt, x < 3, .preserve = TRUE)
  expect_identical(nrow(r2_keys(kept)), 3L)
})

test_that("an equal row count does not mean no group was lost (duplicated slice indices)", {
  skip_if_no_gpu()
  p <- function(d) dplyr::group_by(d, g, .drop = FALSE) |> dplyr::slice(c(1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L))
  expect_same_as_dplyr(r2_df, p)
  expect_same_as_dplyr_lazy(r2_df, p)
  got <- p(tbl_gpu(r2_df))
  expect_identical(dplyr::n_groups(got), 5L)
  p2 <- function(d) dplyr::group_by(d, g) |> dplyr::slice(rep(1L, 8), .preserve = TRUE) |> dplyr::count()
  expect_same_as_dplyr(r2_df, p2)
})

test_that("preserve = TRUE on a .drop = TRUE table: keys only when a group is lost", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.exec_mode = "eager")  # about the eager path
  gt <- dplyr::group_by(tbl_gpu(r2_df), g)
  expect_null(r2_keys(dplyr::filter(gt, x > 0, .preserve = TRUE)))
  kept <- dplyr::filter(gt, x < 3, .preserve = TRUE)
  expect_identical(dplyr::n_groups(kept), dplyr::n_groups(gt))
  expect_identical(nrow(r2_keys(kept)), 5L)
  # .drop = TRUE, preserve = FALSE: the lost groups go
  expect_identical(dplyr::n_groups(dplyr::filter(gt, x < 3)), 1L)
  expect_null(r2_keys(dplyr::filter(gt, x < 3)))
  # dplyr_row_slice()
  expect_null(r2_keys(dplyr::dplyr_row_slice(gt, 1:8, preserve = TRUE)))
  expect_identical(
    dplyr::n_groups(dplyr::dplyr_row_slice(gt, 1:2, preserve = TRUE)),
    dplyr::n_groups(dplyr::dplyr_row_slice(dplyr::group_by(tibble::as_tibble(r2_df), g), 1:2, preserve = TRUE))
  )
})

test_that("keys the input already has are carried through the slice verbs", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    base <- dplyr::group_by(tbl_gpu(r2_df, lazy = lazy), g, .drop = FALSE)
    p <- function(d) dplyr::group_by(d, g, .drop = FALSE) |> dplyr::filter(x < 3) |>
      dplyr::slice_head(n = 1) |> dplyr::semi_join(r2_as(d, data.frame(g = "a")), by = "g") |>
      dplyr::summarise(n = dplyr::n())
    expect_same_as_dplyr(r2_df, p)
    expect_same_as_dplyr_lazy(r2_df, p)
  }
  lz <- dplyr::filter(dplyr::group_by(tbl_gpu(r2_df, lazy = TRUE), g, .drop = FALSE), x < 3)
  expect_identical(nrow(r2_keys(dplyr::slice_head(lz, n = 1))), 5L)
})

test_that("set operations run their join on ungrouped rows and keep groups and .drop", {
  skip_if_no_gpu()
  sub <- r2_df[c(1, 2, 7), ]
  testthat::local_mocked_bindings(
    slice_group_keys = function(...) stop("captured group keys for a set operation"),
    .package = "cuplyr"
  )
  for (op in list(dplyr::intersect, dplyr::setdiff, dplyr::symdiff)) {
    for (lazy in c(FALSE, TRUE)) {
      gt <- dplyr::group_by(tbl_gpu(r2_df, lazy = lazy), g, .drop = FALSE)
      res <- op(gt, tbl_gpu(sub))
      ref <- op(dplyr::group_by(tibble::as_tibble(r2_df), g, .drop = FALSE),
                tibble::as_tibble(sub))
      expect_identical(group_vars(res), "g")
      expect_null(r2_keys(res))
      expect_identical(isFALSE(attr(group_data(res), ".drop")),
                       isFALSE(attr(group_data(ref), ".drop")))
      expect_equal(tibble::as_tibble(dplyr::ungroup(collect(res))),
                   tibble::as_tibble(dplyr::ungroup(ref)))
    }
  }
})

test_that("a lazy table with fixed keys counts its groups without running the plan", {
  skip_if_no_gpu()
  lz <- dplyr::group_by(tbl_gpu(r2_df, lazy = TRUE), g, .drop = FALSE) |>
    dplyr::slice(2)
  expect_true(has_pending_ops(lz))
  expect_false(is.null(r2_keys(lz)))
  testthat::local_mocked_bindings(
    compute = function(...) stop("ran the plan"),
    gpu_group_count = function(...) stop("counted on the GPU"),
    .package = "cuplyr"
  )
  expect_identical(dplyr::n_groups(lz), 5L)
  expect_match(paste(capture.output(print(lz)), collapse = "\n"), "Groups:\\s+g \\[5\\]")
})

test_that("gpu_distinct_key_count() agrees with gpu_group_count()", {
  skip_if_no_gpu()
  df <- data.frame(
    d = c(NaN, NA, 1, NaN, NA, -0, 0, Inf, -Inf, NaN),
    i = c(1L, NA, 1L, NA, 2L, 1L, 1L, 2L, NA, 2L),
    s = c("a", NA, "a", "b", NA, "", "", "a", "b", NA),
    f = factor(c("x", NA, "y", "x", NA, "y", "x", "x", "y", NA)),
    dt = as.Date("2020-01-01") + c(0, NA, 0, 1, 1, 2, 2, NA, 3, 3),
    b = c(TRUE, NA, FALSE, TRUE, NA, FALSE, TRUE, TRUE, NA, FALSE)
  )
  for (tbl in list(df, df[0, ])) {
    gt <- tbl_gpu(tbl)
    for (cols in list("d", "i", "s", "f", "dt", "b", c("d", "i"), c("s", "f"),
                      c("d", "s", "dt"), c("b", "f", "i"))) {
      idx <- as.integer(match(cols, names(tbl)) - 1L)
      expect_identical(gpu_distinct_key_count(.subset2(gt, "ptr"), idx),
                       gpu_group_count(.subset2(gt, "ptr"), idx),
                       info = paste(cols, collapse = ","))
    }
  }
})

test_that("gpu_distinct_key_count() agrees with gpu_group_count() on a GPU-computed NaN key", {
  skip_if_no_gpu()
  # 0/0 is NaN on the GPU; an R NaN would upload as NA
  df <- data.frame(a = c(0, 0, 1, 1, 2, NA, 0), b = c(0, 0, 0, 1, 1, 1, 0))
  gt <- dplyr::compute(dplyr::mutate(tbl_gpu(df), k = a / b))
  kp <- .subset2(gt, "ptr")
  idx <- as.integer(match("k", gt$schema$names) - 1L)
  expect_identical(gpu_distinct_key_count(kp, idx), gpu_group_count(kp, idx))
  idx2 <- as.integer(match(c("k", "b"), gt$schema$names) - 1L)
  expect_identical(gpu_distinct_key_count(kp, idx2), gpu_group_count(kp, idx2))
})

test_that("lazy .drop = FALSE factor, no-NA filter() that empties a level keeps summarise() native", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("x", "x", "y", "z", "x", "y"), levels = c("x", "y", "z", "w")),
    x = as.double(1:6)
  )
  pipe <- function(d) {
    d |>
      dplyr::group_by(f, .drop = FALSE) |>
      dplyr::filter(x < 3) |>
      dplyr::summarise(n = dplyr::n(), s = sum(x))
  }
  got <- NULL
  fell <- r2_fell_back(got <- pipe(tbl_gpu(df, lazy = TRUE)))
  expect_false(fell)
  expect_equal(tibble::as_tibble(collect(got)), pipe(tibble::as_tibble(df)))
})
