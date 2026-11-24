# #17: extra checks beside the acceptance file (qualified dplyr calls, windows
# that run on the GPU, a repeat that must not be shared across mutate() calls
# whose rows differ, case_when() releasing its clauses' columns), and the rule
# for a repeated checked integer subtree: shared where some use evaluates it
# on its own anyway (typed as at run time), where it doesn't fit the chain, or
# where its range can overflow (with the overflow's NA and its one warning);
# fused into every use otherwise.

cse_n <- function(expr) {
  n0 <- gpu_cse_reuse_count()
  force(expr)
  gpu_cse_reuse_count() - n0
}

# gpu_cse_enable() sets a process-wide switch: this restores it when `env`
# (the calling test) ends, however it ends.
local_cse_enable <- function(on, env = parent.frame()) {
  prev <- gpu_cse_enable(on)
  withr::defer(gpu_cse_enable(prev), envir = env)
  invisible(prev)
}

# The result and the warnings of `expr`.
with_warnings <- function(expr) {
  msgs <- character()
  val <- withCallingHandlers(expr, warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(val = val, msgs = msgs)
}

test_that("a repeated %in% under if_else() is computed once", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(1.5, NA, 3, 4), y = c(1, 2, NA, 4), i = c(1L, 5L, NA, 9L)
  )
  st <- c(1L, 5L, 77L)
  p <- function(d) {
    dplyr::mutate(d,
      a = dplyr::if_else(i %in% st, x, y),
      b = dplyr::if_else(i %in% st, 1L, 0L)
    )
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a repeat around a window aggregate that runs on the GPU is shared", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(1.234, -2.5, 0.75, 0.31, 0.5, 0.06),
    y = c(3, 4, 5, 6, 7, 8),
    g = c("a", "b", "a", "b", "a", "b")
  )
  p <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::mutate(
        a = round(x - mean(y), 2) + 1,
        b = round(x - mean(y), 2) * 2
      )
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_gte(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a repeat is only shared within one mutate() call's rows", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.123, 0.456, 0.789), k = c(1L, 2L, 3L))
  p <- function(d) {
    d |>
      dplyr::mutate(a = round(x, 2) + 1, b = round(x, 2) * 2) |>
      dplyr::filter(k > 1L) |>
      dplyr::mutate(c = round(x, 2) - 1, d = round(x, 2) / 2)
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  # each mutate() shares its own pair (the filter in between changes the rows,
  # so a memo that outlived the first call would hand back a 3-row column)
  expect_equal(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 2)
})

test_that("a shared checked chain's proven range does not hide a later overflow", {
  skip_if_no_gpu()
  # `q` is proven in range from the columns' ranges (its interval is filed under
  # its slot), but `r = q * 1000000L` leaves the int range for the larger rows:
  # the interval of `q`, not an assumption that it is small, must send `r` to
  # the exact path.
  df <- data.frame(
    a = c(1L, 2L, 3L, NA, 1000L, 7L),
    b = c(4L, NA, 6L, 7L, 900L, 1L),
    c = c(1L, 1L, NA, 2L, 5L, 3L)
  )
  p <- function(d) {
    dplyr::mutate(d, s = a + b, q = (a + b) * 2L - c, r = q * 1000000L)
  }
  expected <- suppressWarnings(p(df))
  expect_warning(p(df), "NAs produced by integer overflow")
  got <- NULL
  expect_warning(got <- collect(p(tbl_gpu(df, lazy = FALSE))), "NAs produced by integer overflow")
  expect_equal(tibble::as_tibble(got), tibble::as_tibble(expected))
  expect_equal(cse_n(suppressWarnings(collect(p(tbl_gpu(df, lazy = FALSE))))), 1)
})

test_that("gpu_cse_enable(FALSE) turns the memo off and gives the same answer", {
  skip_if_no_gpu()
  df <- data.frame(x = c(0.123, 0.456, NA, 0.789))
  p <- function(d) dplyr::mutate(d, a = round(x, 2) + 1, b = round(x, 2) * 2)
  on <- collect(p(tbl_gpu(df, lazy = FALSE)))
  local_cse_enable(FALSE)
  n0 <- gpu_cse_reuse_count()
  off <- collect(p(tbl_gpu(df, lazy = FALSE)))
  expect_equal(gpu_cse_reuse_count() - n0, 0)
  expect_equal(on, off)
})

test_that("case_when() releasing a clause's columns drops their memo entries", {
  skip_if_no_gpu()
  # a clause's scratch columns are freed after the clause (release_from()); a
  # memo entry that outlived its slot would hand `b` a freed column (input
  # column 0, `w`, sits in the freed slots)
  df <- data.frame(
    w = c(100, -100, 100, -100, 100, -100, 100),
    x = c(0.53, -0.04, 0.47, NA, NaN, 0.66, -0.31)
  )
  a_first <- function(d) {
    dplyr::mutate(d,
      a = dplyr::case_when(round(x, 1) > 0.5 ~ round(x, 1) * 10, round(x, 1) < 0 ~ -1, .default = 0),
      b = round(x, 1) + 1
    )
  }
  b_first <- function(d) {
    dplyr::mutate(d,
      b = round(x, 1) + 1,
      a = dplyr::case_when(round(x, 1) > 0.5 ~ round(x, 1) * 10, round(x, 1) < 0 ~ -1, .default = 0)
    )
  }
  expect_same_as_dplyr(df, a_first)
  expect_same_as_dplyr_lazy(df, a_first)
  expect_same_as_dplyr(df, b_first)
  expect_same_as_dplyr_lazy(df, b_first)
  expect_equal(cse_n(collect(a_first(tbl_gpu(df, lazy = FALSE)))), 1)
  expect_equal(cse_n(collect(b_first(tbl_gpu(df, lazy = FALSE)))), 3)
})

test_that("an overflow behind a shared slot with a proven interval is caught", {
  skip_if_no_gpu()
  # `/` materializes a + b, whose interval [2, 3900] is filed under its slot;
  # the second predicate reads the slot and leaves the int range
  df <- data.frame(a = c(1L, 2L, 3L, NA, 3000L, 7L), b = c(4L, NA, 6L, 7L, 900L, 1L))
  pf <- function(d) dplyr::filter(d, (a + b) / 2 > 0, (a + b) * 1000000L > 0L)
  pm <- function(d) dplyr::mutate(d, s = (a + b) / 2, t = (a + b) * 1000000L, u = t + 1L)
  for (p in list(pf, pm)) {
    want <- with_warnings(p(df))
    for (lazy in c(FALSE, TRUE)) {
      n0 <- gpu_cse_reuse_count()
      got <- with_warnings(collect(p(tbl_gpu(df, lazy = lazy))))
      expect_equal(tibble::as_tibble(got$val), tibble::as_tibble(want$val))
      expect_gte(length(want$msgs), 1)
      expect_length(got$msgs, 1)
      expect_match(got$msgs, "NAs produced by integer overflow", fixed = TRUE)
      expect_equal(gpu_cse_reuse_count() - n0, 1)
    }
  }
})

test_that("a proven integer subtree that every use fuses is not shared", {
  skip_if_no_gpu()
  df <- data.frame(
    i = c(1L, -3L, NA, 7L, 100L), j = c(2L, 3L, 4L, NA, -5L), k = c(5L, 6L, 7L, 8L, NA)
  )
  for (p in list(
    function(d) dplyr::mutate(d, s = (i + j) * (i + j)),
    function(d) dplyr::mutate(d, s = (i + j) * k + (i + j)),
    function(d) dplyr::mutate(d, s = (i + j) * 2L, t = (i + j) - k)
  )) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    expect_equal(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 0)
  }
})

test_that("an integer subtree under a double operand is shared, typed as at run time", {
  skip_if_no_gpu()
  df <- data.frame(
    i = c(1L, -3L, NA, 7L, 100L), j = c(2L, 3L, 4L, NA, -5L), x = c(0.5, 1, 2, NA, -1)
  )
  # `* x` (a double) materializes i + j; the earlier `* 2L` use, which would
  # fuse it, reads that column instead of computing it a second time
  p1 <- function(d) dplyr::mutate(d, a = (i + j) * 2L, b = (i + j) * x)
  # the same through a column made by an earlier dot: a double one ...
  p2 <- function(d) dplyr::mutate(d, z = x * 1, a = (i + j) * 2L, b = (i + j) * z)
  # ... and an integer one, under which i + j fuses again
  p3 <- function(d) dplyr::mutate(d, z = i * 1L, a = (i + j) * 2L, b = (i + j) * z)
  # an earlier dot that pushes scratch columns moves z's real slot away from
  # the index the IR binds (remap_virtual_col_indices()): z must still read as
  # the double it is
  p4 <- function(d) {
    dplyr::mutate(d,
      w = dplyr::if_else(i %in% c(1L, 7L), j, 0L), z = x * 1,
      a = (i + j) * 2L, b = (i + j) * z
    )
  }
  for (p in list(p1, p2, p3, p4)) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
  expect_equal(cse_n(collect(p1(tbl_gpu(df, lazy = FALSE)))), 1)
  expect_equal(cse_n(collect(p2(tbl_gpu(df, lazy = FALSE)))), 1)
  expect_equal(cse_n(collect(p3(tbl_gpu(df, lazy = FALSE)))), 0)
  expect_equal(cse_n(collect(p4(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("an integer subtree that doesn't fit the room left in a chain is shared", {
  skip_if_no_gpu()
  df <- data.frame(i = c(1L, -3L, NA, 7L, 100L), j = c(2L, 3L, 4L, NA, -5L))
  # two nodes fit alone, but the chain (... - i) * 2L - j has used two of its four
  p <- function(d) dplyr::mutate(d, a = ((i * j + i) - i) * 2L - j, b = (i * j + i) + 1L)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  expect_equal(cse_n(collect(p(tbl_gpu(df, lazy = FALSE)))), 1)
})

test_that("a shared subtree that can overflow keeps a later dot's own range check", {
  skip_if_no_gpu()
  df <- data.frame(
    bi = c(2000000000L, 1L, NA, -2000000000L, 5L),
    bj = c(200000000L, 2L, 3L, -200000000L, NA)
  )
  # `s` is the shared, unproven bi + bj; `t` reads it by name, `u` as a shared
  # leaf; every result and the single warning are dplyr's
  p <- function(d) dplyr::mutate(d, s = bi + bj, t = s * 1L, u = (bi + bj) - 0L, v = (bi + bj) * 2L)
  want <- with_warnings(p(df))
  for (lazy in c(FALSE, TRUE)) {
    n0 <- gpu_cse_reuse_count()
    got <- with_warnings(collect(p(tbl_gpu(df, lazy = lazy))))
    expect_equal(tibble::as_tibble(got$val), tibble::as_tibble(want$val))
    expect_length(got$msgs, 1)
    expect_match(got$msgs, "NAs produced by integer overflow", fixed = TRUE)
    expect_equal(gpu_cse_reuse_count() - n0, 2)
  }
})

test_that("a shared integer subtree that overflows gives every reader R's NA", {
  skip_if_no_gpu()
  # (rows 1 and 4 overflow, rows 3 and 5 have an NA operand; the chains that
  # share bi + bj read its wrapped values and its range tests, `/` reads the
  # column with R's NA in it)
  df <- data.frame(
    bi = c(2000000000L, 1L, NA, -2000000000L, 5L, 7L),
    bj = c(200000000L, 2L, 3L, -200000000L, NA, -9L)
  )
  fused_first <- function(d) dplyr::mutate(d, a = (bi + bj) * 1L, b = (bi + bj) / 2, c = (bi + bj) - bi)
  other_first <- function(d) dplyr::mutate(d, b = (bi + bj) / 2, a = (bi + bj) * 1L, c = (bi + bj) - bi)
  both_fused <- function(d) dplyr::mutate(d, a = (bi + bj) * 3L, c = (bi + bj) - bi)
  pf <- function(d) dplyr::filter(d, (bi + bj) * 1L < 10L, (bi + bj) / 2 > -1)
  cases <- list(
    list(p = fused_first, n = 2), list(p = other_first, n = 2),
    list(p = both_fused, n = 1), list(p = pf, n = 1)
  )
  for (case in cases) {
    want <- suppressWarnings(case$p(df))
    for (lazy in c(FALSE, TRUE)) {
      n0 <- gpu_cse_reuse_count()
      got <- with_warnings(collect(case$p(tbl_gpu(df, lazy = lazy))))
      expect_equal(tibble::as_tibble(got$val), tibble::as_tibble(want))
      expect_length(got$msgs, 1)
      expect_match(got$msgs, "NAs produced by integer overflow", fixed = TRUE)
      expect_equal(gpu_cse_reuse_count() - n0, case$n)
    }
  }
})

test_that("a shared subtree with NA operands but no overflow does not warn", {
  skip_if_no_gpu()
  # bi + bj's interval can leave the int range but no row does: shared, no
  # warning (a stale or wrapped read would invent one)
  df <- data.frame(
    bi = c(2000000000L, 1L, NA, -2000000000L, 5L),
    bj = c(-200000000L, 2L, 3L, 200000000L, NA)
  )
  p <- function(d) dplyr::mutate(d, a = (bi + bj) * 1L, c = (bi + bj) - 7L)
  for (lazy in c(FALSE, TRUE)) {
    n0 <- gpu_cse_reuse_count()
    got <- with_warnings(collect(p(tbl_gpu(df, lazy = lazy))))
    expect_equal(tibble::as_tibble(got$val), tibble::as_tibble(p(df)))
    expect_length(got$msgs, 0)
    expect_equal(gpu_cse_reuse_count() - n0, 1)
  }
})
