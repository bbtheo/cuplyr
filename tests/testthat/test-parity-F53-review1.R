# F53 acceptance tests added after the round-1 review (B1, B2, B4, B5).
# dplyr behaviour verified by the reviewer's probes
# (scratchpad/parity/features/F53/runs/review-1/) and by the PM with tools/r.

# ---- B1: na.omit() drops a NaN computed on the GPU ------------------------------

test_that("na.omit() drops rows holding a NaN computed on the GPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(0, 1, 2, NA), y = c(0, 1, 1, 1))
  want <- stats::na.omit(dplyr::mutate(df, r = x / y))
  attr(want, "na.action") <- NULL
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(stats::na.omit(dplyr::mutate(tbl_gpu(df, lazy = lazy), r = x / y)))
    expect_equal(got, want, info = paste("lazy =", lazy))
    expect_equal(nrow(got), 2L)
  }
})

# ---- B2: a column-free summarise() argument is evaluated like dplyr -------------

test_that("a column-free aggregate argument is evaluated once per group, like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  run <- function(tbl, grouped) {
    i <- 0
    bump <- function() {
      i <<- i + 1
      i
    }
    if (grouped) tbl <- dplyr::group_by(tbl, g)
    out <- dplyr::summarise(tbl, m = max(bump()))
    if (is_tbl_gpu(out)) out <- collect(out)
    list(m = dplyr::ungroup(out)$m, calls = i)
  }
  for (grouped in c(FALSE, TRUE)) {
    want <- run(df, grouped)
    for (lazy in c(FALSE, TRUE)) {
      got <- run(tbl_gpu(df, lazy = lazy), grouped)
      expect_identical(got, want, info = paste("grouped =", grouped, "lazy =", lazy))
    }
  }
})

test_that("a column-free aggregate that warns warns like dplyr (once, same text)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  th <- numeric(0)
  collect_warnings <- function(expr) {
    ws <- character()
    value <- withCallingHandlers(expr, warning = function(w) {
      ws <<- c(ws, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
    list(value = value, warnings = ws)
  }
  for (grouped in c(FALSE, TRUE)) {
    prep <- function(t) if (grouped) dplyr::group_by(t, g) else t
    want <- collect_warnings(dplyr::summarise(prep(df), m = max(th)))
    for (lazy in c(FALSE, TRUE)) {
      got <- collect_warnings(collect(dplyr::summarise(prep(tbl_gpu(df, lazy = lazy)), m = max(th))))
      expect_equal(dplyr::ungroup(got$value), dplyr::ungroup(want$value))
      # same number of warnings (dplyr raises one per group that warns, each
      # wrapped once); the header names the verb call, which may be spelled
      # differently through the CPU route, so only the cause is compared
      info <- paste("grouped =", grouped, "lazy =", lazy)
      expect_length(got$warnings, length(want$warnings))
      expect_true(all(grepl("no non-missing arguments to max", got$warnings)), info = info)
      expect_true(all(grepl("In argument: `m = max(th)`", got$warnings, fixed = TRUE)),
                  info = info)
    }
  }
})

# ---- B4: an empty subscript slot in a column-free argument ----------------------

test_that("summarise()/mutate() of sum(m[, 1]) with m an R matrix match dplyr", {
  skip_if_no_gpu()
  m <- matrix(c(1, 2, 3, 4), nrow = 2)
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(3, 1, 2))
  pipes <- list(
    function(t) dplyr::summarise(t, s = sum(m[, 1])),
    function(t) dplyr::summarise(dplyr::group_by(t, g), s = sum(m[, 1])),
    function(t) dplyr::mutate(t, y = sum(m[, 1]))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

# ---- B5: literal folding keeps a NaN a NaN ------------------------------------------

# F53-a2: split from one mutate() into two. What B5 pins is that literal folding
# must not turn the NaN of `0/0` into NA (the `a` check). A call that also names
# is.nan() runs wholly on the CPU (dev's route), and that route re-uploads `a`,
# which maps NaN to null; that is pre-existing GPU NaN handling (F55), not F53's.

test_that("a literal-only sub-expression that is NaN stays NaN", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1L, 2L, 3L))
  want <- dplyr::mutate(df, a = x + 0/0)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), a = x + 0/0))
    expect_identical(is.nan(got$a), is.nan(want$a))
    expect_identical(is.na(got$a), is.na(want$a))
  }
})

test_that("is.nan() of a column plus a literal-only NaN matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1L, 2L, 3L))
  want <- dplyr::mutate(df, c = is.nan(x + 0/0))
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), c = is.nan(x + 0/0)))
    expect_identical(got$c, want$c)
  }
})
