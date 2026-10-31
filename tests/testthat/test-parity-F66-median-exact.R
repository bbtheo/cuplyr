# F66 acceptance tests: roll_median() is exact by default.
#
# Default engine: roll_median_sorted(), identical() to stats::median() per
# right-aligned window (R averages the two middle values of an even window in
# long double; data.table::frollmedian() averages in double and can be one
# rounding step off). Opt-in: options(cuplyr.roll_median = "frollmedian") (or
# CUPLYR_ROLL_MEDIAN) uses data.table::frollmedian() when data.table >= 1.18.4
# is available, else falls back to the exact engine with a one-time message of
# class "cuplyr_message_roll_median_exact".

f66_ref <- function(x, n, na.rm = FALSE) {
  n <- as.integer(n)
  nx <- length(x)
  out <- rep(NA_real_, nx)
  if (nx == 0 || n > nx) return(out)
  x <- as.double(x)
  for (i in seq.int(n, nx)) {
    w <- x[(i - n + 1L):i]
    if (na.rm) {
      w <- w[!is.na(w)]
      if (length(w) == 0) next
    } else if (anyNA(w)) {
      next
    }
    out[i] <- stats::median(w)
  }
  out
}

# Values whose pairwise midpoints round differently in double (data.table)
# and in R's long-double mean(): data.table is off on 21/13/5 rows at
# w = 2/4/6 on this vector (verified on data.table 1.18.4).
f66_trap <- function(n = 400) {
  set.seed(66)
  sample(c(1, 2^-53 + 2^-70, 3, -2^-52 - 2^-71, 1e-300, 0.1), n, TRUE) *
    sample(c(1, 2, 0.5), n, TRUE)
}

# Default engine: no option, no env var.
f66_local_default <- function(env = parent.frame()) {
  withr::local_options(cuplyr.roll_median = NULL, .local_envir = env)
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA, .local_envir = env)
}

f66_local_reset_note <- function(env = parent.frame()) {
  old <- roll_median_state$noted
  roll_median_state$noted <- FALSE
  withr::defer(roll_median_state$noted <- old, envir = env)
}

# -----------------------------------------------------------------------------
# Default: exact
# -----------------------------------------------------------------------------

test_that("F66: default roll_median() is R's median on the ulp-trap pair", {
  f66_local_default()
  x <- c(1, 2^-53 + 2^-70)
  expect_identical(roll_median(x, 2), c(NA, 0.5))
  expect_identical(roll_median(x, 2), f66_ref(x, 2))
})

test_that("F66: default roll_median() never takes the data.table path", {
  f66_local_default()
  local_mocked_bindings(roll_median_dt = function(...) stop("data.table path used"))
  x <- f66_trap()
  for (w in c(1L, 2L, 3L, 4L, 6L, 10L)) {
    for (na_rm in c(FALSE, TRUE)) {
      expect_identical(roll_median(x, w, na.rm = na_rm), f66_ref(x, w, na_rm),
                       info = sprintf("w=%d na.rm=%s", w, na_rm))
    }
  }
})

test_that("F66: default roll_median() is identical to the reference where data.table is not", {
  f66_local_default()
  skip_if_not_installed("data.table", "1.18.4")
  x <- f66_trap()
  for (w in c(2L, 4L, 6L)) {
    ref <- f66_ref(x, w)
    # the trap has teeth: data.table differs from R here
    expect_false(identical(data.table::frollmedian(x, w), ref), info = sprintf("w=%d", w))
    expect_identical(roll_median(x, w), ref, info = sprintf("w=%d", w))
  }
})

test_that("F66: default roll_median() handles NA/NaN/Inf/int/lgl/empty exactly", {
  f66_local_default()
  set.seed(6601)
  cases <- list(
    c(NA, NA, NA, 1, NaN, NaN, 2, 3),
    c(Inf, -Inf, 1, 2, Inf, NA, -Inf, 0),
    c(1e308, 1.5e308, 1.7e308, -1.7e308, -1.6e308, 1e308),
    c(3L, NA, 1L, 4L, 1L, 5L, 9L, 2L, 6L),
    c(TRUE, FALSE, NA, TRUE, TRUE, FALSE),
    sample(c(f66_trap(60), NA, NaN, Inf), 80, TRUE)
  )
  for (y in cases) {
    for (w in c(1L, 2L, 3L, 4L, 5L)) {
      for (na_rm in c(FALSE, TRUE)) {
        got <- roll_median(y, w, na.rm = na_rm)
        expect_type(got, "double")
        expect_identical(got, f66_ref(y, w, na_rm),
                         info = sprintf("w=%d na.rm=%s x=%s", w, na_rm,
                                        paste(utils::head(y, 8), collapse = ",")))
      }
    }
  }
  expect_identical(roll_median(numeric(0), 3), numeric(0))
  expect_identical(roll_median(integer(0), 1, na.rm = TRUE), numeric(0))
  expect_identical(roll_median(c(1, 2), 5), c(NA_real_, NA_real_))
  expect_error(roll_median(1:5, 0), "`n` must be a single positive whole number.", fixed = TRUE)
})

test_that("F66: default roll_median() in tbl_gpu pipelines (grouped/ungrouped, eager/lazy) is exact", {
  skip_if_no_gpu()
  f66_local_default()
  withr::local_options(cuplyr.fallback = "silent")
  n <- 600
  set.seed(6602)
  df <- data.frame(g = sample(c(1:3, NA), n, TRUE), x = f66_trap(n))
  df$x[sample.int(n, 20)] <- NA
  df$x[sample.int(n, 8)] <- NaN
  df$x[sample.int(n, 5)] <- Inf
  for (lazy in c(FALSE, TRUE)) {
    for (na_rm in c(FALSE, TRUE)) {
      for (w in c(2L, 4L)) {
        info <- sprintf("lazy=%s na.rm=%s w=%d", lazy, na_rm, w)
        got <- tbl_gpu(df, lazy = lazy) |>
          dplyr::group_by(g) |>
          dplyr::mutate(m = roll_median(x, w, na.rm = na_rm)) |>
          collect()
        ref <- rep(NA_real_, n)
        for (idx in split(seq_len(n), factor(df$g, exclude = NULL))) {
          ref[idx] <- f66_ref(df$x[idx], w, na_rm)
        }
        expect_identical(got$m, ref, info = paste("grouped", info))
        expect_identical(dplyr::group_vars(got), "g", info = info)

        got_u <- tbl_gpu(df, lazy = lazy) |>
          dplyr::mutate(m = roll_median(x, w, na.rm = na_rm)) |>
          collect()
        expect_identical(got_u$m, f66_ref(df$x, w, na_rm), info = paste("ungrouped", info))
      }
    }
  }
  # zero rows
  empty <- df[0, ]
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(empty, lazy = lazy) |>
      dplyr::mutate(m = roll_median(x, 3)) |>
      collect()
    expect_identical(nrow(got), 0L)
    expect_type(got$m, "double")
  }
})

test_that("F66: data.table::frollmedian() written in mutate() keeps data.table's own result", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table", "1.18.4")
  f66_local_default()
  withr::local_options(cuplyr.fallback = "silent")
  df <- data.frame(g = rep(1:2, each = 200), x = f66_trap(400))
  oracle <- tibble::as_tibble(df) |>
    dplyr::group_by(g) |>
    dplyr::mutate(b = data.table::frollmedian(x, 2), m = roll_median(x, 2))
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(df, lazy = lazy) |>
      dplyr::group_by(g) |>
      dplyr::mutate(b = data.table::frollmedian(x, 2), m = roll_median(x, 2)) |>
      collect()
    expect_identical(got$b, oracle$b, info = sprintf("lazy=%s", lazy))
    expect_identical(got$m, oracle$m, info = sprintf("lazy=%s", lazy))
  }
  # and they really are different engines on this data
  expect_false(identical(oracle$b, oracle$m))
})

# -----------------------------------------------------------------------------
# Opt-in: data.table::frollmedian()
# -----------------------------------------------------------------------------

test_that("F66: cuplyr.roll_median = 'frollmedian' gives data.table's result", {
  skip_if_not_installed("data.table", "1.18.4")
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  withr::local_options(cuplyr.roll_median = "frollmedian")
  x <- c(1, 2^-53 + 2^-70)
  expect_identical(roll_median(x, 2), data.table::frollmedian(x, 2))
  expect_false(identical(roll_median(x, 2), f66_ref(x, 2)))
  y <- f66_trap()
  y[c(5, 50, 51, 200)] <- NA
  for (w in c(2L, 3L, 4L, 6L, 11L)) {
    for (na_rm in c(FALSE, TRUE)) {
      expect_identical(roll_median(y, w, na.rm = na_rm),
                       data.table::frollmedian(y, w, na.rm = na_rm),
                       info = sprintf("w=%d na.rm=%s", w, na_rm))
    }
  }
  # NaN is missing (F55) on this path too
  expect_identical(roll_median(c(1, NaN, 3, 4), 2), c(NA, NA, NA, 3.5))
  expect_identical(roll_median(c(1, NaN, 3, 4), 2, na.rm = TRUE), c(NA, 1, 3, 3.5))
  # midpoint overflow is still recomputed exactly
  big <- c(1e308, 1.5e308, 1.7e308)
  expect_identical(roll_median(big, 2), f66_ref(big, 2))
})

test_that("F66: CUPLYR_ROLL_MEDIAN selects the engine when the option is unset; the option wins", {
  skip_if_not_installed("data.table", "1.18.4")
  x <- c(1, 2^-53 + 2^-70)
  withr::local_options(cuplyr.roll_median = NULL)
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = "frollmedian")
  expect_identical(roll_median(x, 2), data.table::frollmedian(x, 2))
  withr::local_options(cuplyr.roll_median = "exact")
  expect_identical(roll_median(x, 2), c(NA, 0.5))
  withr::local_options(cuplyr.roll_median = NULL)
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = "exact")
  expect_identical(roll_median(x, 2), c(NA, 0.5))
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = "")
  expect_identical(roll_median(x, 2), c(NA, 0.5))
})

test_that("F66: the opt-in reaches tbl_gpu pipelines (grouped, eager and lazy)", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table", "1.18.4")
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  withr::local_options(cuplyr.roll_median = "frollmedian", cuplyr.fallback = "silent")
  df <- data.frame(g = rep(c(2L, 1L), each = 200), x = f66_trap(400))
  df$x[c(7, 210, 211)] <- NA
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(df, lazy = lazy) |>
      dplyr::group_by(g) |>
      dplyr::mutate(m = roll_median(x, 4, na.rm = TRUE)) |>
      collect()
    ref <- c(data.table::frollmedian(df$x[1:200], 4, na.rm = TRUE),
             data.table::frollmedian(df$x[201:400], 4, na.rm = TRUE))
    expect_identical(got$m, ref, info = sprintf("lazy=%s", lazy))
  }
})

test_that("F66: opt-in without a usable data.table falls back to exact with a one-time message", {
  f66_local_reset_note()
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  withr::local_options(cuplyr.roll_median = "frollmedian")
  local_mocked_bindings(
    roll_has_frollmedian = function() FALSE,
    roll_median_dt = function(...) stop("data.table path used")
  )
  x <- f66_trap(60)
  got <- NULL
  cnd <- expect_message(got <- roll_median(x, 2), class = "cuplyr_message_roll_median_exact")
  expect_match(conditionMessage(cnd), "frollmedian", fixed = TRUE)
  expect_match(conditionMessage(cnd), "data.table", fixed = TRUE)
  expect_identical(got, f66_ref(x, 2))
  # once per session
  expect_no_message(got2 <- roll_median(x, 4, na.rm = TRUE))
  expect_identical(got2, f66_ref(x, 4, TRUE))
})

test_that("F66: the default engine never shows the fallback message", {
  f66_local_reset_note()
  f66_local_default()
  local_mocked_bindings(roll_has_frollmedian = function() FALSE)
  expect_no_message(roll_median(c(1, 2, 3), 2))
  withr::local_options(cuplyr.roll_median = "exact")
  expect_no_message(roll_median(c(1, 2, 3), 2))
})

test_that("F66: an invalid cuplyr.roll_median value warns, naming the option, and the exact engine runs", {
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  # An invalid value warns once per session; NA, 1 and TRUE may have been used
  # by another file, so forget what was warned about before each one.
  forget <- function() rm(list = ls(cuplyr_opt_state, all.names = TRUE), envir = cuplyr_opt_state)
  for (bad in list("fast-f66a", NA_character_, c("exact", "frollmedian"), 1, TRUE)) {
    forget()
    withr::local_options(cuplyr.roll_median = bad)
    expect_warning(res <- roll_median(c(1, 2, 3), 2), "cuplyr.roll_median", fixed = TRUE,
                   class = "cuplyr_warning_invalid_option",
                   info = paste(deparse(bad), collapse = " "))
    expect_identical(res, c(NA, 1.5, 2.5))
  }
  withr::local_options(cuplyr.roll_median = NULL)
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = "bogus-f66a")
  expect_warning(res <- roll_median(c(1, 2, 3), 2), "cuplyr.roll_median", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  expect_identical(res, c(NA, 1.5, 2.5))
})

# -----------------------------------------------------------------------------
# Cost of the exact default
# -----------------------------------------------------------------------------

test_that("F66: the exact default stays fast at 1e6 rows for w = 11/101/1001", {
  f66_local_default()
  set.seed(6603)
  x <- rnorm(1e6)
  x[sample.int(1e6, 1e4)] <- NA
  roll_median(x[1:1000], 11) # warm up
  for (w in c(11L, 101L, 1001L)) {
    for (na_rm in c(FALSE, TRUE)) {
      t <- system.time(r <- roll_median(x, w, na.rm = na_rm))[["elapsed"]]
      expect_lt(t, 2)
      idx <- 20000:20400
      expect_identical(r[idx], f66_ref(x[(20000 - w + 1):20400], w, na_rm)[-(1:(w - 1))],
                       info = sprintf("w=%d na.rm=%s", w, na_rm))
    }
  }
})

# -----------------------------------------------------------------------------
# Added after round-1 review B1: na.rm is coerced the way the reference
# (roll_apply_generic(), `if (na.rm)`) and roll_mean() coerce it, on both
# engines; a non-flag na.rm is an error, never a silent na.rm = FALSE.
# -----------------------------------------------------------------------------

test_that("F66: roll_median() coerces na.rm like the reference and roll_mean()", {
  f66_local_default()
  x <- c(1, NA, 3, 4)
  for (v in list(1, "TRUE", 0, "FALSE", TRUE, FALSE)) {
    expect_identical(roll_median(x, 2, na.rm = v), f66_ref(x, 2, na.rm = if (v) TRUE else FALSE),
                     info = paste(deparse(v), collapse = " "))
  }
  expect_identical(roll_median(x, 2, na.rm = 1), c(NA, 1, 3, 3.5))
  expect_error(roll_median(x, 2, na.rm = NA))
  expect_error(roll_median(x, 2, na.rm = c(TRUE, TRUE)))
  skip_if_not_installed("data.table", "1.18.4")
  withr::local_options(cuplyr.roll_median = "frollmedian")
  expect_identical(roll_median(x, 2, na.rm = 1), c(NA, 1, 3, 3.5))
  expect_identical(roll_median(x, 2, na.rm = "FALSE"), c(NA, NA, NA, 3.5))
  expect_error(roll_median(x, 2, na.rm = NA))
})

test_that("F66: na.rm = 1 inside tbl_gpu mutate() agrees with roll_mean() (eager and lazy)", {
  skip_if_no_gpu()
  f66_local_default()
  withr::local_options(cuplyr.fallback = "silent")
  df <- data.frame(x = c(1, NA, 3, 4))
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(df, lazy = lazy) |>
      dplyr::mutate(a = roll_mean(x, 2, na.rm = 1), m = roll_median(x, 2, na.rm = 1)) |>
      collect()
    expect_identical(got$m, c(NA, 1, 3, 3.5), info = sprintf("lazy=%s", lazy))
    expect_identical(got$a, c(NA, 1, 3, 3.5), info = sprintf("lazy=%s", lazy))
  }
})

# -----------------------------------------------------------------------------
# Added after round-2 review B1: on the opt-in, a window whose two middle
# values overflow is recomputed exactly even when the window also holds +-Inf
# (data.table returns +-Inf there; stats::median() is finite). Also pins the
# na.rm = TRUE recompute on a window holding an NA, and that a non-flag na.rm
# on short input returns all-NA like the reference.
# -----------------------------------------------------------------------------

test_that("F66: opt-in overflow recompute holds next to Inf and with NA in the window", {
  skip_if_not_installed("data.table", "1.18.4")
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  withr::local_options(cuplyr.roll_median = "frollmedian")
  expect_identical(roll_median(c(1, 1.5e308, 1.7e308, Inf), 4), c(NA, NA, NA, 1.6e308))
  expect_identical(roll_median(c(-1.5e308, -1.7e308, -Inf, 0), 4), c(NA, NA, NA, -1.6e308))
  expect_identical(roll_median(c(1, 1.5e308, NA, 1.7e308, Inf), 5, na.rm = TRUE),
                   c(NA, NA, NA, NA, 1.6e308))
  expect_identical(roll_median(c(1.5e308, NA, 1.7e308), 3, na.rm = TRUE), c(NA, NA, 1.6e308))
  # genuinely infinite / NaN medians stay as R gives them
  expect_identical(roll_median(c(Inf, Inf, 1, 2), 2), c(NA, Inf, Inf, 1.5))
  expect_identical(roll_median(c(Inf, -Inf, 1), 2), c(NA, NaN, -Inf))
  y <- c(1, 1.5e308, 1.7e308, Inf, -1.7e308, -1.5e308, -Inf, NA, 2, 1.6e308, 1.65e308)
  for (w in 2:6) {
    for (na_rm in c(FALSE, TRUE)) {
      expect_identical(roll_median(y, w, na.rm = na_rm), f66_ref(y, w, na_rm),
                       info = sprintf("w=%d na.rm=%s", w, na_rm))
    }
  }
})

test_that("F66: opt-in overflow next to Inf inside tbl_gpu mutate() (eager and lazy)", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table", "1.18.4")
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = NA)
  withr::local_options(cuplyr.roll_median = "frollmedian", cuplyr.fallback = "silent")
  df <- data.frame(x = c(1, 1.5e308, 1.7e308, Inf))
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(df, lazy = lazy) |> dplyr::mutate(m = roll_median(x, 4)) |> collect()
    expect_identical(got$m, c(NA, NA, NA, 1.6e308), info = sprintf("lazy=%s", lazy))
  }
})

test_that("F66: a non-flag na.rm on short input returns all-NA like the reference", {
  f66_local_default()
  expect_identical(roll_median(c(1, 2), 5, na.rm = NA), c(NA_real_, NA_real_))
  expect_identical(roll_mean(c(1, 2), 5, na.rm = NA), c(NA_real_, NA_real_))
})
