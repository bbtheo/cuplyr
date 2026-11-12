# F81 acceptance tests: a Date/POSIXct value the GPU type can't hold becomes NA
# with one classed warning per upload.
#
# Contract (see the F81 design, `?tbl_gpu`):
#   - +-Inf: `cuplyr_warning_infinite_timestamp`.
#   - Finite beyond TIMESTAMP_DAYS' int32 days / TIMESTAMP_MICROSECONDS' int64
#     microseconds: `cuplyr_warning_timestamp_out_of_range`.
#   - Both also inherit `cuplyr_warning_timestamp_unrepresentable`.
#   - One warning per class per upload, naming every affected column with its
#     count: `d` (2 values), `t` (1 value).
#   - NA and NaN are legitimate nulls: no warning.

f81_inf <- "cuplyr_warning_infinite_timestamp"
f81_oor <- "cuplyr_warning_timestamp_out_of_range"
f81_parent <- "cuplyr_warning_timestamp_unrepresentable"

f81_cap <- function(expr) {
  warnings <- list()
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      warnings[[length(warnings) + 1L]] <<- w
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = warnings)
}

f81_of <- function(cap, class) {
  Filter(function(w) inherits(w, class), cap$warnings)
}

# Exactly one warning of `class`, and return its message.
f81_one <- function(cap, class) {
  got <- f81_of(cap, class)
  expect_length(got, 1L)
  if (length(got) == 0L) return("")
  # (cli may wrap a long message on spaces)
  gsub("\\s+", " ", conditionMessage(got[[1L]]))
}

f81_none <- function(cap) {
  expect_length(f81_of(cap, f81_parent), 0L)
}

f81_ct <- function(x) as.POSIXct(x, origin = "1970-01-01", tz = "UTC")
f81_vals <- function(x) as.double(unclass(x))

f81_lazy_flags <- c(FALSE, TRUE)

f81_df <- tibble::tibble(
  a = 1:3,
  t = f81_ct(c(1, 2, NA)),
  d = as.Date(c(1, 2, NA))
)

# ---- +-Inf ----------------------------------------------------------------

test_that("+-Inf in a Date column warns once and collects NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(d = as.Date(c(1, Inf, -Inf, NA, NaN)))
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(collect(tbl_gpu(df, lazy = lazy)))
    expect_match(f81_one(cap, f81_inf), "`d` \\(2 values\\)")
    expect_length(cap$warnings, 1L)
    expect_s3_class(cap$warnings[[1L]], f81_parent)
    expect_s3_class(cap$warnings[[1L]], "rlang_warning")
    expect_s3_class(cap$value$d, "Date")
    expect_identical(f81_vals(cap$value$d), c(1, NA, NA, NA, NA))
  }
})

test_that("+-Inf in a POSIXct column warns once, collects NA, keeps the zone", {
  skip_if_no_gpu()
  df <- tibble::tibble(t = as.POSIXct(c(1.5, Inf, -Inf, NA, NaN), origin = "1970-01-01",
                                      tz = "America/New_York"))
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(collect(tbl_gpu(df, lazy = lazy)))
    expect_match(f81_one(cap, f81_inf), "`t` \\(2 values\\)")
    expect_length(cap$warnings, 1L)
    expect_s3_class(cap$value$t, "POSIXct")
    expect_identical(attr(cap$value$t, "tzone"), "America/New_York")
    expect_identical(f81_vals(cap$value$t), c(1.5, NA, NA, NA, NA))
  }
})

test_that("a single infinite value is counted in the singular", {
  skip_if_no_gpu()
  cap <- f81_cap(tbl_gpu(tibble::tibble(d = as.Date(c(1, Inf)))))
  expect_match(f81_one(cap, f81_inf), "`d` \\(1 value\\)")
})

test_that("the upload warns once, later verbs and collect() do not warn again", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:3, d = as.Date(c(1, Inf, 3)))
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap({
      g <- tbl_gpu(df, lazy = lazy)
      g |>
        mutate(b = a + 1L) |>
        filter(a > 0L) |>
        arrange(a) |>
        group_by(a) |>
        summarise(m = max(d), .groups = "drop") |>
        collect()
      collect(g)
    })
    f81_one(cap, f81_inf)
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$d), c(1, NA, 3))
  }
})

test_that("several affected columns give one warning naming all of them", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    ok = as.Date(c(1, 2, 3)),
    d = as.Date(c(Inf, -Inf, 3)),
    x = c(1, Inf, 3),
    t = f81_ct(c(1, 2, Inf))
  )
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(collect(tbl_gpu(df, lazy = lazy)))
    msg <- f81_one(cap, f81_inf)
    expect_match(msg, "`d` \\(2 values\\)")
    expect_match(msg, "`t` \\(1 value\\)")
    expect_no_match(msg, "`ok`")
    expect_no_match(msg, "`x`")
    expect_length(cap$warnings, 1L)
    # A plain double keeps its Inf.
    expect_identical(cap$value$x, c(1, Inf, 3))
  }
})

# ---- legitimate nulls -----------------------------------------------------

test_that("NA and NaN alone do not warn", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    d = as.Date(c(1, NA, NaN)),
    t = f81_ct(c(1, NA, NaN)),
    id = structure(c(1L, NA, 3L), class = "Date"),
    it = structure(c(1L, NA, 3L), class = c("POSIXct", "POSIXt"), tzone = "UTC"),
    allna = as.Date(c(NA_real_, NA_real_, NA_real_))
  )
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(collect(tbl_gpu(df, lazy = lazy)))
    expect_length(cap$warnings, 0L)
    expect_identical(f81_vals(cap$value$d), c(1, NA, NA))
    expect_identical(f81_vals(cap$value$t), c(1, NA, NA))
    expect_identical(f81_vals(cap$value$id), c(1, NA, 3))
    expect_identical(f81_vals(cap$value$it), c(1, NA, 3))
  }
})

test_that("zero-row uploads do not warn", {
  skip_if_no_gpu()
  df <- tibble::tibble(d = as.Date(double()), t = f81_ct(double()))
  cap <- f81_cap(collect(tbl_gpu(df)))
  expect_length(cap$warnings, 0L)
  expect_identical(nrow(cap$value), 0L)
  expect_s3_class(cap$value$d, "Date")
  expect_s3_class(cap$value$t, "POSIXct")
})

test_that("ordinary and far-future values round-trip without a warning", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    d = as.Date(c("0001-01-01", "1969-12-31", "9999-12-31")),
    t = f81_ct(c(-1e10, 0.25, 1e10))
  )
  cap <- f81_cap(collect(tbl_gpu(df)))
  expect_length(cap$warnings, 0L)
  expect_identical(f81_vals(cap$value$d), f81_vals(df$d))
  expect_identical(f81_vals(cap$value$t), f81_vals(df$t))
})

# ---- finite but out of range ----------------------------------------------

test_that("Date values beyond int32 days warn and collect NA, not garbage", {
  skip_if_no_gpu()
  df <- tibble::tibble(d = as.Date(c(0, 3e9, -3e9, 1e300, -1e300, 5)))
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(collect(tbl_gpu(df, lazy = lazy)))
    expect_match(f81_one(cap, f81_oor), "`d` \\(4 values\\)")
    expect_length(cap$warnings, 1L)
    expect_s3_class(cap$warnings[[1L]], f81_parent)
    expect_identical(f81_vals(cap$value$d), c(0, NA, NA, NA, NA, 5))
  }
})

test_that("a Date at the edge of the int32 range round-trips", {
  skip_if_no_gpu()
  edge <- c(-2147483648, -2147483647, 2147483646, 2147483647)
  cap <- f81_cap(collect(tbl_gpu(tibble::tibble(d = as.Date(edge)))))
  expect_length(cap$warnings, 0L)
  expect_identical(f81_vals(cap$value$d), edge)
})

test_that("just past the int32 edge is out of range, on the floored value", {
  skip_if_no_gpu()
  df <- tibble::tibble(d = as.Date(c(2147483648, -2147483649, -2147483648.5)))
  cap <- f81_cap(collect(tbl_gpu(df)))
  expect_match(f81_one(cap, f81_oor), "`d` \\(3 values\\)")
  # Out of range is not also reported as fractional.
  expect_length(f81_of(cap, "cuplyr_warning_fractional_date"), 0L)
  expect_identical(f81_vals(cap$value$d), c(NA_real_, NA_real_, NA_real_))
})

test_that("a fraction inside the top day still floors with the fractional warning", {
  skip_if_no_gpu()
  df <- tibble::tibble(d = as.Date(2147483647.5))
  cap <- f81_cap(collect(tbl_gpu(df)))
  expect_length(f81_of(cap, f81_parent), 0L)
  expect_length(f81_of(cap, "cuplyr_warning_fractional_date"), 1L)
  expect_identical(f81_vals(cap$value$d), 2147483647)
})

test_that("POSIXct values beyond int64 microseconds warn and collect NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(t = f81_ct(c(1, 1e16, -1e16, 1e300, 9223372036855, -9223372036855)))
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(collect(tbl_gpu(df, lazy = lazy)))
    expect_match(f81_one(cap, f81_oor), "`t` \\(5 values\\)")
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$t), c(1, NA, NA, NA, NA, NA))
  }
})

test_that("a POSIXct near the int64 microsecond edge round-trips", {
  skip_if_no_gpu()
  edge <- c(-9223372036854, -9.2e12, 9.2e12, 9223372036854)
  cap <- f81_cap(collect(tbl_gpu(tibble::tibble(t = f81_ct(edge)))))
  expect_length(cap$warnings, 0L)
  expect_identical(f81_vals(cap$value$t), edge)
})

test_that("each kind has its own warning in one upload, fractions too", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    inf = as.Date(c(Inf, 1, 2)),
    far = as.Date(c(3e9, 1, 2)),
    both = f81_ct(c(Inf, 1e16, 1)),
    frac = as.Date(c(1.5, 2, 3))
  )
  cap <- f81_cap(collect(tbl_gpu(df)))
  expect_length(cap$warnings, 3L)
  m_inf <- f81_one(cap, f81_inf)
  m_oor <- f81_one(cap, f81_oor)
  expect_match(m_inf, "`inf` \\(1 value\\)")
  expect_match(m_inf, "`both` \\(1 value\\)")
  expect_no_match(m_inf, "`far`")
  expect_match(m_oor, "`far` \\(1 value\\)")
  expect_match(m_oor, "`both` \\(1 value\\)")
  expect_no_match(m_oor, "`inf`")
  expect_length(f81_of(cap, "cuplyr_warning_fractional_date"), 1L)
  expect_identical(f81_vals(cap$value$both), c(NA, NA, 1))
})

# ---- re-uploads of CPU results --------------------------------------------

f81_inf_date <- function(x) rep(as.Date(Inf), length(x))
f81_far_date <- function(x) rep(as.Date(3e9), length(x))
f81_inf_ct <- function(x) rep(f81_ct(Inf), length(x))
f81_far_ct <- function(x) rep(f81_ct(1e16), length(x))

test_that("a CPU-fallback result holding Inf re-uploads with the warning", {
  skip_if_no_gpu()
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(
      tbl_gpu(f81_df, lazy = lazy) |>
        mutate(e = f81_inf_date(a), s = f81_inf_ct(a)) |>
        collect()
    )
    msg <- f81_one(cap, f81_inf)
    expect_match(msg, "`e` \\(3 values\\)")
    expect_match(msg, "`s` \\(3 values\\)")
    expect_length(cap$warnings, 1L)
    expect_true(all(is.na(cap$value$e)))
    expect_true(all(is.na(cap$value$s)))
    expect_identical(f81_vals(cap$value$d), c(1, 2, NA))
  }
})

test_that("a CPU-fallback result out of range re-uploads with the warning", {
  skip_if_no_gpu()
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(
      tbl_gpu(f81_df, lazy = lazy) |>
        mutate(e = f81_far_date(a), s = f81_far_ct(a)) |>
        collect()
    )
    msg <- f81_one(cap, f81_oor)
    expect_match(msg, "`e` \\(3 values\\)")
    expect_match(msg, "`s` \\(3 values\\)")
    expect_length(cap$warnings, 1L)
    expect_true(all(is.na(cap$value$e)))
    expect_true(all(is.na(cap$value$s)))
  }
})

test_that("a grouped CPU result warns once for the whole re-upload", {
  skip_if_no_gpu()
  for (lazy in f81_lazy_flags) {
    cap <- f81_cap(
      tbl_gpu(f81_df, lazy = lazy) |>
        group_by(a) |>
        group_modify(~ tibble::tibble(q = as.Date(Inf))) |>
        collect()
    )
    expect_match(f81_one(cap, f81_inf), "`q` \\(3 values\\)")
    expect_length(cap$warnings, 1L)
  }
})

test_that("other uploads of local data warn under the column's own name", {
  skip_if_no_gpu()
  g <- tbl_gpu(f81_df)

  y <- tibble::tibble(a = 1:3, far = as.Date(c(Inf, 1, 2)))
  cap <- f81_cap(collect(left_join(g, y, by = "a", copy = TRUE)))
  expect_match(f81_one(cap, f81_inf), "`far` \\(1 value\\)")
  expect_length(cap$warnings, 1L)

  more <- tibble::tibble(a = 9L, t = f81_ct(Inf), d = as.Date(3e9))
  cap <- f81_cap(collect(bind_rows(g, more)))
  expect_match(f81_one(cap, f81_inf), "`t` \\(1 value\\)")
  expect_match(f81_one(cap, f81_oor), "`d` \\(1 value\\)")
  expect_length(cap$warnings, 2L)
  expect_true(is.na(cap$value$t[4L]))
  expect_true(is.na(cap$value$d[4L]))

  cap <- f81_cap({
    g2 <- g
    g2$e <- as.Date(c(1, Inf, 3))
    collect(g2)
  })
  expect_match(f81_one(cap, f81_inf), "`e` \\(1 value\\)")
  expect_length(cap$warnings, 1L)
  expect_identical(f81_vals(cap$value$e), c(1, NA, 3))
})

# ---- literals -------------------------------------------------------------

test_that("comparing with an infinite or out-of-range literal matches dplyr", {
  skip_if_no_gpu()
  pinf <- f81_ct(Inf)
  pbig <- f81_ct(1e16)
  dinf <- as.Date(Inf)
  dbig <- as.Date(3e9)
  pipelines <- list(
    function(x) filter(x, t < pinf),
    function(x) filter(x, t < pbig),
    function(x) filter(x, t > pbig),
    function(x) filter(x, d < dinf),
    function(x) filter(x, d < dbig),
    function(x) filter(x, d == dinf),
    function(x) filter(x, t %in% c(pinf, pbig, f81_ct(1))),
    function(x) filter(x, d %in% c(dinf, dbig, as.Date(1)))
  )
  for (p in pipelines) {
    cap <- f81_cap(expect_same_as_dplyr(f81_df, p))
    f81_none(cap)
    cap <- f81_cap(expect_same_as_dplyr_lazy(f81_df, p))
    f81_none(cap)
  }
})

test_that("a constant that can't be held warns under its output name", {
  skip_if_no_gpu()
  pinf <- f81_ct(Inf)
  pbig <- f81_ct(1e16)
  dinf <- as.Date(Inf)
  dbig <- as.Date(3e9)
  for (lazy in f81_lazy_flags) {
    g <- tbl_gpu(f81_df, lazy = lazy)

    cap <- f81_cap(collect(mutate(g, e = dinf)))
    expect_match(f81_one(cap, f81_inf), "`e` \\(3 values\\)")
    expect_length(cap$warnings, 1L)
    expect_true(all(is.na(cap$value$e)))

    cap <- f81_cap(collect(mutate(g, e = pinf)))
    expect_match(f81_one(cap, f81_inf), "`e` \\(3 values\\)")
    expect_length(cap$warnings, 1L)
    expect_true(all(is.na(cap$value$e)))
    expect_s3_class(cap$value$e, "POSIXct")

    cap <- f81_cap(collect(mutate(g, e = dbig)))
    expect_match(f81_one(cap, f81_oor), "`e` \\(3 values\\)")
    expect_length(cap$warnings, 1L)
    expect_true(all(is.na(cap$value$e)))

    cap <- f81_cap(collect(mutate(g, e = pbig)))
    expect_match(f81_one(cap, f81_oor), "`e` \\(3 values\\)")
    expect_length(cap$warnings, 1L)
    expect_true(all(is.na(cap$value$e)))
  }
})

test_that("a literal inside an expression gives NA for that row, with the warning", {
  skip_if_no_gpu()
  pinf <- f81_ct(Inf)
  pbig <- f81_ct(1e16)
  for (lazy in f81_lazy_flags) {
    g <- tbl_gpu(f81_df, lazy = lazy)

    cap <- f81_cap(collect(mutate(g, e = dplyr::if_else(a > 1L, t, pinf))))
    expect_match(f81_one(cap, f81_inf), "`e` \\(1 value\\)")
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$e), c(NA, 2, NA))

    cap <- f81_cap(collect(mutate(g, e = dplyr::coalesce(t, pbig))))
    expect_match(f81_one(cap, f81_oor), "`e` \\(1 value\\)")
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$e), c(1, 2, NA))

    cap <- f81_cap(collect(mutate(g, e = dplyr::lag(t, default = pinf))))
    expect_match(f81_one(cap, f81_inf), "`e` \\(1 value\\)")
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$e), c(NA, 1, 2))

    cap <- f81_cap(collect(mutate(g, e = dplyr::if_else(a > 1L, d, as.Date(3e9)))))
    expect_match(f81_one(cap, f81_oor), "`e` \\(1 value\\)")
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$e), c(NA, 2, NA))
  }
})

test_that("a whole-column Date constant warns under its output name", {
  skip_if_no_gpu()
  for (lazy in f81_lazy_flags) {
    g <- tbl_gpu(f81_df, lazy = lazy)

    cap <- f81_cap(collect(mutate(g, e = as.Date(c(1, Inf, 3)))))
    msg <- f81_one(cap, f81_inf)
    expect_match(msg, "`e` \\(1 value\\)")
    expect_no_match(msg, "\\.\\.vec")
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$e), c(1, NA, 3))

    cap <- f81_cap(collect(mutate(g, e = as.Date(c(1, 3e9, 3)))))
    msg <- f81_one(cap, f81_oor)
    expect_match(msg, "`e` \\(1 value\\)")
    expect_no_match(msg, "\\.\\.vec")
    expect_length(cap$warnings, 1L)
    expect_identical(f81_vals(cap$value$e), c(1, NA, 3))

    # The fractional warning shares this upload and its naming.
    cap <- f81_cap(collect(mutate(g, e = as.Date(c(1.5, 2, 3)))))
    expect_length(cap$warnings, 1L)
    fw <- f81_of(cap, "cuplyr_warning_fractional_date")
    expect_length(fw, 1L)
    expect_match(conditionMessage(fw[[1L]]), "`e`")
    expect_no_match(conditionMessage(fw[[1L]]), "\\.\\.vec")
  }
})

# ---- bind_cols() (round 1 review, B1) --------------------------------------

test_that("bind_cols() warns under the result's column names", {
  skip_if_no_gpu()
  zv <- as.Date(c(1, Inf, 3))
  sv <- f81_ct(c(1, 1e16, 3))
  for (lazy in f81_lazy_flags) {
    g <- tbl_gpu(tibble::tibble(a = 1:3), lazy = lazy)

    # Named vectors.
    cap <- f81_cap(collect(bind_cols(g, z = zv, s = sv)))
    expect_named(cap$value, c("a", "z", "s"))
    expect_match(f81_one(cap, f81_inf), "`z` \\(1 value\\)")
    expect_match(f81_one(cap, f81_oor), "`s` \\(1 value\\)")
    expect_length(cap$warnings, 2L)
    expect_identical(f81_vals(cap$value$z), c(1, NA, 3))
    expect_identical(f81_vals(cap$value$s), c(1, NA, 3))

    # An unnamed vector carries its repaired name.
    cap <- suppressMessages(f81_cap(collect(bind_cols(g, zv))))
    new_name <- setdiff(names(cap$value), "a")
    expect_length(new_name, 1L)
    msg <- f81_one(cap, f81_inf)
    expect_true(grepl(paste0("`", new_name, "` (1 value)"), msg, fixed = TRUE), info = msg)
    expect_no_match(msg, "`value`")

    # A name repaired because it clashes, from a vector and from a data frame.
    cap <- suppressMessages(f81_cap(collect(bind_cols(g, a = zv))))
    new_name <- names(cap$value)[2L]
    expect_false(identical(new_name, "a"))
    msg <- f81_one(cap, f81_inf)
    expect_true(grepl(paste0("`", new_name, "` (1 value)"), msg, fixed = TRUE), info = msg)

    cap <- suppressMessages(f81_cap(collect(bind_cols(g, tibble::tibble(a = zv)))))
    new_name <- names(cap$value)[2L]
    expect_false(identical(new_name, "a"))
    msg <- f81_one(cap, f81_inf)
    expect_true(grepl(paste0("`", new_name, "` (1 value)"), msg, fixed = TRUE), info = msg)
  }
})

test_that("a named bind_cols() argument leaves the column names plain (round 2, B2)", {
  skip_if_no_gpu()
  tb <- tibble::tibble(a = 1:2)
  named_df <- function(x) bind_cols(x, foo = tibble::tibble(p = 3:4))
  for (lazy in f81_lazy_flags) {
    g <- tbl_gpu(tb, lazy = lazy)
    expect_identical(names(named_df(g)), c("a", "p"))
    expect_identical(names(bind_cols(g, z = 3:4)), c("a", "z"))
  }
  later <- list(
    function(x) dplyr::setdiff(named_df(x), named_df(x)),
    function(x) select(named_df(x), dplyr::all_of(c("a", "p"))),
    function(x) rows_update(named_df(x), tibble::tibble(a = 1L, p = 9L), by = "a", copy = TRUE)
  )
  for (p in later) {
    expect_same_as_dplyr(tb, p)
    expect_same_as_dplyr_lazy(tb, p)
  }
})
