# Issue #24 acceptance tests, part 2: Date/POSIXct vector constants the GPU
# can't hold exactly, as.POSIXct() of a Date beyond the POSIXct range, and
# bind_cols() name edges.
#
# Contract:
# - dplyr keeps a Date/POSIXct vector constant exactly (+-Inf, a fractional
#   day, a date beyond the GPU's range), so a later expression reading it sees
#   that value. The GPU stores such a column with NA (or floored) and warns
#   (`cuplyr_warning_timestamp_unrepresentable` / `cuplyr_warning_fractional_date`);
#   the later expressions must still give dplyr's values.
# - That warning is given once per mutate()/transmute() call: in lazy mode when
#   the call is made, never again when the plan runs (collect(), compute(),
#   print(), joins of the plan).
# - as.POSIXct() of a Date is a double in R and never overflows. A GPU POSIXct
#   holds int64 microseconds, i.e. dates within +-106751991 days. A Date beyond
#   that gives what the CPU route would give after upload: NA with
#   `cuplyr_warning_timestamp_out_of_range`. Never a wrapped value.
# - bind_cols() is vctrs::vec_cbind() then dplyr_reconstruct(out, first)
#   (dplyr 1.2.1 R/bind-cols.R): "check_unique" raises vctrs'
#   `vctrs_error_names_must_be_unique` / `vctrs_error_names_cannot_be_empty`; a
#   grouped first input gives a grouped result and fails on blank names with
#   `tibble_error_column_names_cannot_be_empty` before any data is touched; a
#   blank-named factor keeps its levels.

# The dplyr side evaluates these by name, as in a session with dplyr attached;
# cuplyr reads the calls by name.
if_any <- dplyr::if_any
if_all <- dplyr::if_all
if_else <- dplyr::if_else

p24_modes <- c(eager = FALSE, lazy = TRUE)

p24_ts_classes <- c("cuplyr_warning_timestamp_unrepresentable", "cuplyr_warning_fractional_date")

# Run `expr`, muffling and counting the timestamp warnings it gives.
p24_count_ts <- function(expr) {
  msgs <- character()
  handler <- function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
  withCallingHandlers(
    expr,
    cuplyr_warning_timestamp_unrepresentable = handler,
    cuplyr_warning_fractional_date = handler
  )
  msgs
}

# --- item 6 -------------------------------------------------------------------

test_that("a later expression reading a lossy Date/POSIXct vector constant matches dplyr", {
  skip_if_no_gpu()
  g <- tibble::tibble(a = 1:3)
  pipes <- list(
    inf = function(t) mutate(t, e = as.Date(c(1, Inf, 3)), f = is.na(e), h = e > as.Date(2)),
    neg_inf = function(t) mutate(t, e = as.Date(c(1, -Inf, 3)), f = is.na(e), h = e < as.Date(2)),
    beyond = function(t) mutate(t, e = as.Date(c(1, 3e9, 3)), f = is.na(e), h = e > as.Date(2)),
    posixct_beyond = function(t) mutate(t, e = as.POSIXct(c(1, 1e17, 3), tz = "UTC"), f = is.na(e)),
    posixct_inf = function(t) mutate(t, e = as.POSIXct(c(1, Inf, 3), tz = "UTC"), f = is.na(e)),
    trans_inf = function(t) transmute(t, a, e = as.Date(c(1, Inf, 3)), f = is.na(e)),
    if_else_reader = function(t) mutate(t, e = as.Date(c(1, Inf, 3)), f = if_else(is.na(e), 0L, a))
  )
  for (nm in names(pipes)) {
    want <- pipes[[nm]](g)
    for (lz in p24_modes) {
      msgs <- p24_count_ts(got <- collect(pipes[[nm]](tbl_gpu(g, lazy = lz))))
      expect_length(msgs, 1L)
      expect_equal(dplyr::select(got, -e), dplyr::select(want, -e), info = paste(nm, "lazy =", lz))
      # the stored column itself is NA where the GPU can't hold the value
      expect_identical(is.na(got$e), c(FALSE, TRUE, FALSE), info = paste(nm, "lazy =", lz))
    }
  }
})

test_that("a later expression reading a fractional Date vector constant still matches dplyr", {
  skip_if_no_gpu()
  g <- tibble::tibble(a = 1:3)
  p <- function(t) mutate(t, e = as.Date(c(1.5, 2, 3)), f = as.numeric(e))
  for (lz in p24_modes) {
    msgs <- p24_count_ts(got <- collect(p(tbl_gpu(g, lazy = lz))))
    expect_length(msgs, 1L)
    expect_equal(got$f, p(g)$f)
  }
})

# --- item 7 -------------------------------------------------------------------

test_that("a Date vector constant warns once per mutate() call, not once per run", {
  skip_if_no_gpu()
  g <- tibble::tibble(a = 1:3)
  consts <- list(
    inf = quote(as.Date(c(1, Inf, 3))),
    beyond = quote(as.Date(c(1, 3e9, 3))),
    fractional = quote(as.Date(c(1.5, 2, 3))),
    posixct_inf = quote(as.POSIXct(c(1, Inf, 3), tz = "UTC"))
  )
  for (nm in names(consts)) {
    cst <- consts[[nm]]
    # lazy: once when the plan is built, then silent however often it runs
    msgs <- p24_count_ts(lz <- mutate(tbl_gpu(g, lazy = TRUE), e = !!cst))
    expect_length(msgs, 1L)
    expect_match(msgs, "`e`", fixed = TRUE)
    expect_length(p24_count_ts(r1 <- collect(lz)), 0L)
    expect_length(p24_count_ts(collect(lz)), 0L)
    expect_length(p24_count_ts(compute(lz)), 0L)
    expect_length(p24_count_ts(utils::capture.output(print(lz))), 0L)
    expect_length(p24_count_ts(collect(inner_join(lz, lz, by = "a"))), 0L)
    expect_length(p24_count_ts(collect(filter(lz, a > 1L))), 0L)
    expect_identical(nrow(r1), 3L)

    # transmute(), lazy
    msgs <- p24_count_ts(lt <- transmute(tbl_gpu(g, lazy = TRUE), a, e = !!cst))
    expect_length(msgs, 1L)
    expect_length(p24_count_ts(collect(lt)), 0L)

    # eager: once, at the call
    msgs <- p24_count_ts(ea <- mutate(tbl_gpu(g, lazy = FALSE), e = !!cst))
    expect_length(msgs, 1L)
    expect_length(p24_count_ts(collect(ea)), 0L)

    # same stored values either way
    expect_equal(collect(ea), r1, info = nm)
  }
})

# --- item 8 -------------------------------------------------------------------

test_that("as.POSIXct() of a Date beyond the POSIXct range is NA with a warning, never wrapped", {
  skip_if_no_gpu()
  days <- c(1e8, 106751991, -106751991, 106751992, -106751992, 1.1e8, 2e9, NA, 0, -2e9)
  dd <- tibble::tibble(d = as.Date(days))
  beyond <- !is.na(days) & abs(days) > 106751991
  want <- as.POSIXct(dd$d)
  want[beyond] <- NA
  for (lz in p24_modes) {
    expect_warning(
      got <- collect(mutate(tbl_gpu(dd, lazy = lz), p = as.POSIXct(d))),
      class = "cuplyr_warning_timestamp_out_of_range"
    )
    expect_equal(got$p, want, info = paste("lazy =", lz))
    expect_equal(got$d, dd$d)

    # transmute(), and a nested use
    expect_warning(
      got2 <- collect(transmute(tbl_gpu(dd, lazy = lz), p = as.POSIXct(d), q = is.na(p))),
      class = "cuplyr_warning_timestamp_out_of_range"
    )
    expect_equal(got2$p, want)
    expect_identical(got2$q, is.na(want))
  }
})

test_that("as.POSIXct() of in-range Dates is exact and silent", {
  skip_if_no_gpu()
  dd <- tibble::tibble(d = as.Date(c(0, 19000, -7000, 106751991, -106751991, NA)))
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    expect_no_warning(check(dd, function(t) mutate(t, p = as.POSIXct(d))))
    expect_no_warning(check(dd, function(t) filter(t, as.POSIXct(d) > as.POSIXct("2000-01-01", tz = "UTC"))))
  }
})

test_that("filter() on as.POSIXct() of a Date beyond the range warns", {
  skip_if_no_gpu()
  dd <- tibble::tibble(d = as.Date(c(1, 2e9, 19000)))
  for (lz in p24_modes) {
    expect_warning(
      got <- collect(filter(tbl_gpu(dd, lazy = lz), as.POSIXct(d) > as.POSIXct("2000-01-01", tz = "UTC"))),
      class = "cuplyr_warning_timestamp_out_of_range"
    )
    # the in-range rows are dplyr's
    expect_true(as.Date(19000) %in% got$d)
    expect_false(as.Date(1) %in% got$d)
  }
})

# --- item 9 -------------------------------------------------------------------

test_that("bind_cols() keeps the levels of a blank-named factor", {
  skip_if_no_gpu()
  a <- tibble::tibble(a = 1:2, k = c(1, 2))
  dfb <- tibble::tibble(f = factor(c("x", "y"), levels = c("y", "x", "z")))
  names(dfb) <- ""
  for (lz in p24_modes) {
    want <- bind_cols(a, factor(c("x", "y")), .name_repair = "minimal")
    got <- collect(bind_cols(tbl_gpu(a, lazy = lz), factor(c("x", "y")), .name_repair = "minimal"))
    expect_identical(names(got), names(want))
    expect_identical(as.list(got), as.list(want))

    want <- bind_cols(a, dfb, .name_repair = "minimal")
    got <- collect(bind_cols(tbl_gpu(a, lazy = lz), dfb, .name_repair = "minimal"))
    expect_identical(names(got), names(want))
    expect_identical(as.list(got), as.list(want))
  }
})

test_that("bind_cols(.name_repair = 'check_unique') raises vctrs' errors", {
  skip_if_no_gpu()
  a <- tibble::tibble(a = 1:2, k = c(1, 2))
  b <- tibble::tibble(b = c(3, 4))
  dfb <- tibble::tibble(f = c(5, 6))
  names(dfb) <- ""
  # (the classes are dplyr's own)
  expect_error(bind_cols(a, a, .name_repair = "check_unique"), class = "vctrs_error_names_must_be_unique")
  expect_error(bind_cols(a, 5:6, .name_repair = "check_unique"), class = "vctrs_error_names_cannot_be_empty")
  for (lz in p24_modes) {
    ga <- tbl_gpu(a, lazy = lz)
    expect_error(bind_cols(ga, ga, .name_repair = "check_unique"),
                 class = "vctrs_error_names_must_be_unique")
    expect_error(bind_cols(ga, a, .name_repair = "check_unique"),
                 class = "vctrs_error_names_must_be_unique")
    expect_error(bind_cols(ga, 5:6, .name_repair = "check_unique"),
                 class = "vctrs_error_names_cannot_be_empty")
    expect_error(bind_cols(ga, dfb, .name_repair = "check_unique"),
                 class = "vctrs_error_names_cannot_be_empty")
    expect_equal(collect(bind_cols(ga, tbl_gpu(b, lazy = lz), .name_repair = "check_unique")),
                 bind_cols(a, b, .name_repair = "check_unique"))
  }
})

test_that("bind_cols() with a grouped first input keeps dplyr's grouping", {
  skip_if_no_gpu()
  a <- tibble::tibble(a = 1:2, k = c(1, 2))
  b <- tibble::tibble(b = c(3, 4))
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    check(a, function(t) bind_cols(group_by(t, k), b))
  }
  for (lz in p24_modes) {
    got <- collect(bind_cols(group_by(tbl_gpu(a, lazy = lz), k), tbl_gpu(b, lazy = lz)))
    expect_s3_class(got, "grouped_df")
    expect_identical(dplyr::group_vars(got), "k")
  }
})

test_that("bind_cols() with a grouped first input fails on blank names before uploading", {
  skip_if_no_gpu()
  g2 <- tibble::tibble(a = 1:2)
  expect_error(bind_cols(group_by(g2, a), 3:4, .name_repair = "minimal"),
               class = "tibble_error_column_names_cannot_be_empty")
  expect_error(bind_cols(group_by(g2, a), 3:4, 5:6, .name_repair = "minimal"),
               class = "tibble_error_column_names_cannot_be_empty")
  expect_error(bind_cols(group_by(g2, a), tibble::tibble(a = 3:4), .name_repair = "minimal"),
               class = "tibble_error_column_names_must_be_unique")
  for (lz in p24_modes) {
    gg <- group_by(tbl_gpu(g2, lazy = lz), a)
    # raised by the bind_cols() call itself, with no upload warning first
    msgs <- p24_count_ts(
      expect_error(bind_cols(gg, as.Date(c(Inf, 2)), .name_repair = "minimal"),
                   class = "tibble_error_column_names_cannot_be_empty")
    )
    expect_length(msgs, 0L)
    expect_error(bind_cols(gg, 3:4, .name_repair = "minimal"),
                 class = "tibble_error_column_names_cannot_be_empty")
    expect_error(bind_cols(gg, 3:4, 5:6, .name_repair = "minimal"),
                 class = "tibble_error_column_names_cannot_be_empty")
    expect_error(bind_cols(gg, tibble::tibble(a = 3:4), .name_repair = "minimal"),
                 class = "tibble_error_column_names_must_be_unique")
    # an ungrouped first input may have blank names under "minimal"
    got <- collect(bind_cols(tbl_gpu(g2, lazy = lz), 3:4, .name_repair = "minimal"))
    expect_identical(as.list(got), as.list(bind_cols(g2, 3:4, .name_repair = "minimal")))
  }
})
