# Issue #24 acceptance tests, part 3: cases found in review.
#
# - The vctrs::unspecified() marker also survives a window mutate() (cumsum(),
#   a grouped n()) and a window filter(), as dplyr keeps the column as is.
# - A table with a blank-named column next to a factor (bind_cols() under
#   "minimal") can be sliced and printed; two blank-named factors keep their
#   own levels.
# - A lossy Date vector constant inside a staged across() (one whose output
#   names collide with inputs) warns once per call, naming the user's columns.

# The dplyr side evaluates `lag` by name, as in a session with dplyr attached
# (stats::lag otherwise); cuplyr reads the call by name.
lag <- dplyr::lag

p24_modes <- c(eager = FALSE, lazy = TRUE)

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

test_that("the unspecified marker survives a window mutate() and a window filter()", {
  skip_if_no_gpu()
  du <- tibble::tibble(u = vctrs::unspecified(3), y = 1:3, g = c(1L, 1L, 2L))
  abbr <- function(t) vapply(t, vctrs::vec_ptype_abbr, "")
  pipes <- list(
    cumsum = function(t) mutate(t, z = cumsum(y)),
    grouped_n = function(t) mutate(group_by(t, g), n = n()),
    lag = function(t) mutate(t, z = lag(y), w = u),
    window_filter = function(t) filter(t, y == max(y)),
    grouped_window_filter = function(t) filter(group_by(t, g), y == max(y))
  )
  for (nm in names(pipes)) {
    want <- pipes[[nm]](du)
    for (lz in p24_modes) {
      got <- collect(pipes[[nm]](tbl_gpu(du, lazy = lz)))
      expect_identical(abbr(got), abbr(want), info = paste(nm, "lazy =", lz))
      expect_equal(tibble::as_tibble(dplyr::ungroup(got)), tibble::as_tibble(dplyr::ungroup(want)),
                   info = paste(nm, "lazy =", lz))
    }
  }
})

test_that("a blank-named column next to a factor survives slicing and printing", {
  skip_if_no_gpu()
  a <- tibble::tibble(a = 1:2, f = factor(c("x", "y"), levels = c("y", "x")))
  want_all <- bind_cols(a, 3:4, .name_repair = "minimal")
  for (lz in p24_modes) {
    xb <- bind_cols(tbl_gpu(a, lazy = lz), 3:4, .name_repair = "minimal")
    expect_identical(as.list(collect(xb)), as.list(want_all))
    expect_identical(as.list(collect(xb[1, ])), as.list(want_all[1, ]), info = paste("lazy =", lz))
    expect_identical(as.list(collect(xb[2:1, ])), as.list(want_all[2:1, ]), info = paste("lazy =", lz))
    expect_no_error(utils::capture.output(print(xb[1, ])))
    expect_no_error(utils::capture.output(print(xb)))
  }
})

test_that("two blank-named factors keep their own levels", {
  skip_if_no_gpu()
  a1 <- tibble::tibble(a = 1:2)
  f1 <- factor(c("x", "y"))
  f2 <- factor(c("q", "p"), levels = c("q", "p", "r"))
  want <- bind_cols(a1, f1, f2, .name_repair = "minimal")
  dfb <- tibble::tibble(f = f2, h = f1)
  names(dfb) <- c("", "")
  want2 <- bind_cols(a1, dfb, .name_repair = "minimal")
  for (lz in p24_modes) {
    got <- collect(bind_cols(tbl_gpu(a1, lazy = lz), f1, f2, .name_repair = "minimal"))
    expect_identical(as.list(got), as.list(want), info = paste("lazy =", lz))
    got2 <- collect(bind_cols(tbl_gpu(a1, lazy = lz), dfb, .name_repair = "minimal"))
    expect_identical(as.list(got2), as.list(want2), info = paste("lazy =", lz))
  }
})

test_that("a lossy Date vector in a staged across() warns once, naming the user's columns", {
  skip_if_no_gpu()
  d <- tibble::tibble(x = 1:3, y = 4:6)
  # (`c(y = x, z = y)` writes `y` while another piece reads it: staged)
  p <- function(t) mutate(t, across(c(y = x, z = y), ~ as.Date(c(1, Inf, 3))))
  pt <- function(t) transmute(t, across(c(y = x, z = y), ~ as.Date(c(1, Inf, 3))))
  for (fn in list(p, pt)) {
    # eager: one warning at the call
    msgs <- p24_count_ts(ea <- fn(tbl_gpu(d, lazy = FALSE)))
    expect_length(msgs, 1L)
    expect_match(msgs, "`y`", fixed = TRUE)
    expect_match(msgs, "`z`", fixed = TRUE)
    expect_false(any(grepl("..across", msgs, fixed = TRUE)))
    got <- collect(ea)
    expect_identical(is.na(got$y), c(FALSE, TRUE, FALSE))
    expect_identical(is.na(got$z), c(FALSE, TRUE, FALSE))
    expect_s3_class(got$y, "Date")
    # lazy: one warning at the call, none when the plan runs
    msgs <- p24_count_ts(lz <- fn(tbl_gpu(d, lazy = TRUE)))
    expect_length(msgs, 1L)
    expect_match(msgs, "`y`", fixed = TRUE)
    expect_length(p24_count_ts(got_l <- collect(lz)), 0L)
    expect_equal(got_l, got)
  }
  # the unstaged form keeps working the same way
  pu <- function(t) mutate(t, across(c(x, y), ~ as.Date(c(1, Inf, 3))))
  for (lz in p24_modes) {
    msgs <- p24_count_ts(res <- collect(pu(tbl_gpu(d, lazy = lz))))
    expect_length(msgs, 1L)
    expect_identical(is.na(res$x), c(FALSE, TRUE, FALSE))
  }
})
