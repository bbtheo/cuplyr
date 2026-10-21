# F45 acceptance tests: across()/if_any()/if_all() `.cols` resolved against the
# running schema at its dot.
#
# dplyr 1.2.1 contract (R/mutate.R::mutate_col() expands each dot when it is
# reached; R/across.R::across_setup() runs tidyselect on
# mask$get_current_data(groups = FALSE)), verified on plain tibbles:
# - an across()/if_any()/if_all() in dot k selects among the columns as they
#   stand at dot k: columns created or modified by dots 1..k-1 are visible,
#   with their NEW types; columns removed with `x = NULL` are gone; group
#   columns stay excluded.
# - summarise() works the same way: a later across() sees earlier summaries.
# - the expansions of one dot still see the data as it was before that dot.
# - a selection naming a column an earlier dot removed is dplyr's error.

f45_df <- function() {
  data.frame(
    g = c(1L, 1L, 2L, 2L),
    x = c(1, 2, NA, 4),
    y = c(10L, 20L, NA, 40L),
    s = c("a", "b", NA, "d"),
    d = c(4, 5, 6, 7),
    f = factor(c("u", "v", "u", "w")),
    stringsAsFactors = FALSE
  )
}

f45_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# Selections over columns whose type the native path knows must stay on the GPU
f45_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f45_both(df, pipeline, ...)
}

test_that("mutate(): a name-based across() selects columns created by earlier dots", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_native(df, function(d) mutate(d, y2 = y * 2L, across(starts_with("y"), ~ .x + 1L)))
  f45_native(df, function(d) mutate(d, k = x + 1, across(c(x, k), ~ .x * 10)))
  f45_native(df, function(d) mutate(d, m = mean(d), across(starts_with("m"), ~ .x < d)))
  f45_native(df, function(d) mutate(d, k = x + 1, across(c(k, x), ~ .x * 10, .names = "{.col}_p")))
  f45_both(df, function(d) mutate(d, k = x + 1, across(any_of(c("k", "zz")), ~ .x - 1)))
})

test_that("mutate(): a type-based across() sees the new and changed types", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_native(df, function(d) mutate(d, z = "c", across(where(is.character), toupper)))
  f45_native(df, function(d) mutate(d, b = x > 1, across(where(is.logical), ~ !.x)))
  f45_native(df, function(d) mutate(d, i = y + 1L, across(where(is.integer), ~ .x * 2L)))
  f45_native(df, function(d) mutate(d, x = x > 1, across(where(is.double), ~ .x * 2)))
  f45_native(df, function(d) mutate(d, y = y * 1.5, across(where(is.integer), ~ .x + 1L)))
  f45_both(df, function(d) mutate(d, f2 = f, across(where(is.factor), as.character)))
})

test_that("mutate(): columns removed by an earlier dot are not selected", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_both(df, function(d) mutate(d, x = NULL, across(where(is.double), ~ .x + 1)))
  f45_both(df, function(d) mutate(d, s = NULL, across(everything(), ~ dplyr::coalesce(as.numeric(.x), 0))))
})

test_that("mutate(): selecting a column removed by an earlier dot is dplyr's error", {
  skip_if_no_gpu()
  df <- f45_df()
  expect_error(mutate(df, x = NULL, across(x, ~ .x + 1)), "Column `x` doesn't exist")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(mutate(tbl_gpu(df, lazy = lazy), x = NULL, across(x, ~ .x + 1))),
      "Column `x` doesn't exist"
    )
  }
})

test_that("mutate(): grouped and .by calls see created columns, group columns stay excluded", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_native(df, function(d) mutate(group_by(d, g), z = x * 2, across(where(is.double), ~ .x - 1)))
  f45_native(df, function(d) mutate(d, z = x * 2, across(where(is.double), ~ .x - 1), .by = g))
  f45_both(df, function(d) mutate(group_by(d, g), s = NULL, f = NULL, z = 1, across(everything(), ~ .x * 2)))
  f45_both(df, function(d) mutate(group_by(d, g), z = x * 2, across(c(z, x), ~ .x - mean(.x))))
})

test_that("mutate(): .keep, .before and chained lazy calls", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_both(df, function(d) mutate(d, z = x * 2, across(c(x, z), ~ .x + 1), .keep = "none"))
  f45_both(df, function(d) mutate(d, z = x * 2, across(c(x, z), ~ .x + 1), .keep = "used"))
  f45_both(df, function(d) mutate(d, z = x * 2, across(starts_with("z"), ~ .x + 1, .names = "{.col}_p"),
                                  .before = 1))
  f45_both(df, function(d) d |> mutate(z = x * 2) |> mutate(w = z + 1, across(c(w, z), ~ .x * 10)))
})

test_that("mutate(): a later across() sees an earlier across()'s outputs", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_native(df, function(d) mutate(d, across(x, ~ .x + 1, .names = "x1"),
                                    across(starts_with("x"), ~ .x * 10)))
  f45_both(df, function(d) mutate(d, across(c(x, d), ~ .x > 4, .names = "{.col}_big"),
                                  across(where(is.logical), ~ !.x)))
})

test_that("mutate(): one across() dot still reads the data before the dot", {
  skip_if_no_gpu()
  df <- f45_df()
  # x = x + z (z = 2x) and z = z + z, both from the pre-dot values
  f45_both(df, function(d) mutate(d, z = x * 2, across(c(x, z), ~ .x + z)))
  # the renamed output w is created by this dot; z is read from before it
  f45_both(df, function(d) mutate(d, z = x * 2, across(c(w = x, z), ~ .x + 1)))
})

test_that("mutate(): value predicates over created columns match dplyr", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_both(df, function(d) mutate(d, z = c(NA, 1, 2, 3), across(where(~ anyNA(.x)), ~ 0)))
  f45_both(df, function(d) mutate(d, z = d * 0, across(where(~ is.numeric(.x) && all(.x == 0, na.rm = TRUE)),
                                                       ~ .x - 1)))
})

test_that("transmute(): across() sees created columns and the remaining input columns", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_both(df, function(d) transmute(d, z = x * 2, across(c(z, x), ~ .x + 1)))
  f45_both(df, function(d) transmute(d, z = "c", across(where(is.character), toupper)))
  f45_both(df, function(d) transmute(d, x = x, s = NULL, f = NULL, across(everything(), ~ .x + 1)))
  f45_both(df, function(d) transmute(group_by(d, g), z = x * 2, across(where(is.double), ~ .x - 1)))
})

test_that("if_any()/if_all() in mutate()/transmute() see created columns", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_native(df, function(d) mutate(d, z = x > 2, flag = dplyr::if_any(c(z), ~ .x)))
  f45_native(df, function(d) mutate(d, z = x > 1, w = y > 15L, flag = dplyr::if_all(c(z, w))))
  f45_both(df, function(d) transmute(d, z = x > 1, w = y > 15L, flag = dplyr::if_all(c(z, w))))
  f45_both(df, function(d) mutate(d, k = x * 2, dplyr::if_any(starts_with("k"), ~ .x > 3)))
  f45_both(df, function(d) mutate(d, b = x > 1, flag = dplyr::if_all(where(is.logical), ~ .x)))
})

test_that("summarise(): a later across()/if_any() sees earlier summary columns", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_both(df, function(d) summarise(d, m = mean(x, na.rm = TRUE), across(starts_with("m"), ~ .x * 2)))
  f45_both(df, function(d) summarise(group_by(d, g), m = mean(d), across(c(m), ~ .x * 2, .names = "{.col}2")),
           arrange_by = "g")
  f45_both(df, function(d) summarise(d, m = mean(d), across(where(is.double), ~ max(.x, na.rm = TRUE))))
  f45_both(df, function(d) summarise(group_by(d, g), m = mean(d), a = dplyr::if_any(c(m), ~ .x > 5)),
           arrange_by = "g")
  f45_both(df, function(d) summarise(d, m = mean(d), across(c(m, d), ~ max(.x))))
})

test_that("filter(): if_any()/if_all() still resolve against the input columns", {
  skip_if_no_gpu()
  df <- f45_df()
  f45_native(df, function(d) filter(d, if_any(c(x, d), ~ .x > 4)))
  f45_native(df, function(d) filter(group_by(d, g), if_all(where(is.double), ~ .x > 1)))
})

test_that("zero-row inputs", {
  skip_if_no_gpu()
  df <- f45_df()[0, ]
  f45_both(df, function(d) mutate(d, z = "c", across(where(is.character), toupper)))
  f45_both(df, function(d) mutate(d, k = x + 1, across(c(x, k), ~ .x * 10)))
})

test_that("B14: a later across() over a fractional-Date column created earlier sees dplyr's value", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))
  floor_dates <- function(t) {
    t[] <- lapply(t, function(v) if (inherits(v, "Date")) as.Date(floor(unclass(v))) else v)
    t
  }
  pipelines <- list(
    function(t) dplyr::mutate(t, dd = as.Date(19723.5),
                              dplyr::across(starts_with("d"), ~ .x > 19723.2, .names = "{.col}_gt")),
    function(t) dplyr::mutate(t, dd = as.Date(19723.5),
                              dplyr::across(any_of("dd"), ~ .x > 19723.2, .names = "{.col}_gt"))
  )
  for (pipeline in pipelines) {
    expected <- floor_dates(pipeline(df))
    expect_true(all(expected$dd_gt))
    for (lazy in c(FALSE, TRUE)) {
      expect_warning(
        res <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
        class = "cuplyr_warning_fractional_date"
      )
      expect_equal(tibble::as_tibble(res), expected)
    }
  }
})

# --- added after round-1 review (B1-B3) ---------------------------------------

test_that("if_any()/if_all() without .fns over a created column the GPU can't type match dplyr", {
  skip_if_no_gpu()
  df <- f45_df()[c("g", "x", "y")]
  f45_both(df, function(d) mutate(d, z = c(TRUE, FALSE, NA, TRUE), flag = dplyr::if_any(c(z))))
  f45_both(df, function(d) mutate(d, z = c(TRUE, FALSE, NA, TRUE), flag = dplyr::if_all(c(z, z))))
  # a non-logical column is dplyr's error, not a native result
  expect_error(dplyr::mutate(df, z = c(1, 2, 3, 4), flag = dplyr::if_any(c(z))), "must be a logical vector")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), z = c(1, 2, 3, 4), flag = dplyr::if_any(c(z)))),
      "must be a logical vector"
    )
  }
})

test_that("dots after one that needs dplyr see what the earlier dots did", {
  skip_if_no_gpu()
  df <- f45_df()[c("g", "x", "y")]
  f45_both(df, function(d) mutate(d, across(where(~ anyNA(.x)), ~ .x > 1), flag = dplyr::if_any(c(x))))
  f45_both(df, function(d) mutate(d, across(where(~ anyNA(.x)), ~ .x * 2, .names = "{.col}_2"),
                                  across(c(x_2), ~ .x + 1)))
})

test_that("a named dot that evaluates to NULL creates no column for a later across()", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  v <- NULL
  f45_both(df, function(d) mutate(d, z = v, across(everything(), ~ .x * 2)))
  f45_both(df, function(d) mutate(d, z = v, across(everything(), ~ 0)))
  f45_both(df, function(d) transmute(d, z = v, across(everything(), ~ .x + 1)))
})

# --- added after round-2 review (B4, N1) --------------------------------------

test_that("a column re-created after a NULL-valued dot sits at the end, like dplyr's mask", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  v <- NULL
  f45_both(df, function(d) mutate(d, x = v, w = y * 2L, x = y + 1L, across(last_col(), ~ .x * 10L)))
  f45_both(df, function(d) mutate(d, z = v, w = x * 2, z = x + 1, across(last_col(), ~ .x * 10)))
  f45_both(df, function(d) mutate(d, z = v, w = x * 2, z = x + 1,
                                  across(-c(x, y), ~ .x * 10, .names = "{.col}_t")))
})

test_that("a maybe-absent column: selections that can't pick it stay native, others follow dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  v <- NULL
  f45_native(df, function(d) mutate(d, z = v, across(c(x), ~ .x * 2)))
  expect_error(dplyr::mutate(df, x = v, across(2, ~ .x * 10L)), "past the end")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), x = v, across(2, ~ .x * 10L))),
      "past the end"
    )
  }
})

# --- added after round-3 review (B5, B6) --------------------------------------
# B5: the running schema must record R's type for a created column, not
# cuplyr's static IR type. median() of an integer is an integer for odd n
# (double for even n); lubridate's year() is a double; max() of a logical is
# an integer. Where R's type isn't known statically, a type predicate has to
# follow dplyr (CPU route), while name-only selections stay native.

test_that("a created column whose R type isn't cuplyr's static type: where() follows dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3), y = c(10L, 20L, 30L))
  f45_both(df, function(d) mutate(d, m = median(y), across(where(is.double), ~ .x * 2)))
  f45_both(df, function(d) mutate(d, m = median(y), across(where(is.integer), ~ .x + 1L)))
  dg <- data.frame(g = c(1L, 1L, 1L, 2L, 2L, 2L), x = c(1, 2, 3, 4, 5, 6),
                   y = c(10L, 20L, 30L, 40L, 50L, 70L))
  f45_both(dg, function(d) mutate(group_by(d, g), m = median(y), across(where(is.double), ~ .x * 2)))
  f45_both(df, function(d) mutate(d, b = x > 1, mx = max(b), across(where(is.logical), ~ !.x)))
  # name-only selections after such a column stay on the GPU
  f45_native(df, function(d) mutate(d, m = dplyr::ntile(y, 2), across(c(x, m), ~ .x * 2)))
})

test_that("lubridate accessors: a later where() sees R's type, not INT32", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  year <- lubridate::year
  df <- data.frame(x = c(1, 2, 3), d = as.Date(c("2024-01-01", "2024-06-15", "2024-12-31")))
  f45_both(df, function(d) mutate(d, yr = year(d), across(where(is.integer), ~ .x + 1L)))
  f45_both(df, function(d) mutate(d, yr = year(d) + 1L, across(where(is.integer), ~ .x + 1L)))
})

test_that("an unnamed bare dot re-creating a NULL-removed column moves it to the end", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2), y = c(3L, 4L))
  v <- NULL
  x <- c(5, 6)
  # cuplyr's own "Unnamed mutate expression" warning is not part of the contract
  withCallingHandlers(
    f45_both(df, function(d) mutate(d, x = v, x, across(last_col(), ~ .x * 10))),
    warning = function(w) if (grepl("Unnamed mutate expression", conditionMessage(w))) invokeRestart("muffleWarning")
  )
})
