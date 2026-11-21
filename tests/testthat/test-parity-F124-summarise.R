# F124: summarise()/reframe() min()/max() over an empty or all-NA group give
# dplyr's value, type and warning (one condition per verb call, with dplyr's
# text and count), eager and lazy. Helper: helper-parity-F124.R.

f124_types_0row <- list(
  int = integer(0), dbl = double(0), lgl = logical(0), chr = character(0),
  date = as.Date(character(0)), posix = as.POSIXct(character(0), tz = "UTC")
)

test_that("F124: ungrouped summarise min()/max() of zero rows, every native type", {
  skip_if_no_gpu()
  withr::local_collate("C")
  for (t in names(f124_types_0row)) {
    df <- tibble::tibble(x = f124_types_0row[[t]])
    f124_check(df, function(d) summarise(d, m = min(x)))
    f124_check(df, function(d) summarise(d, m = max(x, na.rm = TRUE)))
  }
})

test_that("F124: grouped summarise with all-NA groups and na.rm = TRUE, every native type", {
  skip_if_no_gpu()
  withr::local_collate("C")
  vals <- list(
    int = c(NA, 5L, NA, NA, 7L),
    dbl = c(NA, 5.5, NA, NA, -1),
    lgl = c(NA, TRUE, NA, NA, FALSE),
    chr = c(NA, "b", NA, NA, "a"),
    date = as.Date(c(NA, "2020-01-05", NA, NA, "2020-01-01")),
    posix = as.POSIXct(c(NA, "2020-01-05 10:00:00", NA, NA, "2020-01-01"), tz = "UTC")
  )
  # group 1 has a value, groups 2 and 3 are all NA: two warnings, the first
  # one about group 2
  g <- c(2L, 1L, 2L, 3L, 1L)
  for (t in names(vals)) {
    df <- tibble::tibble(g = g, x = vals[[t]])
    f124_check(df, function(d) summarise(group_by(d, g), m = max(x, na.rm = TRUE), .groups = "drop"))
    f124_check(df, function(d) summarise(group_by(d, g), m = min(x, na.rm = TRUE)))
  }
})

test_that("F124: na.rm = FALSE over an all-NA group is NA with no warning, type kept", {
  skip_if_no_gpu()
  withr::local_collate("C")
  vals <- list(int = c(NA, 5L, NA), dbl = c(NA, 5.5, NA), lgl = c(NA, TRUE, NA),
               chr = c(NA, "b", NA), date = as.Date(c(NA, "2020-01-05", NA)))
  for (t in names(vals)) {
    df <- tibble::tibble(g = c(2L, 1L, 2L), x = vals[[t]])
    f124_check(df, function(d) summarise(group_by(d, g), m = min(x), n = max(x), .groups = "drop"),
               lazy_warns_at = "collect")
  }
})

test_that("F124: non-empty, NA-free input keeps its type and does not warn", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(g = c(1L, 2L, 1L, 3L), i = c(4L, 2L, 9L, 1L), d = c(1.5, 2, -3, 4),
                       b = c(TRUE, FALSE, FALSE, TRUE), s = c("q", "a", "z", "m"))
  f124_check(df, function(d) {
    summarise(group_by(d, g), a = min(i), b2 = max(d), c = min(b), e = max(s), .groups = "drop")
  }, lazy_warns_at = "collect")
  f124_check(df, function(d) summarise(d, a = min(i), b2 = max(d), c = max(b), e = min(s)),
             lazy_warns_at = "collect")
  # NAs, but no all-NA group: still integer, no warning
  df2 <- tibble::tibble(g = c(1L, 1L, 2L), i = c(NA, 3L, 4L), s = c("x", NA, "y"))
  f124_check(df2, function(d) {
    summarise(group_by(d, g), a = min(i, na.rm = TRUE), e = max(s, na.rm = TRUE), .groups = "drop")
  })
})

test_that("F124: a double or string summarise on a lazy table warns at collect()", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(g = c(1L, 2L, 2L), x = c(NA, 1, 2), s = c(NA, "a", "b"))
  f124_check(df, function(d) summarise(group_by(d, g), m = max(x, na.rm = TRUE), .groups = "drop"),
             lazy_warns_at = "collect")
  f124_check(df, function(d) summarise(group_by(d, g), m = min(s, na.rm = TRUE), .groups = "drop"),
             lazy_warns_at = "collect")
  # and on every run of the plan
  lz <- summarise(group_by(tbl_gpu(df, lazy = TRUE), g), m = max(x, na.rm = TRUE), .groups = "drop")
  expect_warning(collect(lz), "no non-missing arguments to max")
  expect_warning(collect(lz), "no non-missing arguments to max")
})

test_that("F124: several aggregations: one warning for the verb, the count of all", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(g = c(1L, 2L, 3L, 3L), x = c(NA, 1L, NA, NA), y = c(2, NA, NA, 1),
                       s = c("a", NA, NA, NA))
  f124_check(df, function(d) {
    summarise(group_by(d, g), a = min(x, na.rm = TRUE), b = max(y, na.rm = TRUE),
              c = max(s, na.rm = TRUE), .groups = "drop")
  })
  # the first warning is the first dot that warns, even when a later one hits
  # an earlier group
  f124_check(df, function(d) {
    summarise(group_by(d, g), b = max(y, na.rm = TRUE), a = min(x, na.rm = TRUE), .groups = "drop")
  })
})

test_that("F124: summarise(.by) labels the group in order of first appearance", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(g = c(3L, 1L, 2L, 1L), x = c(NA, 1L, NA, 2L), s = c("b", NA, NA, "c"))
  f124_check(df, function(d) summarise(d, a = min(x, na.rm = TRUE), .by = g))
  f124_check(df, function(d) summarise(d, a = max(s, na.rm = TRUE), .by = g))
  f124_check(df[0, ], function(d) summarise(d, a = min(x), .by = g))
})

test_that("F124: group labels format keys like dplyr (strings, dates, NA, several keys)", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(
    g = as.Date(c("2020-01-02", "2020-01-01", "2020-01-02")),
    h = c("u", "v", NA),
    x = c(NA, 1, NA)
  )
  f124_check(df, function(d) summarise(group_by(d, g, h), a = min(x, na.rm = TRUE), .groups = "drop"))
  f124_check(df, function(d) summarise(group_by(d, h), a = max(x, na.rm = TRUE)))
})

test_that("F124: .drop = FALSE empty factor groups", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c", "d")),
                       x = c(1L, 2L, NA), s = c("p", "q", NA))
  f124_check(df, function(d) summarise(group_by(d, f, .drop = FALSE), m = max(x)))
  f124_check(df, function(d) summarise(group_by(d, f, .drop = FALSE), m = max(x, na.rm = TRUE)))
  # (a string min()/max() over a .drop = FALSE empty group was a CPU route
  # before F124 and may stay one)
  f124_check(df, function(d) summarise(group_by(d, f, .drop = FALSE), m = min(s, na.rm = TRUE)),
             native = FALSE)
  # zero rows: dplyr evaluates every empty group, without a group label
  f124_check(df[0, ], function(d) summarise(group_by(d, f, .drop = FALSE), m = max(x)))
})

test_that("F124: grouped summarise of zero rows", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(g = integer(0), x = integer(0), s = character(0))
  f124_check(df, function(d) summarise(group_by(d, g), m = max(x)))
  f124_check(df, function(d) summarise(group_by(d, g), m = max(s)))
})

test_that("F124: the fuzz cases (seeds 567, 1126, 4119)", {
  skip_if_no_gpu()
  withr::local_collate("C")
  f124_check(tibble::tibble(k = character(0)),
             function(d) summarise(filter(d), s2 = min(k, na.rm = TRUE)))
  f124_check(tibble::tibble(a = TRUE, d = NA_character_),
             function(d) summarise(group_by(d, a, d), s2 = min(d, na.rm = TRUE)))
  v2 <- "c"
  f124_check(tibble::tibble(b = 1L, c = NA_character_),
             function(d) summarise(summarise(group_by(d, c, b)), s1 = max(.data[[v2]], na.rm = TRUE)))
})

test_that("F124: reframe() min()/max()/range() over empty input", {
  skip_if_no_gpu()
  withr::local_collate("C")
  f124_check(tibble::tibble(x = integer(0)), function(d) reframe(d, m = min(x)))
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), x = c(NA, NA, 5L, 6L, NA))
  f124_check(df, function(d) reframe(group_by(d, g), m = range(x, na.rm = TRUE)))
  f124_check(df, function(d) reframe(group_by(d, g), lo = min(x, na.rm = TRUE), hi = max(x, na.rm = TRUE)))
})
