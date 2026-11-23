# F124: window min()/max() in mutate()/filter() over an empty group (a
# zero-row table, or the empty groups of a `.drop = FALSE` grouping) give
# dplyr's type (an integer/logical min()/max() is a double there) and warning,
# eager and lazy. Helper: helper-parity-F124.R.

test_that("F124: mutate() min()/max() of a zero-row table, every native type", {
  skip_if_no_gpu()
  withr::local_collate("C")
  cols <- list(int = integer(0), dbl = double(0), lgl = logical(0), chr = character(0),
               date = as.Date(character(0)))
  for (t in names(cols)) {
    df <- tibble::tibble(g = integer(0), x = cols[[t]])
    f124_check(df, function(d) mutate(d, m = min(x)))
    f124_check(df, function(d) mutate(group_by(d, g), m = max(x)))
    f124_check(df, function(d) mutate(d, m = max(x), .by = g))
  }
})

test_that("F124: zero-row mutate(): expressions around min()/max(), several calls", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = integer(0), x = integer(0), y = double(0))
  f124_check(df, function(d) mutate(d, m = min(x) + 1L))
  f124_check(df, function(d) mutate(d, m = max(x) - min(x)))
  f124_check(df, function(d) mutate(group_by(d, g), a = max(x), b = x * 2L, c = min(y)))
  # a later dot reads the double
  f124_check(df, function(d) mutate(d, m = max(x), k = m + 1L))
})

test_that("F124: mutate() over a table a lazy filter empties", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 2L, 2L), x = c(1L, 2L, 3L), y = c(1, 2, 3))
  # integer: the type is the data's to decide (a plan break is allowed)
  f124_check(df, function(d) mutate(group_by(filter(d, x > 10L), g), m = max(x)))
  # double: only the warning depends on the data, so the plan stays lazy and
  # warns when it runs
  f124_check(df, function(d) mutate(group_by(filter(d, x > 10L), g), m = max(y)),
             lazy_warns_at = "collect")
  f124_check(df, function(d) mutate(filter(d, y > 10), m = min(y)), lazy_warns_at = "collect")
})

test_that("F124: the fuzz case seed 451 (count() |> filter() empties it, then min())", {
  skip_if_no_gpu()
  # (native = FALSE: the constant-only summarise() at the start is a CPU
  # route of its own, before and after F124)
  f124_check(tibble::tibble(k = character(0)), function(d) {
    mutate(filter(count(summarise(d, s1 = TRUE, s2 = TRUE)), n != n), m3 = min(n))
  }, native = FALSE)
  # the min() part on its own stays on the GPU
  f124_check(tibble::tibble(n = 1L), function(d) mutate(filter(d, n != n), m3 = min(n)))
})

test_that("F124: mutate() over the empty groups of a .drop = FALSE grouping", {
  skip_if_no_gpu()
  withr::local_collate("C")
  fdf <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c", "d")),
                        x = c(1L, 2L, 3L), y = c(1, 2, 3), s = c("p", "q", "r"))
  # two empty groups (b, d): an integer max() is a double, one warning per group
  f124_check(fdf, function(d) mutate(group_by(d, f, .drop = FALSE), m = max(x)))
  f124_check(fdf, function(d) mutate(group_by(d, f, .drop = FALSE), m = max(y), n = min(x)))
  f124_check(fdf, function(d) mutate(group_by(d, f, .drop = FALSE), m = min(s)))
  # every level observed: nothing changes
  full <- tibble::tibble(f = factor(c("a", "b"), levels = c("a", "b")), x = c(1L, 2L))
  f124_check(full, function(d) mutate(group_by(d, f, .drop = FALSE), m = max(x)),
             lazy_warns_at = "collect")
})

test_that("F124: the fuzz case seed 2180 (a filter on a .drop = FALSE table keeps its empty groups)", {
  skip_if_no_gpu()
  y <- tibble::tibble(k = 1, j1 = NA_integer_, j2 = 5L)
  v1 <- "j1"
  f124_check(tibble::tibble(k = 4L, b = TRUE), function(d) {
    yy <- if (is_tbl_gpu(d)) tbl_gpu(y, lazy = identical(d$exec_mode, "lazy")) else y
    mutate(filter(full_join(group_by(d, b, .drop = FALSE), yy, by = "k"), j2 <= .data[[v1]]),
           m1 = max(k))
  })
  # the same with integer `k` (the type changes too)
  f124_check(tibble::tibble(k = c(4L, 5L), b = c(TRUE, FALSE)), function(d) {
    mutate(filter(group_by(d, b, .drop = FALSE), k > 10L), m1 = max(k))
  })
})

test_that("F124: filter() with min()/max() over an empty group warns like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = integer(0), x = integer(0), y = double(0))
  f124_check(df, function(d) filter(d, x == min(x)))
  f124_check(df, function(d) filter(group_by(d, g), x == min(x), y < max(y)))
  f124_check(df, function(d) filter(d, y < max(y)), lazy_warns_at = "collect")
  fdf <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), x = c(1L, 2L, 3L))
  f124_check(fdf, function(d) filter(group_by(d, f, .drop = FALSE), x == max(x)))
})

test_that("F124: non-empty input keeps its type, does not warn, and stays lazy", {
  skip_if_no_gpu()
  withr::local_collate("C")
  df <- tibble::tibble(g = c(1L, 2L, 1L), x = c(3L, NA, 5L), b = c(TRUE, FALSE, NA),
                       y = c(1, 2, 3), s = c("u", "v", "w"))
  f124_check(df, function(d) {
    mutate(group_by(d, g), a = max(x), c = min(b), e = max(y), h = min(s))
  }, lazy_warns_at = "collect")
  f124_check(df, function(d) filter(group_by(d, g), y == max(y)), lazy_warns_at = "collect")
  # row-preserving pending steps keep the row count known: no plan break
  f124_check(df, function(d) mutate(group_by(mutate(d, z = x + 1L), g), m = max(z)),
             lazy_warns_at = "collect")
  # a lazy filter before a double max() doesn't run the plan early
  f124_check(df, function(d) mutate(filter(d, y > 1), m = max(y)), lazy_warns_at = "collect")
})
