# F24 acceptance tests, tier B: STRING columns in non-equi and closest()
# join conditions, natively.
#
# vctrs::vec_locate_matches() (dplyr 1.2.1's join engine) orders strings
# bytewise on UTF-8, i.e. in the C locale: "" < "B" < "Z" < "a" < "e" < "é"
# (verified with tools/r, scratchpad/parity/features/F24/probe/p1.R and
# p2.R). Missing values follow the numeric rules: with na_matches = "na",
# NA matches NA for `>=`/`<=` only.
#
# "Native" is pinned with options(cuplyr.fallback = "error").

join_by <- dplyr::join_by

f24s_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f24s_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f24s_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f24s_both(df, pipeline, ...)
}

f24s_ops <- c(">=", ">", "<=", "<")

f24s_by <- function(fmt, op) {
  eval(parse(text = sprintf(fmt, op)))
}

f24s_mutating <- list(
  left = dplyr::left_join, inner = dplyr::inner_join,
  right = dplyr::right_join, full = dplyr::full_join
)
f24s_filtering <- list(semi = dplyr::semi_join, anti = dplyr::anti_join)

# Case, a leading space, the empty string, a shared prefix, non-ASCII UTF-8,
# NA on both sides, and ties.
f24s_x <- data.frame(
  g = c(1, 1, 1, 1, 2, 2, 2, 1),
  s = c("b", "B", "a ", "é", NA, "", "ab", "zz"),
  xv = 1:8,
  stringsAsFactors = FALSE
)
f24s_y <- data.frame(
  g = c(1, 1, 1, 1, 1, 2, 2, 2, 2),
  r = c("a", "B", "c", "Z", "e", NA, "", "a", "a"),
  yv = 1:9,
  stringsAsFactors = FALSE
)

test_that("closest() on STRING rolling keys uses the C-locale order", {
  skip_if_no_gpu()
  for (op in f24s_ops) {
    by_g <- f24s_by("dplyr::join_by(g, closest(s %s r))", op)
    by_0 <- f24s_by("dplyr::join_by(closest(s %s r))", op)
    for (join in c(f24s_mutating, f24s_filtering)) {
      f24s_native(f24s_x, function(d) join(d, f24s_like(d, f24s_y), by = by_g))
      f24s_native(f24s_x, function(d) join(d, f24s_like(d, f24s_y), by = by_0))
    }
    for (join in f24s_mutating) {
      f24s_native(f24s_x, function(d) {
        join(d, f24s_like(d, f24s_y), by = by_g, na_matches = "never")
      })
    }
  }
})

test_that("plain STRING inequality conditions, with and without an equi key", {
  skip_if_no_gpu()
  for (op in f24s_ops) {
    by_0 <- f24s_by("dplyr::join_by(s %s r)", op)
    by_g <- f24s_by("dplyr::join_by(g, s %s r)", op)
    for (join in c(f24s_mutating, f24s_filtering)) {
      f24s_native(f24s_x, function(d) join(d, f24s_like(d, f24s_y), by = by_0))
      f24s_native(f24s_x, function(d) join(d, f24s_like(d, f24s_y), by = by_g))
      f24s_native(f24s_x, function(d) {
        join(d, f24s_like(d, f24s_y), by = by_g, na_matches = "never")
      })
    }
  }
})

test_that("between() and two STRING conditions on the same columns", {
  skip_if_no_gpu()
  x <- data.frame(s = c("apple", "banana", "cherry", NA, "Banana", "date"),
                  xv = 1:6, stringsAsFactors = FALSE)
  y <- data.frame(lo = c("a", "b", "c", NA, "B"),
                  hi = c("azz", "c", "cherry", "z", "b"),
                  yv = 1:5, stringsAsFactors = FALSE)
  bys <- list(
    join_by(between(s, lo, hi)),
    join_by(s > lo, s < hi),
    join_by(within(s, s, lo, hi))
  )
  for (by in bys) {
    for (join in c(f24s_mutating, f24s_filtering)) {
      f24s_native(x, function(d) join(d, f24s_like(d, y), by = by))
    }
  }
})

test_that("same-named STRING columns on both sides are suffixed", {
  skip_if_no_gpu()
  x <- data.frame(id = c(1, 1, 2), s = c("m", "a", "q"), stringsAsFactors = FALSE)
  y <- data.frame(id = c(1, 1, 2), s = c("b", "n", "p"), stringsAsFactors = FALSE)
  for (join in f24s_mutating) {
    f24s_native(x, function(d) join(d, f24s_like(d, y), by = join_by(id, s >= s)))
    f24s_native(x, function(d) join(d, f24s_like(d, y), by = join_by(id, closest(s >= s))))
    f24s_native(x, function(d) {
      join(d, f24s_like(d, y), by = join_by(id, closest(s < s)), keep = TRUE)
    })
  }
})

test_that("multiple= and lazy filters on top of a STRING inequality join", {
  skip_if_no_gpu()
  by <- join_by(g, s >= r)
  for (m in c("first", "last")) {
    for (join in f24s_mutating) {
      f24s_native(f24s_x, function(d) {
        join(d, f24s_like(d, f24s_y), by = by, multiple = m)
      })
    }
  }
  f24s_native(f24s_x, function(d) {
    dplyr::inner_join(d, f24s_like(d, f24s_y), by = by) |>
      dplyr::filter(r != "a", xv > 1) |>
      dplyr::select(xv, s, r, yv)
  })
  f24s_native(f24s_x, function(d) {
    dplyr::left_join(d, f24s_like(d, f24s_y), by = join_by(g, closest(s >= r))) |>
      dplyr::filter(is.na(r) | r != "a")
  })
})

test_that("0-row inputs with STRING conditions", {
  skip_if_no_gpu()
  for (join in c(f24s_mutating, f24s_filtering)) {
    f24s_native(f24s_x[0, ], function(d) join(d, f24s_like(d, f24s_y), by = join_by(s >= r)))
    f24s_native(f24s_x, function(d) join(d, f24s_like(d, f24s_y[0, ]), by = join_by(g, s < r)))
    f24s_native(f24s_x, function(d) {
      join(d, f24s_like(d, f24s_y[0, ]), by = join_by(g, closest(s < r)))
    })
  }
})

test_that("factor against STRING in an inequality stays correct", {
  skip_if_no_gpu()
  x <- data.frame(f = factor(c("b", "a", "c"), levels = c("c", "b", "a")), xv = 1:3)
  y <- data.frame(r = c("a", "b", "bb"), yv = 1:3, stringsAsFactors = FALSE)
  for (join in list(dplyr::left_join, dplyr::inner_join)) {
    f24s_both(x, function(d) join(d, f24s_like(d, y), by = join_by(f >= r)))
    f24s_both(x, function(d) join(d, f24s_like(d, y), by = join_by(closest(f >= r))))
  }
})
