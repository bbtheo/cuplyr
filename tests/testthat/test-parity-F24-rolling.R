# F24 acceptance tests, tier A: native closest() rolling joins.
#
# dplyr 1.2.1 semantics (R/join-by.R parse_join_by_closest(), R/join-rows.R
# join_rows() -> vctrs::vec_locate_matches(filter = "max"/"min")), verified
# with tools/r (scratchpad/parity/features/F24/probe/p1.R, p2.R):
#   - per x row, keep the matching y rows whose rolling value is the max
#     (`>=`, `>`) or min (`<=`, `<`); every tied y row is returned, in y order;
#   - multiple=/unmatched=/relationship= act on those post-filter matches;
#   - na_matches = "na": a missing x rolling value matches a missing y rolling
#     value for `>=`/`<=` only; "never": rows with a missing key never match;
#   - no many-to-many warning (inequality joins never check for it).
# cuplyr uploads NaN as NA, so NaN is only tested where GPU arithmetic
# produces it ("NaN computed on the GPU in a rolling key is a missing value").
#
# "Native" is pinned with options(cuplyr.fallback = "error"): any CPU
# fallback inside these pipelines raises.

join_by <- dplyr::join_by

# `df` as the same kind of table as `d`: a tibble for the dplyr oracle, a
# tbl_gpu (lazy if `d` is lazy) for the GPU side.
f24_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f24_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f24_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f24_both(df, pipeline, ...)
}

# join_by(<prefix>closest(<lhs> <op> <rhs>)<suffix>) for one operator.
f24_by <- function(op, lhs = "t", rhs = "ty", before = "id, ", after = "") {
  eval(parse(text = sprintf("dplyr::join_by(%sclosest(%s %s %s)%s)",
                            before, lhs, op, rhs, after)))
}

f24_ops <- c(">=", ">", "<=", "<")

f24_mutating <- list(
  left = dplyr::left_join, inner = dplyr::inner_join,
  right = dplyr::right_join, full = dplyr::full_join
)
f24_filtering <- list(semi = dplyr::semi_join, anti = dplyr::anti_join)

# Did evaluating `expr` go through the CPU fallback?
f24_fell_back <- function(expr) {
  fell <- FALSE
  withCallingHandlers(
    force(expr),
    cuplyr_cpu_fallback = function(cnd) fell <<- TRUE
  )
  fell
}

# Ties (ty = 4, 4), exact hits, NA rolling values on both sides, NA equi keys
# on both sides, an x group with no y rows (id 3), a y group with no x rows
# (id 4), y rows that are nobody's closest match.
f24_x <- data.frame(
  id = c(1, 1, 1, 2, 2, 3, NA, 1, 2),
  t = c(5, 10, 0, 3, NA, 7, 4, 4, 100),
  xv = 1:9
)
f24_y <- data.frame(
  id = c(1, 1, 1, 2, 2, NA, 1, 4),
  ty = c(4, 4, 9, 3, NA, 1, -1, 2),
  yv = c("a", "b", "c", "d", "e", "f", "g", "h")
)

# No equi key; same rolling column name on both sides (t.x / t.y).
f24_x2 <- data.frame(t = c(NA, 2, Inf, -Inf, 5.5, 5, 0), xv = 1:7)
f24_y2 <- data.frame(t = c(1, NA, Inf, 5, 5, -Inf, 3), yv = 1:7)

# -----------------------------------------------------------------------------
# Operators x join types
# -----------------------------------------------------------------------------

test_that("closest() with an equi key matches dplyr for every operator and mutating join", {
  skip_if_no_gpu()
  for (op in f24_ops) {
    by <- f24_by(op)
    for (join in f24_mutating) {
      f24_native(f24_x, function(d) join(d, f24_like(d, f24_y), by = by))
    }
  }
})

test_that("closest() with no equi key handles ties, NA and +/-Inf", {
  skip_if_no_gpu()
  for (op in f24_ops) {
    by <- f24_by(op, lhs = "t", rhs = "t", before = "")
    for (join in f24_mutating) {
      f24_native(f24_x2, function(d) join(d, f24_like(d, f24_y2), by = by))
    }
  }
})

test_that("na_matches = 'never' drops missing keys for every operator and mutating join", {
  skip_if_no_gpu()
  for (op in f24_ops) {
    by <- f24_by(op)
    by2 <- f24_by(op, lhs = "t", rhs = "t", before = "")
    for (join in f24_mutating) {
      f24_native(f24_x, function(d) {
        join(d, f24_like(d, f24_y), by = by, na_matches = "never")
      })
      f24_native(f24_x2, function(d) {
        join(d, f24_like(d, f24_y2), by = by2, na_matches = "never")
      })
    }
  }
})

test_that("semi_join()/anti_join() accept closest()", {
  skip_if_no_gpu()
  for (op in f24_ops) {
    by <- f24_by(op)
    by2 <- f24_by(op, lhs = "t", rhs = "t", before = "")
    for (join in f24_filtering) {
      for (na in c("na", "never")) {
        f24_native(f24_x, function(d) {
          join(d, f24_like(d, f24_y), by = by, na_matches = na)
        })
        f24_native(f24_x2, function(d) {
          join(d, f24_like(d, f24_y2), by = by2, na_matches = na)
        })
      }
    }
  }
})

# -----------------------------------------------------------------------------
# Equi keys and key types
# -----------------------------------------------------------------------------

test_that("several equi keys of mixed types, in any condition order", {
  skip_if_no_gpu()
  lv <- c("p", "q", "r")
  x <- data.frame(
    g = c("a", "a", "b", "b", NA, "a"),
    f = factor(c("p", "q", "p", "p", "r", "p"), levels = lv),
    k = c(1L, 1L, 2L, 2L, 3L, 1L),
    t = c(10, 10, 5, 1, 7, 3),
    xv = 1:6,
    stringsAsFactors = FALSE
  )
  y <- data.frame(
    g = c("a", "a", "a", "b", NA, "b", "a"),
    f2 = factor(c("p", "q", "p", "p", "r", "p", "p"), levels = lv),
    k = c(1L, 1L, 1L, 2L, 3L, 2L, 1L),
    ty = c(9, 9, 2, 5, 6, 4, 9),
    yv = 1:7,
    stringsAsFactors = FALSE
  )
  bys <- list(
    join_by(g, closest(t >= ty), f == f2, k),
    join_by(closest(t > ty), g, k, f == f2),
    join_by(g, f == f2, k, closest(t <= ty)),
    join_by(k, closest(y$ty > x$t), g, f == f2)
  )
  for (by in bys) {
    for (join in c(f24_mutating, f24_filtering)) {
      f24_native(x, function(d) join(d, f24_like(d, y), by = by))
    }
  }
})

test_that("rolling keys of type integer, Date and POSIXct (with a time zone)", {
  skip_if_no_gpu()
  xi <- data.frame(id = c(1L, 1L, 2L, 2L), t = c(5L, NA, -3L, 8L), xv = 1:4)
  yi <- data.frame(id = c(1L, 1L, 2L, 2L, 2L), s = c(0L, 5L, 8L, NA, -3L), yv = 1:5)

  xd <- data.frame(t = as.Date(c("2024-01-05", "2024-01-01", NA, "2024-03-01")), xv = 1:4)
  yd <- data.frame(s = as.Date(c("2024-01-02", "2024-01-04", "2024-01-04", NA, "2024-03-01")),
                   yv = 1:5)

  p <- as.POSIXct("2024-03-10 01:00:00", tz = "America/New_York")
  xp <- data.frame(t = p + c(10, 100, NA, 3600 * 3), xv = 1:4)
  yp <- data.frame(s = p + c(5, 50, 100, 3600 * 2, NA), yv = 1:5)

  for (op in f24_ops) {
    by_i <- f24_by(op, lhs = "t", rhs = "s")
    by_0 <- f24_by(op, lhs = "t", rhs = "s", before = "")
    for (join in f24_mutating) {
      f24_native(xi, function(d) join(d, f24_like(d, yi), by = by_i))
      f24_native(xd, function(d) join(d, f24_like(d, yd), by = by_0))
      f24_native(xp, function(d) join(d, f24_like(d, yp), by = by_0))
    }
  }
})

test_that("integer and double rolling keys are compared in their common type", {
  skip_if_no_gpu()
  xi <- data.frame(t = c(1L, 3L, 5L, NA), xv = 1:4)
  yd <- data.frame(s = c(0.5, 2.5, 3, 5.25, NA), yv = 1:5)
  xd <- data.frame(t = c(1.5, 3, 4.99, NA), xv = 1:4)
  yi <- data.frame(s = c(0L, 3L, 3L, 5L, NA), yv = 1:5)
  for (op in f24_ops) {
    by <- f24_by(op, lhs = "t", rhs = "s", before = "")
    for (join in f24_mutating) {
      f24_native(xi, function(d) join(d, f24_like(d, yd), by = by))
      f24_native(xd, function(d) join(d, f24_like(d, yi), by = by))
    }
  }
})

# -----------------------------------------------------------------------------
# Output columns
# -----------------------------------------------------------------------------

test_that("column naming: shared rolling name, keep = TRUE, custom suffix", {
  skip_if_no_gpu()
  x <- data.frame(id = c(1, 1, 2), t = c(1, 5, 2), v = 1:3)
  y <- data.frame(id = c(1, 1, 2), t = c(0, 3, 5), v = 4:6)
  for (join in f24_mutating) {
    f24_native(x, function(d) join(d, f24_like(d, y), by = join_by(id, closest(t >= t))))
    f24_native(x, function(d) {
      join(d, f24_like(d, y), by = join_by(id, closest(t >= t)), keep = TRUE)
    })
    f24_native(x, function(d) {
      join(d, f24_like(d, y), by = join_by(id, closest(t <= t)), suffix = c("_x", "_y"))
    })
  }
})

test_that("keep = FALSE raises dplyr's error", {
  skip_if_no_gpu()
  x <- tibble::as_tibble(f24_x)
  y <- tibble::as_tibble(f24_y)
  expected <- tryCatch(
    dplyr::left_join(x, y, by = f24_by(">="), keep = FALSE),
    error = function(e) e
  )
  expect_s3_class(expected, "error")
  for (lazy in c(FALSE, TRUE)) {
    actual <- tryCatch(
      collect(dplyr::left_join(tbl_gpu(f24_x, lazy = lazy), tbl_gpu(f24_y, lazy = lazy),
                               by = f24_by(">="), keep = FALSE)),
      error = function(e) e
    )
    expect_s3_class(actual, "error")
    expect_identical(conditionMessage(actual), conditionMessage(expected))
  }
})

# -----------------------------------------------------------------------------
# multiple= / relationship= / unmatched=
# -----------------------------------------------------------------------------

# x rows 1 and 2 share the tied closest pair (ty = 4, rows 2 and 3); y row 1
# matches before the filter but is nobody's closest row.
f24_tx <- data.frame(id = c(1, 1, 1), t = c(5, 6, 1), xv = 1:3)
f24_ty <- data.frame(id = c(1, 1, 1, 1), ty = c(2, 4, 4, 0), yv = c("a", "b", "c", "d"))

test_that("multiple = 'first'/'last'/'any' pick among the tied closest rows", {
  skip_if_no_gpu()
  for (m in c("first", "last", "any")) {
    for (op in c(">=", "<")) {
      by <- f24_by(op)
      for (join in f24_mutating) {
        f24_native(f24_tx, function(d) {
          join(d, f24_like(d, f24_ty), by = by, multiple = m)
        })
      }
    }
  }
})

# Run `f(x, y)` on tibbles and on eager and lazy tbl_gpus; the condition
# (error, or warning if there is no error) must have dplyr's class and text.
# `native = FALSE` allows the CPU fallback.
f24_same_condition <- function(x, y, f, native = TRUE) {
  capture <- function(expr) {
    w <- NULL
    e <- tryCatch(
      withCallingHandlers({
        force(expr)
        NULL
      }, warning = function(cnd) {
        w <<- cnd
        invokeRestart("muffleWarning")
      }),
      error = function(cnd) cnd
    )
    e %||% w
  }
  expected <- capture(f(tibble::as_tibble(x), tibble::as_tibble(y)))
  expect_false(is.null(expected), info = "dplyr raised no condition: the fixture is wrong")
  withr::local_options(cuplyr.fallback = if (native) "error" else "silent")
  for (lazy in c(FALSE, TRUE)) {
    actual <- capture(collect(f(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy))))
    expect_false(is.null(actual), info = sprintf("cuplyr raised no condition (lazy = %s)", lazy))
    expect_identical(class(actual), class(expected))
    expect_identical(rlang::cnd_message(actual), rlang::cnd_message(expected))
  }
}

# Every warning's class, in order, from evaluating `expr`.
f24_warning_classes <- function(expr) {
  classes <- list()
  withCallingHandlers(force(expr), warning = function(w) {
    classes[[length(classes) + 1L]] <<- class(w)
    invokeRestart("muffleWarning")
  })
  classes
}

# multiple = "error"/"warning" are undocumented in dplyr 1.2.1 (NEWS lists
# them as defunct) but still reach vctrs and work. cuplyr may run them on
# the CPU fallback; the conditions and data must match dplyr whatever the
# route (round-1 review B1/B3: a native "warning" dropped rows on a non-equi
# join and lost the many-to-many warning on an equi join).
test_that("multiple = 'error'/'warning' match dplyr (CPU fallback allowed)", {
  skip_if_no_gpu()
  by <- f24_by(">=")
  f24_same_condition(f24_tx, f24_ty, function(x, y) {
    dplyr::left_join(x, y, by = by, multiple = "error")
  }, native = FALSE)
  f24_same_condition(f24_tx, f24_ty, function(x, y) {
    dplyr::inner_join(x, y, by = by, multiple = "warning")
  }, native = FALSE)
  # The data under multiple = "warning" is every match.
  xa <- data.frame(a = c(1, 5), xv = 1:2)
  ya <- data.frame(b = c(0, 2, 3), yv = 1:3)
  suppressWarnings({
    f24_both(f24_tx, function(d) {
      dplyr::left_join(d, f24_like(d, f24_ty), by = by, multiple = "warning")
    })
    for (join in f24_mutating) {
      f24_both(xa, function(d) {
        join(d, f24_like(d, ya), by = join_by(a >= b), multiple = "warning")
      })
    }
  })
  # An equi join under multiple = "warning" also raises the many-to-many
  # warning, after the multiple-matches one.
  xk <- data.frame(k = c(2, 1, 1), xv = 1:3)
  yk <- data.frame(k = c(1, 1, 2, 2), yv = 1:4)
  expected <- f24_warning_classes(
    dplyr::inner_join(tibble::as_tibble(xk), tibble::as_tibble(yk), by = "k", multiple = "warning")
  )
  expect_length(expected, 2L)
  for (lazy in c(FALSE, TRUE)) {
    actual <- f24_warning_classes(collect(dplyr::inner_join(
      tbl_gpu(xk, lazy = lazy), tbl_gpu(yk, lazy = lazy), by = "k", multiple = "warning"
    )))
    expect_identical(actual, expected)
  }
})

test_that("unmatched = 'error' counts y rows that multiple = 'first'/'last' capped out", {
  skip_if_no_gpu()
  # Round-1 review B2: vctrs decides the remaining y rows after `multiple`.
  x <- data.frame(t = 5, xv = 1L)
  y <- data.frame(ty = c(4, 4), yv = 1:2)
  by <- join_by(closest(t >= ty))
  f24_same_condition(x, y, function(x, y) {
    dplyr::left_join(x, y, by = by, multiple = "first", unmatched = "error")
  })
  f24_same_condition(x, y, function(x, y) {
    dplyr::inner_join(x, y, by = by, multiple = "last", unmatched = "error")
  })
  f24_same_condition(f24_tx, f24_ty, function(x, y) {
    dplyr::left_join(x, y, by = f24_by(">="), multiple = "first", unmatched = "error")
  })
  # No tie is capped out: no error, same data.
  y1 <- data.frame(ty = c(4, 1), yv = 1:2)
  x1 <- data.frame(t = c(5, 2), xv = 1:2)
  for (m in c("first", "last", "any")) {
    f24_native(x1, function(d) {
      dplyr::left_join(d, f24_like(d, y1), by = by, multiple = m, unmatched = "error")
    })
  }
})

# cuplyr uploads NaN as NA, but GPU arithmetic produces real NaN. dplyr
# (vctrs, nan_distinct = TRUE) treats NaN as a missing value distinct from
# NA: it never matches a number, and matches NaN only for `>=`/`<=` under
# na_matches = "na" (scratchpad/parity/features/F24/probe/p4.R).
test_that("NaN computed on the GPU in a rolling key is a missing value", {
  skip_if_no_gpu()
  x <- data.frame(id = c(1, 1, 1, 1, 2, 2, 2), t = c(0, 5, 3, 0, 0, 7, NA),
                  dv = c(0, 1, 1, 0, 1, 0, 1), xv = 1:7)
  y <- data.frame(id = c(1, 1, 1, 2, 2, 2), ty = c(0, 4, 9, 0, 2, NA),
                  e = c(0, 1, 1, 0, 1, 1), yv = 1:6)
  for (op in f24_ops) {
    by <- f24_by(op)
    for (na in c("na", "never")) {
      for (join in c(f24_mutating, f24_filtering)) {
        f24_native(x, function(d) {
          join(dplyr::mutate(d, t = t / dv), dplyr::mutate(f24_like(d, y), ty = ty / e),
               by = by, na_matches = na)
        })
      }
    }
  }
})

test_that("relationship= checks run on the closest matches only", {
  skip_if_no_gpu()
  by <- f24_by(">=")
  for (rel in c("one-to-one", "one-to-many", "many-to-one")) {
    f24_same_condition(f24_tx, f24_ty, function(x, y) {
      dplyr::left_join(x, y, by = by, relationship = rel)
    })
  }
  # Before the filter, y row 1 (ty = 2) is matched by x rows 1 and 2; after
  # it, every y row is matched at most once and every x row once.
  x <- data.frame(id = c(1, 1), t = c(3, 9), xv = 1:2)
  y <- data.frame(id = c(1, 1), ty = c(2, 8), yv = 1:2)
  for (rel in c("one-to-one", "one-to-many", "many-to-one", "many-to-many")) {
    f24_native(x, function(d) dplyr::left_join(d, f24_like(d, y), by = by, relationship = rel))
  }
})

test_that("unmatched = 'error' sees x rows without a closest match and y rows nobody picked", {
  skip_if_no_gpu()
  by <- f24_by(">=")
  # left: y row 1 matches before the filter but is nobody's closest row.
  f24_same_condition(f24_tx, f24_ty, function(x, y) {
    dplyr::left_join(x, y, by = by, unmatched = "error")
  })
  f24_same_condition(f24_tx, f24_ty, function(x, y) {
    dplyr::inner_join(x, y, by = by, unmatched = c("drop", "error"))
  })
  # inner/right: x row 3 (t = 1) has no y row with ty <= 1 in its group.
  x <- data.frame(id = c(1, 1, 1), t = c(5, 6, -5), xv = 1:3)
  f24_same_condition(x, f24_ty, function(x, y) {
    dplyr::inner_join(x, y, by = by, unmatched = "error")
  })
  f24_same_condition(x, f24_ty, function(x, y) {
    dplyr::right_join(x, y, by = by, unmatched = "error")
  })
  # Every x row has a match: no error, and the data matches.
  f24_native(f24_tx, function(d) {
    dplyr::inner_join(d, f24_like(d, f24_ty), by = by, unmatched = c("error", "drop"))
  })
})

test_that("tied closest rows don't raise the many-to-many warning", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  x <- data.frame(id = c(1, 1), t = c(5, 5))
  y <- data.frame(id = c(1, 1), ty = c(4, 4))
  for (lazy in c(FALSE, TRUE)) {
    expect_no_warning(collect(dplyr::left_join(
      tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy), by = f24_by(">=")
    )))
  }
})

# -----------------------------------------------------------------------------
# Empty inputs, groups, lazy pipelines
# -----------------------------------------------------------------------------

test_that("0-row x and/or y", {
  skip_if_no_gpu()
  x0 <- f24_x[0, ]
  y0 <- f24_y[0, ]
  for (op in c(">=", "<")) {
    by <- f24_by(op)
    for (join in c(f24_mutating, f24_filtering)) {
      f24_native(x0, function(d) join(d, f24_like(d, f24_y), by = by))
      f24_native(f24_x, function(d) join(d, f24_like(d, y0), by = by))
      f24_native(x0, function(d) join(d, f24_like(d, y0), by = by))
    }
  }
})

test_that("a grouped x keeps its groups", {
  skip_if_no_gpu()
  by <- f24_by(">=")
  for (join in c(f24_mutating, f24_filtering)) {
    f24_native(f24_x, function(d) {
      join(dplyr::group_by(d, id), f24_like(d, f24_y), by = by)
    })
  }
})

test_that("lazy pipelines: a filter on y columns is never pushed below a closest() join", {
  skip_if_no_gpu()
  # x's closest row is yv = "a" (ty = 4). Filtering y first would make
  # yv = "b" (ty = 2) the closest one instead.
  x <- data.frame(id = c(1, 1, 2), t = c(5, 3, 1), xv = 1:3)
  y <- data.frame(id = c(1, 1, 2), ty = c(4, 2, 0), yv = c("a", "b", "c"))
  by <- f24_by(">=")
  f24_native(x, function(d) {
    dplyr::inner_join(d, f24_like(d, y), by = by) |> dplyr::filter(yv != "a")
  })
  f24_native(x, function(d) {
    dplyr::right_join(d, f24_like(d, y), by = by) |> dplyr::filter(ty < 4)
  })
  f24_native(x, function(d) {
    dplyr::left_join(d, f24_like(d, y), by = by) |>
      dplyr::filter(yv != "a" | is.na(yv), xv > 1)
  })
  # Filtering y before the join is a different query, and also native.
  f24_native(x, function(d) {
    dplyr::inner_join(d, dplyr::filter(f24_like(d, y), yv != "a"), by = by)
  })
  # A longer chain: x-side filter (pushable), mutate, arrange, select.
  f24_native(x, function(d) {
    dplyr::left_join(dplyr::filter(d, xv < 3), f24_like(d, y), by = by) |>
      dplyr::filter(t > 2) |>
      dplyr::mutate(gap = t - ty) |>
      dplyr::arrange(dplyr::desc(gap)) |>
      dplyr::select(id, gap, yv)
  })
})

# -----------------------------------------------------------------------------
# Scale
# -----------------------------------------------------------------------------

test_that("random data with ties and NAs: exact rows and row order", {
  skip_if_no_gpu()
  set.seed(2024)
  nx <- 20000L
  ny <- 2000L
  x <- data.frame(
    id = sample(c(1:50, NA), nx, replace = TRUE),
    t = sample(c(0:400, NA), nx, replace = TRUE),
    xv = seq_len(nx)
  )
  y <- data.frame(
    id = sample(c(1:55, NA), ny, replace = TRUE),
    ty = sample(c(0:400, NA), ny, replace = TRUE) + 0.5 * sample(0:1, ny, replace = TRUE),
    yv = seq_len(ny)
  )
  for (op in f24_ops) {
    by <- f24_by(op)
    f24_native(x, function(d) dplyr::left_join(d, f24_like(d, y), by = by))
  }
  by <- f24_by(">=")
  for (join in list(dplyr::inner_join, dplyr::right_join, dplyr::full_join,
                    dplyr::semi_join, dplyr::anti_join)) {
    f24_native(x, function(d) join(d, f24_like(d, y), by = by))
  }
})

test_that("a rolling join with no equi key scales without materialising the inequality join", {
  skip_if_no_gpu()
  # 1e5 x 1e5 rows with no equi key: about 5e9 pairs satisfy t >= ty, so an
  # inequality join followed by a per-row max filter can't run here.
  set.seed(7)
  n <- 100000L
  x <- data.frame(t = runif(n, 0, 1e6), xv = seq_len(n))
  y <- data.frame(ty = round(runif(n, 0, 1e6)), yv = seq_len(n))
  by <- join_by(closest(t >= ty))
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(x, function(d) dplyr::left_join(d, f24_like(d, y), by = by))
})

# -----------------------------------------------------------------------------
# Outside the native allowlist: CPU fallback, still exact
# -----------------------------------------------------------------------------

test_that("two closest() conditions, or closest() with another inequality, fall back and stay correct", {
  skip_if_no_gpu()
  x <- data.frame(id = c(1, 1, 2), a = c(5, 2, 7), b = c(5, 1, 3), lo = c(3, 0, 1))
  y <- data.frame(id = c(1, 1, 2, 2), c = c(4, 6, 2, 7), d = c(6, 4, 3, 9), v = 1:4)
  bys <- list(
    join_by(closest(a >= c), closest(b <= d)),
    join_by(id, closest(a >= c), lo <= c)
  )
  for (by in bys) {
    expect_true(f24_fell_back(collect(dplyr::left_join(tbl_gpu(x), tbl_gpu(y), by = by))))
    for (join in f24_mutating) {
      f24_both(x, function(d) join(d, f24_like(d, y), by = by))
    }
  }
})

test_that("factor, logical and Date-vs-POSIXct rolling keys stay correct", {
  skip_if_no_gpu()
  lv <- c("c", "b", "a")
  xf <- data.frame(f = factor(c("b", "c", "a"), levels = lv))
  yf <- data.frame(g = factor(c("a", "b", "c"), levels = lv), v = 1:3)
  xl <- data.frame(f = c(TRUE, FALSE, NA))
  yl <- data.frame(g = c(FALSE, TRUE, NA), v = 1:3)
  xd <- data.frame(d = as.Date(c("2024-01-05", "2024-01-01")))
  yd <- data.frame(e = as.POSIXct(c("2024-01-02", "2024-01-04"), tz = "UTC"), v = 1:2)
  for (op in c(">=", "<")) {
    for (join in list(dplyr::left_join, dplyr::full_join)) {
      f24_both(xf, function(d) join(d, f24_like(d, yf), by = f24_by(op, "f", "g", "")))
      f24_both(xl, function(d) join(d, f24_like(d, yl), by = f24_by(op, "f", "g", "")))
      f24_both(xd, function(d) join(d, f24_like(d, yd), by = f24_by(op, "d", "e", "")))
    }
  }
})
