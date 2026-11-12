# F71 acceptance tests: on a group_by(.drop = FALSE) table, the filtering
# joins (semi_join()/anti_join()) and the slice family keep every group of the
# input, including the groups they empty, as dplyr does.
#
# dplyr 1.2.1 (R/generics.R, dplyr_row_slice.grouped_df()): join_filter()
# (R/join.R) and slice() (R/slice.R, which slice_head/tail/min/max call with
# .preserve = FALSE) both end in dplyr_row_slice(), which keeps the input's
# group_data() and trims empty groups only when `.drop` is TRUE. So after
# semi_join() empties a group, n_groups() still counts it, and summarise()/
# count()/tally() give it a row (n = 0, sum = 0). Mutating joins and set
# operations regroup instead (dplyr_reconstruct()), so they lose the emptied
# groups: those are guards here.

f71_df <- data.frame(
  g = c("a", "a", "b", "c", "d", NA, "b"),
  f = factor(c("x", "x", "y", "z", "x", "y", "y"), levels = c("x", "y", "z", "w")),
  x = c(1L, 2L, 3L, 4L, 5L, 6L, 7L),
  stringsAsFactors = FALSE
)
f71_y <- data.frame(g = c("a", "b"), stringsAsFactors = FALSE)

# y as the same kind of table as d: a tibble for the dplyr oracle, an eager
# tbl_gpu for the GPU side.
f71_as <- function(d, ydf) {
  if (is_tbl_gpu(d)) tbl_gpu(ydf) else tibble::as_tibble(ydf)
}

f71_group_struct <- function(gd) {
  keys <- tibble::as_tibble(gd[setdiff(names(gd), ".rows")])
  attr(keys, ".drop") <- NULL
  list(keys = keys, rows = lapply(gd$.rows, as.integer))
}

f71_drop_of <- function(x) isFALSE(attr(group_data(x), ".drop"))

# Checks the whole group structure of `pipeline(input)` against dplyr, eager
# and lazy: group_data() (keys and rows), n_groups(), group_keys(),
# group_size(), the .drop setting, the collected grouped_df's group_data() and
# the rows.
f71_check_groups <- function(pipeline, df = f71_df, lazies = c(FALSE, TRUE),
                             check_drop = TRUE) {
  expected <- pipeline(tibble::as_tibble(df))
  for (lazy in lazies) {
    label <- if (lazy) "lazy" else "eager"
    got <- pipeline(tbl_gpu(df, lazy = lazy))
    expect_true(is_tbl_gpu(got), info = label)
    expect_identical(group_vars(got), group_vars(expected), info = label)
    expect_equal(f71_group_struct(group_data(got)),
                 f71_group_struct(group_data(expected)),
                 info = paste(label, "group_data"))
    expect_identical(dplyr::n_groups(got), dplyr::n_groups(expected), info = label)
    expect_equal(tibble::as_tibble(dplyr::group_keys(got)),
                 tibble::as_tibble(dplyr::group_keys(expected)),
                 info = paste(label, "group_keys"))
    expect_identical(dplyr::group_size(got), dplyr::group_size(expected),
                     info = paste(label, "group_size"))
    if (check_drop) {
      expect_identical(f71_drop_of(got), f71_drop_of(expected), info = paste(label, ".drop"))
      expect_identical(f71_drop_of(collect(got)), f71_drop_of(expected),
                       info = paste(label, "collected .drop"))
    }
    collected <- collect(got)
    expect_equal(f71_group_struct(dplyr::group_data(collected)),
                 f71_group_struct(dplyr::group_data(expected)),
                 info = paste(label, "collected group_data"))
    expect_equal(tibble::as_tibble(dplyr::ungroup(collected)),
                 tibble::as_tibble(dplyr::ungroup(expected)),
                 info = paste(label, "rows"))
  }
}

# ---------------------------------------------------------------------------
# semi_join() / anti_join(): group structure
# ---------------------------------------------------------------------------

test_that("semi_join() keeps the groups it empties on a .drop = FALSE table (character key)", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_y), by = "g")
  })
})

test_that("anti_join() keeps the groups it empties on a .drop = FALSE table (character key)", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    anti_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_y), by = "g")
  })
})

test_that("filtering joins keep emptied groups with a factor key, unused levels included", {
  skip_if_no_gpu()
  # the NA-g row and the "z" level row are emptied; "w" never had rows
  f71_check_groups(function(d) {
    semi_join(group_by(d, f, .drop = FALSE), f71_as(d, f71_y), by = "g")
  })
  f71_check_groups(function(d) {
    anti_join(group_by(d, f, .drop = FALSE), f71_as(d, f71_y), by = "g")
  })
})

test_that("filtering joins keep emptied groups with two grouping columns", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, f, .drop = FALSE), f71_as(d, f71_y), by = "g")
  })
  f71_check_groups(function(d) {
    anti_join(group_by(d, f, g, .drop = FALSE), f71_as(d, f71_y), by = "g")
  })
  # character + integer keys (no factor at all)
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, x, .drop = FALSE), f71_as(d, f71_y), by = "g")
  })
})

test_that("the NA group survives being emptied", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE),
              f71_as(d, data.frame(x = c(1L, 2L, 3L))), by = "x")
  })
})

test_that("every by= form keeps the emptied groups", {
  skip_if_no_gpu()
  yk <- data.frame(k = c("a", "b"), stringsAsFactors = FALSE)
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_y), by = dplyr::join_by(g))
  })
  f71_check_groups(function(d) {
    anti_join(group_by(d, g, .drop = FALSE), f71_as(d, yk), by = c("g" = "k"))
  })
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f71_as(d, yk), by = dplyr::join_by(g == k))
  })
  f71_check_groups(function(d) {
    suppressMessages(semi_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_y)))
  })
  f71_check_groups(function(d) {
    suppressMessages(anti_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_y), by = NULL))
  })
})

test_that("non-equi join_by() filtering joins keep the emptied groups", {
  skip_if_no_gpu()
  ylo <- data.frame(lo = 4L)
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f71_as(d, ylo), by = dplyr::join_by(x >= lo))
  })
  f71_check_groups(function(d) {
    anti_join(group_by(d, g, .drop = FALSE), f71_as(d, ylo), by = dplyr::join_by(x >= lo))
  })
})

test_that("a filtering join with a tibble y and copy = TRUE keeps the emptied groups", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), tibble::as_tibble(f71_y), by = "g",
              copy = TRUE)
  })
})

test_that("a lazy y and a lazy x with pending ops keep the emptied groups", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    yy <- if (is_tbl_gpu(d)) as_lazy(tbl_gpu(f71_y)) else tibble::as_tibble(f71_y)
    d |>
      mutate(x2 = x * 2L) |>
      filter(x < 7L) |>
      group_by(g, .drop = FALSE) |>
      semi_join(yy, by = "g")
  })
})

test_that("an empty y empties every group but keeps them all", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_y[0, , drop = FALSE]), by = "g")
  })
  f71_check_groups(function(d) {
    anti_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_df[, "g", drop = FALSE]), by = "g")
  })
})

test_that("zero-row inputs keep the right groups", {
  skip_if_no_gpu()
  df0 <- f71_df[0, ]
  f71_check_groups(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f71_as(d, f71_y), by = "g")
  }, df = df0)
  f71_check_groups(function(d) {
    anti_join(group_by(d, f, .drop = FALSE), f71_as(d, f71_y), by = "g")
  }, df = df0)
})

test_that("chained filtering joins and a following filter() keep the groups", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |>
      anti_join(f71_as(d, data.frame(x = 3L)), by = "x")
  })
  f71_check_groups(function(d) {
    group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |>
      filter(x > 1L)
  })
  f71_check_groups(function(d) {
    group_by(d, g, .drop = FALSE) |>
      filter(x > 1L) |>
      anti_join(f71_as(d, f71_y), by = "g")
  })
})

test_that("verbs after a filtering join keep the emptied groups", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |>
      mutate(z = x * 2L)
  })
  f71_check_groups(function(d) {
    group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |>
      arrange(desc(x))
  })
  f71_check_groups(function(d) {
    group_by(d, g, .drop = FALSE) |>
      anti_join(f71_as(d, f71_y), by = "g") |>
      select(g, x)
  })
})

test_that("lazy filtering joins stay lazy and keep the groups through compute()", {
  skip_if_no_gpu()
  expected <- semi_join(group_by(tibble::as_tibble(f71_df), g, .drop = FALSE),
                        tibble::as_tibble(f71_y), by = "g")
  got <- tbl_gpu(f71_df, lazy = TRUE) |>
    group_by(g, .drop = FALSE) |>
    semi_join(tbl_gpu(f71_y), by = "g")
  expect_identical(got$exec_mode, "lazy")
  computed <- compute(got)
  expect_identical(dplyr::n_groups(computed), dplyr::n_groups(expected))
  expect_equal(f71_group_struct(group_data(computed)),
               f71_group_struct(group_data(expected)))
})

# ---------------------------------------------------------------------------
# semi_join() / anti_join(): what follows sees the emptied groups
# ---------------------------------------------------------------------------

test_that("summarise() after a filtering join gives the emptied groups a row", {
  skip_if_no_gpu()
  pipes <- list(
    function(d) group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |>
      summarise(n = n(), s = sum(x)),
    function(d) group_by(d, g, .drop = FALSE) |>
      anti_join(f71_as(d, f71_y), by = "g") |>
      summarise(n = n(), s = sum(x), m = mean(x), mx = max(x, na.rm = FALSE)),
    function(d) group_by(d, f, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |>
      summarise(n = n(), s = sum(x)),
    function(d) group_by(d, g, f, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |>
      summarise(n = n(), s = sum(x), .groups = "drop"),
    function(d) group_by(d, g, x, .drop = FALSE) |>
      anti_join(f71_as(d, f71_y), by = "g") |>
      summarise(n = n(), .groups = "keep")
  )
  for (p in pipes) {
    suppressWarnings({
      expect_same_as_dplyr(f71_df, p)
      expect_same_as_dplyr_lazy(f71_df, p)
    })
  }
})

test_that("summarise() with the default .groups after a two-key filtering join", {
  skip_if_no_gpu()
  p <- function(d) group_by(d, g, f, .drop = FALSE) |>
    semi_join(f71_as(d, f71_y), by = "g") |>
    summarise(n = n())
  expect_same_as_dplyr(f71_df, function(d) suppressMessages(p(d)))
  expect_same_as_dplyr_lazy(f71_df, function(d) suppressMessages(p(d)))
  # the remaining grouping keeps every g, emptied ones included
  f71_check_groups(function(d) suppressMessages(p(d)))
})

test_that("count() and tally() after a filtering join count the emptied groups as 0", {
  skip_if_no_gpu()
  pipes <- list(
    function(d) group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |> count(),
    function(d) group_by(d, g, .drop = FALSE) |>
      anti_join(f71_as(d, f71_y), by = "g") |> tally(),
    function(d) group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |> tally(x),
    function(d) group_by(d, f, .drop = FALSE) |>
      anti_join(f71_as(d, f71_y), by = "g") |> count(),
    function(d) group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, f71_y), by = "g") |> count(name = "rows")
  )
  for (p in pipes) {
    expect_same_as_dplyr(f71_df, p)
    expect_same_as_dplyr_lazy(f71_df, p)
  }
})

test_that("group_split() after a filtering join returns the emptied groups", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    expected <- group_by(tibble::as_tibble(f71_df), g, .drop = FALSE) |>
      semi_join(tibble::as_tibble(f71_y), by = "g") |>
      group_split()
    got <- tbl_gpu(f71_df, lazy = lazy) |>
      group_by(g, .drop = FALSE) |>
      semi_join(tbl_gpu(f71_y), by = "g") |>
      group_split()
    expect_identical(length(got), length(expected))
    if (length(got) != length(expected)) next
    for (i in seq_along(expected)) {
      expect_equal(tibble::as_tibble(dplyr::ungroup(collect(got[[i]]))),
                   tibble::as_tibble(dplyr::ungroup(expected[[i]])),
                   info = paste("lazy", lazy, "piece", i))
    }
  }
})

test_that("printing a filtering join result reports dplyr's group count", {
  skip_if_no_gpu()
  expected <- group_by(tibble::as_tibble(f71_df), g, .drop = FALSE) |>
    anti_join(tibble::as_tibble(f71_y), by = "g")
  header <- grep("^# Groups:", utils::capture.output(print(expected)), value = TRUE)
  expect_identical(header, "# Groups:   g [5]")
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(f71_df, lazy = lazy) |>
      group_by(g, .drop = FALSE) |>
      anti_join(tbl_gpu(f71_y), by = "g")
    out <- utils::capture.output(print(got))
    expect_true(any(grepl("# Groups:   g [5]", out, fixed = TRUE)),
                info = paste("lazy", lazy, paste(out, collapse = "\n")))
  }
})

# ---------------------------------------------------------------------------
# .drop = TRUE and ungrouped: unchanged
# ---------------------------------------------------------------------------

test_that("with .drop = TRUE the filtering joins still drop the emptied groups", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    semi_join(group_by(d, g), f71_as(d, f71_y), by = "g")
  })
  f71_check_groups(function(d) {
    anti_join(group_by(d, f), f71_as(d, f71_y), by = "g")
  })
  p <- function(d) group_by(d, g) |> semi_join(f71_as(d, f71_y), by = "g") |>
    summarise(n = n(), s = sum(x))
  expect_same_as_dplyr(f71_df, p)
  expect_same_as_dplyr_lazy(f71_df, p)
})

test_that("ungrouped filtering joins are unchanged", {
  skip_if_no_gpu()
  for (p in list(
    function(d) semi_join(d, f71_as(d, f71_y), by = "g"),
    function(d) anti_join(d, f71_as(d, f71_y), by = "g"),
    function(d) semi_join(d, f71_as(d, data.frame(lo = 4L)), by = dplyr::join_by(x >= lo))
  )) {
    expect_same_as_dplyr(f71_df, p)
    expect_same_as_dplyr_lazy(f71_df, p)
  }
})

# ---------------------------------------------------------------------------
# Guards: verbs that regroup must keep dplyr's regrouping
# ---------------------------------------------------------------------------

test_that("mutating joins on a .drop = FALSE table still regroup like dplyr", {
  skip_if_no_gpu()
  yv <- data.frame(g = c("a", "b"), v = c(10L, 20L), stringsAsFactors = FALSE)
  f71_check_groups(function(d) {
    inner_join(group_by(d, g, .drop = FALSE), f71_as(d, yv), by = "g")
  })
  f71_check_groups(function(d) {
    right_join(group_by(d, g, .drop = FALSE), f71_as(d, yv), by = "g")
  })
  f71_check_groups(function(d) {
    inner_join(group_by(d, g, .drop = FALSE), f71_as(d, yv), by = "g") |>
      filter(x > 1L)
  })
})

test_that("set operations on a .drop = FALSE table still regroup like dplyr", {
  skip_if_no_gpu()
  # check_drop = FALSE: the set operations lose `.drop = FALSE` on dev
  # (pre-existing, a different cause; an F71 open note), not checked here.
  sub <- f71_df[c(1, 2, 7), ]
  for (op in list(dplyr::intersect, dplyr::setdiff, dplyr::union, dplyr::symdiff)) {
    f71_check_groups(function(d) {
      op(group_by(d, g, .drop = FALSE), f71_as(d, sub))
    }, check_drop = FALSE)
  }
})

# ---------------------------------------------------------------------------
# The slice family shares the cause (dplyr_row_slice() keeps .drop = FALSE
# groups whatever .preserve says)
# ---------------------------------------------------------------------------

test_that("slice() keeps the groups it empties on a .drop = FALSE table", {
  skip_if_no_gpu()
  f71_check_groups(function(d) slice(group_by(d, g, .drop = FALSE), 2))
  f71_check_groups(function(d) slice(group_by(d, g, .drop = FALSE), -1))
  f71_check_groups(function(d) slice(group_by(d, g, .drop = FALSE), 0))
  f71_check_groups(function(d) slice(group_by(d, g, f, .drop = FALSE), 2))
  f71_check_groups(function(d) slice(group_by(d, g, .drop = FALSE), n() + 1L))
})

test_that("slice_head()/slice_tail() keep the groups they empty on a .drop = FALSE table", {
  skip_if_no_gpu()
  f71_check_groups(function(d) slice_head(group_by(d, g, .drop = FALSE), n = -1))
  f71_check_groups(function(d) slice_head(group_by(d, g, .drop = FALSE), n = 0))
  f71_check_groups(function(d) slice_tail(group_by(d, g, .drop = FALSE), n = 0))
  f71_check_groups(function(d) slice_tail(group_by(d, g, .drop = FALSE), prop = 0.5))
  f71_check_groups(function(d) slice_head(group_by(d, f, .drop = FALSE), n = -1))
})

test_that("slice_min()/slice_max() keep the groups they empty on a .drop = FALSE table", {
  skip_if_no_gpu()
  f71_check_groups(function(d) slice_min(group_by(d, g, .drop = FALSE), x, n = 0))
  f71_check_groups(function(d) slice_max(group_by(d, g, .drop = FALSE), x, n = -1))
  f71_check_groups(function(d) slice_max(group_by(d, g, .drop = FALSE), x, prop = 0.5))
})

test_that("summarise() and count() after an emptying slice include the emptied groups", {
  skip_if_no_gpu()
  for (p in list(
    function(d) group_by(d, g, .drop = FALSE) |> slice(2) |> summarise(n = n(), s = sum(x)),
    function(d) group_by(d, g, .drop = FALSE) |> slice_head(n = -1) |> count(),
    function(d) group_by(d, g, .drop = FALSE) |> slice_max(x, n = -1) |> tally()
  )) {
    expect_same_as_dplyr(f71_df, p)
    expect_same_as_dplyr_lazy(f71_df, p)
  }
})

test_that("with .drop = TRUE or .by, slices still lose the emptied groups", {
  skip_if_no_gpu()
  f71_check_groups(function(d) slice(group_by(d, g), 2))
  f71_check_groups(function(d) slice_head(group_by(d, g), n = -1))
  f71_check_groups(function(d) slice_min(group_by(d, g), x, n = 0))
  for (p in list(
    function(d) slice(d, 2, .by = g),
    function(d) slice_head(d, n = -1, by = g)
  )) {
    expect_same_as_dplyr(f71_df, p)
    expect_same_as_dplyr_lazy(f71_df, p)
  }
})

# ---------------------------------------------------------------------------
# Round-1 review additions (B1, N3): keys only when a group can be lost, so a
# later summarise() stays on the GPU where dev already got it right; more
# coverage of the NA factor group, pending ops before a slice, head()/tail()
# and the set-op key leak.
# ---------------------------------------------------------------------------

# TRUE if evaluating `expr` (pipeline + collect) signalled a CPU fallback.
f71_fell_back <- function(expr) {
  fell <- FALSE
  withCallingHandlers(expr, cuplyr_cpu_fallback = function(cnd) fell <<- TRUE)
  fell
}

f71_big <- data.frame(
  g = rep(c("a", "b", "c", "d"), times = 5),
  f = factor(rep(c("x", "y", "z", "x", "y"), times = 4), levels = c("x", "y", "z", "w")),
  x = as.double(1:20),
  stringsAsFactors = FALSE
)

test_that("factor-only keys without NA: summarise() after an emptying semi_join() stays on the GPU", {
  skip_if_no_gpu()
  yf <- data.frame(f = factor(c("x", "y"), levels = c("x", "y", "z", "w")))
  p <- function(d) group_by(d, f, .drop = FALSE) |>
    semi_join(f71_as(d, yf), by = "f") |>
    summarise(n = n(), s = sum(x))
  expect_same_as_dplyr(f71_big, p)
  expect_same_as_dplyr_lazy(f71_big, p)
  for (lazy in c(FALSE, TRUE)) {
    expect_false(f71_fell_back(collect(p(tbl_gpu(f71_big, lazy = lazy)))),
                 info = paste("lazy", lazy))
  }
  p2 <- function(d) group_by(d, f, .drop = FALSE) |>
    anti_join(f71_as(d, yf), by = "f") |>
    count()
  expect_same_as_dplyr(f71_big, p2)
  expect_false(f71_fell_back(collect(p2(tbl_gpu(f71_big)))))
})

test_that("eager: a filtering join or slice that empties no group keeps summarise() on the GPU", {
  skip_if_no_gpu()
  yall <- data.frame(g = c("a", "b", "c", "d"), stringsAsFactors = FALSE)
  pipes <- list(
    function(d) group_by(d, g, .drop = FALSE) |>
      semi_join(f71_as(d, yall), by = "g") |>
      summarise(n = n(), s = sum(x)),
    function(d) group_by(d, g, .drop = FALSE) |>
      anti_join(f71_as(d, data.frame(x = 1)), by = "x") |>
      summarise(n = n(), s = sum(x)),
    function(d) group_by(d, g, .drop = FALSE) |> slice_head(n = 1) |> count(),
    function(d) group_by(d, g, f, .drop = FALSE) |> slice_max(x, n = 1) |> tally()
  )
  for (p in pipes) {
    expect_same_as_dplyr(f71_big, p)
    expect_false(f71_fell_back(collect(p(tbl_gpu(f71_big)))))
  }
})

test_that("an emptied NA factor group survives (factor keys with NA need the keys)", {
  skip_if_no_gpu()
  dna <- data.frame(
    f = factor(c("x", NA, "y", "x"), levels = c("x", "y", "z")),
    x = c(1L, 2L, 3L, 4L)
  )
  yf <- data.frame(f = factor(c("x", "y"), levels = c("x", "y", "z")))
  f71_check_groups(function(d) {
    semi_join(group_by(d, f, .drop = FALSE), f71_as(d, yf), by = "f")
  }, df = dna)
  f71_check_groups(function(d) slice(group_by(d, f, .drop = FALSE), 2), df = dna)
  p <- function(d) group_by(d, f, .drop = FALSE) |>
    semi_join(f71_as(d, yf), by = "f") |>
    summarise(n = n(), s = sum(x))
  expect_same_as_dplyr(dna, p)
  expect_same_as_dplyr_lazy(dna, p)
})

test_that("slices after pending ops keep the emptied groups", {
  skip_if_no_gpu()
  f71_check_groups(function(d) {
    d |> mutate(x2 = x + 1L) |> group_by(g, .drop = FALSE) |> slice(2)
  })
  f71_check_groups(function(d) {
    d |> group_by(g, .drop = FALSE) |> filter(x > 1L) |> slice_head(n = -1)
  })
  f71_check_groups(function(d) {
    d |> group_by(g, .drop = FALSE) |> mutate(x2 = x * 2L) |> slice_min(x2, n = 0)
  })
})

test_that("head() and tail() on a .drop = FALSE table regroup like dplyr", {
  skip_if_no_gpu()
  f71_check_groups(function(d) head(group_by(d, g, .drop = FALSE), 2))
  f71_check_groups(function(d) tail(group_by(d, g, .drop = FALSE), 2))
})

test_that("set operations after a .drop = FALSE filter() regroup like dplyr", {
  skip_if_no_gpu()
  sub <- f71_df[c(1, 2, 7), ]
  for (op in list(dplyr::intersect, dplyr::setdiff, dplyr::symdiff)) {
    f71_check_groups(function(d) {
      group_by(d, g, .drop = FALSE) |> filter(x < 7L) |> op(f71_as(d, sub))
    })
  }
})

test_that("dplyr_row_slice() keeps .drop = FALSE groups and honours preserve =", {
  skip_if_no_gpu()
  f71_check_groups(function(d) dplyr::dplyr_row_slice(group_by(d, g, .drop = FALSE), 1:2),
                   lazies = FALSE)
  f71_check_groups(function(d) dplyr::dplyr_row_slice(group_by(d, g), 1:2, preserve = TRUE),
                   lazies = FALSE)
  f71_check_groups(function(d) dplyr::dplyr_row_slice(group_by(d, g), 1:2),
                   lazies = FALSE)
  f71_check_groups(function(d) dplyr::dplyr_row_slice(group_by(d, g, .drop = FALSE), c(1:3, 7L)),
                   lazies = FALSE)
})
