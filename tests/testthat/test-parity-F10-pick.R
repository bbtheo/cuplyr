# F10 acceptance tests: pick() indirection (gap A6h + Tier 3 "group_by()'s own
# dots don't expand pick()").
#
# Contract:
# - group_by()/arrange() with a top-level pick(<selection>) stay on the GPU
#   (checked with `cuplyr.fallback = "error"`): pick() reduces to "these
#   columns", resolved on the full (ungrouped) data in the quosure's env.
# - Every other pick() shape (inside user functions, column-free wrapper
#   calls, desc(pick()), named group_by(x = pick()), ...) may use the CPU
#   fallback, but must never hard-error or give a different result where
#   dplyr succeeds, and errors must be dplyr's own.

f10_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    a = c(3L, 1L, NA, 2L, 1L),
    b = c("x", "y", "x", NA, "z"),
    c = c(TRUE, NA, FALSE, TRUE, FALSE),
    f = factor(c("u", "v", "u", "w", "v"))
  )
}

f10_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# Same pipeline, GPU forced native (any CPU fallback is an error).
f10_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f10_both(df, pipeline, ...)
}

# dplyr and cuplyr (eager and lazy) both error, with the same message.
f10_same_error <- function(df, pipeline) {
  msg <- function(x) tryCatch({
    out <- pipeline(x)
    if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
    NA_character_
  }, error = function(e) conditionMessage(e))
  cpu <- msg(df)
  expect_false(is.na(cpu), info = "dplyr itself should error here")
  expect_identical(msg(tbl_gpu(df)), cpu)
  expect_identical(msg(tbl_gpu(df, lazy = TRUE)), cpu)
}

# pick() wrapped in user functions: dplyr can't see these at expansion time
# and evaluates them inside the data mask.
f10_pw <- function(...) dplyr::pick(...)
f10_my_pick <- function() dplyr::pick(a, c)
f10_pick_of <- function(x) dplyr::pick(tidyselect::all_of(x))
f10_where <- function(fn) dplyr::pick(tidyselect::where(fn))
f10_my_n <- function() dplyr::n()

# =============================================================================
# group_by(): native pick()
# =============================================================================

test_that("group_by(pick()) groups by the picked columns on the GPU", {
  skip_if_no_gpu()
  f10_native(f10_df(), function(d) dplyr::group_by(d, dplyr::pick(a, b)))
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(g, f)))
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(c(b, a))))
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(a), g))
  f10_native(f10_df(), function(d) dplyr::group_by(d, a, pick(a, b)))
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(tidyselect::where(is.numeric))))
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(tidyselect::starts_with("b"))))
  vars <- c("f", "g")
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(tidyselect::all_of(vars))))
})

test_that("group_by(pick()) result groups correctly downstream", {
  skip_if_no_gpu()
  f10_native(f10_df(), function(d) {
    d |> dplyr::group_by(pick(g, b)) |> dplyr::summarise(s = sum(a, na.rm = TRUE), n = dplyr::n())
  }, arrange_by = c("g", "b"))
  # grouped max(na.rm = TRUE) in mutate() falls back for reasons unrelated to
  # pick() (same with group_by(g)), so only the result is checked here
  f10_both(f10_df(), function(d) {
    d |> dplyr::group_by(pick(g)) |> dplyr::mutate(m = max(a, na.rm = TRUE))
  })
})

test_that("group_by(pick()) on grouped data selects on the ungrouped data", {
  skip_if_no_gpu()
  f10_native(f10_df(), function(d) dplyr::group_by(dplyr::group_by(d, g), pick(tidyselect::everything())))
  f10_native(f10_df(), function(d) {
    dplyr::group_by(dplyr::group_by(d, g), pick(a, b), .add = TRUE)
  })
  f10_native(f10_df(), function(d) {
    dplyr::group_by(dplyr::group_by(d, g), pick(tidyselect::any_of(c("g", "b"))))
  })
})

test_that("group_by(pick(NULL)) groups by nothing", {
  skip_if_no_gpu()
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(NULL)))
  f10_native(f10_df(), function(d) dplyr::group_by(dplyr::group_by(d, g), pick(NULL)))
  f10_native(f10_df(), function(d) dplyr::group_by(dplyr::group_by(d, g), pick(NULL), .add = TRUE))
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(tidyselect::starts_with("zzz")), b))
})

test_that("group_by(pick()) inside a wrapper with {{ }}", {
  skip_if_no_gpu()
  by_pick <- function(data, groups) dplyr::group_by(data, pick({{ groups }}))
  f10_native(f10_df(), function(d) by_pick(d, c(a, g)))
  f10_native(f10_df(), function(d) by_pick(d, NULL))
  f10_native(f10_df(), function(d) by_pick(d, tidyselect::where(is.character)))
})

test_that("group_by(pick()) keeps .drop, mixes with named dots, handles 0 rows", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gd <- dplyr::group_by(tbl_gpu(f10_df()), pick(f), .drop = FALSE)
  expect_s3_class(gd, "tbl_gpu")
  expect_identical(dplyr::group_vars(gd), "f")
  expect_false(dplyr::group_by_drop_default(gd))

  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(a), a2 = a * 2L))
  f10_native(f10_df(), function(d) dplyr::group_by(d, a2 = a * 2L, pick(b)))
  f10_native(f10_df()[0, ], function(d) dplyr::group_by(d, pick(a, b)))
})

# =============================================================================
# group_by(): shapes that may fall back, but must match dplyr
# =============================================================================

test_that("group_by() with pick() wrapped in a user function", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::group_by(d, f10_pw(a, b)))
  f10_both(f10_df(), function(d) dplyr::group_by(dplyr::group_by(d, g), f10_pw(b), .add = TRUE))
  by_pw <- function(data, groups) dplyr::group_by(data, f10_pw({{ groups }}))
  f10_both(f10_df(), function(d) by_pw(d, c(a, g)))
  f10_both(f10_df(), function(d) by_pw(d, NULL))
})

test_that("group_by() with a named pick() makes a df-column like dplyr", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::group_by(d, x = pick(a, b)))
})

test_that("group_by() pick() errors are dplyr's own", {
  skip_if_no_gpu()
  f10_same_error(f10_df(), function(d) dplyr::group_by(d, pick(zz)))
  f10_same_error(f10_df(), function(d) dplyr::group_by(d, pick(a, zz)))
  f10_same_error(f10_df(), function(d) dplyr::group_by(d, pick(q = a)))
  f10_same_error(f10_df(), function(d) dplyr::group_by(d, pick()))
})

# =============================================================================
# arrange(): native pick()
# =============================================================================

test_that("arrange(pick()) sorts by the picked columns on the GPU", {
  skip_if_no_gpu()
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(g, a)))
  f10_native(f10_df(), function(d) dplyr::arrange(d, dplyr::pick(b, a)))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(c), a))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(tidyselect::everything())))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(f), dplyr::desc(a)))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(b), pick(g, a)))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(tidyselect::where(is.character)), -a))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(NULL)))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(NULL), dplyr::desc(g)))
  f10_native(f10_df()[0, ], function(d) dplyr::arrange(d, pick(a, b)))
})

test_that("arrange(pick()) on grouped data and in lazy chains", {
  skip_if_no_gpu()
  f10_native(f10_df(), function(d) dplyr::arrange(dplyr::group_by(d, g), pick(tidyselect::everything())))
  f10_native(f10_df(), function(d) dplyr::arrange(dplyr::group_by(d, b), pick(a), .by_group = TRUE))
  f10_native(f10_df(), function(d) {
    d |> dplyr::filter(g < 3) |> dplyr::arrange(pick(b, a)) |> dplyr::mutate(h = a + 1L)
  })
  sort_by <- function(data, cols) dplyr::arrange(data, pick({{ cols }}))
  f10_native(f10_df(), function(d) sort_by(d, c(c, a)))
})

# =============================================================================
# arrange(): shapes that may fall back, but must match dplyr
# =============================================================================

test_that("arrange() with wrapped, transformed or constant pick() dots", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::arrange(d, f10_pw(g, a)))
  f10_both(f10_df(), function(d) dplyr::arrange(d, f10_pw(b), a))
  f10_both(f10_df(), function(d) dplyr::arrange(d, dplyr::desc(pick(g, a))))
  f10_both(f10_df(), function(d) dplyr::arrange(d, TRUE, pick(tidyselect::everything())))
  f10_both(f10_df(), function(d) dplyr::arrange(d, NULL, pick(b, a)))
  f10_both(f10_df(), function(d) dplyr::arrange(d, dplyr::across(c(g, a), .fns = identity)))
  f10_both(f10_df(), function(d) dplyr::arrange(d, dplyr::across(c(b, a), .fns = dplyr::desc)))
  f10_both(f10_df(), function(d) dplyr::arrange(dplyr::group_by(d, g), f10_pw(a), .by_group = TRUE))
})

test_that("arrange() pick() errors are dplyr's own", {
  skip_if_no_gpu()
  f10_same_error(f10_df(), function(d) dplyr::arrange(d, pick(zz)))
  f10_same_error(f10_df(), function(d) dplyr::arrange(d, f10_undefined_fn(pick(a))))
  f10_same_error(f10_df(), function(d) dplyr::arrange(d, pick(q = a)))
})

# =============================================================================
# mutate()/transmute()/summarise(): pick() inside user functions
# =============================================================================

test_that("mutate() with pick() inside a column-free user-function call", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::mutate(d, n = ncol(f10_my_pick())))
  f10_both(f10_df(), function(d) dplyr::mutate(d, p = f10_my_pick()))
  f10_both(f10_df(), function(d) dplyr::mutate(d, x = f10_pick_of("a"), y = f10_pick_of("b")))
  f10_both(f10_df(), function(d) dplyr::mutate(d, x = f10_where(is.numeric), y = f10_where(is.character)))
  f10_both(f10_df(), function(d) dplyr::mutate(d, n = nrow(f10_pw(tidyselect::everything()))))
  f10_both(f10_df()[0, ], function(d) dplyr::mutate(d, n = ncol(f10_my_pick())))
})

test_that("grouped mutate() evaluates wrapped pick() per group, like dplyr", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), n = nrow(f10_pw(tidyselect::everything())))
  })
  f10_both(f10_df(), function(d) {
    dplyr::mutate(dplyr::group_by(d, g), k = ncol(f10_pw(tidyselect::starts_with("zzz"))))
  })
  f10_both(f10_df(), function(d) dplyr::mutate(dplyr::group_by(d, g), k = f10_my_n()))
})

test_that("transmute() and summarise() with pick() inside user functions", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::transmute(d, n = ncol(f10_my_pick())))
  f10_both(f10_df(), function(d) dplyr::summarise(d, k = ncol(f10_pw(tidyselect::everything()))))
  f10_both(f10_df(), function(d) {
    dplyr::summarise(dplyr::group_by(d, g), n = nrow(f10_my_pick()), k = f10_my_n())
  }, arrange_by = "g")
})

test_that("mutate()/summarise() wrapped-pick() errors are dplyr's own", {
  skip_if_no_gpu()
  f10_same_error(f10_df(), function(d) dplyr::mutate(d, y = f10_pw()))
  # `a` is a grouping column: pick() can't select it
  f10_same_error(f10_df(), function(d) dplyr::mutate(dplyr::group_by(d, a), y = f10_my_pick()))
  f10_same_error(f10_df(), function(d) dplyr::summarise(d, y = nrow(f10_pick_of("zz"))))
  # a plain column-free error is reported the way dplyr reports it
  f10_same_error(f10_df(), function(d) dplyr::mutate(d, y = f10_undefined_fn()))
})

# =============================================================================
# filter(): pick() literal, wrapped, and column-free
# =============================================================================

test_that("filter() with pick() (literal, wrapped, column-free wrapper)", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::filter(d, vctrs::vec_detect_complete(pick(a, c))))
  f10_both(f10_df(), function(d) dplyr::filter(d, vctrs::vec_detect_complete(f10_pw(a, c))))
  f10_both(f10_df(), function(d) dplyr::filter(d, vctrs::vec_detect_complete(f10_my_pick())))
  f10_both(f10_df(), function(d) {
    dplyr::filter(dplyr::group_by(d, g), vctrs::vec_detect_complete(f10_my_pick()))
  })
  f10_both(f10_df(), function(d) dplyr::filter(dplyr::group_by(d, g), f10_my_n() > 1L))
  f10_both(f10_df(), function(d) dplyr::filter(d, g > 1, vctrs::vec_detect_complete(f10_my_pick())))
})

test_that("filter() pick() errors are dplyr's own", {
  skip_if_no_gpu()
  f10_same_error(f10_df(), function(d) dplyr::filter(d, pick(a, zz)))
  f10_same_error(f10_df(), function(d) dplyr::filter(d, f10_pw(a, zz)))
  f10_same_error(f10_df(), function(d) dplyr::filter(d, pick(a, c)))
  f10_same_error(f10_df(), function(d) dplyr::filter(d, f10_my_pick()))
})

# =============================================================================
# reframe(): stays on the CPU path, must work with pick()
# =============================================================================

test_that("reframe() with pick() and wrapped pick()", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::reframe(dplyr::group_by(d, g), pick(a)))
  f10_both(f10_df(), function(d) dplyr::reframe(dplyr::group_by(d, g), f10_my_pick()))
  f10_both(f10_df(), function(d) dplyr::reframe(d, f10_pw(a, b)))
  f10_both(f10_df(), function(d) dplyr::reframe(d, n = nrow(pick(tidyselect::everything()))))
})

# =============================================================================
# Round 1 review additions
# =============================================================================

f10_strs <- function() {
  tibble::tibble(x = c("b", "A", "a", "B", "ø", "o", "z"), n = 1:7)
}

test_that("arrange() honours .locale (native only for the C locale)", {
  skip_if_no_gpu()
  f10_native(f10_strs(), function(d) dplyr::arrange(d, x, .locale = "C"))
  f10_native(f10_strs(), function(d) dplyr::arrange(d, pick(x), .locale = "C"))
  f10_both(f10_strs(), function(d) dplyr::arrange(d, x, .locale = "en"))
  f10_both(f10_strs(), function(d) dplyr::arrange(d, x, .locale = "da"))
  f10_both(f10_strs(), function(d) dplyr::arrange(d, pick(x), .locale = "en"))
  f10_both(f10_strs(), function(d) dplyr::arrange(d, dplyr::desc(x), .locale = "en"))
  f10_same_error(f10_strs(), function(d) dplyr::arrange(d, x, .locale = 1))
  f10_same_error(f10_strs(), function(d) dplyr::arrange(d, x, .locale = "x"))
})

test_that("pick() with a value-dependent where() predicate matches dplyr", {
  skip_if_no_gpu()
  # g and f have no NA; a, b and c do
  f10_both(f10_df(), function(d) dplyr::group_by(d, pick(tidyselect::where(~ !anyNA(.x)))))
  f10_both(f10_df(), function(d) dplyr::arrange(d, pick(tidyselect::where(~ !anyNA(.x))), a))
  big <- function(x) is.numeric(x) && max(x, na.rm = TRUE) > 2
  f10_both(f10_df(), function(d) dplyr::group_by(d, pick(tidyselect::where(big))))
  f10_both(f10_df(), function(d) dplyr::arrange(d, pick(tidyselect::where(big)), b))
})

test_that("named group_by() dots on grouped input are computed on ungrouped data", {
  skip_if_no_gpu()
  gd <- function(d) dplyr::group_by(d, g)
  f10_both(f10_df(), function(d) dplyr::group_by(gd(d), k = dplyr::n()))
  f10_both(f10_df(), function(d) dplyr::group_by(gd(d), k = dplyr::n(), .add = TRUE))
  f10_both(f10_df(), function(d) dplyr::group_by(gd(d), k = f10_my_n()))
  f10_both(f10_df(), function(d) dplyr::group_by(gd(d), k = nrow(f10_pw(tidyselect::everything()))))
  f10_both(f10_df(), function(d) dplyr::group_by(gd(d), m = mean(a, na.rm = TRUE), pick(b)))
})

test_that("group_by(pick()) sees the columns as they are at its position", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) {
    dplyr::group_by(d, pick(tidyselect::where(is.character)), a = as.character(a))
  })
  f10_both(f10_df(), function(d) {
    dplyr::group_by(d, b = nchar(b), pick(tidyselect::where(is.character)))
  })
})

test_that("arrange(.by_group = TRUE) with an empty pick() sorts by the groups", {
  skip_if_no_gpu()
  f10_both(f10_df(), function(d) dplyr::arrange(dplyr::group_by(d, b), pick(NULL), .by_group = TRUE))
  f10_both(f10_df(), function(d) dplyr::arrange(dplyr::group_by(d, g), pick(NULL), .by_group = FALSE))
})

# =============================================================================
# Round 2 review additions
# =============================================================================

test_that("pick() with a bare or injected predicate matches dplyr", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  is_complete <- function(x) !anyNA(x)
  p <- tidyselect::where(~ !anyNA(.x))
  f10_both(f10_df(), function(d) dplyr::arrange(d, pick(is_complete), a))
  f10_both(f10_df(), function(d) dplyr::group_by(d, pick(is_complete)))
  f10_both(f10_df(), function(d) dplyr::arrange(d, pick(!!p), a))
  f10_both(f10_df(), function(d) dplyr::group_by(d, pick(!!p)))
  # a bare column name stays native, even when it shadows a function
  f10_native(f10_df(), function(d) dplyr::group_by(d, pick(c, f)))
  f10_native(f10_df(), function(d) dplyr::arrange(d, pick(c), pick(f, a)))
})

test_that("lazy .by_group arrange followed by a join that drops the group column", {
  skip_if_no_gpu()
  x <- tibble::tibble(g = c(2, 1, 2, 1), k = 1:4, v = c(10, 20, 30, 40))
  y <- tibble::tibble(k = 1:4, w = c("a", "b", "c", "d"))
  pipes <- list(
    function(x, y) {
      x |> dplyr::group_by(g) |> dplyr::arrange(.by_group = TRUE) |> dplyr::ungroup() |>
        dplyr::left_join(y, by = "k") |> dplyr::select(k, v, w)
    },
    function(x, y) {
      x |> dplyr::group_by(g) |> dplyr::arrange(dplyr::desc(v), .by_group = TRUE) |> dplyr::ungroup() |>
        dplyr::left_join(y, by = "k") |> dplyr::select(k, v, w)
    },
    function(x, y) {
      x |> dplyr::group_by(g) |> dplyr::arrange(pick(v), .by_group = TRUE) |> dplyr::ungroup() |>
        dplyr::left_join(y, by = "k") |> dplyr::select(k, v, w)
    }
  )
  for (f in pipes) {
    expected <- f(x, y)
    eager <- dplyr::collect(f(tbl_gpu(x, lazy = FALSE), tbl_gpu(y, lazy = FALSE)))
    lazy <- dplyr::collect(f(tbl_gpu(x, lazy = TRUE), tbl_gpu(y, lazy = TRUE)))
    expect_equal(tibble::as_tibble(eager), expected)
    expect_equal(tibble::as_tibble(lazy), expected)
  }
})
