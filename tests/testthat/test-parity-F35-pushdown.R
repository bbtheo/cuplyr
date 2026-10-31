# F35 acceptance tests: lazy filter pushdown below a join must not change the
# result. Pushdown is unsafe when the join's `multiple=`, `unmatched=` or a
# constraining `relationship=` looks at rows the filter would remove, and a
# predicate on a suffixed column must still find its column after pushdown.
# Helpers are in helper-parity-F35.R.

px <- tibble::tibble(k = c(1L, 2L, 3L, 4L, NA), a = c(10, 20, 30, 40, 50),
                     v = c(1, 2, 3, 4, 5))
py <- tibble::tibble(k = c(1L, 1L, 2L, 5L, NA), b = c("p", "q", "r", "s", "t"),
                     v = c(11, 12, 13, 14, 15))

# ---- multiple = "first" / "last": a y-side filter changes which match wins --

test_that("inner_join(multiple = 'last') then a y-side filter", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> inner_join(y, by = "k", multiple = "last") |> filter(b != "q")
  })
})

test_that("inner_join(multiple = 'first') then a y-side filter", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> inner_join(y, by = "k", multiple = "first") |> filter(v.y > 11)
  })
})

test_that("right_join(multiple = 'first') then a y-side filter", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> right_join(y, by = "k", multiple = "first") |> filter(b != "p")
  })
})

test_that("left_join(multiple = 'last') with filters on both sides", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> left_join(y, by = "k", multiple = "last") |> filter(a > 5, b != "q")
  })
})

test_that("multiple = 'first' with an x-side filter stays correct", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> inner_join(y, by = "k", multiple = "first") |> filter(a > 15)
  })
})

# ---- unmatched = "error": pushdown can hide, create or reclassify the error -

test_that("left_join(unmatched = 'error'): x-side filter must not create an error", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:4, a = c(10, 20, 30, 40))
  y <- tibble::tibble(k = c(1L, 2L), w = c(1, 2))
  f35_expect_parity(list(x, y), function(x, y) {
    x |> left_join(y, by = "k", unmatched = "error") |> filter(a > 15)
  })
})

test_that("right_join(unmatched = 'error'): y-side filter must not create an error", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:2, a = c(10, 20))
  y <- tibble::tibble(k = 1:3, w = c(1, 2, 3))
  f35_expect_parity(list(x, y), function(x, y) {
    x |> right_join(y, by = "k", unmatched = "error") |> filter(w > 1)
  })
})

test_that("inner_join(unmatched = 'error'): y-side filter must not hide the error", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:2, a = c(10, 20))
  y <- tibble::tibble(k = c(1L, 2L, 7L), w = c(1, 2, 3))
  f35_expect_error_parity(list(x, y), function(x, y) {
    x |> inner_join(y, by = "k", unmatched = "error") |> filter(w < 3)
  }, class = "dplyr_error_join_matches_remaining", regexp = "Row 3 of `y`")
})

test_that("inner_join(unmatched = 'error'): x-side filter must not hide the error", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = c(1L, 2L, 9L), a = c(1, 2, 3))
  y <- tibble::tibble(k = 1:2, w = c(5, 6))
  f35_expect_error_parity(list(x, y), function(x, y) {
    x |> inner_join(y, by = "k", unmatched = "error") |> filter(a < 3)
  }, class = "dplyr_error_join_matches_nothing", regexp = "Row 3 of `x`")
})

test_that("inner_join(unmatched = 'error'): x-side check reported before y-side", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = c(1L, 2L, 3L, 4L), a = c(10, 20, 30, 40))
  y <- tibble::tibble(k = c(1L, 1L, 2L, 5L), w = c(1, 2, 3, 4))
  f35_expect_error_parity(list(x, y), function(x, y) {
    x |> inner_join(y, by = "k", unmatched = "error") |> filter(a < 25)
  }, class = "dplyr_error_join_matches_nothing", regexp = "Row 3 of `x`")
})

test_that("inner_join(unmatched = c('drop', 'error')) then an x-side filter", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = c(1L, 2L, 3L), a = c(10, 20, 30))
  y <- tibble::tibble(k = c(1L, 2L), w = c(1, 2))
  f35_expect_parity(list(x, y), function(x, y) {
    x |> inner_join(y, by = "k", unmatched = c("drop", "error")) |> filter(a > 15)
  })
})

# ---- relationship = constraining values: pushdown can hide the error --------

test_that("relationship = 'one-to-one': y-side filter must not hide the error", {
  skip_if_no_gpu()
  f35_expect_error_parity(list(px, py), function(x, y) {
    x |> inner_join(y, by = "k", relationship = "one-to-one") |> filter(b != "p")
  }, class = "dplyr_error_join_relationship_one_to_one", regexp = "Row 1 of `x`")
})

test_that("relationship = 'one-to-many': x-side filter must not hide the error", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = c(1L, 1L, 2L), a = c(1, 2, 3))
  y <- tibble::tibble(k = 1:2, w = c(5, 6))
  f35_expect_error_parity(list(x, y), function(x, y) {
    x |> inner_join(y, by = "k", relationship = "one-to-many") |> filter(a > 1)
  }, class = "dplyr_error_join_relationship_one_to_many", regexp = "Row 1 of `y`")
})

test_that("relationship = 'many-to-many' with filters on both sides", {
  skip_if_no_gpu()
  f35_expect_parity(list(py, py), function(x, y) {
    x |> inner_join(y, by = "k", relationship = "many-to-many") |>
      filter(b.x != "p", v.y > 11)
  })
})

# ---- predicates on suffixed columns -----------------------------------------

test_that("filter on suffixed columns from both sides after inner_join", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> inner_join(y, by = "k") |> filter(v.x > 1, v.y < 15)
  })
})

test_that("filter on a suffixed x column after left_join", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> left_join(y, by = "k") |> filter(v.x > 2)
  })
})

test_that("filter on a suffixed y column after right_join", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> right_join(y, by = "k") |> filter(v.y > 11)
  })
})

test_that("filter on keep = TRUE keys and custom suffixes", {
  skip_if_no_gpu()
  f35_expect_parity(list(px, py), function(x, y) {
    x |> inner_join(y, by = "k", keep = TRUE) |> filter(k.x > 1, k.y < 5)
  })
  f35_expect_parity(list(px, py), function(x, y) {
    x |> inner_join(y, by = "k", suffix = c("_l", "_r")) |> filter(v_l > 1, v_r > 12)
  })
})

test_that("filter on suffixed columns of a self-join", {
  skip_if_no_gpu()
  f35_expect_parity(list(py), function(y) {
    y |> inner_join(y, by = "k", relationship = "many-to-many") |>
      filter(v.x > 11, b.y != "q")
  })
})

# ---- pushdown stays narrow: the safe cases still push -----------------------

test_that("a join with a data-reading check runs at the verb, like dplyr", {
  skip_if_no_gpu()
  lx <- tbl_gpu(px, lazy = TRUE)
  ly <- tbl_gpu(py, lazy = TRUE)

  # data that fails the check: dplyr's error at the verb
  for (join in list(
    function() inner_join(lx, ly, by = "k", unmatched = "error"),
    function() left_join(lx, ly, by = "k", unmatched = "error"),
    function() inner_join(lx, ly, by = "k", relationship = "one-to-one")
  )) {
    expect_error(join(), class = "dplyr_error_join")
  }

  # data that passes it: the join has run (nothing pending), and a filter
  # after it matches dplyr
  x <- tibble::tibble(k = 1:3, a = c(10, 20, 30))
  y <- tibble::tibble(k = 1:3, b = c("p", "q", "r"))
  joined <- inner_join(tbl_gpu(x, lazy = TRUE), tbl_gpu(y, lazy = TRUE), by = "k",
                       unmatched = "error")
  expect_false(has_pending_ops(joined))
  expect_true(is_lazy(joined))
  expect_equal(
    tibble::as_tibble(collect(filter(joined, a > 15, b != "q"))),
    tibble::as_tibble(filter(inner_join(x, y, by = "k", unmatched = "error"), a > 15, b != "q"))
  )
})

test_that("pushdown is blocked only where it is unsafe", {
  skip_if_no_gpu()
  lx <- tbl_gpu(px, lazy = TRUE)
  ly <- tbl_gpu(py, lazy = TRUE)
  both <- function(j) j |> filter(a > 15, b != "q")

  # default inner join (relationship = NULL, multiple = "all"): it may raise
  # dplyr's many-to-many warning, which looks at the FULL inputs, so nothing
  # is pushed (F53 item 6; was "both sides pushed" before F53)
  p <- f35_plan(both(inner_join(lx, ly, by = "k")))
  expect_equal(p, list(above = TRUE, left = FALSE, right = FALSE))

  # multiple = "first"/"last"/"any": x side still pushed, y side stays
  for (m in c("first", "last", "any")) {
    p <- f35_plan(both(inner_join(lx, ly, by = "k", multiple = m)))
    expect_equal(p, list(above = TRUE, left = TRUE, right = FALSE), info = m)
  }

  # unmatched = "error" and a constraining relationship check the joined rows,
  # which dplyr raises at the verb: the verb computes such a join (F80), so a
  # filter after it is a plain filter on a table with nothing pending, and the
  # optimizer rule below only guards an AST that reaches it some other way.
  # (A node built with the check set by hand, off a many-to-many join:)
  checked <- function(field, value) {
    ops <- .subset2(lx |> inner_join(ly, by = "k", relationship = "many-to-many") |>
                      filter(a > 15, b != "q"), "lazy_ops")
    ops$input[[field]] <- value
    f35_plan(list(lazy_ops = ops))
  }
  expect_equal(checked("unmatched", c("error", "error")),
               list(above = TRUE, left = FALSE, right = FALSE))
  for (rel in c("one-to-one", "one-to-many", "many-to-one")) {
    expect_equal(checked("relationship", rel),
                 list(above = TRUE, left = FALSE, right = FALSE), info = rel)
  }
  # many-to-many: both pushed
  p <- f35_plan(both(inner_join(lx, ly, by = "k", relationship = "many-to-many")))
  expect_equal(p, list(above = FALSE, left = TRUE, right = TRUE))

  # left join default: may warn, so nothing pushed (F53 item 6)
  p <- f35_plan(lx |> left_join(ly, by = "k") |> filter(a > 15))
  expect_equal(p, list(above = TRUE, left = FALSE, right = FALSE))
  # left join that can't warn: x side pushed
  p <- f35_plan(lx |> left_join(ly, by = "k", relationship = "many-to-many") |>
                  filter(a > 15))
  expect_equal(p, list(above = FALSE, left = TRUE, right = FALSE))
})
