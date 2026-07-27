# Mirrored dplyr-oracle tests for transmute() semantics (Phase 3, task 4)
#
# Native implementation: mutate() node + ast_select down to the kept set
# (R/mutate.R::transmute.tbl_gpu(), sharing parse_mutate_dots() with
# mutate.tbl_gpu()). Semantics verified empirically against dplyr 1.2.1
# before implementation:
#
#   - transmute(df, a = x+1, y): keeps ONLY the columns .../... names or
#     creates, in literal DOT ORDER (whether a dot is a bare existing
#     column or a brand-new expression) -- unlike mutate(.keep = "none"),
#     whose order keeps existing columns in their ORIGINAL relative
#     position and appends new ones at the end (see test-dplyr-mutate.R's
#     ".keep" coverage for that contrast).
#   - Grouped: group columns NOT explicitly named in `...` are prepended at
#     the very front (in group_vars() order); a group column that IS named
#     in `...` takes its dot position instead of being prepended.
#   - transmute() is superseded (lifecycle) in dplyr but not deprecated --
#     no warning is expected from a normal call, including its most common
#     idiom, an unnamed bare-column dot (verified: dplyr emits no warning
#     there, so cuplyr's transmute() uses a silent auto-naming path
#     instead of mutate()'s warning one -- see
#     R/mutate.R::name_dots_silently()).

transmute_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    x = c(10, 20, 30, 40, 50),
    y = c(1, 2, 3, 4, 5),
    z = c(100, 200, 300, 400, 500)
  )
}

# =============================================================================
# Basic shape: expression-only, bare-column dots, mixed
# =============================================================================

test_that("transmute() with only computed expressions matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, a = x + 1, b = y * 2)
  expect_same_as_dplyr(df, pipeline)
})

test_that("transmute() with only computed expressions matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, a = x + 1, b = y * 2)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("transmute() with a bare column dot keeps that column matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, a = x + 1, y)
  expect_same_as_dplyr(df, pipeline)
})

test_that("transmute() with a bare column dot keeps that column matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, a = x + 1, y)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("transmute() column order follows dot order, not original position (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  # y is an original column (position 3) but appears AFTER the new column
  # `a` in the dots -- transmute()'s output order must follow the dots, not
  # y's original position (unlike mutate(.keep = "none")).
  pipeline <- function(d) dplyr::transmute(d, a = x + 1, y)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "y"))
})

test_that("transmute() replacing an existing column keeps only that name (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, x = x * 2)
  expect_same_as_dplyr(df, pipeline)
})

test_that("transmute() replacing an existing column keeps only that name (lazy)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, x = x * 2)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("transmute() with multiple bare column dots matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, z, y, w = x + y)
  expect_same_as_dplyr(df, pipeline)
})

test_that("transmute() with multiple bare column dots matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, z, y, w = x + y)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("transmute() cross-referencing an earlier dot's output matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, a = x + 1, b = a * 2)
  expect_same_as_dplyr(df, pipeline)
})

test_that("transmute() cross-referencing an earlier dot's output matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) dplyr::transmute(d, a = x + 1, b = a * 2)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# Grouped retention
# =============================================================================

test_that("transmute() on a grouped table prepends un-dotted group columns (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::transmute(a = x + 1)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("g", "a"))
})

test_that("transmute() on a grouped table prepends un-dotted group columns (lazy)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::transmute(a = x + 1)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("transmute() explicitly naming a group column uses its dot position (eager)", {
  skip_if_no_gpu()
  df <- transmute_df()
  # g is the group column, but appears LAST in the dots -- the group column
  # must NOT be prepended when it's explicitly named.
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::transmute(a = x + 1, b = y + 1, g = g * 2)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("a", "b", "g"))
})

test_that("transmute() explicitly naming a group column uses its dot position (lazy)", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::transmute(a = x + 1, b = y + 1, g = g * 2)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("transmute() on a table grouped by multiple columns preserves group_vars() order (eager)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g1 = c(1, 1, 2), g2 = c(9, 9, 8), x = c(1, 2, 3))
  pipeline <- function(d) d |> dplyr::group_by(g2, g1) |> dplyr::transmute(a = x + 1)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_identical(names(result$gpu), c("g2", "g1", "a"))
})

test_that("transmute() on a table grouped by multiple columns preserves group_vars() order (lazy)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g1 = c(1, 1, 2), g2 = c(9, 9, 8), x = c(1, 2, 3))
  pipeline <- function(d) d |> dplyr::group_by(g2, g1) |> dplyr::transmute(a = x + 1)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("transmute() on a grouped table preserves group_vars() itself", {
  skip_if_no_gpu()
  df <- transmute_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::transmute(a = x + 1)
  # expect_same_as_dplyr()/_lazy() already assert group_vars() equality
  # between the GPU and oracle results before collecting (helper-oracle.R).
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

# =============================================================================
# No fallback / no warning
# =============================================================================

test_that("transmute() is GPU-native and never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(transmute_df())

  expect_no_error(gt |> dplyr::transmute(a = x + 1, y))
  expect_no_error(gt |> dplyr::group_by(g) |> dplyr::transmute(a = x + 1))
})

test_that("transmute() with an unnamed bare column dot does not warn (matches dplyr)", {
  skip_if_no_gpu()
  gt <- tbl_gpu(transmute_df())

  expect_no_warning(gt |> dplyr::transmute(y))
})

test_that("transmute() with an unnamed computed expression does not warn (matches dplyr)", {
  skip_if_no_gpu()
  gt <- tbl_gpu(transmute_df())

  expect_no_warning(gt |> dplyr::transmute(x + 1))
})
