# Smoke tests for the mirrored oracle helpers (helper-oracle.R)
#
# These exercise `expect_same_as_dplyr()` / `expect_same_as_dplyr_lazy()` on
# the currently-supported dplyr surface only, so this file must pass today.
# It is not itself a parity suite -- it just proves the harness works before
# later phases add one `test-dplyr-<file>.R` per dplyr test file.

test_that("filter() + select() pipeline matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(1, 5, 3, 8, 2, 9, 4),
    y = c(10, 20, 30, 40, 50, 60, 70),
    z = c("a", "b", "c", "d", "e", "f", "g"),
    stringsAsFactors = FALSE
  )

  pipeline <- function(tbl) {
    tbl |>
      dplyr::filter(x > 2) |>
      dplyr::select(x, y)
  }

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() pipeline matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(
    a = c(1, 2, 3, 4, 5),
    b = c(10, 20, 30, 40, 50)
  )

  pipeline <- function(tbl) {
    tbl |>
      dplyr::mutate(c = a + b, d = a * 2)
  }

  # mutate() currently tags arithmetic results as FLOAT64 regardless of
  # input types (scratchpad/todo.md Phase 0: "Fix mutate output-type
  # inference divergence"), so column types are compared loosely here.
  expect_same_as_dplyr(df, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(df, pipeline, ignore_col_types = TRUE)
})

test_that("arrange() pipeline matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(3, 1, 4, 1, 5, 9, 2, 6),
    y = c("i", "h", "g", "f", "e", "d", "c", "b")
  )

  pipeline <- function(tbl) {
    tbl |>
      dplyr::arrange(x, y)
  }

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("group_by() + summarise() pipeline matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(
    grp = c("a", "a", "b", "b", "b", "c"),
    val = c(1, 2, 3, 4, 5, 6)
  )

  # cuplyr's summarise() always drops all grouping, unlike dplyr's default
  # `.groups = "drop_last"` (see scratchpad/todo.md roadmap Phase 6, which
  # will bring cuplyr's grouping semantics fully in line). Passing
  # `.groups = "drop"` on the oracle side makes the two sides comparable
  # until that phase lands.
  #
  # `n` is bound locally so the bare `n()` call resolves for the dplyr
  # oracle even though this package doesn't attach dplyr (tests only use
  # `dplyr::` qualified calls elsewhere); cuplyr's own aggregation parser
  # matches on the deparsed `n()` text and never evaluates this binding, so
  # it's inert on the GPU side.
  pipeline <- function(tbl) {
    n <- dplyr::n
    tbl |>
      dplyr::group_by(grp) |>
      dplyr::summarise(total = sum(val), n = n(), .groups = "drop")
  }

  # Group-by aggregation order is not guaranteed, so arrange before compare.
  expect_same_as_dplyr(df, pipeline, arrange_by = "grp")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "grp")
})

test_that("join pipeline matches dplyr", {
  skip_if_no_gpu()

  left_df <- data.frame(
    id = c(1, 2, 3, 4),
    x = c(10, 20, 30, 40)
  )
  right_df <- data.frame(
    id = c(2, 3, 4, 5),
    y = c(200, 300, 400, 500)
  )

  # The pipeline receives only the left table; the right table is joined
  # from the closure, converted to the matching representation (tibble for
  # the CPU oracle, tbl_gpu for the GPU side).
  pipeline <- function(left) {
    right <- if (is_tbl_gpu(left)) {
      tbl_gpu(right_df, lazy = is_lazy(left))
    } else {
      dplyr::as_tibble(right_df)
    }
    dplyr::left_join(left, right, by = "id")
  }

  # cuDF join output order is not guaranteed to match dplyr's, so arrange.
  expect_same_as_dplyr(left_df, pipeline, arrange_by = "id")
  expect_same_as_dplyr_lazy(left_df, pipeline, arrange_by = "id")
})
