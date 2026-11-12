# F24 acceptance tests, tier C: degenerate `suffix =` values.
#
# dplyr 1.2.1 (R/join-cols.R add_suffixes(), R/join.R join_mutate()),
# verified with tools/r (scratchpad/parity/features/F24/probe/p2.R):
#   - with ONE empty suffix the other side is suffixed against the first
#     side's names until unique, so the output names never collide;
#   - only suffix = c("", "") collides: y's column overwrites the same-named
#     x column in place, and the output has fewer columns.
# The single-empty forms are native; c("", "") may use the CPU fallback but
# must match dplyr.

f24c_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f24c_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f24c_mutating <- list(
  left = dplyr::left_join, inner = dplyr::inner_join,
  right = dplyr::right_join, full = dplyr::full_join
)

test_that("one empty suffix never collides and runs natively", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  x <- data.frame(id = c(1, 2, 3), a = c(1, 2, 3), a.y = c(7, 8, 9))
  y <- data.frame(id = c(2, 3, 4), a = c(20, 30, 40), a.x = c(5, 6, 0))
  for (join in f24c_mutating) {
    for (sfx in list(c("", ".y"), c(".x", ""))) {
      f24c_both(x, function(d) join(d, f24c_like(d, y), by = "id", suffix = sfx))
      f24c_both(x, function(d) {
        join(d, f24c_like(d, y), by = "id", suffix = sfx, keep = TRUE)
      })
    }
  }
})

test_that("suffix = c('', '') overwrites x's same-named columns like dplyr", {
  skip_if_no_gpu()
  x <- data.frame(k = c(1, 2, 3), id = c(1L, 2L, 3L), z = c("p", "q", "r"), w = 1:3,
                  stringsAsFactors = FALSE)
  y <- data.frame(k = c(2, 3, 4), id = c(20L, 30L, 40L), z = c("Q", "R", "S"),
                  stringsAsFactors = FALSE)
  for (join in f24c_mutating) {
    f24c_both(x, function(d) join(d, f24c_like(d, y), by = "k", suffix = c("", "")))
    f24c_both(x, function(d) {
      join(d, f24c_like(d, y), by = "k", suffix = c("", ""), keep = TRUE)
    })
    f24c_both(x, function(d) {
      join(d, f24c_like(d, y), by = dplyr::join_by(k, closest(id >= id)), suffix = c("", ""))
    })
  }
})
