# Factor levels stay with their column by position once names repeat
# (round-3 review B5 and N1). `levels_by_position()` only trusts a list laid
# out by `levels_from_positions()`.

f40p_dd <- function() {
  tibble::tibble(x = factor("a"), x = factor("z", levels = c("y", "z")), y = 2.5,
                 .name_repair = "minimal")
}

test_that("names<- on duplicated names re-keys levels by position (N1a)", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(f40p_dd(), lazy = lazy)
    names(g) <- c("a", "b", "y")
    expected <- f40p_dd()
    names(expected) <- c("a", "b", "y")
    expect_identical(collect(g), expected, info = paste("lazy =", lazy))
    # and back to duplicated names, through pending lazy ops
    g2 <- relocate(g, b)
    names(g2) <- c("x", "x", "y")
    exp2 <- dplyr::relocate(expected, b)
    names(exp2) <- c("x", "x", "y")
    expect_identical(collect(g2), exp2, info = paste("lazy =", lazy))
  }
})

test_that("bind_cols() reads levels by position on a duplicated-name input (N1b)", {
  skip_if_no_gpu()
  d <- tibble::tibble(x = factor("a"), x = 2.5, .name_repair = "minimal")
  out <- suppressMessages(cuplyr::bind_cols(tbl_gpu(d), tbl_gpu(tibble::tibble(z = 1))))
  expected <- suppressMessages(dplyr::bind_cols(d, tibble::tibble(z = 1)))
  expect_identical(collect(out), expected)
  out_min <- cuplyr::bind_cols(tbl_gpu(d), tbl_gpu(d), .name_repair = "minimal")
  expect_identical(collect(out_min),
                   dplyr::bind_cols(d, d, .name_repair = "minimal"))
})

test_that("head()/tail() keep factor levels on duplicated names", {
  skip_if_no_gpu()
  d <- tibble::tibble(x = factor(c("a", "b")), x = factor(c("z", "y"), levels = c("y", "z")),
                      .name_repair = "minimal")
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(d, lazy = lazy)
    expect_identical(collect(head(g, 1)), d[1, ], info = paste("lazy =", lazy))
    expect_identical(collect(tail(g, 1)), d[2, ], info = paste("lazy =", lazy))
  }
})

test_that("an untracked factor_levels list on duplicated names fails closed", {
  schema <- list(names = c("x", "x"), types = c("DICTIONARY32", "DICTIONARY32"),
                 factor_levels = list(x = c("y", "z"), x = c("a", "b")))
  expect_error(levels_by_position(schema), class = "cuplyr_error_internal")
  tagged <- schema
  tagged$factor_levels <- levels_from_positions(c("x", "x"), list("a", "z"))
  expect_identical(levels_by_position(tagged), list("a", "z"))
  # a tagged list that has lost an entry is refused too
  tagged$factor_levels <- tagged$factor_levels[1]
  attr(tagged$factor_levels, "by_position") <- TRUE
  expect_error(levels_by_position(tagged), class = "cuplyr_error_internal")
  # unique names keep the by-name lookup, whatever the list order
  uniq <- list(names = c("f", "h"), types = c("DICTIONARY32", "DICTIONARY32"),
               factor_levels = list(h = "z", f = "a"))
  expect_identical(levels_by_position(uniq), list("a", "z"))
})
