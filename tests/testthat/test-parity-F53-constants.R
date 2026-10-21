# F53 items 3-5 acceptance tests.
#
# 3. An aggregate whose argument references no column (`sum(1L)`,
#    `mean(c(2, 4))`, `max(th)` with `th` an R variable) is evaluated like
#    dplyr: once per group, the same value in every group. Never "Column '1L'
#    not found".
# 4. A literal-only sub-expression (`k == ""` with `k` an R variable) is a
#    scalar; it must be constant-folded before it reaches the GPU.
# 5. `min_rank({{ v }})` with `v` unsupplied raises dplyr's error.

f53c_df <- function() {
  tibble::tibble(g = c(1, 1, 2, 3, 3), x = c(1L, 2L, 3L, NA, 5L),
                 y = c(0.5, 1.5, -2, 4, NA), s = c("a", "", "b", "", "c"))
}

f53c_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# ---- item 3: constant aggregate arguments ----------------------------------------

test_that("constant aggregate arguments, ungrouped, match dplyr", {
  skip_if_no_gpu()
  th <- 5
  d <- f53c_df()
  f53c_both(d, function(t) summarise(t, a = sum(1L)))
  f53c_both(d, function(t) summarise(t, a = sum(1L), b = sum(1), c = mean(c(2, 4)),
                                     m = max(th), mn = min(th + 1), nd = dplyr::n_distinct(1)))
  f53c_both(d, function(t) summarise(t, a = sum(1L) + sum(x, na.rm = TRUE)))
  f53c_both(d, function(t) summarise(t, a = sum(NA), b = mean(NA), c = max(NA_real_),
                                     e = sum(NA, na.rm = TRUE)))
  f53c_both(d, function(t) summarise(t, an = any(TRUE), al = all(FALSE),
                                     md = median(3), fi = dplyr::first(7L), la = dplyr::last(c(1L, 2L))))
  f53c_both(d, function(t) summarise(t, m = max(.env$th), n = n()))
})

test_that("constant aggregate arguments, grouped, broadcast to every group", {
  skip_if_no_gpu()
  th <- 5
  d <- f53c_df()
  f53c_both(d, function(t) summarise(group_by(t, g), a = sum(1L)))
  f53c_both(d, function(t) summarise(group_by(t, g), a = sum(1L), b = sum(1),
                                     c = mean(c(2, 4)), m = max(th), mn = min(th + 1),
                                     nd = dplyr::n_distinct(1)))
  f53c_both(d, function(t) summarise(group_by(t, g), a = sum(1L) + sum(x, na.rm = TRUE),
                                     n = n()))
  f53c_both(d, function(t) summarise(group_by(t, g), v = var(c(1, 2)), s = sd(1),
                                     a = mean(NA)))
  f53c_both(d, function(t) summarise(t, a = sum(1L), .by = g))
  f53c_both(d, function(t) summarise(group_by(t, g, s), m = max(th), .groups = "drop"))
})

test_that("constant aggregate arguments on empty inputs match dplyr", {
  skip_if_no_gpu()
  th <- 5
  d0 <- f53c_df()[0, ]
  f53c_both(d0, function(t) summarise(t, a = sum(1L), c = mean(c(2, 4)), m = max(th)))
  f53c_both(d0, function(t) summarise(group_by(t, g), a = sum(1L), m = max(th)))
})

test_that("a column name shadows an R variable inside a constant-looking aggregate", {
  skip_if_no_gpu()
  x <- 100L
  d <- f53c_df()
  # `x` is the column (data mask first), not the R value
  f53c_both(d, function(t) summarise(group_by(t, g), m = max(x, na.rm = TRUE)))
  f53c_both(d, function(t) summarise(t, m = sum(.env$x), n = sum(x, na.rm = TRUE)))
})

# ---- item 4: literal-only sub-expressions -----------------------------------------

test_that("a literal-only string comparison inside a column expression is folded", {
  skip_if_no_gpu()
  k <- ""
  d <- f53c_df()
  f53c_both(d, function(t) mutate(t, n = k == "" & x > 0))
  f53c_both(d, function(t) mutate(t, n = k != "" | y > 0))
  f53c_both(d, function(t) filter(t, k == "" & x > 1))
  f53c_both(d, function(t) filter(t, k == "a" | s == ""))
  f53c_both(d, function(t) mutate(t, i = dplyr::if_else(k == "" & x > 1L, x, 0L)))
  f53c_both(d, function(t) mutate(t, w = dplyr::case_when(k != "" ~ 1L, x > 2L ~ 2L, .default = x)))
  f53c_both(d, function(t) mutate(t, z = (k < "b") & (s == k)))
  f53c_both(d, function(t) summarise(t, n = sum(k == ""), m = sum(x[k == ""], na.rm = TRUE)))
  f53c_both(d, function(t) summarise(group_by(t, g), n = sum(x > 1L & k == "", na.rm = TRUE)))
})

test_that("other literal-only sub-expressions fold with R's types", {
  skip_if_no_gpu()
  k <- "abc"
  n0 <- 2L
  d <- f53c_df()
  f53c_both(d, function(t) mutate(t, a = x + (n0 * 3L), b = y * (1 / 4), c = x > nchar(k)))
  f53c_both(d, function(t) mutate(t, a = x + (5L %/% n0), b = is.na(NA_character_) & x > 1L))
  f53c_both(d, function(t) mutate(t, p = paste0(s, toupper(k)), q = s == paste0(k, "")))
  f53c_both(d, function(t) filter(t, x > n0 + 1L, !(k %in% c("zz"))))
  f53c_both(d, function(t) mutate(t, na = (NA_character_ == k) | x > 2L))
})

# ---- item 5: min_rank({{ v }}) with v unsupplied -----------------------------------

test_that("an unsupplied {{ v }} inside a rank function raises dplyr's error", {
  skip_if_no_gpu()
  d <- f53c_df()
  # bare names, as a user with dplyr attached writes them
  min_rank <- dplyr::min_rank
  desc <- dplyr::desc
  row_number <- dplyr::row_number
  ranker <- function(df, v) mutate(df, r = min_rank({{ v }}))
  ranker_desc <- function(df, v) mutate(df, r = min_rank(desc({{ v }})))
  rn <- function(df, v) mutate(df, r = row_number({{ v }}))
  for (f in list(ranker, ranker_desc, rn)) {
    want <- tryCatch(f(d), error = identity)
    expect_s3_class(want, "dplyr:::mutate_error")
    for (lazy in c(FALSE, TRUE)) {
      got <- tryCatch(collect(f(tbl_gpu(d, lazy = lazy))), error = identity)
      expect_s3_class(got, "dplyr:::mutate_error")
      expect_identical(conditionMessage(got), conditionMessage(want))
      expect_s3_class(got$parent, class(want$parent)[1])
      expect_identical(conditionMessage(got$parent), conditionMessage(want$parent))
    }
  }
})
