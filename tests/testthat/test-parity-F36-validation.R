# F36 acceptance tests: join column validation.
#
# 1. Duplicate join columns. dplyr 1.2.1 R/join-cols.R::check_join_vars():
#    per side, `vars <- c(vars[equi], unique(vars[non_equi]))`; any duplicate
#    aborts (plain rlang_error, no dplyr class) with
#    "Join columns in `x` must be unique." + "x Problem with `a`.", x checked
#    before y. A column may repeat across non-equi conditions only
#    (dplyr::join_by(a >= lo, a <= hi) is fine). Raised when the verb is called, so a
#    lazy join errors before collect. Verified with tools/r
#    (scratchpad/parity/features/F36/probe/p1.out).
# 2. Temporary cast-key names (F06 N1). A non-equi Date x POSIXct condition is
#    matched on cast copies named `.cuplyr_key_x<k>`/`.cuplyr_key_y<k>`; a user
#    column with that exact name stood in for the copy and gave wrong rows
#    (probe/p3.out). User column names must never matter.

f36v_x <- data.frame(a = 1:3, b = c(1L, 2L, 2L))
f36v_y <- data.frame(a = c(1L, 2L, 4L), b = c(2L, 2L, 3L), c = 1:3)

# dplyr's error for `call_fn(<tibbles>)` vs cuplyr's for `call_fn(<tbl_gpu>)`:
# same class vector and same message; lazy errors at verb time.
f36v_expect_same_error <- function(call_fn) {
  want <- tryCatch(call_fn(tibble::as_tibble(f36v_x), tibble::as_tibble(f36v_y)),
                   error = identity)
  expect_s3_class(want, "rlang_error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch(
      call_fn(tbl_gpu(f36v_x, lazy = lazy), tbl_gpu(f36v_y, lazy = lazy)),
      error = identity
    )
    expect_s3_class(got, "error")
    if (inherits(got, "error")) {
      expect_identical(class(got), class(want))
      expect_identical(conditionMessage(got), conditionMessage(want))
    }
  }
}

test_that("by = c('a', 'a') errors: join columns in x must be unique", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, right_join, full_join, semi_join, anti_join)) {
    f36v_expect_same_error(function(x, y) join(x, y, by = c("a", "a")))
  }
})

test_that("dplyr::join_by() repeating an equi column errors", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, semi_join)) {
    f36v_expect_same_error(function(x, y) join(x, y, by = dplyr::join_by(a, a)))
    f36v_expect_same_error(function(x, y) join(x, y, by = dplyr::join_by(a, b, a)))
  }
})

test_that("a y column used twice errors on the y side", {
  skip_if_no_gpu()
  for (join in list(left_join, anti_join)) {
    f36v_expect_same_error(function(x, y) join(x, y, by = c(a = "a", b = "a")))
  }
})

test_that("a column in both an equi and a non-equi condition errors", {
  skip_if_no_gpu()
  for (join in list(left_join, inner_join, semi_join)) {
    f36v_expect_same_error(function(x, y) join(x, y, by = dplyr::join_by(a == a, a >= b)))
  }
})

test_that("a column repeated across non-equi conditions is allowed", {
  skip_if_no_gpu()
  y <- data.frame(lo = c(0L, 2L), hi = c(2L, 5L))
  y_like <- function(d) if (is_tbl_gpu(d)) tbl_gpu(y, lazy = identical(d$exec_mode, "lazy")) else tibble::as_tibble(y)
  for (join in list(inner_join, left_join, semi_join)) {
    pipe <- function(d) join(d, y_like(d), by = dplyr::join_by(a >= lo, a <= hi))
    expect_same_as_dplyr(f36v_x, pipe)
    expect_same_as_dplyr_lazy(f36v_x, pipe)
  }
})

# ---- temporary cast-key names -------------------------------------------------

f36v_dx <- data.frame(
  d = as.Date(c("2024-01-01", "2024-01-03", "2024-01-05")),
  .cuplyr_key_x1 = as.POSIXct(rep("2030-01-01", 3), tz = "UTC"),
  .cuplyr_key_y1 = as.POSIXct(rep("1990-01-01", 3), tz = "UTC"),
  check.names = FALSE
)
f36v_dy <- data.frame(
  t = as.POSIXct(c("2024-01-02", "2024-01-04"), tz = "UTC"),
  .cuplyr_key_y1 = as.Date(c("1990-01-01", "1990-01-01")),
  .cuplyr_key_x1 = as.Date(c("2099-01-01", "2099-01-01")),
  check.names = FALSE
)

f36v_dy_like <- function(d) {
  if (!is_tbl_gpu(d)) return(tibble::as_tibble(f36v_dy))
  tbl_gpu(f36v_dy, lazy = identical(d$exec_mode, "lazy"))
}

test_that("user columns named like the temporary cast keys don't change non-equi results", {
  skip_if_no_gpu()
  for (join in list(inner_join, left_join, right_join, full_join)) {
    pipe <- function(d) join(d, f36v_dy_like(d), by = dplyr::join_by(d >= t), relationship = "many-to-many")
    expect_same_as_dplyr(f36v_dx, pipe, arrange_by = c("d", "t"))
    expect_same_as_dplyr_lazy(f36v_dx, pipe, arrange_by = c("d", "t"))
  }
  for (join in list(semi_join, anti_join)) {
    pipe <- function(d) join(d, f36v_dy_like(d), by = dplyr::join_by(d >= t))
    expect_same_as_dplyr(f36v_dx, pipe)
    expect_same_as_dplyr_lazy(f36v_dx, pipe)
  }
})

test_that("cast-key name collisions: y-side cast, and lazy pushdown", {
  skip_if_no_gpu()
  # condition written y-first: the cast copy lands on the other side
  pipe <- function(d) inner_join(f36v_dy_like(d), d, by = dplyr::join_by(t <= d), relationship = "many-to-many")
  expect_same_as_dplyr(f36v_dx, pipe, arrange_by = c("t", "d"))
  expect_same_as_dplyr_lazy(f36v_dx, pipe, arrange_by = c("t", "d"))

  pipe2 <- function(d) {
    d |>
      inner_join(f36v_dy_like(d), by = dplyr::join_by(d >= t), relationship = "many-to-many") |>
      filter(d > as.Date("2024-01-02"))
  }
  expect_same_as_dplyr_lazy(f36v_dx, pipe2, arrange_by = c("d", "t"))
})

test_that("cast-key name collisions with keep = TRUE equi keys", {
  skip_if_no_gpu()
  x <- data.frame(k = 1:3, .cuplyr_key_x1 = c(7, 8, 9), check.names = FALSE)
  y <- data.frame(k = c(1, 2, 2.5), .cuplyr_key_y1 = c("p", "q", "r"), check.names = FALSE)
  y_like <- function(d) if (is_tbl_gpu(d)) tbl_gpu(y, lazy = identical(d$exec_mode, "lazy")) else tibble::as_tibble(y)
  pipe <- function(d) left_join(d, y_like(d), by = "k", keep = TRUE)
  expect_same_as_dplyr(x, pipe)
  expect_same_as_dplyr_lazy(x, pipe)
  pipe2 <- function(d) semi_join(d, y_like(d), by = "k")
  expect_same_as_dplyr(x, pipe2)
  expect_same_as_dplyr_lazy(x, pipe2)
})
