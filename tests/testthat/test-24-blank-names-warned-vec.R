# Issue #24, review round 2: blank-named factors through the other verbs, and the
# rule that lower_mutate() mutes only the vector constants somebody warned for.

test_that("a blank-named factor keeps its levels through distinct(), relocate() and `[`", {
  skip_if_no_gpu()
  a <- tibble::tibble(a = 3:1, f = factor(c("x", "y", "x"), levels = c("y", "x")))
  f2 <- factor(c("q", "p", "q"), levels = c("q", "p", "r"))
  want <- bind_cols(a, f2, .name_repair = "minimal")
  pipes <- list(
    distinct = function(t) distinct(t),
    relocate = function(t) relocate(t, f, .before = 1),
    bracket = function(t) t[2:3, ],
    bracket_cols = function(t) t[, c(3, 2, 1)]
  )
  for (lz in c(FALSE, TRUE)) {
    xb <- bind_cols(tbl_gpu(a, lazy = lz), f2, .name_repair = "minimal")
    for (nm in names(pipes)) {
      expect_identical(as.list(collect(pipes[[nm]](xb))), as.list(pipes[[nm]](want)),
                       info = paste(nm, "lazy =", lz))
    }
  }
})

test_that("an ir_vec() is marked warned only when its recorded loss is lossy", {
  v <- as.Date(c(1, Inf, 3))
  expect_true(ir_vec(v, "TIMESTAMP_DAYS", NULL, "y", loss = timestamp_vector_loss(v))$warned)
  expect_false(ir_vec(v, "TIMESTAMP_DAYS", NULL, "y")$warned)
  ok <- as.Date(1:3)
  expect_false(ir_vec(ok, "TIMESTAMP_DAYS", NULL, "y", loss = timestamp_vector_loss(ok))$warned)
})

test_that("a lazy mutate() with a lossy Date vector warns at the call only, also next to a clean one", {
  skip_if_no_gpu()
  d <- tibble::tibble(x = 1:3)
  count <- function(expr) {
    n <- 0L
    withCallingHandlers(expr, cuplyr_warning_timestamp_unrepresentable = function(w) {
      n <<- n + 1L
      invokeRestart("muffleWarning")
    })
    n
  }
  lz <- NULL
  expect_identical(count(lz <- mutate(tbl_gpu(d, lazy = TRUE), a = as.Date(1:3), b = as.Date(c(1, Inf, 3)))), 1L)
  expect_identical(count(collect(lz)), 0L)
})
