# F111 round 2: the recorded reason of a NULL operand, and the shapes the
# round-1 review turned up that the acceptance files do not pin.

for (.f111r in c("if_any", "if_all", "if_else", "coalesce", "ntile")) {
  if (!exists(.f111r, mode = "function")) assign(.f111r, getExportedValue("dplyr", .f111r))
}
rm(.f111r)

f111r_df <- function() {
  tibble::tibble(x = c(1, 2, 3, NA, 5), y2 = c(5, 4, 3, 2, 1), g = c("a", "a", "b", "b", "b"))
}

f111r_reasons <- function(expr) {
  got <- list()
  withCallingHandlers(
    expr,
    cuplyr_cpu_fallback = function(c) got[[length(got) + 1L]] <<- c,
    warning = function(w) invokeRestart("muffleWarning")
  )
  vapply(got, function(c) c$reason, "")
}

test_that("F111: a literal NULL operand is not blamed on a variable", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  nul <- NULL
  cases <- list(
    function(t) filter(t, x %in% NULL),
    function(t) mutate(t, z = coalesce(x, NULL)),
    function(t) filter(t, x %in% nul)
  )
  for (f in cases) {
    for (lazy in c(FALSE, TRUE)) {
      why <- f111r_reasons(collect(f(tbl_gpu(f111r_df(), lazy = lazy))))
      expect_length(why, 1L)
      expect_match(why, "NULL")
      expect_no_match(why, "variable")
    }
  }
})

test_that("F111: a NULL predicate of a two-column if_any()/if_all() is dplyr's mutate error", {
  skip_if_no_gpu()
  cfg <- list(thr = 2)
  for (lazy in c(FALSE, TRUE)) {
    for (p in list(
      function(t) mutate(t, z = if_any(c(x, y2), ~ cfg$trh)),
      function(t) mutate(t, z = if_all(c(x, y2), ~ cfg$trh))
    )) {
      expect_error(collect(p(tbl_gpu(f111r_df(), lazy = lazy))), class = "dplyr:::mutate_error")
    }
  }
})

test_that("F111: ntile() with a named n and a constant around it stays native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  for (p in list(
    function(t) mutate(t, z = ntile(n = 2) * 10L),
    function(t) mutate(t, z = ntile(y2, n = 2)),
    function(t) mutate(t, z = ntile(y2, 2))
  )) {
    expect_same_as_dplyr(f111r_df(), p)
    expect_same_as_dplyr_lazy(f111r_df(), p)
  }
})
