# F19 acceptance tests: the superseded colwise verbs (`_at`/`_all`/`_if`) run
# natively on the GPU for the common shapes, and keep dplyr's results always.
#
# dplyr 1.2.1 contract (R/colwise*.R), verified on plain tibbles:
# - the colwise verbs are not generics: manip_at()/manip_all()/manip_if()
#   build one dot per (column, function), `name = <fn>(col)`, where <fn> is a
#   function OBJECT (`mean`, `"sum"` looked up, `list(a = abs)`), an
#   `inline_colwise_function` for a formula, or a quosure for `funs()` and
#   extra `...` args; then call mutate()/summarise()/... with those dots.
#   Dots are separate, so they are evaluated sequentially.
# - an inlined formula's body sees the data mask (`~ . + y` reads column y);
#   a plain function value is called normally and does not.
# - `_if` predicates run on the data through dplyr's tbl_ptype() generic
#   (tbl_if_vars()); a type-only predicate needs no data.
# - filter_all()/filter_if()/filter_at() join one call per column with `&`
#   (all_vars()/a function) or `|` (any_vars()).
# - group columns are excluded by _all/_if, included by _at for mutate.

# dplyr is imported, not attached, in the test environment
for (f19_fn in c("vars", "all_vars", "any_vars", "funs", "desc", "n_distinct", "cur_column",
                 "mutate_at", "mutate_all", "mutate_if", "transmute_at", "transmute_all",
                 "transmute_if", "summarise_at", "summarise_all", "summarise_if",
                 "summarize_at", "filter_at", "filter_all", "filter_if", "arrange_at",
                 "arrange_all", "arrange_if", "select_at", "select_if", "rename_all",
                 "rename_if", "group_by_at", "group_by_if", "distinct_at", "distinct_if")) {
  assign(f19_fn, getExportedValue("dplyr", f19_fn))
}
rm(f19_fn)

f19_df <- function() {
  data.frame(
    g = c(1L, 1L, 2L, 2L, 3L),
    x = c(1.5, 2, NA, 4, -1),
    y = c(10L, 20L, NA, 40L, 5L),
    s = c("a", "b", NA, "d", "e"),
    f = factor(c("u", "v", "u", "w", "v")),
    stringsAsFactors = FALSE
  )
}

f19_num <- function() {
  data.frame(x = c(1.5, 2, NA, 4, -1), y = c(10L, 20L, NA, 40L, 5L), z = c(3, 1, 2, 5, 4))
}

f19_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# The common shapes must stay on the GPU
f19_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f19_both(df, pipeline, ...)
}

test_that("mutate_at/_all/_if with a formula, a function, a name or a list are native", {
  skip_if_no_gpu()
  df <- f19_df()
  num <- f19_num()
  f19_native(df, function(d) mutate_at(d, vars(x, y), ~ .x * 2))
  f19_native(df, function(d) mutate_at(d, c("x", "y"), ~ . + 1))
  f19_native(df, function(d) mutate_at(d, vars(x, y), abs))
  f19_native(df, function(d) mutate_at(d, vars(x), "sqrt"))
  f19_native(df, function(d) mutate_if(d, is.numeric, ~ .x * 2))
  f19_native(df, function(d) mutate_if(d, is.double, list(a = ~ .x + 1, b = abs)))
  f19_native(num, function(d) mutate_all(d, ~ .x - 1))
  f19_native(num, function(d) mutate_all(d, list(p = ~ .x + 1, n = ~ -.x)))
  f19_native(df, function(d) mutate_at(d, vars(x), list(~ . * 2, ~ . + 1)))
  f19_native(df, function(d) mutate_at(d, vars(x, z = y), ~ .x * 3))
})

test_that("an inlined colwise formula sees the data mask and dots stay sequential", {
  skip_if_no_gpu()
  df <- f19_df()
  m <- 10
  f19_native(df, function(d) mutate_at(d, vars(x), ~ . + y))
  f19_native(df, function(d) mutate_at(d, vars(x, y), ~ . + x))
  f19_native(df, function(d) mutate_at(d, vars(x), ~ . * m))
  f19_native(df, function(d) transmute_at(d, vars(x, y), ~ . + 1))
  f19_native(f19_num(), function(d) transmute_all(d, abs))
  f19_native(df, function(d) transmute_if(d, is.numeric, ~ .x / 2))
})

test_that("summarise_at/_all/_if are native, with extra args and function lists", {
  skip_if_no_gpu()
  df <- f19_df()
  num <- f19_num()
  f19_native(num, function(d) summarise_all(d, mean))
  f19_native(num, function(d) summarise_all(d, list(mn = min, mx = max)))
  f19_native(num, function(d) summarise_all(d, list(~ mean(., na.rm = TRUE), ~ sum(.x, na.rm = TRUE))))
  f19_native(df, function(d) summarise_at(d, vars(x, y), mean, na.rm = TRUE))
  f19_native(df, function(d) summarise_at(d, c("x", "y"), "sum"))
  f19_native(df, function(d) summarise_if(d, is.numeric, list(m = mean, s = ~ sum(.x, na.rm = TRUE))))
  f19_native(df, function(d) summarize_at(d, vars(y), max, na.rm = TRUE))
})

test_that("grouped colwise verbs exclude or include group columns like dplyr", {
  skip_if_no_gpu()
  df <- f19_df()
  f19_native(df, function(d) suppressMessages(mutate_if(group_by(d, g), is.numeric, ~ .x * 2)))
  f19_native(df, function(d) suppressMessages(mutate_all(group_by(select(d, g, x, y), g), ~ .x + 1)))
  f19_native(df, function(d) suppressMessages(transmute_if(group_by(d, g), is.numeric, ~ .x + 1)))
  f19_native(df, function(d) summarise_all(group_by(select(d, g, x, y), g), mean, na.rm = TRUE))
  f19_native(df, function(d) summarise_if(group_by(d, g), is.numeric, max, na.rm = TRUE))
  f19_native(df, function(d) summarise_at(group_by(d, g), vars(x, y), list(mn = min, mx = max), na.rm = TRUE))
  # mutate_at() includes the group column it names; the result regroups
  f19_both(df, function(d) mutate_at(group_by(d, g), vars(g, x), ~ .x * 2))
  f19_both(df, function(d) mutate_at(group_by(d, g), vars(x), mean, na.rm = TRUE))
})

test_that("filter_at/_all/_if with all_vars()/any_vars() and functions are native", {
  skip_if_no_gpu()
  df <- f19_df()
  num <- f19_num()
  f19_native(df, function(d) filter_at(d, vars(x, y), all_vars(. > 1)))
  f19_native(df, function(d) filter_at(d, vars(x, y), any_vars(is.na(.))))
  f19_native(df, function(d) filter_if(d, is.numeric, all_vars(!is.na(.))))
  f19_native(num, function(d) filter_all(d, any_vars(. > 4)))
  f19_native(num, function(d) filter_all(d, ~ .x > 1))
  f19_native(df, function(d) filter_at(d, vars(x, y), function(v) v > 1.5))
  f19_native(df, function(d) filter_if(d, is.double, ~ !is.na(.x)))
  f19_native(df, function(d) filter_at(group_by(d, g), vars(x), all_vars(. > 0)))
})

test_that("arrange_at/_all/_if with a function are native", {
  skip_if_no_gpu()
  df <- f19_df()
  f19_native(df, function(d) arrange_at(d, vars(y), desc))
  f19_native(f19_num(), function(d) arrange_all(d, desc))
  f19_native(df, function(d) arrange_if(d, is.double, desc))
  f19_native(df, function(d) arrange_at(d, vars(g, x)))
})

test_that("select/rename/group_by/distinct colwise variants keep dplyr's results", {
  skip_if_no_gpu()
  df <- f19_df()
  f19_native(df, function(d) select_at(d, vars(x, y), toupper))
  f19_native(df, function(d) select_if(d, is.numeric))
  f19_native(df, function(d) rename_all(d, toupper))
  f19_native(df, function(d) rename_if(d, is.numeric, ~ paste0(.x, "_n")))
  f19_native(df, function(d) group_by_at(d, vars(g, s)))
  f19_native(df, function(d) group_by_if(d, is.character))
  f19_native(df, function(d) distinct_at(d, vars(g)))
  f19_native(df, function(d) distinct_if(d, is.integer))
  f19_both(df, function(d) group_by_at(d, vars(s), toupper))
  f19_both(df, function(d) distinct_at(d, vars(s), toupper))
})

test_that("type-only `_if` predicates read no data from the GPU", {
  skip_if_no_gpu()
  df <- f19_df()
  pipelines <- list(
    function(d) mutate_if(d, is.numeric, ~ .x * 2),
    function(d) summarise_if(d, is.numeric, mean, na.rm = TRUE),
    function(d) select_if(d, is.character),
    function(d) rename_if(d, is.double, toupper),
    function(d) filter_if(d, is.integer, all_vars(. > 1)),
    function(d) mutate_if(d, ~ is.numeric(.x) && !is.factor(.x), ~ .x + 1),
    function(d) mutate_if(d, function(v) is.integer(v) || is.double(v), ~ .x - 1)
  )
  for (pipeline in pipelines) {
    for (lazy in c(FALSE, TRUE)) {
      gdf <- tbl_gpu(df, lazy = lazy)
      res <- local({
        testthat::local_mocked_bindings(
          collect_frame = function(...) stop("F19: data was downloaded"),
          .package = "cuplyr"
        )
        withr::local_options(cuplyr.fallback = "error")
        pipeline(gdf)
      })
      expect_equal(
        tibble::as_tibble(collect(res)),
        tibble::as_tibble(pipeline(tibble::as_tibble(df)))
      )
    }
  }
})

test_that("value predicates, logical predicates and non-inlinable functions keep dplyr's results", {
  skip_if_no_gpu()
  df <- f19_df()
  f19_both(df, function(d) mutate_if(d, ~ is.numeric(.x) && !anyNA(.x), ~ .x * 2))
  f19_both(df, function(d) select_if(d, ~ n_distinct(.x) > 4))
  f19_both(df, function(d) summarise_if(d, function(v) is.numeric(v) && max(v, na.rm = TRUE) > 5, mean, na.rm = TRUE))
  f19_both(df, function(d) mutate_if(d, c(FALSE, TRUE, TRUE, FALSE, FALSE), ~ .x * 2))
  f19_both(df, function(d) filter_if(d, ~ is.numeric(.x) && anyNA(.x), all_vars(!is.na(.))))
  # a function value is called normally: its free variable is NOT the column
  k <- 100
  plus_k <- function(v) v + k
  dk <- transform(df, k = 1)
  f19_both(dk, function(d) mutate_at(d, vars(x), plus_k))
  f19_both(dk, function(d) filter_at(d, vars(x), function(v) v + k > 101))
  # functions the GPU doesn't know still give dplyr's result
  f19_both(df, function(d) mutate_at(d, vars(s), function(v) paste0(v, "!")))
  f19_both(df, function(d) summarise_at(d, vars(x, y), function(v) median(rev(v), na.rm = TRUE)))
})

test_that("0-row and all-NA inputs are native", {
  skip_if_no_gpu()
  df0 <- f19_df()[0, ]
  num <- data.frame(x = c(NA_real_, NA_real_), y = c(NA_integer_, NA_integer_))
  f19_native(df0, function(d) mutate_if(d, is.numeric, ~ .x * 2))
  f19_native(df0, function(d) mutate_at(d, vars(x, y), ~ . + 1))
  f19_native(df0, function(d) summarise_at(d, vars(x, y), sum))
  f19_native(num, function(d) summarise_all(d, mean, na.rm = TRUE))
  f19_native(num, function(d) mutate_all(d, ~ .x + 1))
  f19_native(df0, function(d) filter_at(d, vars(x, y), all_vars(. > 0)))
})

test_that("dplyr's deprecation warnings and errors are kept", {
  skip_if_no_gpu()
  df <- f19_df()
  gdf <- tbl_gpu(df)
  withr::local_options(lifecycle_verbosity = "warning")
  # funs() is deprecated (dplyr warns before cuplyr sees the dots)
  expect_warning(res <- mutate_at(gdf, vars(x), funs(. * 2)), class = "lifecycle_warning_deprecated")
  expect_equal(tibble::as_tibble(collect(res)),
               suppressWarnings(tibble::as_tibble(mutate_at(df, vars(x), funs(. * 2)))))
  # a quosure in .funs warns, then runs as a formula
  expect_warning(res <- summarise_at(gdf, vars(x, y), rlang::quo(mean(., na.rm = TRUE))),
                 class = "lifecycle_warning_deprecated")
  expect_equal(tibble::as_tibble(collect(res)),
               suppressWarnings(tibble::as_tibble(summarise_at(df, vars(x, y), rlang::quo(mean(., na.rm = TRUE))))))
  # `.y` in a colwise formula is dplyr's error, never a silent result
  expect_error(collect(mutate_at(gdf, vars(x), ~ .y)), class = "dplyr:::mutate_error")
  expect_error(collect(mutate_at(gdf, vars(x), ~ .y)), "fewer than 2 elements")
  expect_error(filter_at(gdf, vars(starts_with("zz")), all_vars(. > 1)),
               "must match at least one column")
})

test_that("arrange colwise desc() on strings follows the session collation like dplyr", {
  skip_if_no_gpu()
  ok <- tryCatch({ withr::local_collate("en_US.UTF-8"); TRUE }, warning = function(w) FALSE,
                 error = function(e) FALSE)
  skip_if_not(ok, "en_US.UTF-8 collation unavailable")
  withr::local_collate("en_US.UTF-8")
  df <- data.frame(s = c("a", "B", "c", "D", NA), x = c(5, 4, 3, 2, 1), stringsAsFactors = FALSE)
  f19_both(df, function(d) arrange_at(d, vars(s), desc))
  f19_both(df, function(d) arrange_if(d, is.character, desc))
  f19_both(df, function(d) arrange_all(d, desc))
  # numeric desc stays native
  f19_native(df, function(d) arrange_at(d, vars(x), desc))
})

test_that("cur_column() and `...` in colwise functions behave like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4), y = c(10L, 20L, NA, 40L), s = c("a", "b", NA, "d"),
                   stringsAsFactors = FALSE)
  f1 <- function(v) paste0(v, cur_column())
  expect_error(collect(mutate_at(tbl_gpu(df), vars(s), f1)), "only be used inside")
  expect_error(collect(mutate_at(tbl_gpu(df, lazy = TRUE), vars(s), f1)), "only be used inside")
  f2 <- function(v) v + nchar(cur_column())
  expect_error(collect(mutate_at(tbl_gpu(df), vars(x), f2)), "only be used inside")
  # inline colwise formulas and rlang lambdas bind `...`
  f19_both(df, function(d) mutate_at(d, vars(x, y), ~ sum(..., na.rm = TRUE)))
  f19_both(df, function(d) filter_at(d, vars(x, y), ~ !is.na(...)))
  # a filter lambda comparing with a string literal still gives dplyr's result
  f19_both(df[c("x", "s")], function(d) filter_all(d, ~ .x != "a"))
  f19_both(df[c("x", "s")], function(d) filter_all(d, function(v) v != "a"))
  lbl <- "a"
  f19_both(df[c("x", "s")], function(d) filter_all(d, ~ .x != lbl))
  f19_both(df, function(d) mutate_at(d, vars(x), function(v) v == "a"))
})

test_that("colwise string min/max and ordering comparisons follow the session collation", {
  skip_if_no_gpu()
  df <- data.frame(s = c("a", "B", "c", "D"), x = c(1, 2, 3, 4), stringsAsFactors = FALSE)
  # C collation (testthat's default): exact on the GPU
  withr::local_collate("C")
  f19_native(df, function(d) summarise_all(d, list(mn = min, mx = max)))
  f19_native(df["s"], function(d) filter_all(d, ~ .x > "a"))
  f19_native(df["s"], function(d) mutate_at(d, vars(s), list(lt = ~ . < "b")))
  ok <- tryCatch({ withr::local_collate("en_US.UTF-8"); TRUE }, warning = function(w) FALSE,
                 error = function(e) FALSE)
  skip_if_not(ok, "en_US.UTF-8 collation unavailable")
  withr::local_collate("en_US.UTF-8")
  f19_both(df, function(d) summarise_all(d, list(mn = min, mx = max)))
  f19_both(df["s"], function(d) filter_all(d, ~ .x > "a"))
  f19_both(df["s"], function(d) mutate_at(d, vars(s), list(lt = ~ . < "b")))
  # numeric shapes stay native whatever the collation
  f19_native(df["x"], function(d) summarise_all(d, list(mn = min, mx = max)))
  f19_native(df["x"], function(d) filter_all(d, ~ .x > 1))
})
