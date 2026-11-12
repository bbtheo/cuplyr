# F122 acceptance: an unnamed `.data` pronoun names the column the way dplyr does.
#
# dplyr's rule (R/group-by.R, add_computed_columns() and
# quo_is_variable_reference(), used by group_by()/count()/add_count() and
# distinct_prepare()): a dot is a "variable reference" if it is a bare symbol,
# `.data$x`, or `.data[[<symbol or string>]]`. If every dot is unnamed and a
# variable reference, the columns are looked up by as_label(dot), with no
# evaluation. Otherwise (any named or computed dot) every dot goes through
# mutate(), named by as_label(dot) with rlang's infix labelling off.

f122_df <- function() {
  tibble::tibble(
    k = c(1L, 1L, 2L, 2L, NA),
    a = c("x", "y", "y", NA, "x"),
    b = c(1.5, 2.5, 2.5, NA, 1.5),
    g = c(1L, 1L, 2L, 2L, 2L)
  )
}

# dplyr's result, eager and lazy, with any CPU fallback an error: every form
# here ran on the GPU before F122.
f122_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# dplyr's result, eager and lazy, CPU fallback allowed.
f122_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# dplyr's error, eager and lazy: same class, and (unless `message = FALSE`)
# the same text.
f122_same_error <- function(df, pipeline, message = TRUE) {
  oracle <- tryCatch(pipeline(tibble::as_tibble(df)), error = identity)
  expect_s3_class(oracle, "error")
  for (lazy in c(FALSE, TRUE)) {
    gpu <- tryCatch(dplyr::collect(pipeline(tbl_gpu(df, lazy = lazy))), error = identity)
    expect_s3_class(gpu, "error")
    if (!inherits(gpu, "error")) next
    expect_identical(class(gpu)[[1L]], class(oracle)[[1L]], info = paste("lazy:", lazy))
    if (message) {
      expect_identical(conditionMessage(gpu), conditionMessage(oracle), info = paste("lazy:", lazy))
    }
  }
}

# A long infix expression, so as_label() has to shorten it. dplyr labels it with
# rlang's infix labelling off (`+...`, not `... + b`).
f122_long <- quote(b + b + b + b + b + b + b + b + b + b + b + b + b + b + b + b + b + b +
                     b + b + b + b + b + b + b)

# -----------------------------------------------------------------------------
# distinct()
# -----------------------------------------------------------------------------

test_that("distinct() names a pronoun key after the column", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::distinct(d, .data$a))
  f122_native(df, function(d) dplyr::distinct(d, .data[["a"]]))
  f122_native(df, function(d) dplyr::distinct(d, .data[[v]]))
  f122_native(df, function(d) dplyr::distinct(d, .data$a, b))
  f122_native(df, function(d) dplyr::distinct(d, b, .data[["a"]]))
  f122_native(df, function(d) dplyr::distinct(d, .data[[v]], .data$b))
})

test_that("distinct() never repeats a key given two ways", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::distinct(d, .data$a, a))
  f122_native(df, function(d) dplyr::distinct(d, a, .data[["a"]]))
  f122_native(df, function(d) dplyr::distinct(d, .data$a, .data[["a"]], .data[[v]]))
})

test_that("distinct(.keep_all = TRUE) with a pronoun key adds no column", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::distinct(d, .data$a, .keep_all = TRUE))
  f122_native(df, function(d) dplyr::distinct(d, .data[["k"]], .keep_all = TRUE))
  f122_native(df, function(d) dplyr::distinct(d, .data[[v]], b, .keep_all = TRUE))
})

test_that("grouped distinct() puts a pronoun key where dplyr does", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  grouped <- function(f) function(d) f(dplyr::group_by(d, g))
  f122_native(df, grouped(function(d) dplyr::distinct(d, .data$a)))
  f122_native(df, grouped(function(d) dplyr::distinct(d, .data[[v]], b)))
  # the group column named as a key keeps the key's position
  f122_native(df, grouped(function(d) dplyr::distinct(d, b, .data$g)))
  f122_native(df, grouped(function(d) dplyr::distinct(d, .data[["g"]], b)))
  f122_native(df, grouped(function(d) dplyr::distinct(d, .data$a, .keep_all = TRUE)))
})

test_that("distinct() with a named or computed key beside a pronoun key", {
  skip_if_no_gpu()
  df <- f122_df()
  f122_native(df, function(d) dplyr::distinct(d, z = .data$a, .data$b))
  f122_native(df, function(d) dplyr::distinct(d, .data$a, k + 1L))
  f122_native(df, function(d) dplyr::distinct(d, .data$k + 1L, .data[["b"]]))
})

test_that("distinct() on pronoun keys handles empty tables and every key type", {
  skip_if_no_gpu()
  f122_native(tibble::tibble(c = integer(0)), function(d) dplyr::distinct(d, .data$c))
  f122_native(tibble::tibble(c = integer(0), e = character(0)),
              function(d) dplyr::distinct(d, .data[["e"]], .data$c))
  typed <- tibble::tibble(
    l = c(TRUE, TRUE, NA),
    d = as.Date("2024-01-01") + c(0L, 0L, 1L),
    s = c("p", "p", NA)
  )
  f122_native(typed, function(d) dplyr::distinct(d, .data$l))
  f122_native(typed, function(d) dplyr::distinct(d, .data[["d"]]))
  f122_native(typed, function(d) dplyr::distinct(d, .data$s, .data$l))
  fct <- tibble::tibble(f = factor(c("u", "v", "u"), levels = c("v", "u", "w")), n = 1:3)
  f122_native(fct, function(d) dplyr::distinct(d, .data$f))
})

test_that("verbs after distinct() find the pronoun key by name", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::relocate(dplyr::distinct(d, .data[["b"]]), b))
  f122_native(df, function(d) dplyr::rename(dplyr::distinct(d, .data[["b"]]), r1 = b))
  f122_native(df, function(d) dplyr::select(dplyr::distinct(d, .data[["a"]], b), a))
  f122_native(df, function(d) dplyr::rename(dplyr::distinct(d, .data$k, b), r1 = k))
  f122_native(df, function(d) dplyr::filter(dplyr::distinct(d, .data[[v]]), a == "x"))
  f122_native(df, function(d) dplyr::mutate(dplyr::distinct(d, .data$b), b2 = b + 1))
  f122_native(df, function(d) dplyr::select(dplyr::distinct(d, .data$a, .keep_all = TRUE), a, k))
  f122_native(df, function(d) dplyr::arrange(dplyr::distinct(d, .data$b, .data$a), a, b))
})

test_that("a pronoun key over a column made by a pending lazy op is named right", {
  skip_if_no_gpu()
  df <- f122_df()
  f122_native(df, function(d) dplyr::distinct(dplyr::mutate(d, a2 = k + 1L), .data$a2))
  f122_native(df, function(d) dplyr::distinct(dplyr::filter(d, b > 1), .data[["a"]]))
})

test_that("pronouns passed through {{ }} and function arguments are named right", {
  skip_if_no_gpu()
  df <- f122_df()
  by_arg <- function(d, col) dplyr::distinct(d, .data[[col]])
  embraced <- function(d, x) dplyr::distinct(d, {{ x }})
  # the argument shadows a column: rlang reads the subscript in the caller
  shadow <- function(d, a) dplyr::distinct(d, .data[[a]])
  f122_native(df, function(d) by_arg(d, "a"))
  f122_native(df, function(d) embraced(d, .data$a))
  f122_native(df, function(d) shadow(d, "b"))
  q <- rlang::quo(.data$a)
  f122_native(df, function(d) dplyr::distinct(d, !!q))
})

# -----------------------------------------------------------------------------
# count(), add_count(), tally(), group_by()
# -----------------------------------------------------------------------------

test_that("count() names a pronoun key after the column", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::count(d, .data$a))
  f122_native(df, function(d) dplyr::count(d, .data[["a"]]))
  f122_native(df, function(d) dplyr::count(d, .data[[v]]))
  f122_native(df, function(d) dplyr::count(d, .data$a, .data[["b"]]))
  f122_native(df, function(d) dplyr::count(d, .data$a, a))
  f122_native(df, function(d) dplyr::count(d, .data$g, name = "z"))
  f122_native(df, function(d) dplyr::count(dplyr::group_by(d, g), .data$a))
  f122_native(df, function(d) dplyr::count(dplyr::group_by(d, g), .data[["g"]]))
  f122_native(df, function(d) dplyr::select(dplyr::count(d, .data[["a"]]), a))
})

test_that("add_count() and tally() with pronoun arguments match dplyr", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::add_count(d, .data$a))
  f122_native(df, function(d) dplyr::add_count(d, .data[[v]], .data$b))
  f122_native(df, function(d) dplyr::add_count(dplyr::group_by(d, g), .data$a))
  f122_both(df, function(d) dplyr::count(d, .data$g, wt = .data$b))
  f122_both(df, function(d) dplyr::tally(dplyr::group_by(d, .data$g), .data$b))
})

test_that("group_by() takes the name of a pronoun key", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::group_by(d, .data$a))
  f122_native(df, function(d) dplyr::group_by(d, .data[["a"]]))
  f122_native(df, function(d) dplyr::group_by(d, .data[[v]], .data$g))
  f122_native(df, function(d) dplyr::summarise(dplyr::group_by(d, .data$g), n = dplyr::n()))
  f122_native(df, function(d) dplyr::summarise(dplyr::group_by(d, .data[[v]]), m = mean(b)))
  f122_native(df, function(d) dplyr::group_by(dplyr::group_by(d, b), .data$g, .add = TRUE))
})

# -----------------------------------------------------------------------------
# dplyr doesn't simplify these, and the verbs that already agree stay agreed
# -----------------------------------------------------------------------------

test_that(".env and computed pronoun expressions keep dplyr's label", {
  skip_if_no_gpu()
  df <- f122_df()
  ev <- 7L
  f122_native(df, function(d) dplyr::distinct(d, .env$ev))
  f122_native(df, function(d) dplyr::distinct(d, .env[["ev"]]))
  f122_native(df, function(d) dplyr::distinct(d, .env$ev + 1L))
  f122_native(df, function(d) dplyr::distinct(d, .data$a == "x"))
  f122_native(df, function(d) dplyr::distinct(d, (.data$a)))
  f122_native(df, function(d) dplyr::distinct(d, -.data$b))
  f122_native(df, function(d) dplyr::count(d, .env$ev))
  f122_native(df, function(d) dplyr::count(d, (.data$a)))
  f122_native(df, function(d) dplyr::group_by(d, .env$ev))
  f122_native(df, function(d) dplyr::group_by(d, .data$a == "x"))
})

test_that("mutate(), transmute() and summarise() keep pronoun names", {
  skip_if_no_gpu()
  v <- "a"
  df <- f122_df()
  f122_native(df, function(d) dplyr::mutate(d, .data$a, .data[["b"]]))
  f122_native(df, function(d) dplyr::transmute(d, .data$b, .data[[v]]))
  # (summarise() warns about an unnamed expression, on the GPU only)
  f122_native(df, function(d) suppressWarnings(dplyr::summarise(d, sum(.data$b, na.rm = TRUE))))
  f122_native(df, function(d) dplyr::summarise(dplyr::group_by(d, .data$g),
                                               m = mean(.data$b, na.rm = TRUE)))
  f122_native(df, function(d) dplyr::mutate(d, n2 = dplyr::n(), .by = .data$g))
  f122_native(df, function(d) dplyr::summarise(d, n2 = dplyr::n(), .by = .data[["g"]]))
  # (tidyselect deprecates .data in selections)
  f122_native(df, function(d) suppressWarnings(dplyr::mutate(d, dplyr::across(.data$b, ~ .x + 1))))
  f122_native(df, function(d) suppressWarnings(dplyr::select(d, z = .data[["a"]], .data$b)))
  # (arrange() on a pronoun takes the CPU route, today as before)
  f122_both(df, function(d) dplyr::arrange(d, .data$b))
})

# -----------------------------------------------------------------------------
# Long expressions: one label everywhere
# -----------------------------------------------------------------------------

test_that("a long unnamed expression has dplyr's label in every verb", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = 1:3, b = c(1, 2, 3))
  e <- f122_long
  f122_native(df, function(d) dplyr::distinct(d, !!e))
  f122_native(df, function(d) dplyr::distinct(d, !!e, .keep_all = TRUE))
  f122_native(df, function(d) dplyr::count(d, !!e))
  f122_native(df, function(d) dplyr::add_count(d, !!e))
  f122_native(df, function(d) dplyr::group_by(d, !!e))
  f122_native(df, function(d) dplyr::summarise(dplyr::group_by(d, !!e), n = dplyr::n()))
  f122_native(df, function(d) dplyr::transmute(d, !!e))
  # mutate() warns about an unnamed expression, on the GPU only
  f122_native(df, function(d) suppressWarnings(dplyr::mutate(d, !!e)))
  f122_native(df, function(d) suppressWarnings(dplyr::mutate(d, dplyr::across(k), !!e)))
})

# -----------------------------------------------------------------------------
# A variable reference names a column; only dplyr's mutate path evaluates it
# -----------------------------------------------------------------------------

test_that("an all-variable-reference key that doesn't exist is dplyr's error", {
  skip_if_no_gpu()
  df <- f122_df()
  outside <- 3L # an environment variable is not a column here
  f122_same_error(df, function(d) dplyr::distinct(d, .data$nope))
  f122_same_error(df, function(d) dplyr::distinct(d, .data[["nope"]], a))
  f122_same_error(df, function(d) dplyr::distinct(d, .data$nope, .keep_all = TRUE))
  f122_same_error(df, function(d) dplyr::distinct(d, nope))
  f122_same_error(df, function(d) dplyr::distinct(d, outside))
  f122_same_error(df, function(d) dplyr::distinct(d, a, outside))
  f122_same_error(df, function(d) dplyr::distinct(dplyr::group_by(d, g), .data$nope))
  f122_same_error(df, function(d) dplyr::count(d, .data$nope))
  f122_same_error(df, function(d) dplyr::count(d, .data[["nope"]], a))
  f122_same_error(df, function(d) dplyr::count(d, nope))
  f122_same_error(df, function(d) dplyr::count(d, outside))
  f122_same_error(df, function(d) dplyr::add_count(d, .data$nope))
  f122_same_error(df, function(d) dplyr::add_count(d, outside))
  # group_by() has its own wording and hint, pre-dating this feature
  f122_same_error(df, function(d) dplyr::group_by(d, .data$nope), message = FALSE)
  f122_same_error(df, function(d) dplyr::group_by(d, outside), message = FALSE)
})

test_that("a pronoun as_label() can't name is dplyr's <unknown> error", {
  skip_if_no_gpu()
  df <- f122_df()
  v <- "a"
  # .data$"a" and a raw `.data[[v]]` (one rlang's capture never rewrote) are
  # variable references that as_label() calls <unknown>
  raw <- rlang::new_quosure(quote(.data[[v]]), environment())
  f122_same_error(df, function(d) dplyr::distinct(d, .data$"a"))
  f122_same_error(df, function(d) dplyr::distinct(d, !!raw))
  f122_same_error(df, function(d) dplyr::count(d, .data$"a"))
  f122_same_error(df, function(d) dplyr::count(d, !!raw))
  f122_same_error(df, function(d) dplyr::group_by(d, .data$"a"), message = FALSE)
  f122_same_error(df, function(d) dplyr::group_by(d, !!raw), message = FALSE)
})

test_that("beside a named or computed dot, a variable reference is evaluated", {
  skip_if_no_gpu()
  df <- f122_df()
  v <- "a"
  outside <- 3L
  raw <- rlang::new_quosure(quote(.data[[v]]), environment())
  # the environment variable becomes a column
  f122_both(df, function(d) dplyr::distinct(d, outside, k + 1L))
  f122_both(df, function(d) dplyr::count(d, outside, k + 1L))
  f122_both(df, function(d) dplyr::add_count(d, outside, k + 1L))
  f122_both(df, function(d) dplyr::group_by(d, outside, k + 1L))
  f122_both(df, function(d) dplyr::group_by(d, outside, z = k))
  # and as_label() names what it can't label <unknown>
  f122_both(df, function(d) dplyr::distinct(d, .data$"a", k + 1L))
  f122_both(df, function(d) dplyr::distinct(d, !!raw, k + 1L))
  f122_both(df, function(d) dplyr::count(d, .data$"a", k + 1L))
  f122_both(df, function(d) dplyr::count(d, !!raw, k + 1L))
  f122_both(df, function(d) dplyr::group_by(d, .data$"a", k + 1L))
  f122_both(df, function(d) dplyr::group_by(d, !!raw, k + 1L))
})

# -----------------------------------------------------------------------------
# Round 1 review: the branch is decided over every dot, NULL included, and a
# reference to an earlier computed dot is not a second computed column
# -----------------------------------------------------------------------------

test_that("a NULL dot counts toward dplyr's mutate branch", {
  skip_if_no_gpu()
  df <- f122_df()
  outside <- 3L
  # NULL is not a variable reference, so every dot is evaluated
  f122_both(df, function(d) dplyr::distinct(d, NULL, outside))
  f122_both(df, function(d) dplyr::count(d, NULL, outside))
  f122_both(df, function(d) dplyr::add_count(d, NULL, outside))
  f122_both(df, function(d) dplyr::group_by(d, NULL, outside))
  f122_both(df, function(d) dplyr::distinct(d, NULL, .data$"a"))
  f122_both(df, function(d) dplyr::count(d, NULL, .data$"a"))
  # an unused {{ }} argument is the usual way to get here
  by_arg <- function(d, g = NULL) dplyr::count(d, {{ g }}, outside)
  f122_both(df, function(d) by_arg(d))
  by_arg2 <- function(d, g = NULL) dplyr::distinct(d, {{ g }}, outside)
  f122_both(df, function(d) by_arg2(d))
  # and an existing column beside NULL is just that column, on the GPU
  f122_native(df, function(d) dplyr::distinct(d, NULL, .data$a))
  f122_native(df, function(d) dplyr::count(d, NULL, .data[["a"]]))
  f122_native(df, function(d) dplyr::group_by(d, NULL, .data$a))
})

test_that("group_by() on a reference to an earlier computed dot stays on the GPU", {
  skip_if_no_gpu()
  df <- f122_df()
  f122_native(df, function(d) dplyr::group_by(d, z = k + 1L, z))
  f122_native(df, function(d) dplyr::group_by(d, z = k + 1L, .data$z))
  f122_native(df, function(d) dplyr::group_by(d, k + 1L, `k + 1L`))
  f122_native(df, function(d) dplyr::summarise(dplyr::group_by(d, z = k + 1L, .data$z), n = dplyr::n()))
})

test_that("a missing variable reference doesn't blame an internal helper", {
  skip_if_no_gpu()
  df <- f122_df()
  for (lazy in c(FALSE, TRUE)) {
    gpu <- tbl_gpu(df, lazy = lazy)
    for (e in list(
      tryCatch(dplyr::distinct(gpu, .data$nope), error = identity),
      tryCatch(dplyr::count(gpu, .data$nope), error = identity),
      tryCatch(dplyr::add_count(gpu, .data$nope), error = identity)
    )) {
      expect_s3_class(e, "rlang_error")
      head <- paste(deparse(rlang::call_name(conditionCall(e))), collapse = "")
      expect_match(head, "^\"(distinct|count|add_count)", info = paste("lazy:", lazy))
    }
  }
})

# -----------------------------------------------------------------------------
# Round 2 review: an empty dot, and two dots that as_label() calls <unknown>
# -----------------------------------------------------------------------------

test_that("an empty dot in distinct() is dplyr's error", {
  skip_if_no_gpu()
  df <- f122_df()
  f122_same_error(df, function(d) dplyr::distinct(d, , g))
  f122_same_error(df, function(d) dplyr::distinct(d, , .data$a))
  f122_same_error(df, function(d) dplyr::distinct(d, , k + 1L))
  unset <- function(x, v) dplyr::distinct(x, {{ v }}, a)
  f122_same_error(df, function(d) unset(d))
  # group_by_prepare() captures with .ignore_empty = "all", so count() drops it
  unset2 <- function(x, v) dplyr::count(x, {{ v }}, a)
  f122_both(df, function(d) unset2(d))
})

test_that("group_by() reads each dot labelled <unknown> for itself", {
  skip_if_no_gpu()
  df <- f122_df()
  v <- "a"
  w <- "b"
  raw_v <- rlang::new_quosure(quote(.data[[v]]), environment())
  raw_w <- rlang::new_quosure(quote(.data[[w]]), environment())
  f122_both(df, function(d) dplyr::group_by(d, .data$"a", .data$"b", k + 1L))
  f122_both(df, function(d) dplyr::group_by(d, NULL, .data$"a", .data$"b"))
  f122_both(df, function(d) dplyr::summarise(dplyr::group_by(d, .data$"a", .data$"b", k + 1L),
                                             n = dplyr::n(), .groups = "drop"))
  f122_both(df, function(d) dplyr::group_by(d, !!raw_v, !!raw_w, k + 1L))
  f122_both(df, function(d) dplyr::count(d, .data$"a", .data$"b", k + 1L))
  f122_both(df, function(d) dplyr::distinct(d, .data$"a", .data$"b", k + 1L))
})

# -----------------------------------------------------------------------------
# Round 3 review: a reference to a column an earlier across() or data-frame dot
# makes, and a table with a column really called <unknown>
# -----------------------------------------------------------------------------

test_that("group_by() on a reference to a column an earlier dot unpacks matches dplyr", {
  skip_if_no_gpu()
  df <- f122_df()
  f122_both(df, function(d) dplyr::group_by(d, dplyr::across(a, .names = "z"), z))
  f122_both(df, function(d) dplyr::group_by(d, dplyr::across(a, .names = "z"), .data$z))
  f122_both(df, function(d) dplyr::summarise(
    dplyr::group_by(d, dplyr::across(k, ~ .x + 1L, .names = "z"), z),
    n = dplyr::n(), .groups = "drop"
  ))
  f122_both(df, function(d) dplyr::group_by(d, tibble::tibble(z = k), z))
  # the earlier dot's column masks an environment variable of the same name
  z <- 9L
  f122_both(df, function(d) dplyr::group_by(d, tibble::tibble(z = k), z))
  f122_both(df, function(d) dplyr::group_by(d, dplyr::across(a, .names = "z"), z))
})

test_that("a column really called <unknown> is found by name", {
  skip_if_no_gpu()
  df <- tibble::tibble(`<unknown>` = c(1L, 1L, 2L), a = c("x", "y", "y"))
  f122_both(df, function(d) dplyr::distinct(d, `<unknown>`))
  f122_both(df, function(d) dplyr::distinct(d, .data[["<unknown>"]], a))
  f122_both(df, function(d) dplyr::count(d, .data$`<unknown>`))
  f122_both(df, function(d) dplyr::add_count(d, `<unknown>`))
  # with nothing to evaluate, as_label() alone decides: this reads the column
  f122_both(df, function(d) dplyr::distinct(d, .data$"a"))
  f122_both(df, function(d) dplyr::count(d, .data$"a"))
})
