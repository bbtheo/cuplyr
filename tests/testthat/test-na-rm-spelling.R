# Each reducer takes only its own spelling of the NA flag (F19-a2 round-2
# review B1/N1). Base R's sum()/mean()/min()/max()/sd()/var()/median()/
# quantile()/any()/all() and dplyr's n_distinct() spell it `na.rm`; dplyr's
# first()/last()/nth() spell it `na_rm`. The other spelling is not a flag:
# sum()/min()/max()/any()/all() take it as one more `...` value, mean()/
# median() swallow it, sd()/var()/first()/last()/nth() error ("unused
# argument"), n_distinct() errors (named `...`). The native summarise() only
# strips the function's own spelling, and only when its value is a single
# TRUE/FALSE that doesn't read the data mask; anything else runs in dplyr.
# The IR parsers (mutate()/filter() windows, when_all()/when_any(),
# roll_*()) follow the same value rule.

nr_df <- function() {
  data.frame(g = c(1L, 1L, 2L, 2L), x = c(1, 2, NA, 4), y = c(1L, 2L, 3L, 4L),
             b = c(TRUE, NA, FALSE, TRUE), z = c(0.1, 0.5, 0.2, 0.3))
}

# Run `pipeline` on a tibble (dplyr) and on an eager and a lazy tbl_gpu. If
# dplyr errors, the GPU must raise dplyr's error (same innermost message);
# otherwise names, column classes, values, grouping and warnings must match.
nr_check <- function(df, pipeline) {
  run <- function(input) {
    warns <- character()
    res <- tryCatch(
      withCallingHandlers(
        {
          out <- pipeline(input)
          groups <- if (is_tbl_gpu(out)) out$groups else dplyr::group_vars(out)
          list(data = dplyr::ungroup(tibble::as_tibble(collect(out))), groups = groups)
        },
        warning = function(w) {
          warns <<- c(warns, conditionMessage(w))
          invokeRestart("muffleWarning")
        },
        message = function(m) invokeRestart("muffleMessage")
      ),
      error = function(e) {
        while (!is.null(e$parent)) e <- e$parent
        list(error = conditionMessage(e))
      }
    )
    res$warnings <- warns
    res
  }
  oracle <- run(tibble::as_tibble(df))
  for (lazy in c(FALSE, TRUE)) {
    label <- if (lazy) "lazy" else "eager"
    gpu <- run(tbl_gpu(df, lazy = lazy))
    if (!is.null(oracle$error)) {
      expect_identical(gpu$error, oracle$error, info = label)
      next
    }
    expect_null(gpu$error, info = label)
    expect_identical(gpu$groups, oracle$groups, info = label)
    expect_identical(names(gpu$data), names(oracle$data), info = label)
    expect_identical(lapply(gpu$data, class), lapply(oracle$data, class), info = label)
    expect_equal(gpu$data, oracle$data, info = label)
    expect_identical(gpu$warnings, oracle$warnings, info = label)
  }
}

# Same, with any CPU fallback an error: the native path must produce it.
nr_native <- function(df, pipeline) {
  withr::local_options(cuplyr.fallback = "error")
  nr_check(df, pipeline)
}

test_that("summarise(): base reducers given na_rm compute what dplyr computes", {
  skip_if_no_gpu()
  df <- nr_df()
  nr_check(df, function(d) summarise(d, s = sum(y, na_rm = TRUE), m = mean(x, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = sum(x, na_rm = TRUE), mx = max(z, na_rm = TRUE),
                                     mn = min(z, na_rm = TRUE), md = median(x, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, a = any(b, na_rm = TRUE), al = all(b, na_rm = FALSE)))
  nr_check(df, function(d) summarise(dplyr::group_by(d, g), s = sum(y, na_rm = TRUE),
                                     m = mean(x, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = base::sum(y, na_rm = TRUE)))
})

test_that("summarise(): a flag the reducer doesn't take is dplyr's error", {
  skip_if_no_gpu()
  df <- nr_df()
  nr_check(df, function(d) summarise(d, s = sd(x, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = var(x, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = n_distinct(x, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = quantile(x, 0.5, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = dplyr::first(x, na.rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = dplyr::last(x, na.rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = dplyr::nth(x, 2, na.rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = n(na.rm = TRUE)))
})

test_that("summarise(): a partially matched flag is not dropped", {
  skip_if_no_gpu()
  df <- nr_df()
  # mean.default()/sd()/quantile.default()/nth() partially match `na.rm`/`na_rm`
  nr_check(df, function(d) summarise(d, m = mean(x, na = TRUE), s = sd(x, na = TRUE)))
  # (`na` alone is ambiguous for quantile.default(): na.rm or names)
  nr_check(df, function(d) summarise(d, q = quantile(x, 0.5, na.r = TRUE, names = FALSE)))
  nr_check(df, function(d) summarise(d, n = dplyr::nth(x, 3, na = TRUE)))
  nr_check(df, function(d) summarise(d, f = dplyr::first(x, na = TRUE)))
})

test_that("summarise(): an NA flag value the native path can't take exactly runs in dplyr", {
  skip_if_no_gpu()
  df <- nr_df()
  nr_check(df, function(d) summarise(d, s = sum(x, na.rm = 1), m = mean(x, na.rm = 0)))
  nr_check(df, function(d) summarise(d, s = sum(x, na.rm = NA)))
  nr_check(df, function(d) summarise(d, s = sum(x, na.rm = "yes")))
  nr_check(df, function(d) summarise(d, s = sum(x, na.rm = c(TRUE, TRUE))))
  nr_check(df, function(d) summarise(d, s = sum(x, na.rm = TRUE, na.rm = FALSE)))
  nr_check(df, function(d) summarise(d, s = dplyr::last(x, na_rm = TRUE, na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = sum(x, na.rm = n() > 0)))
  # a column shadows an environment variable of the same name, as in dplyr
  flag <- TRUE
  nr_check(dplyr::mutate(df, flag = FALSE), function(d) summarise(d, s = sum(x, na.rm = flag)))
  nr_check(dplyr::mutate(df, flag = FALSE), function(d) summarise(d, s = sum(x, na.rm = .data$flag)))
})

test_that("summarise(): each reducer's own spelling stays native", {
  skip_if_no_gpu()
  df <- nr_df()
  for (rm in c(TRUE, FALSE)) {
    nr_native(df, function(d) summarise(d, s = sum(x, na.rm = rm), m = mean(x, na.rm = rm),
                                        mn = min(x, na.rm = rm), mx = max(x, na.rm = rm),
                                        sd = sd(x, na.rm = rm), v = var(x, na.rm = rm),
                                        md = median(x, na.rm = rm), nd = n_distinct(x, na.rm = rm),
                                        a = any(b, na.rm = rm), al = all(b, na.rm = rm)))
    nr_native(df, function(d) summarise(dplyr::group_by(d, g), f = dplyr::first(x, na_rm = rm),
                                        l = dplyr::last(x, na_rm = rm), n = dplyr::nth(x, 1, na_rm = rm)))
  }
  # (names = FALSE: a GPU column can't carry quantile()'s "50%" names)
  nr_native(df, function(d) summarise(d, q = quantile(x, 0.5, na.rm = TRUE, names = FALSE)))
  nr_native(df, function(d) summarise(d, s = sum(x * 2, na.rm = TRUE), m = base::mean(x, na.rm = TRUE)))
  flag <- TRUE
  nr_native(df, function(d) summarise(d, s = sum(x, na.rm = flag), m = mean(x, na.rm = !flag)))
  nr_check(df, function(d) summarise(d, s = sum(x, na.rm = .env$flag)))
})

test_that("colwise and across() with either spelling", {
  skip_if_no_gpu()
  df <- nr_df()
  sat <- dplyr::summarise_at
  nr_check(df, function(d) dplyr::summarise_all(d, sum, na_rm = TRUE))
  nr_check(df, function(d) sat(d, dplyr::vars(x, z), mean, na_rm = TRUE))
  nr_check(df, function(d) sat(d, dplyr::vars(z), list(mx = max, mn = min), na_rm = TRUE))
  nr_check(df, function(d) dplyr::summarise_if(d, is.numeric, sd, na_rm = TRUE))
  nr_check(df, function(d) sat(d, dplyr::vars(x), dplyr::first, na.rm = TRUE))
  nr_check(df, function(d) summarise(d, across(c(x, y), ~ sum(.x, na_rm = TRUE))))
  nr_native(df, function(d) sat(d, dplyr::vars(x, z), mean, na.rm = TRUE))
  # (native only where `last` resolves to dplyr's, i.e. with dplyr attached;
  # the plain summarise(last(x, na_rm =)) native pin is above)
  nr_check(df, function(d) sat(dplyr::group_by(d, g), dplyr::vars(x), dplyr::last, na_rm = TRUE))
  nr_native(df, function(d) summarise(d, across(c(x, y), ~ sum(.x, na.rm = TRUE))))
})

test_that("mutate() and filter() windows with either spelling", {
  skip_if_no_gpu()
  df <- nr_df()
  nr_check(df, function(d) mutate(d, s = sum(x, na_rm = TRUE), m = mean(x, na_rm = TRUE),
                                  mx = max(z, na_rm = TRUE)))
  nr_check(df, function(d) mutate(dplyr::group_by(d, g), s = sum(y, na_rm = TRUE)))
  nr_check(df, function(d) mutate(d, f = dplyr::first(x, na.rm = TRUE)))
  nr_check(df, function(d) filter(d, z < mean(z, na_rm = TRUE)))
  nr_check(df, function(d) mutate(d, s = sum(x, na.rm = 1)))
  nr_native(df, function(d) mutate(d, s = sum(x, na.rm = TRUE), t = sum(y, na.rm = FALSE)))
  nr_native(df, function(d) filter(d, x < sum(x, na.rm = TRUE)))
  nr_native(df, function(d) mutate(dplyr::group_by(d, g), w = dplyr::when_any(b, x > 1, na_rm = TRUE)))
  nr_native(df, function(d) mutate(d, r = roll_mean(x, 2, na.rm = TRUE)))
  nr_check(df, function(d) mutate(d, w = dplyr::when_any(b, x > 1, na.rm = TRUE)))
  # a column shadows an environment variable of the same name, as in dplyr
  flag <- TRUE
  df2 <- dplyr::mutate(df, flag = FALSE)
  nr_check(df2, function(d) mutate(d, s = sum(x, na.rm = flag)))
  nr_check(df2, function(d) mutate(d, w = dplyr::when_any(b, x > 1, na_rm = flag)))
  nr_check(df2, function(d) mutate(d, r = roll_mean(x, 2, na.rm = flag)))
})

test_that("reframe() with either spelling", {
  skip_if_no_gpu()
  df <- nr_df()
  nr_check(df, function(d) reframe(d, r = range(x, na_rm = TRUE)))
  nr_check(df, function(d) reframe(d, q = quantile(x, 0.5, na_rm = TRUE)))
  nr_check(df, function(d) reframe(d, s = sum(y, na_rm = TRUE)))
})

test_that("summarise(): the one argument left must be the data", {
  skip_if_no_gpu()
  df <- nr_df()
  nr_check(df, function(d) summarise(d, s = sum(na_rm = TRUE)))
  nr_check(df, function(d) summarise(d, s = mean(na_rm = x)))
  nr_native(df, function(d) summarise(d, s = sum(x = y), m = mean(x = x, na.rm = TRUE)))
})
