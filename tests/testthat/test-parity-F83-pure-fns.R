# F83 acceptance: more deterministic base/stats functions count as pure, so a
# grouped verb folds a column-free call to them (qnorm(.975), choose(5, 2),
# class(1)) on the GPU instead of sending the whole call to dplyr. And a `wt`
# column named like a mask function (pick, c_across, cur_*) is a column.
#
# Note on the brief's example, summarise(g, mean(x) - qnorm(.975) * sd(x) /
# sqrt(n())): summarise() runs one reducer per expression, so arithmetic around
# reducers is the CPU's whether or not qnorm() is pure (summarise(mean(x) + 1)
# falls back too). The same expression is native in a grouped mutate(), tested
# below; the summarise form is only compared with dplyr.
#
# Tests marked "guard" already pass on dev: the feature must not break them.

f83_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L),
    x = c(1, 2, 3, NA, 5, 6),
    i = 1:6
  )
}

f83_mutate <- function(cl) {
  eval(bquote(function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * .(cl))))
}

# --- the families -------------------------------------------------------------

f83_dist_args <- list(
  norm = list(), t = list(3), chisq = list(2), f = list(2, 3), binom = list(3, 0.5),
  pois = list(3), exp = list(1), gamma = list(2), beta = list(2, 3), unif = list(),
  lnorm = list(), geom = list(0.5), hyper = list(5, 5, 3), nbinom = list(3, 0.5),
  logis = list(), cauchy = list(), weibull = list(2), signrank = list(5), wilcox = list(3, 4)
)
f83_dist_calls <- function() {
  out <- list()
  for (d in names(f83_dist_args)) {
    args <- f83_dist_args[[d]]
    out[[paste0("d", d)]] <- as.call(c(as.name(paste0("d", d)), list(1), args))
    out[[paste0("p", d)]] <- as.call(c(as.name(paste0("p", d)), list(1), args))
    out[[paste0("q", d)]] <- as.call(c(as.name(paste0("q", d)), list(0.5), args))
  }
  out$ptukey <- quote(ptukey(1, 3, 10))
  out$qtukey <- quote(qtukey(0.5, 3, 10))
  out
}
f83_math_calls <- alist(
  sign = sign(-2), gamma = gamma(3), lgamma = lgamma(3), digamma = digamma(1),
  trigamma = trigamma(1), beta = beta(2, 3), lbeta = lbeta(2, 3), choose = choose(5, 2),
  lchoose = lchoose(5, 2), factorial = factorial(3), lfactorial = lfactorial(3)
)
f83_helper_calls <- alist(
  identical = as.integer(identical(1, 1)), NROW = NROW(1:3), NCOL = NCOL(1:3),
  seq.int = seq.int(2, 2), is.data.frame = as.integer(is.data.frame(1)),
  is.list = as.integer(is.list(1))
)

test_that("F83: every new name is on the allowlist and resolves to the real function", {
  skip_if_no_gpu()
  names <- c(names(f83_dist_calls()), names(f83_math_calls), names(f83_helper_calls))
  expect_equal(length(names), 59L + 11L + 6L)
  expect_identical(setdiff(names, cuplyr:::ir_pure_names()), character())
  resolves <- vapply(names, cuplyr:::ir_head_is_pure, logical(1), env = globalenv())
  expect_identical(names[!resolves], character())
})

test_that("F83: random, clock and higher-order functions are never on the allowlist (guard)", {
  skip_if_no_gpu()
  never <- c(
    "rnorm", "rt", "rchisq", "rf", "rbinom", "rpois", "rexp", "rgamma", "rbeta", "runif",
    "rlnorm", "rgeom", "rhyper", "rnbinom", "rlogis", "rcauchy", "rweibull", "rsignrank",
    "rwilcox", "rmultinom", "sample", "sample.int", "set.seed", "Sys.time", "date",
    "sapply", "lapply", "vapply", "mapply", "Map", "Reduce", "Filter", "do.call", "match.fun",
    "all.equal", # all.equal.numeric() takes `formatFUN`
    # can fold to a string, which reaches native ops that do not guard a string
    # operand: left off until folded strings are guarded
    "class", "typeof", "mode", "storage.mode", "abbreviate", "unname"
  )
  expect_identical(intersect(never, cuplyr:::ir_pure_names()), character())
})

test_that("F83: the mean/qnorm interval expression is native in a grouped mutate()", {
  skip_if_no_gpu()
  df <- f83_df()
  f48_native(df, function(d) {
    dplyr::mutate(dplyr::group_by(d, g),
      ci = mean(x) - qnorm(0.975) * sd(x) / sqrt(n())
    )
  })
})

test_that("F83: the summarise() interval expression equals dplyr", {
  skip_if_no_gpu()
  df <- f83_df()
  p <- function(d) {
    dplyr::summarise(dplyr::group_by(d, g),
      ci = mean(x, na.rm = TRUE) - qnorm(0.975) * sd(x, na.rm = TRUE) / sqrt(n())
    )
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("F83: each distribution function folds in a grouped mutate()", {
  skip_if_no_gpu()
  df <- f83_df()
  for (cl in f83_dist_calls()) {
    f48_native(df, f83_mutate(cl))
  }
})

test_that("F83: each math function folds in a grouped mutate()", {
  skip_if_no_gpu()
  df <- f83_df()
  for (cl in f83_math_calls) {
    f48_native(df, f83_mutate(cl))
  }
})

test_that("F83: each helper folds in a grouped mutate()", {
  skip_if_no_gpu()
  df <- f83_df()
  for (cl in f83_helper_calls) {
    f48_native(df, f83_mutate(cl))
  }
})

test_that("F83: families stay native in filter(), summarise(), .by and transmute-free shapes", {
  skip_if_no_gpu()
  df <- f83_df()
  pipes <- list(
    function(d) dplyr::filter(dplyr::group_by(d, g), x > qnorm(0.9)),
    function(d) dplyr::filter(dplyr::group_by(d, g), x > pt(1, 3)),
    function(d) dplyr::filter(d, x > dnorm(0), .by = g),
    function(d) dplyr::filter(dplyr::group_by(d, g), x > choose(2, 1)),
    function(d) dplyr::filter(dplyr::group_by(d, g), x > NROW(1:3)),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x * qnorm(0.9), na.rm = TRUE)),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = mean(x * gamma(3), na.rm = TRUE)),
    function(d) dplyr::summarise(d, v = sum(x * qnorm(0.9), na.rm = TRUE), .by = g),
    function(d) dplyr::summarise(d, v = sum(x * sign(-2), na.rm = TRUE)),
    function(d) dplyr::mutate(d, v = x * pchisq(1, 2), .by = g)
  )
  for (p in pipes) {
    f48_native(df, p)
  }
})

test_that("F83: ungrouped calls stay native (guard)", {
  skip_if_no_gpu()
  df <- f83_df()
  f48_native(df, function(d) dplyr::mutate(d, v = x * qnorm(0.9)))
  f48_native(df, function(d) dplyr::filter(d, x > pbinom(1, 3, 0.5)))
})

test_that("F83: a whole column-free dot keeps its value and type", {
  skip_if_no_gpu()
  df <- f83_df()
  pipes <- list(
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = qnorm(0.9)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = identical(1, 1)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = sign(-2L)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = NROW(1:3))
  )
  for (p in pipes) {
    f48_native(df, p)
  }
})

test_that("F83: integer folds keep integer columns", {
  skip_if_no_gpu()
  df <- f83_df()
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = i + NROW(1:3)))
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = i + sign(-2L)))
})

test_that("F83: NA and empty inputs", {
  skip_if_no_gpu()
  df <- f83_df()
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * qnorm(NA_real_)))
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * sign(NA_real_)))
  f48_native(df[0, ], function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * qnorm(0.9)))
  f48_native(df[0, ], function(d) dplyr::filter(dplyr::group_by(d, g), x > choose(2, 1)))
})

test_that("F83: a column named like a new function does not get in the way", {
  skip_if_no_gpu()
  df <- f83_df()
  df$qnorm <- 2
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * qnorm(0.9)))
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * qnorm))
})

test_that("F83: pkg::fn resolves to the same function", {
  skip_if_no_gpu()
  df <- f83_df()
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * stats::qnorm(0.9)))
  f48_native(df, function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * base::choose(5, 2)))
})

# --- what must keep running in dplyr ------------------------------------------

test_that("F83: a user function shadowing a new name is called, as in dplyr (guard)", {
  skip_if_no_gpu()
  df <- f83_df()
  calls <- 0L
  qnorm <- function(p) {
    calls <<- calls + 1L
    0
  }
  gamma <- function(x) {
    calls <<- calls + 1L
    10
  }
  NROW <- function(x) {
    calls <<- calls + 1L
    100L
  }
  pipes <- list(
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * qnorm(0.9)),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * gamma(3)),
    function(d) dplyr::filter(dplyr::group_by(d, g), x > NROW(1:3)),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x * qnorm(0.9), na.rm = TRUE))
  )
  withr::local_options(cuplyr.fallback = "silent")
  for (p in pipes) {
    want <- p(df)
    want_calls <- calls
    expect_equal(want_calls, 3L)
    for (lazy in c(FALSE, TRUE)) {
      calls <- 0L
      got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))
      expect_identical(calls, 3L, info = sprintf("lazy = %s", lazy))
      expect_equal(dplyr::ungroup(got), dplyr::ungroup(want), ignore_attr = TRUE)
    }
    calls <- 0L
  }
})

test_that("F83: random draws stay on the CPU and match dplyr with a seed (guard)", {
  skip_if_no_gpu()
  df <- f83_df()
  pipes <- list(
    function(d) {
      set.seed(83)
      dplyr::summarise(dplyr::group_by(d, g), v = sum(x * rnorm(1), na.rm = TRUE))
    },
    function(d) {
      set.seed(83)
      dplyr::summarise(dplyr::group_by(d, g), v = mean(x, na.rm = TRUE) + runif(1))
    },
    function(d) {
      set.seed(83)
      dplyr::summarise(dplyr::group_by(d, g), v = sample(1:100, 1))
    },
    function(d) {
      set.seed(83)
      dplyr::mutate(dplyr::group_by(d, g), v = x * runif(1))
    }
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    expect_error(
      withr::with_options(list(cuplyr.fallback = "error"), p(tbl_gpu(df))),
      class = "cuplyr_error_fallback"
    )
  }
})

test_that("F83: a function that takes a function stays on the CPU (guard)", {
  skip_if_no_gpu()
  df <- f83_df()
  calls <- list(
    quote(sapply(1, sqrt)), quote(vapply(1, sqrt, 1)), quote(Reduce(`+`, 1:2)),
    quote(do.call(sum, list(1))), quote(unlist(Map(sqrt, 1)))
  )
  for (cl in calls) {
    p <- f83_mutate(cl)
    expect_same_as_dplyr(df, p)
    expect_error(
      withr::with_options(list(cuplyr.fallback = "error"), p(tbl_gpu(df))),
      class = "cuplyr_error_fallback"
    )
  }
})

test_that("F83: a call reading a column is not folded (guard)", {
  skip_if_no_gpu()
  df <- f83_df()
  p <- function(d) dplyr::mutate(dplyr::group_by(d, g), v = qnorm(x / 10))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

# A fold that warns must leave the warning to dplyr, which reports it per group
# and names the argument. Passes on dev through the CPU route; the allowlist
# must not turn it into one bare "NaNs produced".
test_that("F83: a pure call that warns keeps dplyr's warning", {
  skip_if_no_gpu()
  df <- f83_df()
  pipes <- list(
    mutate_part = function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * qnorm(2)),
    mutate_whole = function(d) dplyr::mutate(dplyr::group_by(d, g), v = qnorm(2)),
    gamma = function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * gamma(-1)),
    filter = function(d) dplyr::filter(dplyr::group_by(d, g), x > qnorm(2)),
    summarise = function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x * qnorm(2))),
    by = function(d) dplyr::mutate(d, v = x * qnorm(2), .by = g),
    transmute = function(d) dplyr::transmute(dplyr::group_by(d, g), v = x * qnorm(2)),
    probs = function(d) {
      dplyr::summarise(dplyr::group_by(d, g), q = quantile(x, dpois(1.5, 1), na.rm = TRUE, names = FALSE))
    },
    type = function(d) {
      dplyr::summarise(dplyr::group_by(d, g),
        q = quantile(x, 0.5, type = 7 + dpois(1.5, 1), na.rm = TRUE, names = FALSE)
      )
    },
    nth = function(d) dplyr::summarise(dplyr::group_by(d, g), v = dplyr::nth(x, 1 + dpois(1.5, 1))),
    top_n = function(d) dplyr::top_n(dplyr::group_by(d, g), 1 + dpois(1.5, 1), x)
  )
  withr::local_options(cuplyr.fallback = "silent")
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    want <- suppressWarnings(p(df))
    for (lazy in c(FALSE, TRUE)) {
      got <- NULL
      expect_warning(
        got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy))),
        "There were 3 warnings",
        class = "rlang_warning"
      )
      expect_equal(dplyr::ungroup(got), dplyr::ungroup(want), ignore_attr = TRUE, info = nm)
    }
  }
})

# The decline re-runs the whole call on the CPU, so it is only allowed when
# every column-free call in it is pure. An ungrouped mutate()/filter() is not
# screened: a sibling that is not pure must still run exactly once.
test_that("F83: a pure warning does not re-run an impure sibling (ungrouped)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  k <- 0L
  bump <- function() {
    k <<- k + 1L
    1
  }
  forms <- list(
    function(d) dplyr::mutate(d, v = x * bump() + dpois(1.5, 1)),
    function(d) dplyr::mutate(d, a = bump(), v = x + dpois(1.5, 1)),
    function(d) dplyr::filter(d, x > bump() + dpois(1.5, 1))
  )
  withr::local_options(cuplyr.fallback = "silent")
  for (p in forms) {
    k <- 0L
    want <- suppressWarnings(p(df))
    expect_identical(k, 1L)
    for (lazy in c(FALSE, TRUE)) {
      k <- 0L
      got <- suppressWarnings(dplyr::collect(p(tbl_gpu(df, lazy = lazy))))
      expect_identical(k, 1L, info = sprintf("lazy = %s", lazy))
      expect_equal(tibble::as_tibble(got), want, ignore_attr = TRUE)
    }
  }
  seeded <- function(d) {
    set.seed(42)
    dplyr::mutate(d, v = x + runif(1) + dpois(1.5, 1))
  }
  want <- suppressWarnings(seeded(df))
  for (lazy in c(FALSE, TRUE)) {
    got <- suppressWarnings(dplyr::collect(seeded(tbl_gpu(df, lazy = lazy))))
    expect_equal(tibble::as_tibble(got), want, ignore_attr = TRUE)
  }
})

test_that("F83: a call that is not pure and warns runs as often as in dplyr", {
  skip_if_no_gpu()
  df <- f83_df()
  k <- 0L
  f <- function() {
    k <<- k + 1L
    warning("from f")
    2
  }
  withr::local_options(cuplyr.fallback = "silent")
  forms <- list(
    list(p = function(d) dplyr::mutate(d, v = x * f()), n = 1L),
    list(p = function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * f()), n = 3L)
  )
  for (form in forms) {
    k <- 0L
    want <- suppressWarnings(form$p(df))
    expect_identical(k, form$n)
    for (lazy in c(FALSE, TRUE)) {
      k <- 0L
      got <- suppressWarnings(dplyr::collect(form$p(tbl_gpu(df, lazy = lazy))))
      expect_identical(k, form$n, info = sprintf("lazy = %s", lazy))
      expect_equal(dplyr::ungroup(got), dplyr::ungroup(want), ignore_attr = TRUE)
    }
  }
})

test_that("F83: a vector-valued fold inside a reducer equals dplyr", {
  skip_if_no_gpu()
  df <- f83_df()
  pipes <- list(
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x * dnorm(c(0, 1)), na.rm = TRUE)),
    function(d) dplyr::summarise(d, v = sum(x * dnorm(c(0, 1)), na.rm = TRUE)),
    function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x * rev(c(1, 2)), na.rm = TRUE))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F83: a string-valued fold in arithmetic keeps dplyr's error", {
  skip_if_no_gpu()
  df <- f83_df()
  withr::local_options(cuplyr.fallback = "silent")
  pipes <- list(
    list(p = function(d) dplyr::mutate(dplyr::group_by(d, g), v = x + class(1)), cl = "dplyr:::mutate_error"),
    list(p = function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * typeof(1)), cl = "dplyr:::mutate_error"),
    list(p = function(d) dplyr::summarise(dplyr::group_by(d, g), v = sum(x + class(1))), cl = "rlang_error")
  )
  for (form in pipes) {
    expect_error(form$p(df), "non-numeric argument", class = form$cl)
    for (lazy in c(FALSE, TRUE)) {
      expect_error(
        dplyr::collect(form$p(tbl_gpu(df, lazy = lazy))),
        "non-numeric argument",
        class = form$cl
      )
    }
  }
})

# Compare an error from the GPU with dplyr's: same leading class and, through
# dplyr's wrapper, the same underlying message.
f83_same_error <- function(df, p, lazy) {
  want <- tryCatch(p(df), error = function(e) e)
  got <- tryCatch(dplyr::collect(p(tbl_gpu(df, lazy = lazy))), error = function(e) e)
  expect_s3_class(want, "error")
  expect_s3_class(got, "error")
  expect_identical(class(got)[[1L]], class(want)[[1L]])
  msg <- function(e) conditionMessage(if (is.null(e$parent)) e else e$parent)
  expect_identical(msg(got), msg(want))
}

test_that("F83: a string-returning call keeps dplyr's error outside arithmetic (guard)", {
  skip_if_no_gpu()
  df <- f83_df()
  withr::local_options(cuplyr.fallback = "silent")
  g <- function(d) dplyr::group_by(d, g)
  pipes <- list(
    function(d) dplyr::mutate(g(d), v = x + sqrt(class(1))),
    function(d) dplyr::mutate(g(d), v = x > 0 & class(1)),
    function(d) dplyr::filter(g(d), x > 0 | class(1)),
    function(d) dplyr::mutate(g(d), v = x * abs(typeof(1))),
    function(d) dplyr::mutate(g(d), v = dplyr::if_else(x > 0, x, class(1))),
    function(d) dplyr::mutate(g(d), v = dplyr::coalesce(x, class(1))),
    function(d) dplyr::mutate(g(d), v = dplyr::case_when(x > 2 ~ class(1), .default = x)),
    function(d) dplyr::mutate(g(d), v = dplyr::if_else(class(1), x, 0)),
    function(d) dplyr::summarise(g(d), v = sum(x * sqrt(class(1))))
  )
  for (p in pipes) {
    f83_same_error(df, p, lazy = FALSE)
    f83_same_error(df, p, lazy = TRUE)
  }
})

test_that("F83: strings already native stay native (guard)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), s = c("numeric", "b", "numeric"))
  label <- "a"
  pipes <- list(
    function(d) dplyr::filter(dplyr::group_by(d, g), s == paste0("num", "eric")),
    function(d) dplyr::filter(dplyr::group_by(d, g), s != toupper("numeric")),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = nchar(toupper("a"))),
    function(d) dplyr::mutate(d, v = dplyr::if_else(g > 1L, toupper("a"), "b")),
    function(d) dplyr::mutate(d, v = dplyr::case_match(s, c("numeric", "x") ~ "g1", .default = "o")),
    function(d) dplyr::mutate(d, v = dplyr::recode_values(s, c("numeric", "x") ~ "g1", default = "o")),
    function(d) dplyr::mutate(d, v = dplyr::replace_values(s, c("b", "x") ~ "z")),
    # a variable read through a pronoun is not a call folding to a string
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = dplyr::if_else(g > 1L, .env$label, "b")),
    function(d) dplyr::mutate(d, v = dplyr::coalesce(s, .env$label))
  )
  for (p in pipes) {
    f48_native(df, p)
  }
})

test_that("F83: an ungrouped top_n() with a pure wt that warns keeps dplyr's warning", {
  skip_if_no_gpu()
  df <- f83_df()
  withr::local_options(cuplyr.fallback = "silent")
  pipes <- list(
    function(d) dplyr::top_n(d, 2, x * dpois(1.5, 1)),
    function(d) dplyr::top_n(d, 2, wt = x * qnorm(2)),
    function(d) dplyr::top_frac(d, 0.5, wt = x * dpois(1.5, 1))
  )
  for (p in pipes) {
    want <- suppressWarnings(p(df))
    for (lazy in c(FALSE, TRUE)) {
      got <- NULL
      expect_warning(
        got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy))),
        "There was 1 warning",
        class = "rlang_warning"
      )
      expect_equal(tibble::as_tibble(got), want, ignore_attr = TRUE)
    }
  }
})

test_that("F83: an ungrouped top_n() with a pure wt and an impure n runs n once", {
  skip_if_no_gpu()
  df <- f83_df()
  k <- 0L
  bump <- function() {
    k <<- k + 1L
    2
  }
  withr::local_options(cuplyr.fallback = "silent")
  p <- function(d) dplyr::top_n(d, bump(), wt = x * qnorm(0.9))
  k <- 0L
  want <- p(df)
  expect_identical(k, 1L)
  for (lazy in c(FALSE, TRUE)) {
    k <- 0L
    got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))
    expect_identical(k, 1L, info = sprintf("lazy = %s", lazy))
    expect_equal(tibble::as_tibble(got), want, ignore_attr = TRUE)
  }
})

test_that("F83: a string-valued call in an unscreened verb runs once (guard)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  k <- 0L
  next_id <- function() {
    k <<- k + 1L
    paste0("id", k)
  }
  withr::local_options(cuplyr.fallback = "silent")
  forms <- list(
    function(d) dplyr::mutate(d, v = dplyr::if_else(x > 0, next_id(), "none")),
    function(d) dplyr::mutate(d, v = dplyr::case_when(x > 0 ~ next_id(), .default = "none")),
    function(d) dplyr::filter(d, next_id() == "id1")
  )
  for (p in forms) {
    k <- 0L
    want <- p(df)
    expect_identical(k, 1L)
    for (lazy in c(FALSE, TRUE)) {
      k <- 0L
      got <- dplyr::collect(p(tbl_gpu(df, lazy = lazy)))
      expect_identical(k, 1L, info = sprintf("lazy = %s", lazy))
      expect_equal(tibble::as_tibble(got), want, ignore_attr = TRUE)
    }
  }
  seeded <- function(d) {
    set.seed(42)
    dplyr::mutate(d, a = x + runif(1), b = dplyr::if_else(x > 0, toupper("a"), "z"))
  }
  want <- seeded(df)
  for (lazy in c(FALSE, TRUE)) {
    got <- dplyr::collect(seeded(tbl_gpu(df, lazy = lazy)))
    expect_equal(tibble::as_tibble(got), want, ignore_attr = TRUE)
  }
})

test_that("F83: a string-returning call on the left of %in% equals dplyr (guard)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), s = c("numeric", "b", "numeric"))
  p <- function(d) dplyr::mutate(dplyr::group_by(d, g), v = class(1) %in% s)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("F83: a pure call that fails keeps dplyr's error (guard)", {
  skip_if_no_gpu()
  df <- f83_df()
  withr::local_options(cuplyr.fallback = "silent")
  pipes <- list(
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = x * qnorm("a")),
    function(d) dplyr::mutate(dplyr::group_by(d, g), v = qnorm("a"))
  )
  for (p in pipes) {
    expect_error(p(df), class = "dplyr:::mutate_error")
    expect_error(dplyr::collect(p(tbl_gpu(df))), class = "dplyr:::mutate_error")
  }
})

# --- the wt column name -------------------------------------------------------

test_that("F83: a wt column named like a mask function stays native", {
  skip_if_no_gpu()
  for (nm in c("pick", "c_across", "cur_group", "cur_group_rows", "cur_data", "cur_data_all")) {
    df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = 1:4)
    df[[nm]] <- c(1, 2, 3, 4)
    wt <- as.name(nm)
    pipes <- list(
      eval(bquote(function(d) dplyr::top_n(d, 2, wt = .(wt)))),
      eval(bquote(function(d) dplyr::top_n(dplyr::group_by(d, g), 1, wt = .(wt)))),
      eval(bquote(function(d) dplyr::top_n(d, 2, wt = .(wt) * 2))),
      eval(bquote(function(d) dplyr::top_frac(d, 0.5, wt = .(wt))))
    )
    for (p in pipes) {
      f48_native(df, p)
    }
  }
})

test_that("F83: a wt that calls a mask function still goes to dplyr (guard)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4))
  pipes <- list(
    function(d) dplyr::top_n(d, 2, wt = rowSums(pick(x))),
    function(d) dplyr::top_n(dplyr::group_by(d, g), 1, wt = dplyr::cur_group_rows())
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    expect_error(
      withr::with_options(list(cuplyr.fallback = "error"), p(tbl_gpu(df))),
      class = "cuplyr_error_fallback"
    )
  }
})
