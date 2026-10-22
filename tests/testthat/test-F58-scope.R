# F58: release-scope behaviour beyond the acceptance tests.
#
# Two kinds of scope (src/gpu_scope.cpp): a step scope per wrap_gpu_call(),
# which records the tables its step creates, and a call scope per user-level
# call, which frees on return what the call's steps made and did not return,
# and never hands what it returns to an enclosing scope (R/utils.R).

# A fresh table made by a raw GPU op (no wrapper, so it is recorded by
# whatever step scope is innermost).
f58_raw_copy <- function(a) cuplyr:::gpu_select(a$ptr, 0L)
f58_raw_values <- function(p) cuplyr:::gpu_collect(p, "x")$x

test_that("a released intermediate errors loudly instead of looking empty", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  id <- cuplyr:::gpu_scope_open()
  p <- f58_raw_copy(a)
  cuplyr:::gpu_scope_close(id, NULL)
  expect_error(f58_raw_values(p), "released")
  expect_error(cuplyr:::gpu_dim(p), "released")
  expect_error(collect(cuplyr:::new_tbl_gpu(ptr = p, schema = a$schema)), "released")
  # the input is untouched
  expect_equal(collect(a)$x, 1:5)
})

test_that("a step scope keeps tables reachable from its value: lists, attributes, ALTREP payloads", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  id <- cuplyr:::gpu_scope_open()
  p1 <- f58_raw_copy(a)
  p2 <- f58_raw_copy(a)
  p3 <- f58_raw_copy(a)
  p4 <- f58_raw_copy(a)
  dropped <- f58_raw_copy(a)
  # p3 in an attribute (a list_of's ptype), p4 behind a vec_proxy() row-handle
  # vector (an ALTREP whose payload holds its source table)
  handles <- cuplyr:::gpu_handle_src(1L, 5, list(ptr = p4))
  with_attr <- list()
  attr(with_attr, "first") <- 1L
  attr(with_attr, "ptype") <- list(ptr = p3)
  # a factor with deferred-string levels: a dotted pairlist inside an ALTREP
  lv <- factor(1:3, levels = 1:4)
  cuplyr:::gpu_scope_close(id, list(lv, list(p1), with_attr, list(p2), handles))
  expect_equal(f58_raw_values(p1), 1:5)
  expect_equal(f58_raw_values(p2), 1:5)
  expect_equal(f58_raw_values(p3), 1:5)
  expect_equal(f58_raw_values(p4), 1:5)
  expect_error(f58_raw_values(dropped), "released")
})

test_that("step scopes nest: a kept table moves to the scope below", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  outer <- cuplyr:::gpu_scope_open()
  inner <- cuplyr:::gpu_scope_open()
  p <- f58_raw_copy(a)
  cuplyr:::gpu_scope_close(inner, p)
  expect_equal(f58_raw_values(p), 1:5)
  cuplyr:::gpu_scope_close(outer, NULL)
  expect_error(f58_raw_values(p), "released")
  expect_identical(cuplyr:::gpu_scope_depth(), 0L)
})

test_that("a failing wrapped call leaves no scope open", {
  skip_if_no_gpu()
  expect_error(cuplyr:::wrap_gpu_call("x", stop("boom")), "boom")
  expect_identical(cuplyr:::gpu_scope_depth(), 0L)
  expect_identical(cuplyr:::gpu_wrap_state$depth, 0L)
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  expect_error(dplyr::left_join(a, a, by = "nope"))
  expect_identical(cuplyr:::gpu_scope_depth(), 0L)
})

test_that("an unclosed inner scope is closed with the outer one; an unknown id is a no-op", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  outer <- cuplyr:::gpu_scope_open()
  inner <- cuplyr:::gpu_scope_open()  # never closed (as after an interrupt)
  p <- f58_raw_copy(a)
  cuplyr:::gpu_scope_close(outer, NULL)
  expect_identical(cuplyr:::gpu_scope_depth(), 0L)
  expect_error(f58_raw_values(p), "released")
  expect_no_error(cuplyr:::gpu_scope_close(inner, NULL))
  expect_equal(collect(a)$x, 1:5)
})

test_that("a call scope left open above a closing scope stops tracking, releases nothing", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  outer <- cuplyr:::gpu_scope_open()
  # a call scope whose close never ran (its call returned `p` to the user)
  frame <- new.env()
  cuplyr:::gpu_call_scope_open(frame)
  step <- cuplyr:::gpu_scope_open()
  p <- f58_raw_copy(a)
  cuplyr:::gpu_scope_close(step, p)  # p moves to the call scope
  # an interrupted step of a later call, left open above it
  cuplyr:::gpu_scope_open()
  dropped <- f58_raw_copy(a)
  cuplyr:::gpu_scope_close(outer, NULL)
  expect_identical(cuplyr:::gpu_scope_depth(), 0L)
  expect_equal(f58_raw_values(p), 1:5)
  expect_error(f58_raw_values(dropped), "released")
  # its own close, should it run late, is a no-op
  expect_no_error(cuplyr:::gpu_call_scope_exit(frame, NULL))
  expect_equal(f58_raw_values(p), 1:5)
})

test_that("a subtree shared many times in a returned value is walked once", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  id <- cuplyr:::gpu_scope_open()
  p <- f58_raw_copy(a)
  dropped <- f58_raw_copy(a)
  # 2^30 paths to `p`: seconds without the visited set, microseconds with it
  shared <- list(p)
  for (i in 1:30) shared <- list(shared, shared)
  elapsed <- system.time(cuplyr:::gpu_scope_close(id, shared))[["elapsed"]]
  expect_lt(elapsed, 1)
  expect_equal(f58_raw_values(p), 1:5)
  expect_error(f58_raw_values(dropped), "released")
})

test_that("a table deep inside a returned value (a long lazy plan) is kept", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  id <- cuplyr:::gpu_scope_open()
  p <- f58_raw_copy(a)
  deep <- list(p)
  for (i in 1:500) deep <- list(deep)
  cuplyr:::gpu_scope_close(id, deep)
  expect_equal(f58_raw_values(p), 1:5)
})

test_that("scope ids are doubles that never repeat", {
  id1 <- cuplyr:::gpu_scope_open()
  id2 <- cuplyr:::gpu_scope_open()
  cuplyr:::gpu_scope_close(id1, NULL)
  expect_type(id1, "double")
  expect_gt(id2, id1)
  expect_identical(cuplyr:::gpu_scope_depth(), 0L)
})

# -----------------------------------------------------------------------------
# Ownership: what a user-level call returns is the user's
# -----------------------------------------------------------------------------

test_that("a verb's result is never freed by a scope around it", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  # an enclosing step scope (as when user code runs inside a GPU step)
  id <- cuplyr:::gpu_scope_open()
  t <- dplyr::mutate(a, y = x + 1L)
  u <- tbl_gpu(data.frame(x = 6:7), lazy = FALSE)
  cuplyr:::gpu_scope_close(id, NULL)
  expect_equal(collect(t)$y, 2:6)
  expect_equal(collect(u)$x, 6:7)
})

test_that("tables user code makes through base callbacks or its own helpers stay valid", {
  skip_if_no_gpu()
  df <- data.frame(k = 1:6, x = c(1, 2, 3, 4, 5, 6))
  a <- tbl_gpu(df, lazy = FALSE)
  la <- tbl_gpu(df, lazy = TRUE)
  # lapply() from user code: each compute() is a user-level call
  outs <- lapply(list(dplyr::filter(la, x > 2), dplyr::filter(la, x < 5)), dplyr::compute)
  invisible(gc(full = TRUE))
  expect_equal(collect(outs[[1]])$x, c(3, 4, 5, 6))
  expect_equal(collect(outs[[2]])$x, c(1, 2, 3, 4))
  # Reduce() with a user function: the step results belong to the user too
  dims <- list(tbl_gpu(data.frame(k = 1:6, v1 = 6:1), lazy = FALSE),
               tbl_gpu(data.frame(k = 2:5, v2 = 1:4), lazy = FALSE))
  joined <- Reduce(function(acc, d) dplyr::inner_join(acc, d, by = "k"), dims, a)
  expect_equal(
    tibble::as_tibble(collect(joined)),
    tibble::as_tibble(Reduce(function(acc, d) dplyr::inner_join(acc, d, by = "k"),
                             list(data.frame(k = 1:6, v1 = 6:1), data.frame(k = 2:5, v2 = 1:4)), df))
  )
  # a table built by a user function called from mutate() and from a join's
  # `y` argument while the verb runs
  made <- new.env()
  helper <- function(n) {
    made[[as.character(n)]] <- tbl_gpu(data.frame(k = seq_len(n)), lazy = FALSE)
    n
  }
  m <- dplyr::mutate(a, z = x + helper(3L))
  sj <- dplyr::semi_join(a, made[[as.character(helper(4L))]], by = "k")
  expect_equal(collect(made[["3"]])$k, 1:3)
  expect_equal(collect(made[["4"]])$k, 1:4)
  expect_equal(collect(m)$z, df$x + 3)
  expect_equal(collect(sj)$k, 1:4)
})

test_that("group_split() keeps the ptype it returns in an attribute", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(g = c(1L, 1L, 2L), x = c(1, 2, 3)), lazy = FALSE)
  parts <- dplyr::group_split(dplyr::group_by(a, g))
  ptype <- attr(parts, "ptype")
  invisible(gc(full = TRUE))
  expect_equal(nrow(collect(ptype)), 0L)
  expect_equal(collect(parts[[2]])$x, 3)
})

test_that("vctrs on lazy tables keeps the computed sources it proxies", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 2, 3, 4))
  la <- dplyr::filter(tbl_gpu(df, lazy = TRUE), x > 1)
  expect_equal(collect(vctrs::vec_c(la, la))$x, c(2, 3, 4, 2, 3, 4))
  expect_equal(collect(vctrs::vec_slice(la, c(3L, 1L)))$x, c(4, 2))
})

test_that("multi-step verbs reached from other verbs free their step tables (1e6 rows)", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(1e9)
  set.seed(5858)
  n <- 1e6
  df <- data.frame(g = sample.int(100L, n, replace = TRUE),
                   f = factor(sample(c("a", "b", "c"), n, replace = TRUE), levels = c("a", "b", "c", "d")),
                   h = factor(sample(c("u", "v"), n, replace = TRUE), levels = c("u", "v", "w")),
                   x = runif(n))
  a <- tbl_gpu(df, lazy = FALSE)
  la <- tbl_gpu(df, lazy = TRUE)
  # the input is built before the measurement: a table the user makes and drops
  # inside the measured call is the user's, freed by R's GC
  fh <- dplyr::group_by(dplyr::filter(a, x > 0.1), f, h)
  cases <- list(
    group_trim_two_factors = function() dplyr::group_trim(fh),
    bind_cols_lazy = function() dplyr::bind_cols(dplyr::select(dplyr::mutate(la, y = x * 2), y),
                                                 dplyr::select(dplyr::mutate(la, z = x * 3), z)),
    cuplyr_bind_rows_lazy = function() cuplyr::bind_rows(dplyr::filter(la, x > 0.5), dplyr::filter(la, x < 0.1)),
    sample_n_lazy = function() dplyr::sample_n(dplyr::filter(la, x > 0.5), 10),
    add_count_computed = function() dplyr::add_count(a, gg = g %% 3L)
  )
  used <- function() gpu_memory_state()$pool_used_bytes
  for (nm in names(cases)) {
    gc_gpu()
    res <- cases[[nm]]()
    u1 <- used()
    invisible(gc(full = TRUE))
    invisible(gc(full = TRUE))
    expect_lte(u1 - used(), 512 * 1024, label = paste(nm, "bytes a gc() still released"))
    expect_gt(nrow(collect(res)), 0, label = nm)
    rm(res)
  }
})

# -----------------------------------------------------------------------------
# wrap_gpu_call() rewords what the GPU layer raised, and nothing else
# -----------------------------------------------------------------------------

test_that("a condition R code classed on purpose passes through every wrap unchanged", {
  for (cls in c("cuplyr_summarise_rerun", "cuplyr_error_rowwise_retype", "dplyr_error_join",
                "some_future_cuplyr_class")) {
    err <- expect_error(
      cuplyr:::wrap_gpu_call("op", rlang::abort("deliberate", class = cls)),
      class = cls
    )
    expect_identical(conditionMessage(err), "deliberate")
    expect_false(inherits(err, "cuplyr_error_gpu"))
  }
  expect_error(
    cuplyr:::wrap_gpu_call("outer", cuplyr:::wrap_gpu_call("inner", rlang::abort("x", class = "mine"))),
    class = "mine"
  )
})

test_that("C++ exceptions and plain stop() are reworded; the message is kept", {
  cpp <- structure(class = c("cudf::logic_error", "C++Error", "error", "condition"),
                   list(message = "bad column", call = NULL))
  err <- expect_error(cuplyr:::wrap_gpu_call("op", stop(cpp)), class = "cuplyr_error_gpu")
  expect_match(conditionMessage(err), "GPU operation 'op' failed", fixed = TRUE)
  expect_match(conditionMessage(err), "bad column", fixed = TRUE)
  err <- expect_error(cuplyr:::wrap_gpu_call("op", stop("plain")), class = "cuplyr_error_gpu")
  expect_match(conditionMessage(err), "plain", fixed = TRUE)
})

test_that("an R condition that merely mentions out of memory is not retried", {
  k <- 0L
  expect_error(
    cuplyr:::wrap_gpu_call("op", {
      k <- k + 1L
      rlang::abort("the allocator is out of memory", class = "not_a_gpu_error")
    }),
    class = "not_a_gpu_error"
  )
  expect_identical(k, 1L)
})

test_that("summarise() still reruns an integer-overflowing window sum on the CPU", {
  skip_if_no_gpu()
  M <- .Machine$integer.max
  d <- data.frame(g = c(1, 1, 2, 2), x = c(M - 5L, 5L, 1L, 2L))
  # dplyr warns about the integer overflow of the other group; so does the rerun
  run <- function(...) suppressWarnings(tibble::as_tibble(collect(...)))
  want <- suppressWarnings(dplyr::summarise(d, m = max(x * sum(x))))
  expect_equal(run(dplyr::summarise(tbl_gpu(d, lazy = FALSE), m = max(x * sum(x)))), tibble::as_tibble(want))
  expect_equal(run(dplyr::summarise(tbl_gpu(d, lazy = TRUE), m = max(x * sum(x)))), tibble::as_tibble(want))
  want_by <- suppressWarnings(dplyr::summarise(d, m = max(x * sum(x)), .by = g))
  expect_equal(run(dplyr::summarise(tbl_gpu(d, lazy = FALSE), m = max(x * sum(x)), .by = g)),
               tibble::as_tibble(want_by))
})

test_that("the lazy rowwise retype error keeps its class", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  lazy <- tbl_gpu(tibble::tibble(a = integer(), b = integer()), lazy = TRUE)
  expect_error(collect(dplyr::mutate(dplyr::rowwise(lazy), m = max(c(a, b)))),
               class = "cuplyr_error_rowwise_retype")
})

test_that("an OOM that survives the retry is not swallowed by a rowwise CPU rerun", {
  skip_if_no_gpu()
  withr::defer(gpu_oom_inject(0L))
  rw <- dplyr::rowwise(tbl_gpu(data.frame(a = 1:20, b = 20:1), lazy = FALSE))
  gpu_oom_inject(3L)
  expect_error(dplyr::mutate(rw, m = max(c(a, b))), class = "cuplyr_error_gpu_oom")
  gpu_oom_inject(0L)
  expect_equal(tibble::as_tibble(collect(dplyr::mutate(rw, m = max(c(a, b))))),
               tibble::as_tibble(dplyr::mutate(dplyr::rowwise(data.frame(a = 1:20, b = 20:1)), m = max(c(a, b)))))
})

# -----------------------------------------------------------------------------
# More multi-step verbs free their intermediates on return (1e6 rows, one
# FLOAT64 column is 8 MB; a 512 KB slack absorbs allocator rounding)
# -----------------------------------------------------------------------------

test_that("set operations, rows_*() and joins with an uploaded y free their intermediates", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(1e9)
  set.seed(5858)
  n <- 1e6
  df <- data.frame(g = sample.int(100L, n, replace = TRUE), k = sample.int(1e5L, n, replace = TRUE),
                   x = runif(n))
  dfy <- data.frame(k = 1:1e5, v = runif(1e5))
  a <- tbl_gpu(df, lazy = FALSE)
  ak <- dplyr::select(a, g, k)
  upd <- tibble::tibble(k = 1:10, x = 0)
  cases <- list(
    union = function() dplyr::union(ak, ak),
    intersect = function() dplyr::intersect(ak, ak),
    union_cast = function() dplyr::union(dplyr::select(a, k), tbl_gpu(data.frame(k = c(1.5, 2)), lazy = FALSE)),
    rows_update = function() dplyr::rows_update(dplyr::select(a, k, x), upd, by = "k", unmatched = "ignore", copy = TRUE),
    left_join_df = function() dplyr::left_join(a, dfy, by = "k", copy = TRUE)
  )
  used <- function() gpu_memory_state()$pool_used_bytes
  for (nm in names(cases)) {
    gc_gpu()
    u1 <- NULL
    res <- cases[[nm]]()
    u1 <- used()
    invisible(gc(full = TRUE))
    invisible(gc(full = TRUE))
    expect_lte(u1 - used(), 512 * 1024, label = paste(nm, "bytes a gc() still released"))
    expect_gt(nrow(collect(res)), 0, label = nm)
    rm(res)
  }
})

# A warning raised in one node of a plan whose later node runs out of memory:
# the retry runs the first node again, and its warning must not reach the user
# twice. The OOM is armed from the warning handler, so it hits the step that is
# already running.
f58_arm_on_first_warning <- function(f) {
  n_warn <- 0L
  armed <- FALSE
  res <- withCallingHandlers(f(), warning = function(w) {
    n_warn <<- n_warn + 1L
    if (!armed) {
      armed <<- TRUE
      gpu_oom_inject(1L)
    }
    invokeRestart("muffleWarning")
  })
  list(res = res, n_warn = n_warn, remaining = gpu_oom_inject_remaining())
}

test_that("a retried step does not signal its first attempt's warnings again", {
  skip_if_no_gpu()
  withr::defer(gpu_oom_inject(0L))
  df <- data.frame(x = c(4, 3, 2, 1), k = c(1L, 2L, 3L, 4L))
  big <- 2147483647L
  # one GPU step (compute()) running a mutate node that overflows, then an
  # arrange node that allocates
  la <- tbl_gpu(df, lazy = TRUE)
  out <- f58_arm_on_first_warning(function() dplyr::compute(dplyr::arrange(dplyr::mutate(la, y = k * big), x)))
  gpu_oom_inject(0L)
  expect_identical(out$remaining, 0L)
  expect_identical(out$n_warn, 1L)
  want <- suppressWarnings(dplyr::arrange(dplyr::mutate(df, y = k * big), x))
  expect_equal(tibble::as_tibble(collect(out$res)), tibble::as_tibble(want))

  # an eager multi-step verb: the computed key's step warns, a later step OOMs
  a <- tbl_gpu(df, lazy = FALSE)
  out <- f58_arm_on_first_warning(function() dplyr::distinct(a, kk = k * big))
  gpu_oom_inject(0L)
  expect_identical(out$remaining, 0L)
  expect_identical(out$n_warn, 1L)
  want <- suppressWarnings(dplyr::distinct(df, kk = k * big))
  expect_equal(tibble::as_tibble(collect(out$res)), tibble::as_tibble(want))
})

test_that("compute() passes lowering errors through unworded, and still retries an OOM", {
  err <- expect_error(cuplyr:::wrap_gpu_call("compute", reword = FALSE, stop("lowering said no")))
  expect_identical(class(err), c("simpleError", "error", "condition"))
  expect_identical(conditionMessage(err), "lowering said no")
  k <- 0L
  out <- cuplyr:::wrap_gpu_call("compute", reword = FALSE, {
    k <- k + 1L
    if (k == 1L) stop("std::bad_alloc: out_of_memory")
    "ok"
  })
  expect_identical(out, "ok")
  expect_identical(k, 2L)
})

test_that("a fallback notice raised as an error is not worded as a GPU failure", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  a <- tbl_gpu(data.frame(x = 1:4, g = c(1L, 1L, 2L, 2L)), lazy = FALSE)
  err <- expect_error(dplyr::filter(a, x > mean(vapply(x, function(z) z, numeric(1)))),
                      class = "cuplyr_error_fallback")
  expect_match(conditionMessage(err), "^cuplyr: ")
})
