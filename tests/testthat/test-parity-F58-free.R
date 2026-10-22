# F58: an eager verb (and compute() of a lazy plan) frees its intermediates
# when it returns, instead of leaving them to R's garbage collector.
#
# Contract (scratchpad/parity/features/F58.md, Design):
# - Right after the call returns, with the result still referenced, a full
#   gc() releases (almost) nothing more at the allocator: whatever the call
#   built besides its result is already gone.
# - Dropping the result and collecting frees everything the call allocated
#   (nothing parked in a cache or registry).
# - Nothing the user still holds is freed: inputs, lazy sources, self-joins,
#   verbs that hand back their input table, results that share input columns.
#
# Measured at the allocator (`pool_used_bytes`, helper-memory.R), which is
# exact under cuplyr's pool. One FLOAT64 column at 1e6 rows is 8 MB; every
# intermediate any of these verbs builds is at least one column, so a 512 KB
# slack only absorbs allocator rounding and tiny metadata buffers.

f58_slack <- 512 * 1024

f58_mem_df <- function(n = 1e6) {
  set.seed(5858)
  data.frame(
    g = sample.int(100L, n, replace = TRUE),
    k = sample.int(1e5L, n, replace = TRUE),
    x = runif(n),
    y = runif(n)
  )
}

f58_mem_dfy <- function() {
  set.seed(5859)
  data.frame(k = 1:1e5, v = runif(1e5))
}

# Run `f()` and report what it holds at the allocator once it has returned,
# what a full gc() then releases while the result is still alive, and what is
# left over after the result is dropped too.
f58_measure <- function(f) {
  gc_gpu()
  u0 <- gpu_alloc_used_bytes()
  res <- f()
  u1 <- gpu_alloc_used_bytes()
  invisible(gc(full = TRUE))
  invisible(gc(full = TRUE))
  u2 <- gpu_alloc_used_bytes()
  out <- tibble::as_tibble(collect(res))
  rm(res)
  invisible(gc(full = TRUE))
  invisible(gc(full = TRUE))
  u3 <- gpu_alloc_used_bytes()
  list(held = u1 - u0, released_by_gc = u1 - u2, leaked = u3 - u0, out = out)
}

f58_expect_freed <- function(m, label) {
  expect_lte(m$released_by_gc, f58_slack, label = paste(label, "bytes a gc() still released"))
  expect_lte(m$leaked, f58_slack, label = paste(label, "bytes left after dropping the result"))
}

# `bk`: `b` with a double key, built before the measurement (the join must free
# its own cast copies, not the caller's temporaries).
f58_mem_cases <- function(bk = NULL) {
  force(bk)
  list(
    filter = list(
      gpu = function(a, b) dplyr::filter(a, x > 0.5),
      cpu = function(df, dfy) dplyr::filter(df, x > 0.5)
    ),
    filter_2pred = list(
      gpu = function(a, b) dplyr::filter(a, x > 0.5, y < 0.5),
      cpu = function(df, dfy) dplyr::filter(df, x > 0.5, y < 0.5)
    ),
    mutate = list(
      gpu = function(a, b) dplyr::mutate(a, z = x + y),
      cpu = function(df, dfy) dplyr::mutate(df, z = x + y)
    ),
    mutate_chained_exprs = list(
      gpu = function(a, b) dplyr::mutate(a, z = x + y, w = z * 2),
      cpu = function(df, dfy) dplyr::mutate(df, z = x + y, w = z * 2)
    ),
    mutate_vector_literal = list(
      gpu = function(a, b) dplyr::mutate(a, i = seq_len(1e6)),
      cpu = function(df, dfy) dplyr::mutate(df, i = seq_len(1e6))
    ),
    mutate_window = list(
      gpu = function(a, b) dplyr::mutate(a, z = cumsum(x)),
      cpu = function(df, dfy) dplyr::mutate(df, z = cumsum(x))
    ),
    mutate_grouped_window = list(
      gpu = function(a, b) dplyr::mutate(dplyr::group_by(a, g), z = x - mean(x)),
      cpu = function(df, dfy) dplyr::mutate(dplyr::group_by(df, g), z = x - mean(x))
    ),
    summarise_grouped = list(
      gpu = function(a, b) dplyr::summarise(dplyr::group_by(a, g), m = mean(x), s = sum(y), n = dplyr::n()),
      cpu = function(df, dfy) dplyr::summarise(dplyr::group_by(df, g), m = mean(x), s = sum(y), n = dplyr::n())
    ),
    summarise_expr = list(
      gpu = function(a, b) dplyr::summarise(dplyr::group_by(a, g), m = mean(x * y)),
      cpu = function(df, dfy) dplyr::summarise(dplyr::group_by(df, g), m = mean(x * y))
    ),
    arrange = list(
      gpu = function(a, b) dplyr::arrange(a, g, dplyr::desc(x)),
      cpu = function(df, dfy) dplyr::arrange(df, g, dplyr::desc(x))
    ),
    left_join = list(
      gpu = function(a, b) dplyr::left_join(a, b, by = "k"),
      cpu = function(df, dfy) dplyr::left_join(df, dfy, by = "k")
    ),
    inner_join = list(
      gpu = function(a, b) dplyr::inner_join(a, b, by = "k"),
      cpu = function(df, dfy) dplyr::inner_join(df, dfy, by = "k")
    ),
    left_join_key_cast = list(
      gpu = function(a, b) dplyr::left_join(a, bk, by = "k"),
      cpu = function(df, dfy) dplyr::left_join(df, dplyr::mutate(dfy, k = as.numeric(k)), by = "k")
    ),
    semi_join = list(
      gpu = function(a, b) dplyr::semi_join(a, b, by = "k"),
      cpu = function(df, dfy) dplyr::semi_join(df, dfy, by = "k")
    ),
    distinct = list(
      gpu = function(a, b) dplyr::distinct(a, g, k),
      cpu = function(df, dfy) dplyr::distinct(df, g, k)
    ),
    slice = list(
      gpu = function(a, b) dplyr::slice(a, 1:5e5),
      cpu = function(df, dfy) dplyr::slice(df, 1:5e5)
    ),
    slice_max_grouped = list(
      gpu = function(a, b) dplyr::slice_max(dplyr::group_by(a, g), x, n = 1),
      cpu = function(df, dfy) dplyr::slice_max(dplyr::group_by(df, g), x, n = 1)
    ),
    count = list(
      gpu = function(a, b) dplyr::count(a, g),
      cpu = function(df, dfy) dplyr::count(df, g)
    ),
    bind_rows = list(
      gpu = function(a, b) dplyr::bind_rows(a, a),
      cpu = function(df, dfy) dplyr::bind_rows(df, df)
    ),
    # Verbs that run several steps from R (round-1 review B2).
    filter_grouped_window = list(
      gpu = function(a, b) dplyr::filter(dplyr::group_by(a, g), x > mean(x)),
      cpu = function(df, dfy) dplyr::filter(dplyr::group_by(df, g), x > mean(x))
    ),
    summarise_by = list(
      gpu = function(a, b) dplyr::summarise(a, m = mean(x), .by = g),
      cpu = function(df, dfy) dplyr::summarise(df, m = mean(x), .by = g)
    ),
    distinct_computed_key = list(
      gpu = function(a, b) dplyr::distinct(a, k2 = k %% 10L),
      cpu = function(df, dfy) dplyr::distinct(df, k2 = k %% 10L)
    ),
    count_computed_key = list(
      gpu = function(a, b) dplyr::count(a, gg = g %% 2L),
      cpu = function(df, dfy) dplyr::count(df, gg = g %% 2L)
    ),
    bind_rows_fill = list(
      gpu = function(a, b) dplyr::bind_rows(a, b),
      cpu = function(df, dfy) dplyr::bind_rows(df, dfy)
    ),
    # Multi-step code reached from verbs other than the ones round 2 scoped
    # (round-2 review B2).
    mutate_cur_group_id = list(
      gpu = function(a, b) dplyr::mutate(dplyr::group_by(a, g), id = dplyr::cur_group_id() * 2L),
      cpu = function(df, dfy) dplyr::mutate(dplyr::group_by(df, g), id = dplyr::cur_group_id() * 2L)
    ),
    mutate_by_cur_group_id = list(
      gpu = function(a, b) dplyr::mutate(a, id = dplyr::cur_group_id() * 2L, .by = g),
      cpu = function(df, dfy) dplyr::mutate(df, id = dplyr::cur_group_id() * 2L, .by = g)
    ),
    transmute_cur_group_id = list(
      gpu = function(a, b) dplyr::transmute(dplyr::group_by(a, g), id = dplyr::cur_group_id() * 2L),
      cpu = function(df, dfy) dplyr::transmute(dplyr::group_by(df, g), id = dplyr::cur_group_id() * 2L)
    ),
    reframe_by_range = list(
      gpu = function(a, b) dplyr::reframe(a, r = range(x), .by = g),
      cpu = function(df, dfy) dplyr::reframe(df, r = range(x), .by = g)
    ),
    reframe_by_quantile = list(
      gpu = function(a, b) dplyr::reframe(a, q = stats::quantile(x, c(0.25, 0.75)), .by = g),
      cpu = function(df, dfy) dplyr::reframe(df, q = stats::quantile(x, c(0.25, 0.75)), .by = g)
    )
  )
}

test_that("eager verbs free their intermediates when they return", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(2e9)
  df <- f58_mem_df()
  dfy <- f58_mem_dfy()
  a <- tbl_gpu(df, lazy = FALSE)
  b <- tbl_gpu(dfy, lazy = FALSE)
  cases <- f58_mem_cases(bk = dplyr::mutate(b, k = as.numeric(k)))
  for (nm in names(cases)) {
    m <- f58_measure(function() cases[[nm]]$gpu(a, b))
    f58_expect_freed(m, nm)
    expect_equal(m$out, tibble::as_tibble(cases[[nm]]$cpu(df, dfy)), label = nm)
  }
  # The inputs survived every call.
  expect_equal(tibble::as_tibble(collect(a)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(b)), tibble::as_tibble(dfy))
})

test_that("compute() of a lazy pipeline frees its intermediates", {
  skip_if_no_gpu()
  skip_if_insufficient_gpu_memory(2e9)
  df <- f58_mem_df()
  dfy <- f58_mem_dfy()
  la <- tbl_gpu(df, lazy = TRUE)
  lb <- tbl_gpu(dfy, lazy = TRUE)

  three <- function(x) x |> dplyr::filter(x > 0.5) |> dplyr::mutate(z = x + y) |> dplyr::arrange(z)
  m <- f58_measure(function() dplyr::compute(three(la)))
  f58_expect_freed(m, "lazy filter/mutate/arrange")
  expect_equal(m$out, tibble::as_tibble(three(df)))

  grouped <- function(x) x |> dplyr::filter(x > 0.2) |> dplyr::group_by(g) |> dplyr::summarise(m = mean(x))
  m <- f58_measure(function() dplyr::compute(grouped(la)))
  f58_expect_freed(m, "lazy filter/summarise")
  expect_equal(m$out, tibble::as_tibble(grouped(df)))

  joined <- function(x, y) x |> dplyr::mutate(z = x * 2) |> dplyr::left_join(y, by = "k") |> dplyr::filter(z > 1)
  m <- f58_measure(function() dplyr::compute(joined(la, lb)))
  f58_expect_freed(m, "lazy mutate/left_join/filter")
  expect_equal(m$out, tibble::as_tibble(joined(df, dfy)))

  # Eager verbs that compute() a lazy input first (round-2 review B2): the
  # computed input is an intermediate of the call too.
  m <- f58_measure(function() dplyr::bind_rows(dplyr::filter(la, x > 0.5), dplyr::filter(la, x < 0.1)))
  f58_expect_freed(m, "bind_rows of lazy inputs")
  expect_equal(m$out, tibble::as_tibble(dplyr::bind_rows(dplyr::filter(df, x > 0.5), dplyr::filter(df, x < 0.1))))

  m <- f58_measure(function() dplyr::slice_sample(dplyr::filter(la, x > 0.5), n = 10))
  f58_expect_freed(m, "slice_sample of a lazy input")
  expect_equal(nrow(m$out), 10L)
  expect_true(all(m$out$x > 0.5))

  # The lazy sources and their pending plans still work.
  expect_equal(tibble::as_tibble(collect(la)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(three(la))), tibble::as_tibble(three(df)))
})

# -----------------------------------------------------------------------------
# Never free what the user still holds
# -----------------------------------------------------------------------------

test_that("verbs that hand back (or share) their input never free it", {
  skip_if_no_gpu()
  df <- f58_mem_df(2000)
  a <- tbl_gpu(df, lazy = FALSE)

  r1 <- dplyr::filter(a, TRUE)
  r2 <- dplyr::rename(a, xx = x)
  r3 <- dplyr::select(a, x, g)
  r4 <- dplyr::arrange(a)
  r5 <- dplyr::group_by(a, g)
  r6 <- dplyr::mutate(a)
  r7 <- dplyr::relocate(a, x)
  invisible(gc(full = TRUE))

  expect_equal(tibble::as_tibble(collect(a)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(r1)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(r2)), tibble::as_tibble(dplyr::rename(df, xx = x)))
  expect_equal(tibble::as_tibble(collect(r3)), tibble::as_tibble(dplyr::select(df, x, g)))
  expect_equal(tibble::as_tibble(collect(r4)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(dplyr::ungroup(r5))), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(r6)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(r7)), tibble::as_tibble(dplyr::relocate(df, x)))

  # A chain of verbs on results that share columns with `a`.
  r8 <- a |> dplyr::select(x, g) |> dplyr::mutate(z = x + 1) |> dplyr::rename(gg = g)
  rm(r1, r2, r3, r4, r5, r6, r7)
  invisible(gc(full = TRUE))
  expect_equal(tibble::as_tibble(collect(r8)),
               tibble::as_tibble(df |> dplyr::select(x, g) |> dplyr::mutate(z = x + 1) |> dplyr::rename(gg = g)))
  expect_equal(tibble::as_tibble(collect(a)), tibble::as_tibble(df))
})

test_that("self-joins and joins of lazy tables never free a source", {
  skip_if_no_gpu()
  df <- f58_mem_df(2000)
  dfy <- data.frame(k = 1:500, v = seq(0, 1, length.out = 500))
  a <- tbl_gpu(df, lazy = FALSE)
  b <- tbl_gpu(dfy, lazy = FALSE)

  self <- dplyr::inner_join(dplyr::select(a, k, x), dplyr::select(a, k, y), by = "k",
                            relationship = "many-to-many")
  expect_equal(nrow(collect(self)),
               nrow(dplyr::inner_join(dplyr::select(df, k, x), dplyr::select(df, k, y), by = "k",
                                      relationship = "many-to-many")))

  la <- tbl_gpu(df, lazy = TRUE)
  lb <- tbl_gpu(dfy, lazy = TRUE)
  plan <- la |> dplyr::left_join(lb, by = "k") |> dplyr::semi_join(lb, by = "k")
  first <- dplyr::compute(plan)
  second <- dplyr::compute(plan)
  invisible(gc(full = TRUE))
  expected <- df |> dplyr::left_join(dfy, by = "k") |> dplyr::semi_join(dfy, by = "k")
  expect_equal(tibble::as_tibble(collect(first)), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(collect(second)), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(collect(la)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(lb)), tibble::as_tibble(dfy))
  expect_equal(tibble::as_tibble(collect(a)), tibble::as_tibble(df))
  expect_equal(tibble::as_tibble(collect(b)), tibble::as_tibble(dfy))
})

test_that("intermediate lazy tables the user holds stay valid", {
  skip_if_no_gpu()
  df <- f58_mem_df(2000)
  la <- tbl_gpu(df, lazy = TRUE)
  l1 <- dplyr::filter(la, x > 0.3)
  l2 <- dplyr::mutate(l1, z = x * y)
  c1 <- dplyr::compute(l1)
  c2 <- dplyr::compute(l2)
  c3 <- dplyr::compute(dplyr::arrange(c1, x))
  invisible(gc(full = TRUE))
  e1 <- dplyr::filter(df, x > 0.3)
  expect_equal(tibble::as_tibble(collect(c1)), tibble::as_tibble(e1))
  expect_equal(tibble::as_tibble(collect(c2)), tibble::as_tibble(dplyr::mutate(e1, z = x * y)))
  expect_equal(tibble::as_tibble(collect(c3)), tibble::as_tibble(dplyr::arrange(e1, x)))
  expect_equal(tibble::as_tibble(collect(l2)), tibble::as_tibble(dplyr::mutate(e1, z = x * y)))
  expect_equal(tibble::as_tibble(collect(la)), tibble::as_tibble(df))
})

# Tables that user code creates while a verb runs (an argument assigned inline,
# a memoised upload, a table built inside a predicate) belong to the user, even
# though they were created during the verb's call (round-2 review B4).
test_that("tables user code creates during a verb call stay valid", {
  skip_if_no_gpu()
  dim_df <- data.frame(k = 1:3, v = c(10, 20, 30))
  adf <- data.frame(k = c(1L, 2L, 2L, 3L), x = c(1, 2, 3, 4))
  a <- tbl_gpu(adf, lazy = FALSE)

  # (a) y assigned inside the call
  r1 <- dplyr::left_join(a, b <- tbl_gpu(dim_df, lazy = FALSE), by = "k")
  invisible(gc(full = TRUE))
  expect_equal(tibble::as_tibble(collect(b)), tibble::as_tibble(dim_df))
  expect_equal(tibble::as_tibble(collect(r1)), tibble::as_tibble(dplyr::left_join(adf, dim_df, by = "k")))

  r2 <- dplyr::semi_join(a, s <- tbl_gpu(dim_df[1:2, ], lazy = FALSE), by = "k")
  expect_equal(tibble::as_tibble(collect(s)), tibble::as_tibble(dim_df[1:2, ]))
  expect_equal(tibble::as_tibble(collect(r2)), tibble::as_tibble(dplyr::semi_join(adf, dim_df[1:2, ], by = "k")))

  # (b) a memoised y, first built inside a join, reused by the next call
  cache <- new.env()
  dim_tbl <- function() {
    if (is.null(cache$d)) cache$d <- tbl_gpu(dim_df, lazy = FALSE)
    cache$d
  }
  expected <- tibble::as_tibble(dplyr::inner_join(adf, dim_df, by = "k"))
  expect_equal(tibble::as_tibble(collect(dplyr::inner_join(a, dim_tbl(), by = "k"))), expected)
  expect_equal(tibble::as_tibble(collect(dplyr::inner_join(a, dim_tbl(), by = "k"))), expected)
  expect_equal(tibble::as_tibble(collect(dim_tbl())), tibble::as_tibble(dim_df))

  # (c) a set operation
  a2df <- data.frame(k = 9L, x = 9)
  u <- dplyr::union(a, a2 <- tbl_gpu(a2df, lazy = FALSE))
  expect_equal(tibble::as_tibble(collect(a2)), tibble::as_tibble(a2df))
  expect_equal(tibble::as_tibble(collect(u)), tibble::as_tibble(dplyr::union(adf, a2df)))

  # (d) a memoised table first built while filter()/mutate() evaluate an expression
  cache2 <- new.env()
  keys_tbl <- function() {
    if (is.null(cache2$d)) cache2$d <- tbl_gpu(data.frame(kk = c(1L, 3L)), lazy = FALSE)
    cache2$d
  }
  f1 <- dplyr::filter(a, k %in% dplyr::pull(keys_tbl(), kk))
  expect_equal(tibble::as_tibble(collect(keys_tbl())), tibble::tibble(kk = c(1L, 3L)))
  expect_equal(tibble::as_tibble(collect(f1)), tibble::as_tibble(dplyr::filter(adf, k %in% c(1L, 3L))))

  cache3 <- new.env()
  keys_tbl3 <- function() {
    if (is.null(cache3$d)) cache3$d <- tbl_gpu(data.frame(kk = c(2L, 3L)), lazy = FALSE)
    cache3$d
  }
  m1 <- dplyr::mutate(dplyr::group_by(a, k), hit = k %in% dplyr::pull(keys_tbl3(), kk), id = dplyr::cur_group_id())
  expect_equal(tibble::as_tibble(collect(keys_tbl3())), tibble::tibble(kk = c(2L, 3L)))
  expect_equal(
    tibble::as_tibble(collect(dplyr::ungroup(m1))),
    tibble::as_tibble(dplyr::ungroup(dplyr::mutate(dplyr::group_by(adf, k), hit = k %in% c(2L, 3L), id = dplyr::cur_group_id())))
  )

  # (e) a lazy plan whose y was created inline, computed later
  la <- tbl_gpu(adf, lazy = TRUE)
  lres <- dplyr::left_join(la, b2 <- tbl_gpu(dim_df, lazy = FALSE), by = "k")
  c1 <- dplyr::compute(lres)
  invisible(gc(full = TRUE))
  expect_equal(tibble::as_tibble(collect(b2)), tibble::as_tibble(dim_df))
  expect_equal(tibble::as_tibble(collect(c1)), tibble::as_tibble(dplyr::left_join(adf, dim_df, by = "k")))
  expect_equal(tibble::as_tibble(collect(lres)), tibble::as_tibble(dplyr::left_join(adf, dim_df, by = "k")))

  expect_equal(tibble::as_tibble(collect(a)), tibble::as_tibble(adf))
})
