# F02 acceptance tests: the vctrs contract on tbl_gpu.
#
# Every case runs the same function on plain tibbles (the dplyr/vctrs oracle)
# and on tbl_gpu inputs, eager and lazy. A data-frame result must come back as
# a tbl_gpu (the common type of a tbl_gpu and anything data-frame-like stays on
# the GPU) and match the oracle after collect(); a vector result must be
# identical to the oracle's.

f02_df <- function() {
  data.frame(
    g = c("a", "b", "a", "c"),
    x = c(1L, NA, 3L, 1L),
    y = c(1.5, 2, NA, 1.5),
    f = factor(c("u", "v", "u", NA), levels = c("u", "v", "w")),
    d = as.Date("2024-01-01") + c(0L, 1L, NA, 3L),
    l = c(TRUE, NA, FALSE, TRUE),
    n = c(10, 20, 30, 40),
    stringsAsFactors = FALSE
  )
}

f02_up <- function(df, lazy = FALSE) {
  g <- tbl_gpu(df)
  if (lazy) as_lazy(g) else g
}

# fn(...) on tibbles vs fn(...) on tbl_gpu. Data-frame arguments are uploaded;
# anything else is passed through unchanged.
f02_expect_same <- function(fn, ..., lazy = FALSE) {
  args <- list(...)
  oracle_args <- lapply(args, function(a) if (is.data.frame(a)) tibble::as_tibble(a) else a)
  gpu_args <- lapply(args, function(a) if (is.data.frame(a)) f02_up(a, lazy) else a)

  oracle <- do.call(fn, oracle_args)
  gpu <- do.call(fn, gpu_args)
  label <- if (lazy) "lazy" else "eager"

  if (is.data.frame(oracle)) {
    expect_true(is_tbl_gpu(gpu), info = paste(label, ": result should stay a tbl_gpu"))
    if (!is_tbl_gpu(gpu)) {
      return(invisible())
    }
    expect_identical(dplyr::group_vars(gpu), dplyr::group_vars(oracle), info = label)
    expect_identical(
      tibble::as_tibble(collect(gpu)),
      tibble::as_tibble(dplyr::ungroup(oracle)),
      info = label
    )
  } else {
    expect_identical(gpu, oracle, info = label)
  }
}

f02_both <- function(fn, ...) {
  f02_expect_same(fn, ..., lazy = FALSE)
  f02_expect_same(fn, ..., lazy = TRUE)
}

# Run `expr` with the host<->device copy primitives disabled, so the test
# fails if the operation collects or re-uploads instead of staying on the GPU.
f02_no_transfer <- function(expr) {
  testthat::local_mocked_bindings(
    gpu_collect = function(...) stop("F02: unexpected gpu_collect()"),
    gpu_head = function(...) stop("F02: unexpected gpu_head()"),
    df_to_gpu = function(...) stop("F02: unexpected df_to_gpu()"),
    .package = "cuplyr"
  )
  force(expr)
}

# ---------------------------------------------------------------------------
# size, length, names

test_that("vec_size() and obj_is_vector() treat a tbl_gpu as a data frame", {
  skip_if_no_gpu()
  df <- f02_df()

  for (lazy in c(FALSE, TRUE)) {
    g <- f02_up(df, lazy)
    expect_true(vctrs::obj_is_vector(g))
    expect_identical(vctrs::vec_size(g), 4L)
    expect_identical(vctrs::vec_size(f02_up(df[0, ], lazy)), 0L)
    expect_identical(vctrs::vec_size_common(g, 1L), 4L)
    expect_error(vctrs::vec_size_common(g, 1:3), class = "vctrs_error_incompatible_size")
  }
})

test_that("vec_size() of a lazy tbl_gpu counts rows after pending ops and leaves it lazy", {
  skip_if_no_gpu()
  g <- f02_up(f02_df(), lazy = TRUE) |> dplyr::filter(x > 1L | is.na(x))

  expect_identical(vctrs::vec_size(g), 2L)
  expect_identical(g$exec_mode, "lazy")
  expect_identical(nrow(collect(g)), 2L)
})

test_that("vec_size() of an eager tbl_gpu is metadata-only", {
  skip_if_no_gpu()
  g <- f02_up(f02_df())
  expect_identical(f02_no_transfer(vctrs::vec_size(g)), 4L)
})

test_that("length() is the number of columns, like a data frame", {
  skip_if_no_gpu()
  df <- f02_df()
  expect_identical(length(f02_up(df)), length(df))
  lz <- f02_up(df, lazy = TRUE) |> dplyr::mutate(z = y * 2)
  expect_identical(length(lz), length(df) + 1L)
})

test_that("rlang::set_names() renames a tbl_gpu like a data frame", {
  skip_if_no_gpu()
  df <- f02_df()
  nm <- c("G", "X", "Y", "F", "D", "L", "N")

  f02_both(function(x) rlang::set_names(x, nm), df)
  f02_both(function(x) rlang::set_names(x, toupper), df)
  f02_both(function(x) rlang::set_names(dplyr::group_by(x, g), toupper), df)

  g <- f02_up(df)
  out <- f02_no_transfer(rlang::set_names(g, nm))
  expect_identical(names(out), nm)

  expect_error(rlang::set_names(g, c("a", "b")), "size of `nm`")
})

# ---------------------------------------------------------------------------
# slicing

test_that("vec_slice() matches vctrs on a data frame", {
  skip_if_no_gpu()
  df <- f02_df()

  f02_both(function(x) vctrs::vec_slice(x, c(3, 1, 1)), df)
  f02_both(function(x) vctrs::vec_slice(x, 2L), df)
  f02_both(function(x) vctrs::vec_slice(x, -2L), df)
  f02_both(function(x) vctrs::vec_slice(x, c(TRUE, FALSE, TRUE, FALSE)), df)
  f02_both(function(x) vctrs::vec_slice(x, integer()), df)
  f02_both(function(x) vctrs::vec_slice(x, c(NA, 2L)), df)
  f02_both(function(x) vctrs::vec_slice(x, integer()), df[0, ])
})

test_that("vec_slice() keeps grouping and the input's execution mode", {
  skip_if_no_gpu()
  df <- f02_df()

  f02_both(function(x) vctrs::vec_slice(dplyr::group_by(x, g), c(4, 1)), df)

  lz <- f02_up(df, lazy = TRUE)
  expect_identical(vctrs::vec_slice(lz, 1:2)$exec_mode, "lazy")
  expect_identical(vctrs::vec_slice(f02_up(df), 1:2)$exec_mode, "eager")

  pending <- lz |> dplyr::filter(!is.na(x)) |> dplyr::mutate(z = x + 1L)
  f02_expect_same(
    function(t) vctrs::vec_slice(dplyr::mutate(dplyr::filter(t, !is.na(x)), z = x + 1L), c(3, 1)),
    df, lazy = TRUE
  )
  expect_identical(names(vctrs::vec_slice(pending, 1)), c(names(df), "z"))
})

test_that("vec_slice() gathers on the GPU without collecting or re-uploading", {
  skip_if_no_gpu()
  df <- f02_df()
  g <- f02_up(df)

  out <- f02_no_transfer(vctrs::vec_slice(g, c(4, NA, 2)))
  expect_true(is_tbl_gpu(out))
  expect_identical(collect(out), vctrs::vec_slice(df, c(4, NA, 2)))
})

test_that("vec_slice() raises vctrs' subscript errors", {
  skip_if_no_gpu()
  g <- f02_up(f02_df())
  expect_error(vctrs::vec_slice(g, 10), class = "vctrs_error_subscript_oob")
  expect_error(vctrs::vec_slice(g, "a"), "character names to index")
})

test_that("vec_init(), vec_rep() and vec_ptype() work on a tbl_gpu", {
  skip_if_no_gpu()
  df <- f02_df()

  f02_both(function(x) vctrs::vec_init(x, 2L), df)
  f02_both(function(x) vctrs::vec_rep(x, 2L), df)
  f02_both(function(x) vctrs::vec_ptype(x), df)
})

# ---------------------------------------------------------------------------
# common type and casting

test_that("vec_ptype2() of tbl_gpu and data-frame-likes is a tbl_gpu", {
  skip_if_no_gpu()
  a <- data.frame(x = 1L, s = "p", stringsAsFactors = FALSE)
  b <- data.frame(x = 2.5, z = TRUE)

  for (lazy in c(FALSE, TRUE)) {
    ga <- f02_up(a, lazy)
    gb <- f02_up(b, lazy)
    ta <- tibble::as_tibble(a)
    tb <- tibble::as_tibble(b)

    # (the tbl_gpu result, vctrs' answer for the same inputs as data frames:
    # two plain data.frames give a data.frame, any tibble a tibble)
    for (case in list(
      list(vctrs::vec_ptype2(ga, gb), vctrs::vec_ptype2(a, b)),
      list(vctrs::vec_ptype2(ga, tb), vctrs::vec_ptype2(a, tb)),
      list(vctrs::vec_ptype2(ta, gb), vctrs::vec_ptype2(ta, b)),
      list(vctrs::vec_ptype2(ga, b), vctrs::vec_ptype2(a, b)),
      list(vctrs::vec_ptype2(a, gb), vctrs::vec_ptype2(a, b))
    )) {
      expect_true(is_tbl_gpu(case[[1]]))
      expect_identical(collect(case[[1]]), case[[2]])
    }
  }

  expect_error(
    vctrs::vec_ptype2(f02_up(a), tibble::tibble(x = "chr")),
    class = "vctrs_error_incompatible_type"
  )
})

test_that("vec_ptype2() of grouped tbl_gpus keeps the grouping, like grouped_df", {
  skip_if_no_gpu()
  df <- f02_df()
  f02_both(function(x) vctrs::vec_ptype2(dplyr::group_by(x, g), x), df)
  f02_both(function(x) vctrs::vec_ptype2(x, dplyr::group_by(x, g)), df)
})

test_that("vec_cast() moves between tbl_gpu and tibble with vctrs' rules", {
  skip_if_no_gpu()
  int_df <- data.frame(x = c(1L, 2L))
  dbl_df <- data.frame(x = c(1.5, 2))

  g_int <- f02_up(int_df)
  g_dbl <- f02_up(dbl_df)

  up <- vctrs::vec_cast(tibble::as_tibble(int_df), g_dbl)
  expect_true(is_tbl_gpu(up))
  expect_identical(collect(up), data.frame(x = c(1, 2)))

  down <- vctrs::vec_cast(g_int, tibble::tibble(x = double()))
  expect_false(is_tbl_gpu(down))
  expect_identical(down, tibble::tibble(x = c(1, 2)))

  widened <- vctrs::vec_cast(g_int, g_dbl)
  expect_true(is_tbl_gpu(widened))
  expect_identical(collect(widened), data.frame(x = c(1, 2)))

  added <- vctrs::vec_cast(g_int, f02_up(data.frame(x = 1.5, z = "a")))
  expect_identical(collect(added), data.frame(x = c(1, 2), z = NA_character_))

  expect_error(vctrs::vec_cast(g_dbl, g_int), class = "vctrs_error_cast_lossy")
  expect_error(
    vctrs::vec_cast(f02_up(data.frame(x = 1L, w = 2L)), g_int),
    class = "vctrs_error_cast_lossy_dropped"
  )
})

# ---------------------------------------------------------------------------
# combining and assignment

test_that("vec_c(), list_unchop() and vec_assign() combine tbl_gpu rows", {
  skip_if_no_gpu()
  df <- f02_df()
  other <- f02_df()[c(2, 4), ]
  other$y <- c(10, 20)

  f02_both(function(a, b) vctrs::vec_c(a, b), df, other)
  f02_both(function(a, b) vctrs::list_unchop(list(b, a)), df, other)
  f02_both(function(a, b) vctrs::vec_c(a, NA, b), df, other)
  f02_both(function(a, b) vctrs::vec_c(vctrs::vec_slice(a, 3), vctrs::vec_slice(b, 2), vctrs::vec_slice(a, 1)), df, other)
  f02_both(function(a) vctrs::vec_assign(a, c(1, 3), NA), df)
  f02_both(function(a, b) vctrs::vec_assign(a, c(4, 1), b), df, other)

  # a tbl_gpu with a tibble: the result stays on the GPU
  out <- vctrs::vec_c(f02_up(df), tibble::as_tibble(other))
  expect_true(is_tbl_gpu(out))
  expect_identical(collect(out), vctrs::vec_c(tibble::as_tibble(df), tibble::as_tibble(other)))
})

# ---------------------------------------------------------------------------
# equality, missingness, ordering

test_that("equality and ordering helpers compare tbl_gpu rows by value", {
  skip_if_no_gpu()
  df <- data.frame(
    a = c(1, 1, NA, 2, 1, NA),
    b = c("x", "x", NA, "y", "z", "q"),
    stringsAsFactors = FALSE
  )
  key <- data.frame(a = 1, b = "x", stringsAsFactors = FALSE)

  f02_both(function(x) vctrs::vec_equal(x, tibble::tibble(a = 1, b = "x")), df)
  f02_both(function(x, k) vctrs::vec_equal(x, k), df, key)
  f02_both(function(x) vctrs::vec_detect_missing(x), df)
  f02_both(function(x) vctrs::vec_detect_complete(x), df)
  f02_both(function(x) vctrs::vec_duplicate_detect(x), df)
  f02_both(function(x) vctrs::vec_unique(x), df)
  f02_both(function(x) vctrs::vec_unique_count(x), df)
  f02_both(function(x, k) vctrs::vec_in(x, k), df, key)
  f02_both(function(x) vctrs::vec_order(x), df)
  f02_both(function(x) vctrs::vec_sort(x), df)
})

# ---------------------------------------------------------------------------
# dplyr helpers that take a whole table (the conformance failures)

test_that("coalesce() fills entirely missing tbl_gpu rows", {
  skip_if_no_gpu()
  x <- data.frame(x = c(1, NA), y = c(NA, NA))
  y <- data.frame(x = c(2, NA), y = c(TRUE, TRUE))
  f02_both(function(a, b) dplyr::coalesce(a, b), x, y)

  x2 <- data.frame(p = c(NA, 2L, NA), q = c(NA, "b", "c"), stringsAsFactors = FALSE)
  y2 <- data.frame(p = c(9L, 9L, 9L), q = c("z", "z", "z"), stringsAsFactors = FALSE)
  f02_both(function(a, b) dplyr::coalesce(a, b), x2, y2)
})

test_that("consecutive_id() and n_distinct() accept a tbl_gpu", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, 1, 1, 1, NA), y = c(1, 2, 2, 1, NA))

  f02_both(function(x) dplyr::consecutive_id(x), df)
  f02_both(function(x) dplyr::n_distinct(x), df)
  f02_both(function(x) dplyr::n_distinct(x, na.rm = TRUE), df)
  f02_both(function(x) dplyr::n_distinct(x), df[0, ])
})

test_that("between() compares tbl_gpu rows lexicographically", {
  skip_if_no_gpu()
  x <- data.frame(year = c(2020, 2020, 2021), month = c(1, 3, 6))
  left <- data.frame(year = c(2019, 2020, 2021), month = c(1, 4, 3))
  right <- data.frame(year = c(2020, 2020, 2022), month = c(1, 6, 3))
  f02_both(function(a, l, r) dplyr::between(a, l, r), x, left, right)
})

test_that("if_else(), lag(), lead() and nth() take tbl_gpu values", {
  skip_if_no_gpu()
  true <- data.frame(x = 1, y = 2)
  false <- data.frame(x = 3, y = 4)
  f02_both(function(t, f) dplyr::if_else(c(TRUE, FALSE, NA, TRUE), t, f), true, false)

  df <- data.frame(a = 1:3, b = letters[1:3], stringsAsFactors = FALSE)
  default <- data.frame(a = 0L, b = "", stringsAsFactors = FALSE)
  f02_both(function(x) dplyr::lag(x), df)
  f02_both(function(x) dplyr::lead(x, n = 2L), df)
  f02_both(function(x, d) dplyr::lag(x, n = 2L, default = d), df, default)

  f02_both(function(x) dplyr::nth(x, 2L), df)
  f02_both(function(x) dplyr::nth(x, -1L), df)
  f02_both(function(x) dplyr::nth(x, 10L), df)
  f02_both(function(x, d) dplyr::nth(x, 10L, default = d), df, default)
})

test_that("na_if() and replace_when() work on tbl_gpu rows", {
  skip_if_no_gpu()
  x <- data.frame(x = c(2, 1, NA, 1), y = c(1, NA, NA, NA), z = c(3, NaN, NA, NaN))
  y <- data.frame(x = 1, y = NA, z = NaN)
  f02_both(function(a, b) dplyr::na_if(a, b), x, y)

  w <- data.frame(a = c(1, 2, 3, 1), b = c(2, 3, 4, 2))
  f02_both(function(a) dplyr::replace_when(a, vctrs::vec_equal(a, tibble::tibble(a = 1, b = 2)) ~ NA), w)
})

test_that("ranking helpers and with_order() accept a tbl_gpu", {
  skip_if_no_gpu()
  df <- data.frame(
    year = c(2020, 2020, 2021, 2020, 2020, NA),
    month = c(3, 2, 1, 2, NA, NA)
  )
  f02_both(function(x) dplyr::row_number(x), df)
  f02_both(function(x) dplyr::min_rank(x), df)
  f02_both(function(x) dplyr::dense_rank(x), df)
  f02_both(function(x) dplyr::percent_rank(x), df)
  f02_both(function(x) dplyr::cume_dist(x), df)
  f02_both(function(x) dplyr::ntile(x, 2), df)

  ob <- data.frame(a = c(1, 1, 2), b = c(2, 1, 1))
  f02_both(function(o) dplyr::with_order(o, dplyr::lag, 1:3), ob)
  f02_both(function(o) dplyr::nth(1:3, 1, order_by = o), ob)
})

# ---------------------------------------------------------------------------
# Round-1 review additions (B1, B2)

# Force a full GC on entry to the tbl_gpu proxy and restore methods, so any
# GPU table that is referenced only through row handles inside a vctrs C loop
# would be collected before vctrs restores it.
f02_local_gc_pressure <- function(env = parent.frame()) {
  force(env)
  vctrs_ns <- asNamespace("vctrs")
  generics <- c("vec_proxy", "vec_restore")
  originals <- lapply(generics, function(g) utils::getS3method(g, "tbl_gpu", envir = vctrs_ns))
  # One deferred restore for both methods, registered before anything is
  # wrapped. `generics`/`originals` are never modified after this point.
  withr::defer(
    for (i in seq_along(generics)) {
      registerS3method(generics[[i]], "tbl_gpu", originals[[i]], envir = vctrs_ns)
    },
    envir = env
  )
  for (i in seq_along(generics)) {
    wrapper <- local({
      orig <- originals[[i]]
      # Full collection before restoring (where a dead source would be
      # dereferenced); a cheaper minor collection before each proxy.
      full <- identical(generics[[i]], "vec_restore")
      function(x, ...) {
        gc(full = full)
        orig(x, ...)
      }
    })
    registerS3method(generics[[i]], "tbl_gpu", wrapper, envir = vctrs_ns)
  }
}

test_that("multi-table combines survive garbage collection in the middle of a vctrs call", {
  skip_if_no_gpu()
  f02_local_gc_pressure()

  ints <- data.frame(x = 1:5)
  dbls <- data.frame(x = c(1.5, 2.5))
  df <- f02_df()

  f02_both(function(a, b) vctrs::vec_c(a, b), ints, dbls)
  f02_both(function(a) vctrs::vec_c(a, NA, a), dbls)
  f02_both(function(b) vctrs::vec_c(tibble::tibble(x = c(7L, 8L)), b), ints)
  f02_both(function(a) vctrs::vec_c(vctrs::vec_slice(a, 2:1), vctrs::vec_slice(a, 4)), df)

  x <- data.frame(p = c(NA, 2, NA, 4), q = c(NA, "b", NA, "d"), stringsAsFactors = FALSE)
  y <- data.frame(p = c(9, 9, 9, 9), q = c("z", "z", "z", "z"), stringsAsFactors = FALSE)
  f02_both(function(a, b) dplyr::coalesce(dplyr::filter(a, !is.na(p) | is.na(p)), b), x, y)
  f02_both(function(a) dplyr::lag(dplyr::mutate(a, r = p * 2)), x)
  f02_both(function(a) vctrs::vec_assign(dplyr::filter(a, !is.na(p) | is.na(p)), c(1, 3), NA), x)
})

test_that("the GC-pressure helper put the original vctrs methods back", {
  skip_if_no_gpu()
  methods <- get(".__S3MethodsTable__.", envir = asNamespace("vctrs"))
  ns <- asNamespace("cuplyr")
  expect_identical(methods[["vec_proxy.tbl_gpu"]], ns$vec_proxy.tbl_gpu)
  expect_identical(methods[["vec_restore.tbl_gpu"]], ns$vec_restore.tbl_gpu)
})

test_that("bind_rows() and bind_cols() handle tbl_gpu inputs passed in lists or after NULL", {
  skip_if_no_gpu()
  d1 <- data.frame(x = 1:3, s = c("a", "b", "c"), stringsAsFactors = FALSE)
  d2 <- data.frame(x = 4:6, s = c("d", "e", "f"), stringsAsFactors = FALSE)
  d3 <- data.frame(y = 7:9)

  f02_both(function(a, b) cuplyr::bind_rows(list(a, b)), d1, d2)
  f02_both(function(a) cuplyr::bind_rows(NULL, a), d1)
  f02_both(function(a, b) cuplyr::bind_rows(NULL, a, NULL, b), d1, d2)
  f02_both(function(b) cuplyr::bind_rows(tibble::tibble(x = 0L, s = "z"), b), d2)
  f02_both(function(a, c) cuplyr::bind_cols(NULL, a, c), d1, d3)
  f02_both(function(a, c) cuplyr::bind_cols(list(a, c)), d1, d3)

  acc <- NULL
  for (d in list(d1, d2)) acc <- cuplyr::bind_rows(acc, tbl_gpu(d))
  expect_true(is_tbl_gpu(acc))
  expect_identical(collect(acc), dplyr::bind_rows(d1, d2))
})
