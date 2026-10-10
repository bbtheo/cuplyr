# F02 extras beyond the PM's acceptance file: registry safety, names<- details.

test_that("restoring handles of a collected source is a loud error", {
  skip_if_no_gpu()
  make <- function() {
    src <- tbl_gpu(data.frame(x = 1:3))
    # a materialized slice of the proxy no longer references `src`
    vctrs::vec_slice(vctrs::vec_proxy(src), 1:2)
  }
  handles <- make()
  keep <- tbl_gpu(data.frame(x = 1:3))
  gc()
  expect_error(
    vctrs::vec_restore(handles, keep),
    class = "cuplyr_error_vctrs_dead_source"
  )
})

test_that("a collected source's handles never resolve to another table", {
  skip_if_no_gpu()
  make <- function() {
    src <- tbl_gpu(data.frame(x = 1:3))
    vctrs::vec_slice(vctrs::vec_proxy(src), 1:2)
  }
  handles <- make()
  gc()
  # enough new sources to force registry sweeps; ids must not be recycled
  others <- lapply(1:300, function(i) tbl_gpu(data.frame(x = c(100L, 200L, 300L) + i)))
  for (o in others) vctrs::vec_size(o)
  gc()
  expect_error(
    vctrs::vec_restore(handles, others[[1]]),
    class = "cuplyr_error_vctrs_dead_source"
  )
})

test_that("running out of source ids is a loud error", {
  skip_if_no_gpu()
  st <- vctrs_state_init()
  withr::defer(st$next_id <- saved)
  saved <- st$next_id
  st$next_id <- vctrs_max_source_id + 1
  expect_error(
    vctrs::vec_size(tbl_gpu(data.frame(x = 1:3))),
    class = "cuplyr_error_vctrs_registry_full"
  )
})

test_that("multi-table combines are stable under natural garbage collection", {
  skip_if_no_gpu()
  gi <- tbl_gpu(data.frame(x = seq_len(5000)))
  gd <- tbl_gpu(data.frame(x = seq_len(5000) + 0.5))
  lz <- tbl_gpu(data.frame(x = c(NA, seq_len(4999) + 0)), lazy = TRUE) |>
    dplyr::filter(!is.na(x) | is.na(x))
  fixed <- tbl_gpu(data.frame(x = rep(9, 5000)))
  for (i in 1:40) {
    expect_identical(nrow(collect(vctrs::vec_c(gi, gd))), 10000L)
    expect_identical(nrow(collect(vctrs::vec_c(gd, NA, gd))), 10001L)
    expect_identical(collect(dplyr::coalesce(lz, fixed))$x[1], 9)
  }
})

test_that("pins taken for a vctrs call are released when it returns", {
  skip_if_no_gpu()
  st <- vctrs_state_init()
  gi <- tbl_gpu(data.frame(x = 1:3))
  gd <- tbl_gpu(data.frame(x = c(1.5, 2.5)))
  lz <- tbl_gpu(data.frame(x = c(NA, 2, 3)), lazy = TRUE) |> dplyr::filter(x > 0 | is.na(x))
  vctrs::vec_c(gi, gd)
  dplyr::coalesce(lz, tbl_gpu(data.frame(x = c(9, 9, 9))))
  expect_error(vctrs::vec_c(gi, tbl_gpu(data.frame(x = "a"))), class = "vctrs_error_incompatible_type")
  expect_length(ls(st$pins), 0L)
})

test_that("vec_rbind()/vec_cbind() on a tbl_gpu is a clear error, not a malformed tibble", {
  skip_if_no_gpu()
  g1 <- tbl_gpu(data.frame(x = 1:3))
  g2 <- tbl_gpu(data.frame(x = 4:6))
  expect_error(vctrs::vec_rbind(g1, g2), class = "cuplyr_error_vctrs_bind_unsupported")
  expect_error(vctrs::vec_cbind(g1, g2), class = "cuplyr_error_vctrs_bind_unsupported")
  expect_error(dplyr::bind_rows(g1, g2), class = "cuplyr_error_vctrs_bind_unsupported")
  expect_error(dplyr::bind_cols(g1, g2), class = "cuplyr_error_vctrs_bind_unsupported")
})

test_that("bind_rows()/bind_cols() accept a tbl_gpu anywhere in their arguments", {
  skip_if_no_gpu()
  d1 <- data.frame(x = 1:2, s = c("a", "b"))
  d2 <- data.frame(x = 3:4, s = c("c", "d"))
  d3 <- data.frame(x = 5L, s = "e")
  g1 <- tbl_gpu(d1)
  g2 <- tbl_gpu(d2)

  out <- cuplyr::bind_rows(g1, list(g2, NULL, tibble::as_tibble(d3)))
  expect_true(is_tbl_gpu(out))
  expect_identical(collect(out), tibble::as_tibble(rbind(d1, d2, d3)))

  named <- cuplyr::bind_rows(list(a = d1, b = g2), .id = "src")
  expect_true(is_tbl_gpu(named))
  expect_identical(
    collect(named),
    dplyr::bind_rows(list(a = tibble::as_tibble(d1), b = tibble::as_tibble(d2)), .id = "src")
  )

  cols <- cuplyr::bind_cols(tibble::tibble(k = 1:2), g1)
  expect_true(is_tbl_gpu(cols))
  expect_identical(collect(cols), dplyr::bind_cols(tibble::tibble(k = 1:2), tibble::as_tibble(d1)))

  # plain data still goes to dplyr
  expect_identical(cuplyr::bind_rows(NULL, tibble::as_tibble(d1)), tibble::as_tibble(d1))
  expect_error(cuplyr::bind_rows(g1, 1:3), "Argument 2 must be a data frame or a named atomic vector.", fixed = TRUE)
})

test_that("a tbl_gpu is rejected as a value inside an expression", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(x = 1:3))
  h1 <- tbl_gpu(data.frame(y = 1L))
  expect_error(dplyr::mutate(g, z = h1), "can't be used as a value")
})

test_that("value-based vctrs paths report their CPU collect through the fallback option", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(x = c(1, 1, 2)))
  withr::local_options(cuplyr.fallback = "error")
  expect_error(dplyr::n_distinct(g), "fell back to CPU")
  # GPU-resident paths stay quiet
  expect_no_error(vctrs::vec_slice(g, 2:3))
  expect_no_error(vctrs::vec_c(g, g))
  expect_no_error(vctrs::vec_size(g))
})

test_that("vec_slice() by column name never indexes rows", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(g = c("a", "b"), x = 1:2))
  expect_error(vctrs::vec_slice(g, "g"), "character names to index")
  expect_null(vctrs::vec_names(g))
  expect_identical(names(g), c("g", "x"))
})

test_that("names<- keeps groups and factor levels in step", {
  skip_if_no_gpu()
  df <- data.frame(g = c("a", "b"), f = factor(c("u", "v")), x = 1:2)
  gg <- dplyr::group_by(tbl_gpu(df), g)
  out <- rlang::set_names(gg, c("G", "F", "X"))
  expect_identical(dplyr::group_vars(out), "G")
  expect_identical(dplyr::ungroup(collect(out)), tibble::tibble(G = df$g, F = df$f, X = df$x))
})

test_that("names<- on a lazy table with pending ops renames through the AST", {
  skip_if_no_gpu()
  lz <- tbl_gpu(data.frame(a = 1:3, b = 4:6), lazy = TRUE) |> dplyr::filter(a > 1L)
  out <- rlang::set_names(lz, c("p", "q"))
  expect_identical(names(out), c("p", "q"))
  expect_identical(collect(out), tibble::tibble(p = 2:3, q = 5:6))
})

test_that("vec_c() of many distinct tables gathers from every source", {
  skip_if_no_gpu()
  parts <- lapply(1:5, function(i) tbl_gpu(data.frame(x = i * 10 + 1:2, s = letters[i + 0:1])))
  out <- do.call(vctrs::vec_c, parts)
  expect_identical(
    collect(out),
    do.call(vctrs::vec_c, lapply(parts, collect))
  )
})

test_that("vec_cast() between layouts goes through the CPU cast and keeps groups", {
  skip_if_no_gpu()
  a <- dplyr::group_by(tbl_gpu(data.frame(g = 1:2, x = c(1L, 2L))), g)
  b <- tbl_gpu(data.frame(g = c(1, 2), x = c(1.5, 2)))
  p <- vctrs::vec_ptype2(a, b)
  expect_identical(dplyr::group_vars(p), "g")
  casted <- vctrs::vec_cast(a, p)
  expect_identical(dplyr::ungroup(collect(casted)), tibble::tibble(g = c(1, 2), x = c(1, 2)))
  expect_identical(dplyr::group_vars(casted), "g")
})

test_that("a lazy input is computed once per vctrs call, and the memo is dropped after", {
  skip_if_no_gpu()
  lz <- dplyr::filter(as_lazy(tbl_gpu(data.frame(x = 1:10))), x > 2L)
  n <- 0L
  orig <- lower_and_execute
  local_mocked_bindings(lower_and_execute = function(...) {
    n <<- n + 1L
    orig(...)
  })

  # lower_and_execute() recurses over the AST; one compute() is `per` calls
  invisible(compute(lz))
  per <- n
  n <- 0L

  expect_identical(vctrs::list_sizes(rep(list(lz), 5)), rep(8L, 5))
  expect_identical(n, per)
  expect_length(ls(vctrs_state$memo), 0L)

  # a new vctrs call recomputes; a combine of the same input computes once
  out <- vctrs::vec_c(lz, lz)
  expect_identical(n, 2L * per)
  expect_identical(collect(out), tibble::tibble(x = c(3:10, 3:10)))
  expect_length(ls(vctrs_state$memo), 0L)
  expect_length(ls(vctrs_state$pins), 0L)

  # a different pending pipeline on the same source is not confused with it
  lz2 <- dplyr::filter(lz, x > 8L)
  out2 <- vctrs::vec_c(lz, lz2)
  expect_identical(collect(out2), tibble::tibble(x = c(3:10, 9:10)))
  expect_length(ls(vctrs_state$memo), 0L)
})
