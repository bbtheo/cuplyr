# Shared helpers for the F35 acceptance tests (test-parity-F35-*.R).
#
# f35_expect_parity() runs one pipeline function on plain tibbles (the dplyr
# oracle), on eager tbl_gpu inputs, on lazy tbl_gpu inputs, and on a mix (odd
# inputs lazy, even inputs eager), and compares groups, `.drop`, names and
# values.

f35_inputs <- function(tables, mode) {
  switch(mode,
    dplyr = lapply(tables, tibble::as_tibble),
    eager = lapply(tables, tbl_gpu, lazy = FALSE),
    lazy = lapply(tables, tbl_gpu, lazy = TRUE),
    mixed = Map(function(t, i) tbl_gpu(t, lazy = i %% 2L == 1L),
                tables, seq_along(tables))
  )
}

f35_expect_parity <- function(tables, fn, modes = c("eager", "lazy", "mixed")) {
  want <- do.call(fn, unname(f35_inputs(tables, "dplyr")))
  want_groups <- dplyr::group_vars(want)
  want_drop <- dplyr::group_by_drop_default(want)
  want_tbl <- oracle_normalize(want)
  for (m in modes) {
    got <- do.call(fn, unname(f35_inputs(tables, m)))
    expect_equal(oracle_group_vars(got), want_groups, info = paste("groups", m))
    expect_equal(dplyr::group_by_drop_default(got), want_drop, info = paste(".drop", m))
    got_tbl <- oracle_normalize(collect(got))
    expect_equal(names(got_tbl), names(want_tbl), info = paste("names", m))
    expect_equal(got_tbl, want_tbl, info = paste("values", m))
  }
}

f35_expect_error_parity <- function(tables, fn, class, regexp = NULL,
                                    modes = c("eager", "lazy", "mixed")) {
  expect_error(do.call(fn, unname(f35_inputs(tables, "dplyr"))),
               class = class, regexp = regexp, info = "dplyr")
  for (m in modes) {
    expect_error(collect(do.call(fn, unname(f35_inputs(tables, m)))),
                 class = class, regexp = regexp, info = m)
  }
}

# Where do filters sit relative to the topmost join after optimize_ast()?
f35_plan <- function(tbl) {
  node <- optimize_ast(.subset2(tbl, "lazy_ops"))
  above <- character()
  while (node$type != "join") {
    above <- c(above, node$type)
    node <- node$input
  }
  has_filter <- function(n) {
    while (!is.null(n)) {
      if (n$type == "filter") return(TRUE)
      if (n$type == "join") return(FALSE)
      n <- n$input
    }
    FALSE
  }
  list(above = "filter" %in% above,
       left = has_filter(node$left),
       right = has_filter(node$right))
}
