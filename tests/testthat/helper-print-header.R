# print(tib, ...) as tibble writes it, with "A tbl_gpu" as the header's first
# name instead of "A tibble": the header print() of the same table on the GPU
# must show. pillar pads every header name to the longest one, so the rename
# has to happen in tbl_sum(), before pillar formats the header; a text
# substitution on tibble's output would get the "# Groups:" padding wrong.
tibble_print_as_tbl_gpu <- function(tib, ...) {
  renamed <- structure(tib, class = c("cuplyr_test_as_tbl_gpu", class(tib)))
  testthat::local_mocked_s3_method(
    "tbl_sum", "cuplyr_test_as_tbl_gpu",
    function(x, ...) {
      sum <- NextMethod()
      names(sum)[1] <- "A tbl_gpu"
      sum
    },
    frame = environment()
  )
  utils::capture.output(print(renamed, ...))
}
