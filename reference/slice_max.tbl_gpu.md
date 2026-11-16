# Select rows with the largest values of a variable

Keeps the `n` (or `prop` fraction of) rows with the largest `order_by`
values, similar to
[`dplyr::slice_max()`](https://dplyr.tidyverse.org/reference/slice.html).
See
[`slice_min.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_min.tbl_gpu.md)
for the shared native-vs-fallback dispatch and `order_by`/`with_ties`/
`na_rm` semantics (identical, except the ranking direction and the
result's row order, which is descending here).

## Usage

``` r
# S3 method for class 'tbl_gpu'
slice_max(
  .data,
  order_by,
  ...,
  n,
  prop,
  by = NULL,
  with_ties = TRUE,
  na_rm = FALSE
)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- order_by:

  Column (or IR-supported expression) to rank by.

- ...:

  Not used; for extensibility, must be empty.

- n, prop:

  As for
  [`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md).

- by:

  Optional on-the-fly grouping columns, GPU-native (Phase 5, task W7) –
  see
  [`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md)'s
  `by` docs.

- with_ties:

  If `TRUE` (the default), ties at the cutoff are all kept (the result
  can have more than `n` rows); if `FALSE`, ties are broken by original
  row position and the result has exactly `n` rows (or fewer if `.data`
  is smaller).

- na_rm:

  If `FALSE` (the default), `NA` values in `order_by` sort last (for
  both [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)
  and [`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html))
  and can appear in the result once `n`/`prop` reaches past the non-`NA`
  count. If `TRUE`, `NA` rows are always excluded.

## Value

A `tbl_gpu` object with the selected rows, in descending `order_by`
order.
