# Select rows with the smallest values of a variable

Keeps the `n` (or `prop` fraction of) rows with the smallest `order_by`
values, as
[`dplyr::slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)
does. It runs on the GPU, ungrouped, after
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) and
with `by`, when `order_by` is a column or an expression
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
runs on the GPU; any other `order_by` runs in dplyr on the CPU (see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)).
Grouped results come group by group, as for
[`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md):
in ascending group-key order after
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html), and
in the order each group first appears in `.data` with `by`.

## Usage

``` r
# S3 method for class 'tbl_gpu'
slice_min(
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

  Column, or expression of columns, to rank by.

- ...:

  Not used; for extensibility, must be empty.

- n, prop:

  As for
  [`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md).

- by:

  Optional grouping columns for this call only, run on the GPU; see
  [`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md)'s
  `by`.

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

A `tbl_gpu` object with the selected rows, in ascending `order_by` order
(grouped: ascending `order_by` order within each group's own block).
