# Select rows with the smallest values of a variable

Keeps the `n` (or `prop` fraction of) rows with the smallest `order_by`
values, similar to
[`dplyr::slice_min()`](https://dplyr.tidyverse.org/reference/slice.html).
Ungrouped,
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped,
and `by=` calls with a plain `order_by` (a bare column, or an expression
the mutate/filter IR recognizes) all run entirely on the GPU (Phase 5,
task W7): ungrouped via a top-k cut plus a sort of the survivors when
the key is numeric or a date and `n` is a small share of the rows,
otherwise a stable sort (with `cudf::rank()` for ties); grouped/`by=` on
such a key by stable radix sorts of the row positions and per-group
ranks, other grouped/`by=` keys via a per-group window desugar (see
R/slice.R's module docs for the exact group-ordering rules, including
`by=`'s first-appearance-order rule, which applies here too, not just to
the positional slice\_\*()s). An unsupported `order_by` expression (one
the IR can't parse) still transparently falls back to CPU evaluation
(see
[`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md)).

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

A `tbl_gpu` object with the selected rows, in ascending `order_by` order
(grouped: ascending `order_by` order within each group's own block).
