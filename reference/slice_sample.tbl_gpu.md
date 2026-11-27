# Sample rows

Keeps a random sample of rows of a GPU table, as
[`dplyr::slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html)
does. The row numbers are drawn in R with the same
[`sample.int()`](https://rdrr.io/r/base/sample.html) calls dplyr makes,
so the same [`set.seed()`](https://rdrr.io/r/base/Random.html) gives the
same rows as dplyr; the rows are then gathered on the GPU. This works
ungrouped, after
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) and
with `by`. A lazy table's pending steps run first, since the draw needs
the row count. `weight_by` runs in dplyr on the CPU (see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)),
since the weights are values of the data.

## Usage

``` r
# S3 method for class 'tbl_gpu'
slice_sample(.data, ..., n, prop, by = NULL, weight_by = NULL, replace = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Not used; for extensibility, must be empty.

- n:

  Number of rows to sample. Default `1` if neither `n` nor `prop` is
  supplied. Negative values are clamped like
  [`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md)'s
  `n`. With `replace = TRUE`, a non-negative `n`/`prop` is not clamped
  to the table's (or group's) row count: oversampling with replacement
  is allowed and gives a larger result; without `replace`, oversampling
  silently clamps (no error).

- prop:

  Proportion of rows to sample (`n = prop * nrow(.data)`, subject to the
  same clamp rule as `n`).

- by:

  Optional grouping columns for this call only, run on the GPU. As in
  dplyr, the groups are drawn (and appear in the result) in the order
  each one first appears in `.data`, and the result is ungrouped.
  Supplying `by` when `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, as in dplyr.

- weight_by:

  Sampling weights. A call with `weight_by` runs in dplyr on the CPU.

- replace:

  Whether to sample with replacement.

## Value

A `tbl_gpu` object with the sampled rows.
