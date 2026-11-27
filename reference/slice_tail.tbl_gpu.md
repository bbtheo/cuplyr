# Select the last rows

Keeps the last `n` (or `prop` fraction of) rows of a GPU table, as
[`dplyr::slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html)
does. It runs on the GPU, ungrouped, after
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) and
with `by`; see
[`slice_head.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md)
for the order of the groups in the result.

## Usage

``` r
# S3 method for class 'tbl_gpu'
slice_tail(.data, ..., n, prop, by = NULL)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Not used; for extensibility, must be empty.

- n:

  Number of rows to keep. Negative `n` keeps all but the last `abs(n)`
  rows. Clamped to `[0, nrow(.data)]` (grouped: `[0, group_size]`, per
  group). Default `1` if neither `n` nor `prop` is supplied.

- prop:

  Proportion of rows to keep (`floor(prop * nrow(.data))`, clamped like
  `n`; grouped: `floor(prop * group_size)`, per group). Exactly one of
  `n`/`prop` may be supplied.

- by:

  Optional grouping columns for this call only, run on the GPU. As in
  dplyr, the result has the groups in the order each one first appears
  in `.data`, and it is ungrouped. Supplying `by` when `.data` is
  already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, as in dplyr.

## Value

A `tbl_gpu` object with the last rows selected.
