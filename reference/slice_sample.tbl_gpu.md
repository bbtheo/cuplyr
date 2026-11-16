# Sample rows

Keeps a random sample of rows of a GPU table, similar to
[`dplyr::slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html).
Ungrouped,
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped,
and `by=` calls all run entirely on the GPU for the actual row gather
(Phase 5, task W8): row indices are generated on the CPU (to reproduce
dplyr's own RNG stream bit-for-bit, see
[`slice_sample_native()`](https://bbtheo.github.io/cuplyr/reference/slice_sample_native.md)'s
own docs for the full recipe/group-ordering rules), then gathered via
`gpu_slice_indices()`. `weight_by=` still transparently falls back to
CPU evaluation (see
[`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md))
– weighted sampling needs the actual (potentially computed) column
values, not just row counts/group columns.

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
  `n`. With `replace = TRUE`, a non-negative `n`/`prop` is NOT clamped
  to the table's (or group's) row count – oversampling with replacement
  is allowed and produces a larger result; without `replace`,
  oversampling silently clamps (no error).

- prop:

  Proportion of rows to sample (`n = prop * nrow(.data)`, subject to the
  same clamp rule as `n`).

- by:

  Optional on-the-fly grouping columns, GPU-native (Phase 5, task W8):
  row indices are drawn (and the result ordered) by each group's
  FIRST-APPEARANCE position in `.data` (see
  [`slice_sample_native()`](https://bbtheo.github.io/cuplyr/reference/slice_sample_native.md)'s
  own docs), and the result is always ungrouped. Supplying `by` when
  `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, matching dplyr.

- weight_by:

  Sampling weights. Always falls back to CPU evaluation (see
  [`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md))
  – not GPU-native.

- replace:

  Whether to sample with replacement.

## Value

A `tbl_gpu` object with the sampled rows.
