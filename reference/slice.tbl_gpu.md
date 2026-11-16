# Select rows by position

Keeps rows of a GPU table by integer position, similar to
[`dplyr::slice()`](https://dplyr.tidyverse.org/reference/slice.html).
Ungrouped,
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped,
and `.by=` calls with plain, data-independent index expressions all run
entirely on the GPU (Phase 5, task W7): ungrouped via a direct
`cudf::gather()`; grouped/ `.by=` via a per-group
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
desugar (R/slice.R's module docs), restricted to a strictly increasing
positive index vector or any set of negative indices (arbitrary
duplication/reordering is ungrouped-only – see `...`'s docs below).
Everything else (an index expression the GPU-native path can't evaluate
without a data mask, e.g. one that references a column or
[`n()`](https://dplyr.tidyverse.org/reference/context.html), or a
grouped/`.by=` index vector with duplicates/reordering) transparently
falls back to CPU evaluation (see
[`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md)).

## Usage

``` r
# S3 method for class 'tbl_gpu'
slice(.data, ..., .by = NULL, .preserve = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Integer row positions, e.g. `slice(df, 1:3)`. Duplicates and
  reordering are allowed for positive indices when `.data` is ungrouped
  and `.by=` isn't supplied (`slice(df, c(3, 1, 1))`); grouped/`.by=`
  calls only run natively for a strictly increasing, duplicate-free
  positive index vector (falling back to CPU otherwise); negative
  indices drop those rows instead (order/duplicates never matter here,
  grouped or not); `0`/`NA` are silently dropped; out-of-range positions
  are silently dropped; positive and negative indices cannot be mixed.

- .by:

  Optional on-the-fly grouping columns, GPU-native (Phase 5, task W7):
  result rows are ordered by each group's FIRST-APPEARANCE position in
  `.data` (see module docs), and the result is always ungrouped.
  Supplying `.by` when `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, matching dplyr.

- .preserve:

  Whether to keep the groups the slice emptied. With `TRUE`, a
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped
  result keeps the input's group keys (empty groups included) for
  [`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html),
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) and
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
  like dplyr (see
  [`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
  for what carries them on). On a `.drop = FALSE` table every group is
  kept whatever `.preserve` says, as in dplyr (also for
  [`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html),
  [`slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html),
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html) and
  [`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)).

## Value

A `tbl_gpu` object with the selected rows.
