# Select rows by position

Keeps rows of a GPU table by integer position, as
[`dplyr::slice()`](https://dplyr.tidyverse.org/reference/slice.html)
does. Constant positions run on the GPU, ungrouped, after
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) and
with `.by`. With groups, only a strictly increasing positive index
vector or a set of negative indices runs on the GPU (an ungrouped table
also takes duplicated and reordered positions; see `...` below).
Anything else runs in dplyr on the CPU (see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)):
an index expression that needs the data, e.g. one that reads a column or
[`n()`](https://dplyr.tidyverse.org/reference/context.html), or a
grouped or `.by` index vector with duplicated or reordered positions.

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
  and `.by=` isn't supplied (`slice(df, c(3, 1, 1))`); grouped or `.by`
  calls run on the GPU only for a strictly increasing, duplicate-free
  positive index vector (and on the CPU otherwise); negative indices
  drop those rows instead (order/duplicates never matter here, grouped
  or not); `0`/`NA` are silently dropped; out-of-range positions are
  silently dropped; positive and negative indices cannot be mixed.

- .by:

  Optional grouping columns for this call only, run on the GPU. As in
  dplyr, the result has the groups in the order each one first appears
  in `.data`, and it is ungrouped. Supplying `.by` when `.data` is
  already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, as in dplyr.

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

## Details

### Row order

As in dplyr: ungrouped, the rows come in the order of the positions.
After
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) the
result comes group by group, in ascending group-key order (an `NA` key
last), and each group's rows keep their order. With `.by` the groups
come in the order each first appears in `.data`.
