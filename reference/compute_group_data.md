# Compute the grouping-metadata tibble shared by the whole query family

Materializes `.data` if needed, then either builds the trivial
single-row ungrouped `.rows` tibble (as in dplyr,
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
on an ungrouped table is one row, `.rows` holding every row 1:n in
original order – no GPU work needed at all), or calls `gpu_group_info()`
(src/ops_groupby.cpp) once and assembles the result: the small
distinct-keys table is collected via the ordinary
[`collect.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
path (reusing its factor-level/INT64 handling rather than duplicating
it), and `.rows` is sliced out of the returned `perm` vector using
`offsets` as 0-based boundary positions (see `gpu_group_info()`'s own
docs for the exact contract).

## Usage

``` r
compute_group_data(.data)
```

## Arguments

- .data:

  A `tbl_gpu` object

## Value

A tibble: grouping columns (if grouped) plus a `.rows` list-of column of
1-based row indices, one row per group (or one row for the whole table
if ungrouped)

## Details

`.drop = FALSE` (see
[`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)'s
own docs) is honored as a post-hoc R-side expansion over the (small,
group-count-sized) result – never over the underlying GPU data – via
`expand_dropped_groups()`.

Not cached across calls (each call re-runs the GPU query). The query
itself is a single GPU groupby, so repeated calls are cheap relative to
any actual aggregation.
