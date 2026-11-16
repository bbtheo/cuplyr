# Split a GPU table into one GPU table per group (dplyr backend method)

Native:
[`dplyr::group_split()`](https://dplyr.tidyverse.org/reference/group_split.html)
on a `tbl_gpu` groups by `...` (or uses the existing groups), computes
the group row indices on the GPU
([`compute_group_data()`](https://bbtheo.github.io/cuplyr/reference/compute_group_data.md):
sorted keys, NA last, `.drop = FALSE` and preserved empty groups),
gathers the rows into group order and cuts the table at the group sizes
(`gpu_split_rows()`). The chunks are `tbl_gpu` tables, as dplyr's chunks
have the class of the ungrouped input, and the `list_of` ptype is a
0-row `tbl_gpu`. A table left with no columns (every column dropped by
`.keep = FALSE`) is the one case that still goes through the CPU
fallback.

## Usage

``` r
# S3 method for class 'tbl_gpu'
group_split(.tbl, ..., .keep = TRUE, keep = lifecycle::deprecated())
```

## Arguments

- .tbl:

  A tbl.

- ...:

  If `.tbl` is an ungrouped data frame, a grouping specification,
  forwarded to
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html).

- .keep:

  Should the grouping columns be kept?

- keep:

  Defunct, use `.keep`.

## Value

A [`vctrs::list_of()`](https://vctrs.r-lib.org/reference/list_of.html)
of `tbl_gpu` tables.
