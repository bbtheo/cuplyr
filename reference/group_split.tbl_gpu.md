# Split a GPU table into one GPU table per group (dplyr backend method)

[`dplyr::group_split()`](https://dplyr.tidyverse.org/reference/group_split.html)
on a `tbl_gpu` runs on the GPU. It groups by `...` (or uses the existing
groups), finds the rows of each group (groups in sorted key order, `NA`
last, as
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
gives them, empty groups included), puts the rows in group order and
cuts the table into one piece per group. The pieces are `tbl_gpu`
tables, as dplyr's pieces have the class of the ungrouped input, and the
`list_of` prototype is a 0-row `tbl_gpu`. A table left with no columns
(every column dropped by `.keep = FALSE`) runs in dplyr on the CPU.

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
