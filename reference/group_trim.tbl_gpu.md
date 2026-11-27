# Drop unused levels of the factor group columns (dplyr backend method)

[`dplyr::group_trim()`](https://dplyr.tidyverse.org/reference/group_trim.html)
on a grouped `tbl_gpu` runs on the GPU. For each factor group column it
finds the levels still in use (only those distinct codes come back to
R), keeps them in their original order (an ordered factor stays
ordered), recodes the column on the GPU and regroups with `.drop`. Empty
groups kept by `.preserve = TRUE` are dropped too. Pending operations of
a lazy table run first, because the new levels depend on the data. An
ungrouped table is returned unchanged, without looking at `.drop`, as in
dplyr. A table with duplicated column names runs in dplyr on the CPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
group_trim(.tbl, .drop = dplyr::group_by_drop_default(.tbl))
```

## Arguments

- .tbl:

  A `tbl_gpu` object.

- .drop:

  Whether the regroup drops empty groups.

## Value

A `tbl_gpu`.
