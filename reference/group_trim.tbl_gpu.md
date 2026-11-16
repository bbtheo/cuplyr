# Drop unused levels of the factor group columns (dplyr backend method)

Native:
[`dplyr::group_trim()`](https://dplyr.tidyverse.org/reference/group_trim.html)
on a grouped `tbl_gpu` finds the codes still used by each factor group
column on the GPU (a distinct over that one column; only the at most
`n_levels + 1` distinct codes reach R), keeps the used levels in their
original order (an ordered factor stays ordered), remaps the codes with
`gpu_remap_codes()` and regroups with
[`ungroup()`](https://dplyr.tidyverse.org/reference/group_by.html) +
`group_by(.drop = )`, which also drops any `fixed_keys` left by
`.preserve = TRUE`. Pending ops are computed first (the new level set
depends on the data). An ungrouped table is returned unchanged, without
looking at `.drop`, like dplyr. Shapes outside the allowlist (duplicated
names, a factor group column without known levels) use the CPU fallback.

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
