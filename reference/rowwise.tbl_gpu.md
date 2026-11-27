# Group a GPU table by rows

Native:
[`rowwise()`](https://dplyr.tidyverse.org/reference/rowwise.html) of a
`tbl_gpu` only records which columns identify a row (metadata, no GPU
work and no download; a lazy table keeps its pending operations). The
result is a *rowwise* `tbl_gpu`:
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
returns dplyr's `rowwise_df`,
[`as_tibble()`](https://tibble.tidyverse.org/reference/as_tibble.html)
and [`ungroup()`](https://dplyr.tidyverse.org/reference/group_by.html)
drop the rowwise-ness,
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
replaces it, and
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) of a
`rowwise_df` uploads one.

## Usage

``` r
# S3 method for class 'tbl_gpu'
rowwise(data, ...)
```

## Arguments

- data:

  A `tbl_gpu`.

- ...:

  \<[`tidy-select`](https://dplyr.tidyverse.org/reference/dplyr_tidy_select.html)\>
  Columns to keep as identifying variables in
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html).
  A grouped table takes its group variables (naming others is an error,
  as in dplyr).

## Value

A rowwise `tbl_gpu`.

## Details

Elementwise expressions in
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)/[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
on a rowwise table run on the GPU as they would on a plain one (every
column is one value per row). The horizontal aggregates
[`sum()`](https://rdrr.io/r/base/sum.html)/[`mean()`](https://rdrr.io/r/base/mean.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`prod()`](https://rdrr.io/r/base/prod.html)
of `c_across(<tidyselect>)`, of `c(a, b, ...)` or of several column
arguments, [`n()`](https://dplyr.tidyverse.org/reference/context.html),
[`cur_group_id()`](https://dplyr.tidyverse.org/reference/context.html)
and `length(c_across(...))` are rewritten into GPU operations over the
columns of each row, with dplyr's NA rules and result types (integer
inputs give an integer
[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html),
a double sum if the integer range is left; an all-`NA` row of
[`min()`](https://rdrr.io/r/base/Extremes.html)/`max(na.rm = TRUE)` is
`Inf`/`-Inf` with dplyr's warning; on a lazy table a call with such an
integer [`sum()`](https://rdrr.io/r/base/sum.html) runs at the verb,
since its type depends on the data). Only logical, integer and double
columns take part. Everything else (a function that needs the row as a
vector, list-columns, window functions, `.by`) runs the whole call in
dplyr on the CPU and comes back as a rowwise `tbl_gpu`.
