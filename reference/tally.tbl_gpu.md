# Count the rows in each group (no additional grouping)

Like
[`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md),
but without any `...`: aggregates exactly the groups `x` already has (or
all rows, if ungrouped). Sugar over
[`summarise.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md).

## Usage

``` r
# S3 method for class 'tbl_gpu'
tally(x, wt = NULL, sort = FALSE, name = NULL)
```

## Arguments

- x:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- wt:

  Optional weight column/expression – as
  [`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md).

- sort:

  If `TRUE`, sorts the result in descending order of the count column.

- name:

  The name of the count column. If `NULL` (default), uses `"n"`, or
  `"nn"`/`"nnn"`/... if `"n"` already names one of `x`'s group columns
  (note: unlike
  [`add_count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/add_count.tbl_gpu.md)/[`dplyr::add_tally()`](https://dplyr.tidyverse.org/reference/count.html),
  this check is against the GROUP columns only, not every column –
  verified empirically against dplyr 1.2.1, see this file's header
  comment).

## Value

A `tbl_gpu` with the count column appended. If `x` was grouped by more
than one column, the result's grouping drops only the LAST grouping
column (matching
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)'s
default `.groups = "drop_last"` behavior for a real `grouped_df` – this
is [`tally()`](https://dplyr.tidyverse.org/reference/count.html)'s own
rule, and is NOT the same as
[`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md)'s
"restore original grouping in full" rule; see this file's header
comment).

## See also

[`count.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md)

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)
  gpu_mtcars |> group_by(cyl) |> tally() |> collect()
}
#> # A tibble: 3 × 2
#>     cyl     n
#>   <dbl> <int>
#> 1     4    11
#> 2     6     7
#> 3     8    14
```
