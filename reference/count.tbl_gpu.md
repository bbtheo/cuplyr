# Count observations by group

Counts the number of rows (or sums a weight column) for each combination
of `...`, similar to
[`dplyr::count()`](https://dplyr.tidyverse.org/reference/count.html).
Sugar over
[`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md),
[`summarise.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md),
and
[`arrange.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/arrange.tbl_gpu.md)
– runs on the GPU. The one exception is a `.drop = FALSE` grouping with
a factor key, or a table with fixed group keys (after
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html),
... on a `.drop = FALSE` table), which also collects the (small) group
keys to add the empty groups.

## Usage

``` r
# S3 method for class 'tbl_gpu'
count(
  x,
  ...,
  wt = NULL,
  sort = FALSE,
  name = NULL,
  .drop = dplyr::group_by_drop_default(x)
)
```

## Arguments

- x:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Column names or expressions to group by (e.g. `count(df, x)` or
  `count(df, g = x %/% 10)`), parsed through the same expression IR
  [`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
  uses for any computed expression. If `x` is already grouped, these are
  added to the existing grouping (matching
  `group_by(x, ..., .add = TRUE)`).

- wt:

  Optional weight column/expression. If supplied, each group's count is
  `sum(wt)` (NA-excluding) instead of the row count.

- sort:

  If `TRUE`, sorts the result in descending order of the count column.

- name:

  The name of the count column in the output. If `NULL` (default), uses
  `"n"`, or `"nn"`/`"nnn"`/... if `"n"` already names one of the
  (post-`...`) grouping columns.

- .drop:

  Passed to
  [`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md);
  with `.drop = FALSE` the result has a zero-count row for every empty
  factor-level group, in
  [`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
  order. Defaults to `group_by_drop_default(x)`.

## Value

A `tbl_gpu` object with one row per group, containing the grouping
columns and the count column. Grouping: if `x` was ungrouped, the result
is ungrouped; if `x` was grouped, the result's grouping is restored to
`x`'s ORIGINAL grouping in full (verified against dplyr 1.2.1 – see this
file's header comment for why this is not simply "drop the last grouping
level").

## See also

[`tally.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tally.tbl_gpu.md),
[`group_by.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md),
[`summarise.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md)

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)
  gpu_mtcars |> count(cyl) |> collect()
  gpu_mtcars |> count(cyl, sort = TRUE) |> collect()
}
#>   cyl  n
#> 1   8 14
#> 2   4 11
#> 3   6  7
```
