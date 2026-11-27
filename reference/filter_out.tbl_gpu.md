# Drop rows that match a condition

The complement of
[`filter.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md),
like
[`dplyr::filter_out()`](https://dplyr.tidyverse.org/reference/filter.html):
all conditions are combined with AND first, then the rows where that
combination is `TRUE` are dropped. Rows where it is `FALSE` or `NA` are
kept (so this is not `filter(!cond)`). With no conditions every row is
dropped. It runs like
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html): the
same conditions run on the GPU, with the same window predicates, `.by`,
CPU fallbacks, and eager and lazy modes.

## Usage

``` r
# S3 method for class 'tbl_gpu'
filter_out(.data, ..., .by = NULL, .preserve = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Logical expressions; rows where all of them are `TRUE` are dropped.

- .by:

  Optional grouping columns for this call only (tidyselect), run on the
  GPU: `filter(df, x > mean(x), .by = g)` gives the same rows as
  `df |> group_by(g) |> filter(x > mean(x)) |> ungroup()`, and the
  result is ungrouped. Supplying `.by` when `.data` is already grouped
  (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, as in dplyr.

- .preserve:

  Whether to keep the groups the filter emptied. With `TRUE`, a
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped
  result keeps the input's group keys (empty groups included) for
  [`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html),
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) and
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
  like dplyr (see
  [`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
  for what carries them on). `FALSE` recomputes the groups from the
  remaining rows.

## Value

A `tbl_gpu` object without the rows where all conditions are TRUE.
