# Drop rows of a GPU table where conditions are TRUE

The complement of
[`filter.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md),
like
[`dplyr::filter_out()`](https://dplyr.tidyverse.org/reference/filter.html):
all conditions are combined with AND first, then the rows where that
combination is `TRUE` are dropped. Rows where it is `FALSE` or `NA` are
kept (so this is not `filter(!cond)`). With no conditions every row is
dropped. It runs on the same engine as
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) (same
expression surface, window predicates, `.by`, CPU fallbacks, eager and
lazy): the predicate pushed to the GPU is `is.na(P) | !P` for the
combined `P`.

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

  Optional on-the-fly grouping columns (tidyselect), GPU-native (Phase
  5, task W6): `filter(df, x > mean(x), .by = g)` behaves identically to
  `df |> group_by(g) |> filter(x > mean(x)) |> ungroup()` – same
  window/aggregate-predicate machinery as
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-driven
  filtering, and the result is always ungrouped. Supplying `.by` when
  `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, matching dplyr exactly.

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
