# Order rows using column values

[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) orders
the rows of a GPU table by the values of the columns you name, as
[`dplyr::arrange()`](https://dplyr.tidyverse.org/reference/arrange.html)
does. The sort runs on the GPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
arrange(.data, ..., .by_group = FALSE, .locale = NULL)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Columns to sort by. A bare column, `desc(column)`,
  [`pick()`](https://dplyr.tidyverse.org/reference/pick.html) of
  columns, and `-column` of a numeric or logical column sort on the GPU.
  Any other expression, such as `arrange(x + y)` or `-column` of a
  factor, string or date column (where R's `-` warns or errors), is
  evaluated by dplyr on the CPU. The first key is the primary sort key,
  the second breaks its ties, and so on.

- .by_group:

  If `TRUE` and `.data` is grouped, sort by the grouping columns first
  (ascending), then by `...`. Default is `FALSE`.

- .locale:

  The locale to sort character vectors in. `NULL` (the default) and
  `"C"` sort by byte order on the GPU; any other locale is sorted by
  dplyr on the CPU (with stringi).

## Value

A `tbl_gpu` with the rows reordered and the groups of `.data`. The
sorted result is new GPU memory (about twice the table size at peak).

## Details

### Sort order

- Ascending by default; `desc(column)` or `-column` sorts descending.

- With several keys, the first is the primary key, the second breaks its
  ties, and so on.

- The sort is stable: rows that tie keep their original order.

### Missing values

`NA` and `NaN` sort last, in ascending and in descending order, as in
dplyr.

### Memory usage

The arrange operation requires approximately 2x the table size in GPU
memory:

- Original table

- Sort indices (4 bytes per row)

- New sorted table

For very large tables, consider filtering to reduce size before sorting.

### Supported column types

Every column type a GPU table holds can be a sort key: double, integer,
integer64, character, logical, factor, `Date` and `POSIXct`. A factor
sorts by the order of its levels, as in dplyr.

Character columns sort by byte (UTF-8) order, which is dplyr's default
`"C"` locale. Another `.locale`, or
`options(dplyr.legacy_locale = TRUE)`, sorts in dplyr on the CPU.

## See also

[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md)
for filtering rows,
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for selecting columns,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Sort by single column (ascending)
  sorted <- gpu_mtcars |>
    arrange(mpg) |>
    collect()

  # Sort descending
  sorted_desc <- gpu_mtcars |>
    arrange(desc(mpg)) |>
    collect()

  # Multiple columns: primary and secondary sort keys
  sorted_multi <- gpu_mtcars |>
    arrange(cyl, desc(mpg)) |>
    collect()

  # With grouped data
  grouped_sort <- gpu_mtcars |>
    group_by(cyl) |>
    arrange(mpg, .by_group = TRUE) |>
    collect()
}
```
