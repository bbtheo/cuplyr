# Select columns from a GPU table

Keeps only the specified columns from a GPU table, similar to
[`dplyr::select()`](https://dplyr.tidyverse.org/reference/select.html).
Supports tidyselect syntax for flexible column selection.

## Usage

``` r
# S3 method for class 'tbl_gpu'
select(.data, ...)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Column names or tidyselect expressions specifying which columns to
  keep. Supports:

  - Column names: `select(x, y, z)`

  - Negative selection: `select(-x)`

  - Range: `select(x:z)`

  - Predicates: `where(is.numeric)`. Base type tests (`is.numeric`,
    `is.double`, `is.integer`, `is.character`, `is.logical`,
    `is.factor`, `is.ordered`, `inherits(x, "Date")`), bare or combined
    in a lambda such as `~ is.numeric(.x) || is.factor(.x)`, are
    answered from the column types. Any other predicate
    (`where(~ anyNA(.x))`,
    [`lubridate::is.Date`](https://lubridate.tidyverse.org/reference/date_utils.html)),
    and any call that is not a tidyselect helper or a base name builder
    (`sort(names(x))`, [`paste0()`](https://rdrr.io/r/base/paste.html),
    [`rev()`](https://rdrr.io/r/base/rev.html),
    [`setdiff()`](https://generics.r-lib.org/reference/setops.html),
    `.data$x`, ... with safe arguments), is resolved on the data: the
    table is collected and the selection runs in dplyr on the CPU (see
    [cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)
    to be told when)

  - Helpers:
    [`starts_with()`](https://tidyselect.r-lib.org/reference/starts_with.html),
    [`ends_with()`](https://tidyselect.r-lib.org/reference/starts_with.html),
    [`contains()`](https://tidyselect.r-lib.org/reference/starts_with.html),
    etc.

  - Renaming while selecting: `select(new_name = old_name)`

## Value

A `tbl_gpu` object containing only the selected columns. Column order
matches the order specified in the selection; any `new = old` selections
are renamed in the result.

## Details

Column selection creates a new GPU table with only the selected columns.
The original data remains in GPU memory until garbage collected.

### Renaming

`select(new_name = old_name)` both selects and renames `old_name`. The
selection runs on the GPU as usual; the rename only changes the column
names, as
[`rename.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/rename.tbl_gpu.md)
does.

### Grouped tables

Grouping columns that the selection leaves out are added in front, with
the message "Adding missing grouping variables", as in dplyr. Groups
follow their column by position, so `select(group_by(df, a), a = b)` is
ungrouped.

### Zero columns

A selection that keeps no column (`select(df, -everything())`) gives a
zero-column table that still knows its row count. A lazy table with
pending operations is computed first to get that count.

### Performance

Select operations involve copying column data to a new table structure.
For very wide tables, selecting fewer columns can significantly reduce
memory usage and improve performance of subsequent operations.

## See also

[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md)
for filtering rows,
[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
for creating columns,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Select specific columns
  result <- gpu_mtcars |>
    select(mpg, cyl, hp) |>
    collect()

  # Select with tidyselect helpers
  result <- gpu_mtcars |>
    select(starts_with("d")) |>
    collect()

  # Reorder columns
  result <- gpu_mtcars |>
    select(hp, mpg, wt) |>
    collect()
}
```
