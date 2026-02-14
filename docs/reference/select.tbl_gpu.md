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

  - Negative selection: `select(-x)` (not yet supported)

  - Range: `select(x:z)` (not yet supported)

  - Helpers: `starts_with()`, `ends_with()`, `contains()`, etc.

## Value

A `tbl_gpu` object containing only the selected columns. Column order
matches the order specified in the selection.

## Details

Column selection creates a new GPU table with only the selected columns.
The original data remains in GPU memory until garbage collected.

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
#> Error in collect(select(gpu_mtcars, mpg, cyl, hp)): could not find function "collect"
```
