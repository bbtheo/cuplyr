# Arrange rows of a GPU table by column values

Orders the rows of a GPU table by the values of specified columns,
similar to
[`dplyr::arrange()`](https://dplyr.tidyverse.org/reference/arrange.html).
Sorting is performed entirely on the GPU using a memory-efficient
two-phase algorithm.

## Usage

``` r
# S3 method for class 'tbl_gpu'
arrange(.data, ..., .by_group = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by [`tbl_gpu()`](tbl_gpu.md).

- ...:

  Column names or expressions to sort by. Use `desc(column)` or
  `-column` for descending order. Multiple columns are sorted in order
  of precedence (first column is primary sort key).

- .by_group:

  If `TRUE` and `.data` is grouped, sort within groups by prepending
  group columns to the sort specification. Default is `FALSE`.

## Value

A `tbl_gpu` object with rows reordered. The GPU memory for the sorted
result is newly allocated (approximately 2x table size peak memory).

## Details

### Sort order

- Default is ascending order

- Use `desc(column)` or `-column` for descending order

- Multiple columns: first column is primary key, second is tiebreaker,
  etc.

- Sorting is stable: ties preserve their original relative order

### NA handling

- `NA` values are placed last for ascending order

- `NA` values are placed first for descending order

### Memory usage

The arrange operation requires approximately 2x the table size in GPU
memory:

- Original table

- Sort indices (4 bytes per row)

- New sorted table

For very large tables, consider filtering to reduce size before sorting.

### Supported column types

All column types supported by `tbl_gpu` can be sorted: numeric, integer,
character, logical, Date, POSIXct.

Note: Character sorting uses binary/UTF-8 ordering, not locale-aware
collation.

## See also

[`filter.tbl_gpu`](filter.tbl_gpu.md) for filtering rows,
[`select.tbl_gpu`](select.tbl_gpu.md) for selecting columns,
[`collect.tbl_gpu`](collect.tbl_gpu.md) for retrieving results

## Examples

``` r
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
#> Error in collect(arrange(gpu_mtcars, mpg)): could not find function "collect"
```
