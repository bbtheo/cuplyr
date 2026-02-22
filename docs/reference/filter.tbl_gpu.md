# Filter rows of a GPU table

Selects rows from a GPU table where conditions are TRUE, similar to
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html).
Filtering is performed entirely on the GPU for maximum performance on
large datasets.

## Usage

``` r
# S3 method for class 'tbl_gpu'
filter(.data, ..., .preserve = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by [`tbl_gpu()`](tbl_gpu.md).

- ...:

  Logical expressions to filter by. Each expression should be a
  comparison of the form `column <op> value` or `column <op> column`.
  Multiple conditions are combined with AND (all must be TRUE).

- .preserve:

  Ignored. Included for compatibility with dplyr generic.

## Value

A `tbl_gpu` object containing only rows where all conditions are TRUE.
The GPU memory for the filtered result is newly allocated.

## Details

### Supported comparison operators

- `==` - equal to

- `!=` - not equal to

- `>` - greater than

- `>=` - greater than or equal to

- `<` - less than

- `<=` - less than or equal to

### Current limitations

- Only simple comparisons are supported (column op value/column)

- Compound expressions with `&` or `|` are not yet supported

- String comparisons are not yet implemented

- Only numeric scalar values on the right-hand side

### Performance

Filtering on GPU is highly parallel and can process billions of rows per
second. For best performance, chain multiple filter conditions rather
than using compound expressions.

## See also

[`mutate.tbl_gpu`](mutate.tbl_gpu.md) for creating new columns,
[`select.tbl_gpu`](select.tbl_gpu.md) for selecting columns,
[`collect.tbl_gpu`](collect.tbl_gpu.md) for retrieving results

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Filter with single condition
  efficient_cars <- gpu_mtcars |>
    filter(mpg > 25)

  # Multiple conditions (combined with AND)
  result <- gpu_mtcars |>
    filter(mpg > 20) |>
    filter(cyl == 4) |>
    collect()

  # Compare two columns
  gpu_cars <- tbl_gpu(cars)
  fast_stops <- gpu_cars |>
    filter(dist < speed) |>
    collect()
}
#> Error in storage.mode(x) <- "double": 'list' object cannot be coerced to type 'double'
```
