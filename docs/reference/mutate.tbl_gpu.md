# Create or modify columns in a GPU table

Adds new columns or modifies existing columns in a GPU table using
arithmetic expressions, similar to
[`dplyr::mutate()`](https://dplyr.tidyverse.org/reference/mutate.html).
All computations are performed on the GPU for maximum performance.

## Usage

``` r
# S3 method for class 'tbl_gpu'
mutate(.data, ...)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Name-value pairs of expressions. The name gives the column name (new
  or existing), and the value is an arithmetic expression involving
  existing columns and/or scalar values.

## Value

A `tbl_gpu` object with the new or modified columns. If a column name
already exists, it is replaced. New columns are appended.

## Details

### Supported arithmetic operators

- `+` - addition

- `-` - subtraction

- `*` - multiplication

- `/` - division

- `^` - exponentiation (power)

### Column replacement behavior

When the output column name matches an existing column, the existing
column is replaced in-place (preserving column order). For example,
`mutate(x = x + 1)` will modify `x` rather than creating a duplicate.

### Current limitations

- Only binary operations are supported (col op value or col op col)

- Complex expressions like `(x + y) * z` are not yet supported

- Functions like [`sqrt()`](https://rdrr.io/r/base/MathFun.html),
  [`log()`](https://rdrr.io/r/base/Log.html),
  [`abs()`](https://rdrr.io/r/base/MathFun.html) are not yet implemented

- Result type is always FLOAT64 (double precision)

### Performance

GPU arithmetic operations are highly vectorized and can process billions
of elements per second. Memory bandwidth is typically the limiting
factor, not compute.

## See also

[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md)
for filtering rows,
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for selecting columns,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Add a new column
  result <- gpu_mtcars |>
    mutate(kpl = mpg * 0.425) |>
    collect()

  # Modify an existing column
  adjusted <- gpu_mtcars |>
    mutate(mpg = mpg + 5) |>
    collect()

  # Combine two columns
  gpu_cars <- tbl_gpu(cars)
  result <- gpu_cars |>
    mutate(ratio = dist / speed) |>
    collect()

  # Chain multiple mutations
  result <- gpu_mtcars |>
    mutate(power_weight = hp / wt) |>
    mutate(efficiency = mpg * power_weight) |>
    collect()
}
#> Error in collect(mutate(gpu_mtcars, kpl = mpg * 0.425)): could not find function "collect"
```
