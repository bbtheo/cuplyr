# Summarise groups in a GPU table

Computes aggregations on groups defined by
[`dplyr::group_by()`](https://dplyr.tidyverse.org/reference/group_by.html).
Operations are performed entirely on the GPU for maximum performance.

## Usage

``` r
# S3 method for class 'tbl_gpu'
summarise(.data, ..., .groups = "drop")

# S3 method for class 'tbl_gpu'
summarize(.data, ..., .groups = "drop")
```

## Arguments

- .data:

  A grouped `tbl_gpu` object created by
  [`dplyr::group_by()`](https://dplyr.tidyverse.org/reference/group_by.html).

- ...:

  Name-value pairs of summary functions. The name will be the name of
  the variable in the result. The value must be a single aggregation
  expression in the form `fun(column)`.

- .groups:

  Controls grouping structure of the result. Currently only "drop" is
  supported (default).

## Value

A `tbl_gpu` object with one row per group containing the grouping
columns and computed aggregations.

## Details

### Supported aggregation functions

- `sum(x)` - Sum of values

- `mean(x)` - Arithmetic mean

- `min(x)` - Minimum value

- `max(x)` - Maximum value

- [`n()`](https://dplyr.tidyverse.org/reference/context.html) - Count of
  rows in each group

- `sd(x)` - Standard deviation

- `var(x)` - Variance

### NA handling

By default, NA values are excluded from aggregations. This matches the
default behavior of R's base aggregation functions.

### Ungrouped summarise

If `.data` is not grouped, summarise will compute aggregations over all
rows, returning a single-row table.

## See also

[`group_by.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
for grouping data,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Simple aggregation over all rows
  total <- gpu_mtcars |>
    summarise(avg_mpg = mean(mpg)) |>
    collect()

  # Grouped aggregation
  by_cyl <- gpu_mtcars |>
    group_by(cyl) |>
    summarise(
      avg_mpg = mean(mpg),
      max_hp = max(hp),
      count = n()
    ) |>
    collect()

  # Multiple grouping columns
  by_cyl_gear <- gpu_mtcars |>
    group_by(cyl, gear) |>
    summarise(
      mean_mpg = mean(mpg),
      min_wt = min(wt)
    ) |>
    collect()
}
#> Error in collect(summarise(gpu_mtcars, avg_mpg = mean(mpg))): could not find function "collect"
```
