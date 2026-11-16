# Switch to eager execution mode

Computes any pending operations and returns a `tbl_gpu` that will
execute all subsequent operations immediately.

## Usage

``` r
as_eager(.data)
```

## Arguments

- .data:

  A `tbl_gpu` object.

## Value

A `tbl_gpu` in eager execution mode.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  # Tables are lazy by default; switch to eager mid-pipeline
  result <- tbl_gpu(mtcars) |>
    filter(mpg > 20) |>
    as_eager() |>
    mutate(x = hp * 2) |>  # executes immediately
    collect()
}
```
