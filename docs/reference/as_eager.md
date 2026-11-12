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
if (has_gpu()) {
  # Start lazy, switch to eager mid-pipeline
  result <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20) |>
    as_eager() |>
    mutate(x = hp * 2) |>  # executes immediately
    collect()
}
#> Error in collect(mutate(as_eager(filter(tbl_gpu(mtcars, lazy = TRUE),     mpg > 20)), x = hp * 2)): could not find function "collect"
```
