# Switch to lazy execution mode

Returns a `tbl_gpu` that will defer operations until
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) or
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) is
called.

## Usage

``` r
as_lazy(.data)
```

## Arguments

- .data:

  A `tbl_gpu` object.

## Value

A `tbl_gpu` in lazy execution mode.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  # Create in eager mode, switch to lazy
  gpu_data <- tbl_gpu(mtcars, lazy = FALSE) |>
    as_lazy() |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425)  # not yet executed

  # Execute with collect
  result <- gpu_data |> collect()
}
```
