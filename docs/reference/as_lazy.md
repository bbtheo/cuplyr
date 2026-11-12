# Switch to lazy execution mode

Returns a `tbl_gpu` that will defer operations until `collect()` or
`compute()` is called.

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
if (has_gpu()) {
  # Create in eager mode, switch to lazy
  gpu_data <- tbl_gpu(mtcars) |>
    as_lazy() |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425)  # not yet executed

  # Execute with collect
  result <- gpu_data |> collect()
}
#> Error in mutate(filter(as_lazy(tbl_gpu(mtcars)), mpg > 20), kpl = mpg *     0.425): could not find function "mutate"
```
