# Check if a tbl_gpu uses lazy execution

Check if a tbl_gpu uses lazy execution

## Usage

``` r
is_lazy(.data)
```

## Arguments

- .data:

  A `tbl_gpu` object.

## Value

Logical. `TRUE` if the table is in lazy execution mode.

## Examples

``` r
if (has_gpu()) {
  lazy_tbl <- tbl_gpu(mtcars)
  is_lazy(lazy_tbl)   # TRUE

  eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)
  is_lazy(eager_tbl)  # FALSE
}
#> [1] FALSE
```
