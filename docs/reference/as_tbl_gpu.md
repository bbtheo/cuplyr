# Coerce to a GPU table

Converts a data frame or compatible object to a `tbl_gpu` object,
transferring data to GPU memory.

## Usage

``` r
as_tbl_gpu(x, ...)
```

## Arguments

- x:

  A data frame or object coercible to a data frame.

- ...:

  Additional arguments passed to
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

## Value

A `tbl_gpu` object with data stored on the GPU.

## See also

[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) for
details on GPU table creation

## Examples

``` r
if (has_gpu()) {
  gpu_df <- as_tbl_gpu(iris)
  print(gpu_df)
}
#> Rows: 150
#> Columns: 5
#> $ Sepal.Length <dbl> 5.1, 4.9, 4.7, 4.6, 5, 5.4, 4.6, 5, 4.4, 4.9
#> $ Sepal.Width  <dbl> 3.5, 3, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1
#> $ Petal.Length <dbl> 1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5
#> $ Petal.Width  <dbl> 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1
#> $ Species      <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
```
