# Create a GPU-backed data frame

Transfers an R data frame to GPU memory using NVIDIA's libcudf library,
enabling high-performance data manipulation operations. The resulting
`tbl_gpu` object can be used with dplyr verbs like
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`select()`](https://dplyr.tidyverse.org/reference/select.html), and
collected back to R with
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

## Usage

``` r
tbl_gpu(data, ...)

# S3 method for class 'data.frame'
tbl_gpu(data, ...)

# S3 method for class 'tbl_gpu'
tbl_gpu(data, ...)
```

## Arguments

- data:

  A data frame or tibble to transfer to GPU memory. Supported column
  types include: numeric (double), integer, character, and logical.

- ...:

  Additional arguments passed to methods (currently unused).

## Value

A `tbl_gpu` object containing:

- `ptr` - External pointer to the GPU table

- `schema` - List with column names and types

- `lazy_ops` - Pending operations (for future lazy evaluation)

- `groups` - Grouping variables (for future group_by support)

## Details

The data is immediately copied to GPU memory when `tbl_gpu()` is called.
GPU memory is automatically freed when the R object is garbage
collected.

Column type mappings from R to GPU:

- `numeric` -\> FLOAT64

- `integer` -\> INT32

- `character` -\> STRING

- `logical` -\> BOOL8 (stored as INT32)

- `Date` -\> TIMESTAMP_DAYS

- `POSIXct` -\> TIMESTAMP_MICROSECONDS

## See also

[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
to transfer data back to R,
[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md),
[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md),
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for data manipulation

## Examples

``` r
if (has_gpu()) {
  # Transfer mtcars to GPU
  gpu_mtcars <- tbl_gpu(mtcars)
  print(gpu_mtcars)

  # Chain operations
  result <- gpu_mtcars |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425) |>
    collect()
}
#> Rows: 32
#> Columns: 11
#> $ mpg  <dbl> 21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2
#> $ cyl  <dbl> 6, 6, 4, 6, 8, 6, 8, 4, 4, 6
#> $ disp <dbl> 160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6
#> $ hp   <dbl> 110, 110, 93, 110, 175, 105, 245, 62, 95, 123
#> $ drat <dbl> 3.9, 3.9, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92
#> $ wt   <dbl> 2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44
#> $ qsec <dbl> 16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20, 22.9, 18.3
#> $ vs   <dbl> 0, 0, 1, 1, 0, 1, 0, 1, 1, 1
#> $ am   <dbl> 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
#> $ gear <dbl> 4, 4, 4, 3, 3, 3, 3, 4, 4, 4
#> $ carb <dbl> 4, 4, 1, 1, 2, 1, 4, 2, 2, 4
#> Error in collect(mutate(filter(gpu_mtcars, mpg > 20), kpl = mpg * 0.425)): could not find function "collect"
```
