# Add a count column without collapsing rows

Like
[`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md),
but keeps every row and column of `x` and adds the group's count (or
weighted sum) as a new column on every row of the group, as
[`dplyr::add_count()`](https://dplyr.tidyverse.org/reference/count.html)
does. It runs on the GPU: rows keep their original order, and `NA` keys
form one group, as in dplyr.

## Usage

``` r
# S3 method for class 'tbl_gpu'
add_count(x, ..., wt = NULL, sort = FALSE, name = NULL, .drop = NULL)
```

## Arguments

- x:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Column names or expressions to group by, as
  [`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md).
  If `x` is already grouped, they are added to its grouping for the
  count; the result keeps `x`'s own grouping.

- wt:

  Optional weight column/expression, as
  [`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md).

- sort:

  If `TRUE`, sorts the result in descending order of the count column.
  No rows are collapsed, so every row of a group ties with the others;
  the sort is stable and keeps their order.

- name:

  The name of the count column. If `NULL` (default), uses `"n"`, or
  `"nn"`/`"nnn"`/... if `"n"` already names any column of `x` (where
  [`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md)
  and
  [`tally.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tally.tbl_gpu.md)
  look at the grouping columns only), as in dplyr.

- .drop:

  Deprecated in dplyr 1.0.0; supplying it is an error, as in dplyr.

## Value

A `tbl_gpu` with the same rows as `x`, plus the count column, grouped as
`x` is.

## See also

[`count.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md)

## Examples

``` r
library(dplyr)
#> 
#> Attaching package: ‘dplyr’
#> The following objects are masked from ‘package:cuplyr’:
#> 
#>     bind_cols, bind_rows
#> The following objects are masked from ‘package:stats’:
#> 
#>     filter, lag
#> The following objects are masked from ‘package:base’:
#> 
#>     intersect, setdiff, setequal, union
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)
  gpu_mtcars |> add_count(cyl) |> collect()
}
#>     mpg cyl  disp  hp drat    wt  qsec vs am gear carb  n
#> 1  21.0   6 160.0 110 3.90 2.620 16.46  0  1    4    4  7
#> 2  21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4  7
#> 3  22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1 11
#> 4  21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1  7
#> 5  18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2 14
#> 6  18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1  7
#> 7  14.3   8 360.0 245 3.21 3.570 15.84  0  0    3    4 14
#> 8  24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2 11
#> 9  22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2 11
#> 10 19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4  7
#> 11 17.8   6 167.6 123 3.92 3.440 18.90  1  0    4    4  7
#> 12 16.4   8 275.8 180 3.07 4.070 17.40  0  0    3    3 14
#> 13 17.3   8 275.8 180 3.07 3.730 17.60  0  0    3    3 14
#> 14 15.2   8 275.8 180 3.07 3.780 18.00  0  0    3    3 14
#> 15 10.4   8 472.0 205 2.93 5.250 17.98  0  0    3    4 14
#> 16 10.4   8 460.0 215 3.00 5.424 17.82  0  0    3    4 14
#> 17 14.7   8 440.0 230 3.23 5.345 17.42  0  0    3    4 14
#> 18 32.4   4  78.7  66 4.08 2.200 19.47  1  1    4    1 11
#> 19 30.4   4  75.7  52 4.93 1.615 18.52  1  1    4    2 11
#> 20 33.9   4  71.1  65 4.22 1.835 19.90  1  1    4    1 11
#> 21 21.5   4 120.1  97 3.70 2.465 20.01  1  0    3    1 11
#> 22 15.5   8 318.0 150 2.76 3.520 16.87  0  0    3    2 14
#> 23 15.2   8 304.0 150 3.15 3.435 17.30  0  0    3    2 14
#> 24 13.3   8 350.0 245 3.73 3.840 15.41  0  0    3    4 14
#> 25 19.2   8 400.0 175 3.08 3.845 17.05  0  0    3    2 14
#> 26 27.3   4  79.0  66 4.08 1.935 18.90  1  1    4    1 11
#> 27 26.0   4 120.3  91 4.43 2.140 16.70  0  1    5    2 11
#> 28 30.4   4  95.1 113 3.77 1.513 16.90  1  1    5    2 11
#> 29 15.8   8 351.0 264 4.22 3.170 14.50  0  1    5    4 14
#> 30 19.7   6 145.0 175 3.62 2.770 15.50  0  1    5    6  7
#> 31 15.0   8 301.0 335 3.54 3.570 14.60  0  1    5    8 14
#> 32 21.4   4 121.0 109 4.11 2.780 18.60  1  1    4    2 11
```
