# Pull a single column from a GPU table

Extracts one column as a plain R vector, similar to
[`dplyr::pull()`](https://dplyr.tidyverse.org/reference/pull.html).
Materializes any pending lazy operations first via
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html), then
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)s the
table and picks out the requested column.

## Usage

``` r
# S3 method for class 'tbl_gpu'
pull(.data, var = -1, name = NULL, ...)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- var:

  A variable, as a name, position (positive or negative, counting from
  the end), or expression evaluated against column *positions* (see
  [`dplyr::pull()`](https://dplyr.tidyverse.org/reference/pull.html)'s
  own contract). Defaults to the last column.

- name:

  An optional column (by the same rules as `var`) whose values become
  the names of the returned vector.

- ...:

  Not used; for extensibility, must be empty.

## Value

A plain R vector (not a `tbl_gpu`), named if `name` was supplied.
Grouping (if any) is ignored, matching
[`dplyr::pull()`](https://dplyr.tidyverse.org/reference/pull.html).

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)
  gpu_mtcars |> pull(mpg)
}
#>  [1] 21.0 21.0 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 17.8 16.4 17.3 15.2 10.4
#> [16] 10.4 14.7 32.4 30.4 33.9 21.5 15.5 15.2 13.3 19.2 27.3 26.0 30.4 15.8 19.7
#> [31] 15.0 21.4
```
