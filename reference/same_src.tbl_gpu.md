# Source compatibility and upload for tbl_gpu

`tbl_gpu` methods for dplyr's
[`dplyr::same_src()`](https://dplyr.tidyverse.org/reference/same_src.html)
and
[`dplyr::auto_copy()`](https://dplyr.tidyverse.org/reference/auto_copy.html)
generics, used by the join verbs' `copy =` handling.

## Usage

``` r
# S3 method for class 'tbl_gpu'
same_src(x, y)

# S3 method for class 'tbl_gpu'
auto_copy(x, y, copy = FALSE, ...)
```

## Arguments

- x:

  A `tbl_gpu`.

- y:

  Object to test or upload.

- copy:

  Passed by dplyr;
  [`auto_copy()`](https://dplyr.tidyverse.org/reference/auto_copy.html)
  only reaches this method when copying is allowed.

- ...:

  Not used.

## Value

[`same_src()`](https://dplyr.tidyverse.org/reference/same_src.html):
`TRUE` when `y` is a `tbl_gpu`.
[`auto_copy()`](https://dplyr.tidyverse.org/reference/auto_copy.html):
`y` as a `tbl_gpu`.
