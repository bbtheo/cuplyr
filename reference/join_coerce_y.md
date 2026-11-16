# Coerce `y` to a `tbl_gpu` (shared by every join verb)

Goes through
[`dplyr::auto_copy()`](https://dplyr.tidyverse.org/reference/auto_copy.html):
a `tbl_gpu` `y` is returned as is, a non-GPU `y` is uploaded when
`copy = TRUE`, and with `copy = FALSE` dplyr's "`x` and `y` must share
the same src" error is raised.

## Usage

``` r
join_coerce_y(x, y, copy)
```
