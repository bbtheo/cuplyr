# Bind multiple data frames/tables by row

Combines objects vertically by stacking rows. For tbl_gpu objects,
operations are performed on the GPU. For other objects, delegates to
dplyr.

## Usage

``` r
bind_rows(..., .id = NULL)
```

## Arguments

- ...:

  Objects to bind (tbl_gpu, data.frame, or lists of these, flattened as
  in
  [`dplyr::bind_rows()`](https://dplyr.tidyverse.org/reference/bind_rows.html)).
  `NULL`s are dropped.

- .id:

  Optional column name to identify source tables

## Value

Combined data frame or tbl_gpu

## Details

If any input (including one inside a list) is a `tbl_gpu`, the result is
a `tbl_gpu`, wherever the `tbl_gpu` sits in the arguments.
[`dplyr::bind_rows()`](https://dplyr.tidyverse.org/reference/bind_rows.html)
and
[`purrr::map_dfr()`](https://purrr.tidyverse.org/reference/map_dfr.html)
on a `tbl_gpu` input are handed to this function and give the same
result;
[`vctrs::vec_rbind()`](https://vctrs.r-lib.org/reference/vec_bind.html)
and
[`purrr::list_rbind()`](https://purrr.tidyverse.org/reference/list_c.html)
can't be intercepted and raise an error for a `tbl_gpu` (use
`dplyr::bind_rows(list)` instead of `list_rbind()`).
