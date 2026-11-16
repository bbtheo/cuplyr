# Bind multiple data frames/tables by column

Combines objects horizontally by adding columns. For tbl_gpu objects,
operations are performed on the GPU. For other objects, delegates to
dplyr.

## Usage

``` r
bind_cols(
  ...,
  .name_repair = c("unique", "universal", "check_unique", "minimal")
)
```

## Arguments

- ...:

  Objects to bind (tbl_gpu, data.frame, vectors, or lists of these,
  flattened recursively as in
  [`dplyr::bind_cols()`](https://dplyr.tidyverse.org/reference/bind_cols.html)).
  A vector becomes one column named by its argument name. `NULL`s are
  dropped.

- .name_repair:

  How to handle duplicate column names: `"unique"`, `"universal"`,
  `"check_unique"` or `"minimal"`

## Value

Combined data frame or tbl_gpu

## Details

If any input (including one inside a list) is a `tbl_gpu`, the result is
a `tbl_gpu`, wherever the `tbl_gpu` sits in the arguments.
[`dplyr::bind_cols()`](https://dplyr.tidyverse.org/reference/bind_cols.html)
on a `tbl_gpu` input is handed to this function and gives the same
result for the four `.name_repair` strings above; any other
`.name_repair` (a `*_quiet` string, a function or a formula) is an error
for a `tbl_gpu`.
[`vctrs::vec_cbind()`](https://vctrs.r-lib.org/reference/vec_bind.html)
and
[`purrr::list_cbind()`](https://purrr.tidyverse.org/reference/list_c.html)
can't be intercepted and raise an error for a `tbl_gpu` (use
`dplyr::bind_cols(list)` instead of `list_cbind()`).
