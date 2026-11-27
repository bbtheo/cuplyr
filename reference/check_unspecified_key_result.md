# After a mutating join: dplyr's error for a merged all-NA x key and no rows

`join_mutate()` casts the merged key of x's ORIGINAL (sliced) column to
the common type; an empty slice of a plain logical column is no longer
unspecified, so dplyr raises `vctrs_error_cast` when the join returns no
rows. A
[`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/vctrs-unspecified.html)
key, `keep = TRUE`, a non-equi pair and the unspecified side being y do
not.

## Usage

``` r
check_unspecified_key_result(ast, ptr)
```
