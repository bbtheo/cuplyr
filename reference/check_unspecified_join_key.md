# Check that a logical join key is all `NA` (#70)

dplyr gives a logical key that is all `NA` (vctrs "unspecified") the
type of the key it is joined to. Whether the column is one is a matter
of its values, so it is checked when the join runs: all null with rows,
or a
[`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/vctrs-unspecified.html)
column (`cast$marker`), passes; anything else has no common type and
raises the incompatible-type error.
[`apply_join_key_casts()`](https://bbtheo.github.io/cuplyr/reference/apply_join_key_casts.md)
checks the pairs in `by` order, each on its source column before any
temp copy is made, so the first bad pair is the one named.

## Usage

``` r
check_unspecified_join_key(ptr, idx, cast, by)
```
