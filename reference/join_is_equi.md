# Is a join spec entirely equi (`==`)?

A spec with no condition pairs at all (`length(spec$op) == 0`) counts as
equi – vacuously true, and matches `join_is_equi()`'s only current
caller
([`resolve_join_keep()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_keep.md))
needing a sensible answer for the natural-join (`by = NULL`, zero common
columns already rejected earlier) edge case.

## Usage

``` r
join_is_equi(spec)
```
