# Validate a join spec's per-side column list exists in that side's schema

Deliberately does not require `cols` to be unique: a repeated column IS
a legitimate non-equi shape – `join_by(between(a, lo, hi))` compares the
SAME left column `a` against TWO different right columns (`lo` via `>=`,
`hi` via `<=`), so `join_spec$left` is `c("a", "a")` by construction
([`within()`](https://rdrr.io/r/base/with.html)/ `overlaps()` similarly
duplicate a column across their pre-expanded condition pairs) – this is
exactly what
[`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/
`overlaps()` need on the native path, not a user error to reject.

## Usage

``` r
validate_join_cols(cols, tbl, side)
```
