# Is a `join_by()` object non-equi?

`TRUE` only for a real `dplyr_join_by` object (class from
[`dplyr::join_by()`](https://dplyr.tidyverse.org/reference/join_by.html))
with at least one condition other than `"=="` (an inequality/rolling
condition, e.g. `join_by(a >= b)`, `join_by(between(...))`) or any
`filter` other than `"none"` (a `closest()` join). Empirically verified
against dplyr 1.2.1 (`str(join_by(...))` for `join_by(a)`,
`join_by(a == b, c >= d)`, `join_by(x$a == y$b)`,
`join_by(between(a, lo, hi))`, `join_by(closest(a >= b))`,
`join_by(within(...))`, `join_by(overlaps(...))`): the object is a
PRE-NORMALIZED `list(exprs, condition, filter, x, y)` where
`condition`/`filter`/`x`/`y` are already parallel, per-condition
character vectors (NOT a single scalar `filter` –
[`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/
`overlaps()` each pre-expand to two `condition` entries with
`filter = c("none", "none")`, and mixing `closest()` with a plain equi
condition in the same call, e.g. `join_by(closest(a >= b), c == d)`,
produces `filter = c("max", "none")`). This function's "any" check is
deliberate: a
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html) call
mixing an equi condition with even ONE non-equi/closest condition must
route the WHOLE join to the CPU fallback (see the callers in this file),
since there is no such thing as a partially-equi join in this package's
hash-join path.

## Usage

``` r
join_by_is_non_equi(by)
```
