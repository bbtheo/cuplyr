# Parse a join specification into a 4-vector spec

Returns `list(left, right, op, filter)`: four parallel character
vectors, one entry per key/condition pair. `op` is each condition's
operator (`"=="` for every key of a character/`NULL` `by`), `filter` its
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
filter: `"none"`, or `"max"`/`"min"` for a `closest()` condition (F24).
Callers run
[`join_by_natively_supported()`](https://bbtheo.github.io/cuplyr/reference/join_by_natively_supported.md)
first, so a
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html) that
reaches here is one the native lowering handles:
[`join_equi_spec()`](https://bbtheo.github.io/cuplyr/reference/join_equi_spec.md)
and
[`join_cond_spec()`](https://bbtheo.github.io/cuplyr/reference/join_cond_spec.md)
split it into its equi/non-equi halves at lowering time
(`R/lower.R::lower_join()`), and a `closest()` condition goes to
`gpu_rolling_join()`.

## Usage

``` r
parse_join_by(by, x, y)
```
