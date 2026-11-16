# Resolve `keep`'s `NULL` default against a join spec

Ported directly from `dplyr:::join_cols()` (verified via
[`body()`](https://rdrr.io/r/base/body.html) against dplyr 1.2.1, Phase
7 J7 finding): `keep = NULL`'s real resolution is NOT a single
whole-join TRUE/FALSE decision – it is PER CONDITION. `join_cols()`'s
own `is_null(keep)` branch drops the right key column (and coalesces, no
suffix distinction) only for conditions whose `by$condition == "=="`; a
genuinely non-equi condition (`join_by(a >= b)`) always keeps BOTH
sides' key columns, suffixed like any other colliding column – verified
empirically: `join_by(c == d, a >= b)` drops `d` (the equi pair) but
keeps both `a` and `b` (the non-equi pair). An all-equi spec's
per-condition resolution collapses to "drop every right key" (this
package's historical `FALSE` behavior, unchanged); an all-non-equi
spec's per-condition resolution collapses to "keep every column,
suffixed uniformly" (this package's historical `TRUE`-forced behavior
for a pure non-equi join, also unchanged) – only a MIXED equi/non-equi
spec actually needs the per-condition machinery to differ from either
extreme, which is why this was never caught before J7 (no spec was ever
both equi and non-equi at once).

## Usage

``` r
resolve_join_keep(keep, spec)
```

## Arguments

- keep:

  `NULL`, `TRUE`, or `FALSE`, as passed by the user

- spec:

  A join spec as returned by
  [`parse_join_by()`](https://bbtheo.github.io/cuplyr/reference/parse_join_by.md)

## Value

`TRUE`/`FALSE`

## Details

Both `NULL` and a *legal* explicit `FALSE` resolve to the same `FALSE`
return value here:
[`build_join_output_info()`](https://bbtheo.github.io/cuplyr/reference/build_join_output_info.md)'s
own `keep != TRUE` branch always applies the per-condition
(`join_spec$op == "=="`) logic, which – per `dplyr:::join_cols()`'s own
`is_null`/`is_false` branches – are byte-identical whenever
`is_false(keep)` is even legal (i.e. an all-equi spec, where
`by$x[equi] == by$x` and `by$y[equi] == by$y` make the two branches'
cross-sets identical); explicit `keep = FALSE` on any spec with a
genuinely non-equi condition is a hard error with dplyr's own text,
exactly as before.
