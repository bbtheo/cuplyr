# Can this `closest()` join run on `gpu_rolling_join()` (F24)?

A POSITIVE allowlist; everything else (several `closest()`, `closest()`
next to another inequality, an unknown operator, factor/logical/Date-vs-
POSIXct rolling keys) takes the CPU fallback:

- exactly one condition has `filter != "none"`, its operator is one of
  `>=`, `>`, `<=`, `<`, and every other condition is `==` with
  `filter == "none"`;

- the rolling pair is two columns of one type in join_rolling_types, or
  INT32 against FLOAT64 (compared as FLOAT64 through `key_casts`). A
  column missing from a schema is left to
  [`validate_join_cols()`](https://bbtheo.github.io/cuplyr/reference/validate_join_cols.md).

## Usage

``` r
join_by_rolling_native(by, x, y)
```
