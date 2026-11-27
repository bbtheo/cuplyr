# Is this `by=` spec natively lowerable end to end?

`TRUE` for everything
[`join_by_is_non_equi()`](https://bbtheo.github.io/cuplyr/reference/join_by_is_non_equi.md)
already called equi (no change there – the native hash-join path). For a
genuinely non-equi
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
object, native lowering additionally requires one of:

- A plain non-equi join (every condition's `filter` is `"none"`): every
  operator is one this package's AST lowering understands
  ([join_native_non_equi_ops](https://bbtheo.github.io/cuplyr/reference/join_native_non_equi_ops.md)).
  A non-equi condition comparing two STRING columns is native too:
  [`lower_join()`](https://bbtheo.github.io/cuplyr/reference/lower_join.md)
  maps both sides onto INT32 codes of one shared order
  ([`apply_join_key_casts()`](https://bbtheo.github.io/cuplyr/reference/apply_join_key_casts.md)),
  because cudf's AST evaluator can't compare strings. A STRING against
  anything else stays on the CPU fallback.

- A rolling join
  ([`join_by_rolling_native()`](https://bbtheo.github.io/cuplyr/reference/join_by_rolling_native.md)):
  exactly one `closest()` condition plus `==` conditions. A
  missing/misspelled column name is NOT checked here – left to
  [`validate_join_cols()`](https://bbtheo.github.io/cuplyr/reference/validate_join_cols.md)'s
  own proper error downstream on the native path (safer than silently
  routing an outright user typo to a fallback that would just re-raise
  the same error one layer later).

## Usage

``` r
join_by_natively_supported(by, x, y)
```
