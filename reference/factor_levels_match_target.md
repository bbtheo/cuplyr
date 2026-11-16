# Do a table's factor columns already carry the target's exact level sets?

Helper for
[`align_to_schema()`](https://bbtheo.github.io/cuplyr/reference/align_to_schema.md)'s
already-aligned fast path:
[`identical()`](https://rdrr.io/r/base/identical.html) on
`names`/`types` alone isn't enough for a `DICTIONARY32` column – two
tables can agree on the type string while their actual level VECTORS
differ (Bug 2, Phase 11 L2), which still needs a code remap even though
no type-string cast is triggered.

## Usage

``` r
factor_levels_match_target(
  source_factor_levels,
  target_factor_levels,
  names,
  types
)
```
