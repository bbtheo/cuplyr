# Lower distinct node

`gpu_distinct()` (`src/ops_distinct.cpp`) always returns every column of
its input, deduped on `ast$key_cols` (`cudf::stable_distinct()`, which
preserves input row order and keeps the first occurrence of each key –
matching dplyr's contract exactly). `ast$key_cols`'s positions in the
(already-mutated, if the caller had computed key expressions) input
schema are unchanged by the dedup itself, so the same `key_idx` used to
build the dedup key can be reused to project down to just those columns,
in the caller's requested order, for the `.keep_all = FALSE` case.

## Usage

``` r
lower_distinct(ast, source_ptr)
```
