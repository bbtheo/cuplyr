# Align one side of a set operation to the unified target schema

A thin, laziness-preserving wrapper over
[`align_to_schema()`](https://bbtheo.github.io/cuplyr/reference/align_to_schema.md)
(R/bind.R): that function assumes an already-*materialized* table (it
reads/casts `tbl$ptr`/`tbl$schema` directly, with no AST awareness – the
same reason
[`bind_rows_gpu()`](https://bbtheo.github.io/cuplyr/reference/bind_rows_gpu.md)
materializes every lazy input before calling it), so this only
materializes (and only then calls
[`align_to_schema()`](https://bbtheo.github.io/cuplyr/reference/align_to_schema.md))
when
[`current_schema()`](https://bbtheo.github.io/cuplyr/reference/current_schema.md)
(the lazy-AST-aware schema, unlike `tbl$schema`) actually differs from
the target – the common case (x/y already schema-identical) skips both
entirely, preserving whatever laziness `tbl` already had. Names/types
alone don't catch every case that needs a real align pass: two
`DICTIONARY32` columns can share the type string while their level
vectors differ (the level-union remap path), so
[`factor_levels_match_target()`](https://bbtheo.github.io/cuplyr/reference/factor_levels_match_target.md)
(R/bind.R) is checked too.
[`align_to_schema()`](https://bbtheo.github.io/cuplyr/reference/align_to_schema.md)'s
own return also resets `$groups` to
[`character()`](https://rdrr.io/r/base/character.html) – restored here
from the caller-supplied `groups`; its `$schema$factor_levels` is reset
to whatever it computed on its own slow path (unset on the fast-path
passthrough), so it's overwritten unconditionally here with the
already-computed `target_schema$factor_levels` for consistency either
way.

## Usage

``` r
align_set_op_side(tbl, target_schema, groups = character())
```
