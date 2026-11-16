# Would this join's output have a duplicate column name?

With the
[`add_suffixes()`](https://bbtheo.github.io/cuplyr/reference/add_suffixes.md)
port above, this can only happen when at least one `suffix` element is
the literal empty string `""` AND that side's would-be-suffixed name
genuinely collides with the other side (see
[`add_suffixes()`](https://bbtheo.github.io/cuplyr/reference/add_suffixes.md)'s
own roxygen for why `""` is special: dplyr itself resolves the resulting
name clash via a name-keyed column *overwrite* downstream in
`join_mutate()`, not a rename or an error) – with any other suffix
value,
[`add_suffixes()`](https://bbtheo.github.io/cuplyr/reference/add_suffixes.md)'s
iterate-until-unique loop guarantees
[`build_join_output_info()`](https://bbtheo.github.io/cuplyr/reference/build_join_output_info.md)'s
`names` has no duplicates. Callers use this to detect the degenerate
case up front (computable from schemas alone, no GPU work needed) and
route the whole join to the CPU fallback instead, since cuplyr's
fixed-column-count gather has no native way to reproduce "the later
column's data silently replaces the earlier same-named column, and the
total column count drops by one".

## Usage

``` r
join_output_would_collide(left_schema, right_schema, join_spec, suffix, keep)
```
