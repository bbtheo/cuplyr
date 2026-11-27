# Parse `replace_values(x, old_values ~ new_value, ..., from=, to=)`

Value-match clauses like
[`ir_parse_value_match_clauses()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_value_match_clauses.md)
(`x %in% old_values_i`), but TYPE STABLE: the result always keeps `x`'s
own type, and unmatched rows keep `x` itself (there is no
`.default=`/`default=` argument at all) – expressed as a dedicated
`ir_call("replace_when", ...)` node (see `type_replace_when()` and this
file's module comment above) rather than reusing `case_when`'s node,
since case_when's own common-type-unify rule is wrong here.

## Usage

``` r
ir_parse_replace_values(expr, env, schema)
```

## Details

Only the two-sided-formula clause interface is supported: `from=`/`to=`
(the vector-lookup-table interface) being present at all makes this
return `NULL` (fall back).
