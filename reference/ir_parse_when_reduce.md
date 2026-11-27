# Parse `when_all(..., na_rm=FALSE, size=NULL)` / `when_any(...)`

See this file's module comment above for the full empirical rationale.
Fully desugars at parse time into a right-folded chain of the existing
`|`/`&` ops – no new IR op, registry entry, or C++ handler needed.

## Usage

``` r
ir_parse_when_reduce(expr, env, schema, fn_name)
```

## Arguments

- fn_name:

  `"when_all"` or `"when_any"`

## Details

`size=` (only meaningful for the zero-input edge case) is NOT supported
– its presence, or zero dots, makes this return `NULL`.
