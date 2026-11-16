# Parse `dplyr::recode(.x, old = new, ..., .default =, .missing =)`

A POSITIVE allowlist (F30): anything that doesn't match exactly returns
`NULL`, i.e. the CPU fallback, which already reproduces dplyr's results,
warnings and errors. Accepted: head resolving to
[`dplyr::recode`](https://dplyr.tidyverse.org/reference/recode.html);
`.x` a bare STRING/INT32/FLOAT64 (non-factor) column; scalar-literal
values of one shared typeof; string keys = names, numeric keys =
`as.double(names)` (whole int32 for an INT32 `.x`) or positions; no
duplicate keys; and, without `.default`, a value type equal to `.x`'s
(otherwise dplyr's data-dependent "Unreplaced values treated as NA"
warning must stay on the CPU). Lowers to one
`recode_lookup(x, keys, values, default, missing)` node (F61: the keys
and values are two vector literals, one hash lookup on the GPU instead
of a clause per key), every literal explicitly typed. See
`features/F30.md`.

## Usage

``` r
ir_parse_recode(expr, env, schema)
```
