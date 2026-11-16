# Shared clause-parsing helper for case_match()/recode_values() (Phase 4 wave 2)

Both functions share an identical clause shape once "match by value" is
expressed as `%in%`:
`f(x, old_values_1 ~ new_value_1, ..., <default_key> = value)`. Builds
`x %in% old_values_i` as each clause's condition (reusing the existing
`%in%` op verbatim, including its NA-matching semantics) and hands the
interleaved cond/value list to `ir_call( "case_when", ...)` directly –
case_match()/recode_values() are, at the IR level, indistinguishable
from case_when() once desugared this way.

## Usage

``` r
ir_parse_value_match_clauses(expr, env, schema, default_key)
```

## Arguments

- expr:

  The raw `case_match(...)`/`recode_values(...)` call

- env:

  The quosure's environment

- schema:

  List with `names`/`types`

- default_key:

  `".default"` (case_match) or `"default"` (recode_values) – the one
  named control argument this wave supports

## Value

An `ir_call("case_when", ...)` node, or `NULL` if unsupported

## Details

Only the two-sided-formula clause interface is implemented this wave:
recode_values()'s `from=`/`to=`/`unmatched=`/`ptype=`
vector-lookup-table interface, and case_match()'s `.ptype=`, are NOT
implemented – ANY named argument other than `default_key` present in the
call makes this return `NULL` (caller falls back).

A clause's LHS ("old value(s)") must resolve to a literal (a constant, a
`c(...)` vector, or an environment lookup) – an LHS referencing a column
is unsupported this wave (mirrors `%in%`'s own RHS restriction, since
that's exactly the op this desugars into). A STRING-vs-non-STRING type
mismatch between `x` and a clause's LHS is diagnosed here (mirroring
[`na_if()`](https://dplyr.tidyverse.org/reference/na_if.html)'s own
STRING compatibility check) rather than surfacing as a raw cudf cast
error at lowering time.
