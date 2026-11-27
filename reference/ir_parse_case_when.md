# Parse a `case_when()` call into IR

`case_when(cond1 ~ val1, cond2 ~ val2, ..., .default = val)` dots are
two-sided formulas (`~` calls), not ordinary expressions – so unlike
every other `ir_call_registry` entry, this is invoked directly from
`ir_parse_expr()`'s call-handling branch (see the
`identical(fn_name, "case_when")` check there) rather than going through
the registry's generic per-arg parse loop. Recognizes exactly: any
number of two-sided `cond ~ value` clauses, plus an optional
`.default = value` named argument. Returns `NULL` (unsupported, caller
falls back per the normal contract) for anything else, which isn't
supported: `.ptype=`, `.size=`, `.unmatched=`, one-sided formulas,
non-formula dots, or zero clauses.

## Usage

``` r
ir_parse_case_when(expr, env, schema)
```

## Arguments

- expr:

  The raw `case_when(...)` call

- env:

  The quosure's environment (conditions/values are parsed against this,
  exactly like any other IR subtree)

- schema:

  List with `names`/`types`

## Value

An `ir_call("case_when", ...)` node, or `NULL` if unsupported

## Details

Normalizes into a single flat `ir_call("case_when", args)` where `args`
is `list(cond1, val1, cond2, val2, ..., valN, default)` – always
terminated by a default (a bare untyped-NA literal is synthesized when
the user didn't supply `.default=`, matching dplyr's "unmatched rows are
a typed NA" rule). `meta$n_when` records the clause count for
diagnostics/[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html);
the C++ side derives the same count from `length(args)` being odd
(`2 * n_when + 1`), so `meta` isn't load-bearing there.

Bare untyped `NA` literals among the value positions (any `val_i` or the
`.default`/synthesized default) are resolved against a sibling value's
type via
[`ir_resolve_value_nas()`](https://bbtheo.github.io/cuplyr/reference/ir_resolve_value_nas.md)
– the same value-position type adoption if_else()/coalesce() use,
generalized to `case_when`'s N+1 value slots.
