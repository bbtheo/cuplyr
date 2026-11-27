# Resolve a bare untyped `NA` literal's type from a sibling value-position arg

[`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)/[`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)/[`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html)/[`na_if()`](https://dplyr.tidyverse.org/reference/na_if.html)
each have several "value position" argument slots that must share a
common output type (if_else's yes/no/missing; case_when's val_i's +
default; coalesce's whole arg list; na_if's x/y). A bare, untyped `NA`
in one of those slots (e.g. `if_else( cond, x, NA)`) must adopt a
*sibling* value-position arg's type – exactly like arithmetic's
`1L + NA` being integer, generalized from 2 args to N.

## Usage

``` r
ir_resolve_value_nas(nodes, schema)
```

## Arguments

- nodes:

  A list of IR nodes – the "value position" args only (e.g. if_else()'s
  yes/no/missing, never its cond)

- schema:

  List with `names`/`types`

## Value

`nodes`, with every isolated bare-NA node's `type` field filled in from
the first non-isolated sibling's inferred type (returned unchanged if
none, or all, of `nodes` are isolated)

## Details

This is done here, at PARSE time (mutating each isolated node's own
`type` field directly, in place), rather than relying on
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)'s
own generic isolated-NA substitution (which already exists for
arithmetic's 2-arg case): that generic mechanism picks the FIRST
non-isolated arg among a call node's ENTIRE `args` list, which for
`if_else` would incorrectly be `cond` (always the first arg, always
non-isolated, and NOT a value-position arg) rather than a true sibling
value. Calling this here on ONLY the value-position args (never `cond`)
avoids that pitfall entirely, and once it runs, the isolated node's
`type` field is no longer `NULL` – so
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)'s
later generic walk of the finished node is a no-op for these args, as
intended.

A typed `NA` (e.g. `NA_character_`, `NA_integer_`) is untouched: it
already carries its own `type` and is not "isolated" by this function's
definition (matching dplyr: `na_if(numeric_x, NA_character_)` still
errors, only a bare untyped `NA` adopts a sibling's type).
