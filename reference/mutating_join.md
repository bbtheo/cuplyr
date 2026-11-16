# Shared body of `left_join()`/`inner_join()`/`full_join()`/`right_join()`

`unmatched = NULL` means the verb has no `unmatched=` argument
([`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)):
it is not validated, not forwarded to the CPU fallback, and the node
gets `"drop"`.

## Usage

``` r
mutating_join(
  join_type,
  x,
  y,
  by,
  copy,
  suffix,
  keep,
  na_matches,
  multiple,
  unmatched,
  relationship,
  user_env
)
```

## Details

Order of checks: the unsupported-`by` CPU fallback; argument validation;
key-type resolution (raises `dplyr_error_join_incompatible_type`, A3);
the factor-level-conflict and suffix-collision fallbacks; then, for
`by = character()` only, dplyr's deprecation warning (emitted last so a
fallback, whose dplyr call warns by itself, never warns twice).
