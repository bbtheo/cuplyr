# Would this join's equi keys compare two factor columns with different level sets?

Native GPU joins match equi keys by comparing their PHYSICAL INT32 codes
directly
([`resolve_join_key_types()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_key_types.md)
only checks
[`identical()`](https://rdrr.io/r/base/identical.html)/numeric-
compatible type strings, never level compatibility) – if both sides of
an equi condition are factor columns whose level VECTORS differ, two
logically-equal labels can be encoded under different codes (e.g. `"a"`
is code 1 on the left but code 2 on the right), so comparing raw codes
would silently produce wrong matches. This is detectable from schemas
alone, before any GPU op runs – every join verb
([`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/
[`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/[`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/[`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)/[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html))
calls this right after
[`resolve_join_key_types()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_key_types.md)
and routes the WHOLE join to the CPU fallback when it's `TRUE`,
mirroring
[`join_output_would_collide()`](https://bbtheo.github.io/cuplyr/reference/join_output_would_collide.md)'s
existing pattern (Phase 8 sets.R precedent:
[`check_set_op_compatible()`](https://bbtheo.github.io/cuplyr/reference/check_set_op_compatible.md)
uses the identical identical-levels-vs-not split to decide native vs.
fallback). Both-factor-with-IDENTICAL-levels keys compare correctly
natively (same code domain) and are not flagged here.

## Usage

``` r
join_factor_key_levels_conflict(x_schema, y_schema, join_spec)
```
