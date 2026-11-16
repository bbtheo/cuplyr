# Parse `case_match(.x, old_values ~ new_value, ..., .default=, .ptype=)`

See
[`ir_parse_value_match_clauses()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_value_match_clauses.md)
–
[`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html)
is deprecated in dplyr 1.2.1 in favor of
[`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)/[`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)
(verified: calling it emits a lifecycle warning), but is still exported
and functional, and is explicitly in this wave's scope, so it's
implemented identically to
[`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)'s
formula interface, just with `.default` as the named key instead of
`default`.

## Usage

``` r
ir_parse_case_match(expr, env, schema)
```
