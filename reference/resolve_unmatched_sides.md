# Split a validated `unmatched=` value into its x-side/y-side components

Mirrors `dplyr:::check_unmatched()`'s own splitting (validation itself
already happened in
[`validate_join_unmatched()`](https://bbtheo.github.io/cuplyr/reference/validate_join_unmatched.md)):
length 1 applies identically to both sides (even though, per join type,
only one side may ever actually be consulted – see
[`check_join_cardinality()`](https://bbtheo.github.io/cuplyr/reference/check_join_cardinality.md));
length 2 (only ever passed for
[`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html))
assigns positionally, `c(x, y)`.

## Usage

``` r
resolve_unmatched_sides(unmatched)
```
