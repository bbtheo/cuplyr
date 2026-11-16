# Validate `unmatched=`, replicating dplyr's exact bad-value text

`max_length` is 2 only for
[`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
(independent x-side/y-side checks); every other join type allows length
1 only.
[`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
has no `unmatched=` parameter at all in dplyr (it never drops rows), so
it never calls this.

## Usage

``` r
validate_join_unmatched(unmatched, max_length = 1L)
```
