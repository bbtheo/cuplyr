# `relationship = "one-to-many"` violation: forbids a `y` row matching multiple `x` rows (i.e. `right_multi_first`) – ALWAYS reported as "y matches multiple in x", regardless of any `x`-side duplication that may also exist (dplyr's own `rethrow_error_join_relationship_one_to_many()` hardcodes this direction, no `which`-based branch, unlike one-to-one).

`relationship = "one-to-many"` violation: forbids a `y` row matching
multiple `x` rows (i.e. `right_multi_first`) – ALWAYS reported as "y
matches multiple in x", regardless of any `x`-side duplication that may
also exist (dplyr's own `rethrow_error_join_relationship_one_to_many()`
hardcodes this direction, no `which`-based branch, unlike one-to-one).

## Usage

``` r
check_relationship_one_to_many(right_multi)
```
