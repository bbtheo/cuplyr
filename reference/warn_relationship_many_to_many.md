# The many-to-many auto-warning (`relationship = NULL`, the default)

Fires only when BOTH directions have a genuine duplicate (an x row
matching multiple y rows, AND a y row matching multiple x rows) –
verified empirically: a one-directional duplication pattern (e.g. many
x-rows sharing a key that matches a single y-row) is a many-to-one
pattern, not many-to-many, and does NOT warn. Mirrors
`dplyr:::rethrow_warning_join_relationship_many_to_many()`'s exact
message (3 `i` bullets) and class.

## Usage

``` r
warn_relationship_many_to_many(left_multi, right_multi)
```
