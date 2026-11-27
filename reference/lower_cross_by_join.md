# Lower a `by = character()` join

dplyr runs its ordinary join machinery with a constant key
(`join_rows(cross = TRUE)`), so every x row matches every y row. That is
a cartesian product when both sides have rows (`gpu_cross_join()`,
x-major), but an empty side behaves like an outer join: left/full keep x
with NA y columns, right/full keep y with NA x columns, inner gives 0
rows; semi returns x iff y has rows, anti the opposite. Row counts are
only known here, at execution time, so the empty-side cases are decided
in R. Deliberately NOT a hash join on a constant key column (a hash
build over n identical keys is quadratic).

## Usage

``` r
lower_cross_by_join(ast, left_ptr, right_ptr, left_schema, right_schema)
```

## Details

`multiple = "first"/"any"/"last"` keep y's first/last row for each x
row; for right/full the other y rows are then unmatched and appended
with NA x columns, like dplyr. `relationship`/`unmatched` run through
[`check_join_cardinality()`](https://bbtheo.github.io/cuplyr/reference/check_join_cardinality.md)
on a diag synthesized from the two row counts; the default relationship
is `"many-to-many"` (no warning).
