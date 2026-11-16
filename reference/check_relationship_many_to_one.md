# `relationship = "many-to-one"` violation: forbids an `x` row matching multiple `y` rows (i.e. `left_multi_first`) – ALWAYS reported as "x matches multiple in y" (mirror image of one-to-many above).

`relationship = "many-to-one"` violation: forbids an `x` row matching
multiple `y` rows (i.e. `left_multi_first`) – ALWAYS reported as "x
matches multiple in y" (mirror image of one-to-many above).

## Usage

``` r
check_relationship_many_to_one(left_multi)
```
