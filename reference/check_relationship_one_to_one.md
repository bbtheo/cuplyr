# `relationship = "one-to-one"` violation

Checks BOTH directions (unlike one-to-many/many-to-one, which each check
only one). When both `left_multi`/`right_multi` indicate a violation,
the SMALLER 0-based index wins (ties go to `left`/x) – empirically
verified against dplyr 1.2.1 via
[`vctrs::vec_locate_matches()`](https://vctrs.r-lib.org/reference/vec_locate_matches.html)
directly (the underlying engine dplyr delegates to): the reported
offender is always whichever side's first duplicate appears earliest in
the (left_map, right_map)-sorted match sequence, not a fixed x-then-y or
y-then-x priority.

## Usage

``` r
check_relationship_one_to_one(left_multi, right_multi)
```
