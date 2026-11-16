# Validate `multiple=`, replicating dplyr's exact bad-value text

Note dplyr's own message (from
[`vctrs::vec_locate_matches()`](https://vctrs.r-lib.org/reference/vec_locate_matches.html))
omits the offending value for this argument specifically (unlike
`unmatched=`/ `relationship=`/`na_matches=`, which all name it) – this
is dplyr's actual behavior, verified empirically against dplyr 1.2.1,
not an oversight here.

## Usage

``` r
validate_join_multiple(multiple)
```
