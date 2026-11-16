# Validate `suffix=`, replicating dplyr's exact bad-value text

Mirrors `dplyr:::standardise_join_suffix()` exactly (verified
empirically against dplyr 1.2.1): must be a length-2 character vector
with no `NA` element. A single literal `""` element is legal (it means
"don't suffix this side at all" – see
[`add_suffixes()`](https://bbtheo.github.io/cuplyr/reference/add_suffixes.md)'s
own roxygen) and is NOT rejected here.

## Usage

``` r
validate_join_suffix(suffix)
```
