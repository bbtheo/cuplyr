# `unmatched = "error"`, x-side: "Each row of x must have a match in y"

Mirrors `dplyr:::rethrow_error_join_matches_nothing()` exactly (also the
handler for `na_matches`-driven "incomplete" x-side violations –
`rethrow_error_join_matches_incomplete()` just delegates to this same
function, so no separate na_matches-specific class/text is needed here).

## Usage

``` r
abort_join_matches_nothing(left_unmatched)
```
