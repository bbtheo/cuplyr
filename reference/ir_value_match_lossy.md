# Would dplyr's `vec_cast(from, x)` fail on a value-match clause's left side?

[`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)
/
[`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html)
/
[`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)
cast the matched values to `x`'s type and raise vctrs' lossy-cast error
(`1.5` for an integer `x`, `4L` for a logical one). The clause desugars
to `x %in% lhs`, which compares in the common type and would silently
match nothing, so a lossy literal makes the caller return `NULL` (CPU
fallback; dplyr raises its own error). Only logical / integer / double
`x` are checked here.

## Usage

``` r
ir_value_match_lossy(x_type, lhs_ir)
```
