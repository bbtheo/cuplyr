# Does `node` have one value per group in dplyr?

TRUE for a scalar literal,
[`n()`](https://dplyr.tidyverse.org/reference/context.html),
[`cur_group_id()`](https://dplyr.tidyverse.org/reference/context.html)
(the helper column `substitute_cur_group_id()` made of it), an aggregate
([`sum()`](https://rdrr.io/r/base/sum.html),
[`mean()`](https://rdrr.io/r/base/mean.html), ... of anything, a column
included: `sum(k)` is one value), an elementwise call whose non-literal
arguments all are, and a shift/scan/rank/rolling call whose value
argument is. FALSE for a bare column (a stored column has n rows in
dplyr), bare
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html),
and an elementwise call with a column or
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
among its arguments.

## Usage

``` r
ir_is_group_scalar(node)
```
