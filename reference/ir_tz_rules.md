# How each POSIXct-valued op gets its zone (F32)

Verified against dplyr 1.2 / vctrs 0.7 on plain tibbles, for local,
absent, UTC and New York operands:

## Usage

``` r
ir_tz_rules
```

## Format

A named list of op names per rule

## Details

- `x`: the result is `x` itself, a slice of it, or other values cast to
  `x`'s type (`vec_cast(y, x)`), so it has `x`'s zone (absent becomes
  `""`): [`na_if()`](https://dplyr.tidyverse.org/reference/na_if.html),
  [`replace_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)
  (also
  [`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)),
  [`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/
  [`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html) (with
  `default = `),
  [`first()`](https://dplyr.tidyverse.org/reference/nth.html)/[`last()`](https://dplyr.tidyverse.org/reference/nth.html)/[`nth()`](https://dplyr.tidyverse.org/reference/nth.html),
  [`min()`](https://rdrr.io/r/base/Extremes.html)/
  [`max()`](https://rdrr.io/r/base/Extremes.html),
  [`mean()`](https://rdrr.io/r/base/mean.html)/[`median()`](https://rdrr.io/r/stats/median.html)
  (F41), and cuplyr's
  [`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md).

- `common`: vctrs' common type of the value operands in order
  ([`tz_fold()`](https://bbtheo.github.io/cuplyr/reference/tz_fold.md)),
  then absent becomes `""`:
  [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)
  (true, false, missing),
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)
  (values in order, then `.default`),
  [`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html).

- `as_posixct`: a POSIXct keeps its zone as is, a Date becomes UTC
  midnight (R's `as.POSIXct.Date`).

- `shift` (F46): `ts_add_us`/`ts_sub_us` keep `x`'s zone attribute as is
  (an absent one stays absent).

- `round_date` (F46): `round_date()` of a local or absent zone returns
  the session zone's name; floor_date()/ceiling_date() (in `x`) keep
  `""`.

Not listed, so CPU: unary `-`,
[`abs()`](https://rdrr.io/r/base/MathFun.html),
[`cummax()`](https://rdrr.io/r/base/cumsum.html)/[`cummin()`](https://rdrr.io/r/base/cumsum.html)
(R's `Math.POSIXt` rejects them, so dplyr errors). A bare `t +/- n`
never reaches here: its type check sends it to the CPU
([`infer_mutate_output_type()`](https://bbtheo.github.io/cuplyr/reference/infer_mutate_output_type.md));
only the rounding calls rewrite it (`ir_posixct_shift()`).
