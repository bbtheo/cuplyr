# Parse a lubridate accessor call, once [`ir_require_lubridate_fn()`](https://bbtheo.github.io/cuplyr/reference/ir_require_lubridate_fn.md) has already confirmed the call really invokes lubridate's own function

Every one of these 9 functions takes `x` as its sole required argument;
each also has 1-4 OPTIONAL controls (`label=`/`abbr=`/`locale=`
(month()/wday()), `week_start=` (wday() only), `with_year=`/
`fiscal_start=` (quarter() only)) that are NOT supported – any of them
present at all (even at its own default value) falls back, matched via
[`rlang::call_match()`](https://rlang.r-lib.org/reference/call_match.html)
against a prototype mirroring each function's REAL lubridate signature
(so `year(x)`/`year(x = d)` both match, but e.g.
`month(x, label = TRUE)` does not).

## Usage

``` r
ir_parse_lubridate_accessor(expr, env, schema, fn_name)
```

## Details

`x` must already be one of the 5 TIMESTAMP\_\* granularities (a STRING
or numeric `x` is not attempted natively, mirroring
[`as.Date()`](https://rdrr.io/r/base/as.Date.html)/
[`as.POSIXct()`](https://rdrr.io/r/base/as.POSIXlt.html)'s own scope).

`wday()`'s own default depends on the GLOBAL `lubridate.week.start`
option (`getOption("lubridate.week.start", 7)`) – verified empirically
(plain Rscript, no GPU needed): with the option at its own default (7,
i.e. Sunday), `lubridate::wday(d)` returns `Sun=1, Mon=2, ..., Sat=7`.
cuDF's `WEEKDAY` component (`datetime.hpp`) is ISO `Mon=1..Sun=7`;
`(w %% 7) + 1` maps cuDF's convention onto lubridate's exactly (verified
algebraically against the empirical ground truth: Mon
`(1 %% 7) + 1 = 2`; Sun `(7 %% 7) + 1 = 1`) – implemented in the C++
handler (`src/expr_eval.hpp`). If the user has changed the global option
away from its own default, this remap would be wrong, so `wday()`
additionally checks the CURRENT option value at parse time and falls
back if it isn't the default 7.

KNOWN, DOCUMENTED DIVERGENCE (see `test-dplyr-datetime.R`):
[`lubridate::second()`](https://lubridate.tidyverse.org/reference/second.html)
returns a FRACTIONAL value for a POSIXct with sub-second precision (e.g.
`second(x)` where `x` is `13:45:30.75` is `30.75`, not `30`) – cuDF's
`SECOND` component is always a whole integer, so `second()` on a
genuinely sub-second- precision column silently truncates natively
instead of falling back. Whole-second POSIXct values (the overwhelming
majority of real usage, and every `Date`/whole-second `POSIXct`) are
unaffected.
