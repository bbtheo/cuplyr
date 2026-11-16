# Parse a lubridate floor_date()/ceiling_date()/round_date() call (F46)

Native only for a UTC-equivalent POSIXct with unit
second/minute/hour/day and a Date with day/week (week starting Sunday,
the default `lubridate.week.start`), `unit` a single character literal
in one of timechange's spellings with no multiplier. Anything else
(other zones, `week_start`/`change_on_boundary` arguments, calendar
units, a Date with a sub-day unit, which lubridate turns into a POSIXct,
unknown spellings) returns `NULL` and runs on the CPU.

## Usage

``` r
ir_parse_lubridate_rounding(expr, env, schema, fn_name)
```
