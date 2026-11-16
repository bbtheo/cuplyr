# Generate a column name guaranteed not to collide with `existing`

Appends an incrementing integer suffix to `base` until the candidate
name is not already present in `existing`. The very first candidate
tried is the bare `base` itself (no suffix); collisions start the suffix
at `1`. Shared by every part of the package that needs a synthetic temp
column name:
[`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)'s
computed `order_by` column (`R/slice.R`, via
`unique_slice_order_name()`), and the window decomposition's
`..win*..`/`..winarg*..` temp columns below.

## Usage

``` r
unique_temp_name(base, existing)
```

## Arguments

- base:

  Character scalar, the desired base name (e.g. `"..win.."`)

- existing:

  Character vector of names already in use

## Value

A character scalar not in `existing`
