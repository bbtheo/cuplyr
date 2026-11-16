# Set operations on `tbl_gpu` tables

GPU-native implementations of dplyr's row-set operations, composed
entirely from existing verbs
([`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)/[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html)/[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)/
[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html))
– see the file header comment in `R/sets.R` for the exact composition
and the empirically-verified row-order/column- compatibility contracts
(mirroring `dplyr:::is_compatible()` / `check_compatible()`).

## Usage

``` r
# S3 method for class 'tbl_gpu'
union(x, y, ...)

# S3 method for class 'tbl_gpu'
union_all(x, y, ...)

# S3 method for class 'tbl_gpu'
intersect(x, y, ...)

# S3 method for class 'tbl_gpu'
setdiff(x, y, ...)

# S3 method for class 'tbl_gpu'
symdiff(x, y, ...)

# S3 method for class 'tbl_gpu'
setequal(x, y, ...)
```

## Arguments

- x, y:

  `tbl_gpu` objects. `y` may also be a plain `data.frame` (uploaded
  automatically via
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)) –
  unlike the join verbs, these generics have no `copy=` argument.

- ...:

  Not used; must be empty (dplyr itself requires this too).

## Value

A `tbl_gpu` with `x`'s column order/grouping (see the file header
comment for the exact row-order contract per verb), except
[`setequal()`](https://generics.r-lib.org/reference/setops.html) which
returns a plain `TRUE`/`FALSE`.

## Details

Two factor columns with different level sets (level-union remap,
`gpu_remap_codes()`) and a factor column vs. a character column (decode
to labels, `gpu_decode_factor()`) are both GPU-native as of Phase 11 L2.
A `Date` column vs. a `POSIXct` column (dplyr promotes to `POSIXct`) is
ALSO GPU-native as of Phase 11 L3 (`gpu_cast_column()` gained the
`TIMESTAMP_DAYS` -\> `TIMESTAMP_MICROSECONDS` cast, among the other 4
timestamp-granularity targets) – no column-type combination that real
dplyr accepts still needs the CPU fallback (`$needs_fallback` is always
`FALSE`, see
[`check_set_op_compatible()`](https://bbtheo.github.io/cuplyr/reference/check_set_op_compatible.md)).
Every column-type mismatch that's a genuine error in real dplyr (e.g.
integer vs. character, or factor vs. integer) errors here too, naming
the offending column and both types.
