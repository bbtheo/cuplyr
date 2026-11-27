# dplyr's "Join columns must be unique" check (`check_join_vars()`)

Per side, equi columns plus the UNIQUE non-equi columns must not repeat:
a column may appear in several non-equi conditions
(`join_by(a >= lo, a <= hi)`) but not twice among the equi ones, nor in
both an equi and a non-equi condition. dplyr raises a plain
`rlang_error`; x is checked before y. Runs on the raw `by` at the verb
call (before routing to the fallback), so eager, lazy and fallback joins
all fail the same way. Also raises dplyr's `keep = FALSE` + non-equi
error first, and, per side after its duplicates, dplyr's "Join columns
in `x` must be present in the data." (so x's missing column precedes y's
duplicates, as in dplyr's `join_cols()`).

## Usage

``` r
check_join_cols_unique(
  by,
  x,
  keep = NULL,
  call = rlang::caller_env(),
  y = NULL
)
```

## Arguments

- by:

  The user's `by=` (`NULL`, character,
  [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html))

- x:

  The left table

- keep:

  The user's `keep=` (`NULL` for filtering joins)

- call:

  The verb's frame, reported as the error call

- y:

  The right table, or `NULL` to leave its columns unchecked
