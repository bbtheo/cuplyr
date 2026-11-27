# Classify a call's effective grouping for the slice() family

Shared by the slice() verbs: resolves `by=`/`.by=` via
[`resolve_by()`](https://bbtheo.github.io/cuplyr/reference/resolve_by.md)
(which itself raises dplyr's own "Can't supply `by`..." error when
`.data` is already grouped) and returns which of three dispatch paths
applies.

## Usage

``` r
slice_group_mode(.data, by_quo, arg_name, call = rlang::caller_env())
```

## Arguments

- .data:

  A `tbl_gpu` object

- by_quo:

  A quosure for the verb's `by=`/`.by=` argument

- arg_name:

  Character scalar, `"by"` or `".by"` (interpolated into
  [`resolve_by()`](https://bbtheo.github.io/cuplyr/reference/resolve_by.md)'s
  grouped-`.data` error message, see its own docs)

- call:

  The calling verb's frame, for the `by`-on-grouped error

## Value

`list(mode = "none"|"group_by"|"by", group_cols = <chr>)`: `"none"` when
`.data` is ungrouped and no `by=`/`.by=` was supplied (the ungrouped GPU
kernels apply); `"group_by"` when `.data` is
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped;
`"by"` when `by=`/`.by=` was supplied (which implies `.data` itself is
ungrouped, since
[`resolve_by()`](https://bbtheo.github.io/cuplyr/reference/resolve_by.md)
would otherwise have already errored)
