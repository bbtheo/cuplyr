# Translate one R-side `window_spec()` into the plain list `gpu_window()` (`src/ops_window.cpp`) expects

`default_value`/`default_valid` collapse
[`window_spec()`](https://bbtheo.github.io/cuplyr/reference/window_spec.md)'s
`default` field (an IR literal node, or `NULL`) into the shape
`build_scalar()` (`src/expr_eval.hpp`) already accepts elsewhere: no
default supplied, and an explicit `default = NA` (`spec$default$na`),
both mean the same thing for
[`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/[`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html)
(an invalid/NA fill scalar of the value column's own type) – so both
collapse to `default_valid = FALSE`, `default_value = NULL`.

## Usage

``` r
build_window_cpp_spec(spec, out_pos, schema)
```

## Arguments

- spec:

  A
  [`window_spec()`](https://bbtheo.github.io/cuplyr/reference/window_spec.md)
  structure

- out_pos:

  0-based position of this spec among the whole `ast_window` node's
  appended columns

- schema:

  The node input's schema (`list(names=, types=)`)

## Value

A plain list:
`list(fn, value_idx, out_type, out_pos, n, default_value, default_valid, na_rm)`,
plus `roll_path` (the internal `cuplyr.roll_path` switch, see
[`?roll`](https://bbtheo.github.io/cuplyr/reference/roll.md)) for a
rolling spec
