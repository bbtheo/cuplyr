# `cross_join()`: native via `cudf::cross_join`

Native GPU implementation (Phase 7 J3). Unlike every other join verb,
dplyr's own
[`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
signature has no `by`/`keep`/`na_matches`/
`multiple`/`unmatched`/`relationship` at all (verified via
`args(dplyr::cross_join)`: `x, y, ..., copy, suffix`) – there are no
keys to match, every row of `x` is paired with every row of `y`.
Verified empirically against dplyr 1.2.1:

- Output column order is left-major: all of `x`'s columns (in their
  original order), then all of `y`'s (in their original order).

- Every COMMON column name (not just would-be join keys, since there are
  none) gets BOTH suffixes – e.g. two tables both having `id`/`val`
  produces `id.x`/`val.x`/`id.y`/`val.y`, never a bare shared name. This
  is exactly
  [`build_join_output_info()`](https://bbtheo.github.io/cuplyr/reference/build_join_output_info.md)'s
  own behavior for an all-common spec with `keep = TRUE` (its
  `drop_right` is always `character(0)` when `keep = TRUE`, regardless
  of what the spec's `left`/`right` contain), so an empty join spec (no
  keys at all) plus `keep = TRUE` reproduces it exactly – see
  [`ast_join()`](https://bbtheo.github.io/cuplyr/reference/ast_join.md)'s
  `by`/`keep` fields below.

- Grouping follows the rule shared by every join
  ([`push_join()`](https://bbtheo.github.io/cuplyr/reference/push_join.md),
  `R/execute.R`): `x`'s groups that survive in the output keep grouping
  (dplyr's `cross_join.data.frame()` ends with
  `dplyr_reconstruct(x_out, x)`); a group column whose name collides
  with `y`'s and gets suffixed away (`"g"` -\> `"g.x"`) drops out. `y`'s
  grouping is irrelevant.

- 0-row `x` or `y` produces a 0-row result with the correct (merged)
  schema – cudf's
  [`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
  handles this natively, no R-side special-casing needed.

- `copy = TRUE` with a plain `data.frame` `y` works like every other
  join.

## Usage

``` r
# S3 method for class 'tbl_gpu'
cross_join(x, y, ..., copy = FALSE, suffix = c(".x", ".y"))
```

## Arguments

- x, y:

  `tbl_gpu` objects. If `y` is not a `tbl_gpu`, set `copy = TRUE` to
  upload it automatically.

- ...:

  Not used; dplyr itself requires these to be empty.

- copy:

  Whether to upload a non-`tbl_gpu` `y` via
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- suffix:

  Length-2 character vector disambiguating every common column name
  (every shared name gets both suffixes, since there are no join keys to
  coalesce). `suffix = c("", ...)` colliding with a genuine duplicate
  name falls back to CPU evaluation, same as the mutating joins'
  degenerate-suffix case.

## Value

A `tbl_gpu` with every row of `x` paired with every row of `y`
(`nrow(x) * nrow(y)` rows), columns from `x` first then `y`. `x`'s
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
grouping is preserved for any group column whose name survives suffixing
unchanged.

## Details

`gpu_cross_join()` (`src/ops_join.cpp`) throws if either side has 0
*columns* (verified in the `cudf/join/join.hpp` doc comment: "throw
cudf::logic_error if the number of columns in either left or right table
is 0") – guarded here with a clearer, dplyr-flavored message before ever
reaching the GPU call, for both the eager and lazy schedules (the
0-column case is knowable from the schemas alone, no execution needed).
