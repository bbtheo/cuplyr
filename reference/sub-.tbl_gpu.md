# Subset a GPU table with tibble's `[` semantics

`x[i, j]` on a `tbl_gpu` follows tibble's `[` rules, and for a grouped
table dplyr's rules for a grouped data frame:

- `x[j]` selects columns; `x[i, j]`, `x[i, ]`, `x[, j]`, `x[]` are the
  2-d forms. `j` is resolved first (names, positions, negative, logical)
  with tibble's `vctrs_error_subscript_*` errors, messages and calls.

- Rows (`i`): positive positions gather in the given order (duplicates
  kept; out-of-range or `NA` gives an all-`NA` row), `0` is dropped,
  negative positions drop rows (out-of-range ones are ignored), a
  logical recycles from length 1 or must have length `nrow` (`NA` gives
  an `NA` row), a character is read as a row position (no row names).

- `drop = TRUE` with exactly one result column returns that column as a
  vector (the only shape that executes a lazy table); with several
  columns it is ignored, except that a grouped table comes back
  ungrouped. In the 1-d form `drop=` warns and is ignored (a grouped
  table is still ungrouped, as in dplyr).

## Usage

``` r
# S3 method for class 'tbl_gpu'
x[i, j, drop = FALSE]
```

## Arguments

- x:

  A `tbl_gpu` object.

- i:

  Row subscript (or, in the 1-d form, the column subscript).

- j:

  Column subscript.

- drop:

  If `TRUE` and the result has one column, return the column.

## Value

A `tbl_gpu`, or a vector when `drop = TRUE` selects one column.

## Details

Row selection runs on the GPU as one gather, and is added to the plan of
a lazy table like any verb; column selection works as
[`select.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
does. The result keeps the group variables that survive, and the table's
`.drop` setting.

Row subscripts are validated against the real row count at call time
whenever that is known without executing anything (an eager table, or a
lazy one with no pending operations). On a lazy table with pending
operations the row subscript is resolved when the plan runs, so its
errors (same class, message and `[` call) surface at
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)/[`compute()`](https://dplyr.tidyverse.org/reference/compute.html).

A zero-column result (`x[, 0]`, `x[character()]`, `x[, FALSE]`) is an
ungrouped zero-column table that keeps the row count (a lazy table with
pending operations is computed first).

Divergences from tibble, all loud errors: duplicated columns
(`x[, c("a", "a")]`) because a `tbl_gpu` cannot represent them, and
matrix subscripts (`x[m]`, and the soft-deprecated one-column matrix row
subscript `x[m, ]`, which tibble still accepts with a deprecation
notice).
