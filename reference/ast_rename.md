# Create a rename AST node

Schema-only: `old_names[i]` in the *input* schema becomes `new_names[i]`
in the output schema; every other column (position, type, and GPU column
data) is untouched. This is the one AST node type where lowering does no
GPU work at all (see
[`lower_rename()`](https://bbtheo.github.io/cuplyr/reference/lower_rename.md),
`R/lower.R`) – column names live entirely in the R-side schema
(`$schema$names`), never on the underlying `cudf::table`/`GpuTablePtr`
itself (confirmed by `gpu_collect()`, `src/transfer_io.cpp`, which takes
the R-side names as a parameter and applies them positionally to the
physical columns), so a pure rename never needs to touch the GPU
pointer.

## Usage

``` r
ast_rename(input, old_names, new_names, positions = NULL)
```

## Arguments

- input:

  Input AST node

- old_names:

  Character vector of column names (in `input`'s output schema) being
  renamed

- new_names:

  Character vector of the same length: `new_names[i]` is what
  `old_names[i]` becomes

- positions:

  `NULL` (rename by `old_names`), or the integer positions of the
  renamed columns: used instead of the names when the input has
  duplicated column names, where a name is ambiguous. Such a node is
  never pruned through by the optimizer.

## Value

An ast_rename node

## Details

Used directly by
[`rename()`](https://dplyr.tidyverse.org/reference/rename.html)/[`rename_with()`](https://dplyr.tidyverse.org/reference/rename.html)
(`R/rename.R`), and stacked on top of an
[`ast_select()`](https://bbtheo.github.io/cuplyr/reference/ast_select.md)
node by [`select()`](https://dplyr.tidyverse.org/reference/select.html)
(the `new = old` rename fix, `R/select.R`) and
[`relocate()`](https://dplyr.tidyverse.org/reference/relocate.html)
(`R/relocate.R`) whenever their tidyselect resolution renames a column.

Treated as an optimizer barrier (see
[`is_barrier()`](https://bbtheo.github.io/cuplyr/reference/is_barrier.md)
below): the simplest safe choice, since letting a filter/projection
above a rename push below it would need to translate column references
through `old_names`/ `new_names` – correct but not worth the complexity
for a schema-only verb. Optimization opportunity for later: translate
`required_cols`/ `pred$cols` through the name map instead of blocking
pushdown entirely.
