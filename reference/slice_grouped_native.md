# GPU-native grouped/`by=` slice() family desugar (Phase 5, task W7)

Implements every
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped
or `by=`/`.by=` on-the-fly-grouped
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)
call by composing the existing window/mutate/filter/arrange/select AST
machinery (Phase 5 W1-W6) – ZERO new GPU primitives. See this file's
module docs for the empirically-verified semantics this reproduces
(group order, NA handling, `by=`'s first-appearance rule).

## Usage

``` r
slice_grouped_native(
  .data,
  group_mode,
  group_cols,
  mode,
  amount = NULL,
  is_prop = NULL,
  raw_indices = NULL,
  order_quo = NULL,
  descending = NULL,
  with_ties = NULL,
  na_rm = NULL
)
```

## Arguments

- .data:

  A `tbl_gpu` object (grouped, or ungrouped with `by=`/`.by=` already
  resolved by the caller)

- group_mode:

  `"group_by"` or `"by"`, as returned by
  [`slice_group_mode()`](https://bbtheo.github.io/cuplyr/reference/slice_group_mode.md)

- group_cols:

  Character vector of group column names

- mode:

  One of `"head"`, `"tail"`, `"index"`, `"rank"`

- amount, is_prop:

  For `mode` `"head"`/`"tail"`/`"rank"`: as resolved by
  `resolve_slice_amount()`

- raw_indices:

  For `mode == "index"`: as returned by `resolve_slice_dots()`

- order_quo:

  For `mode == "rank"`: a quosure for `order_by`

- descending:

  For `mode == "rank"`: `FALSE` for
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html),
  `TRUE` for
  [`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)

- with_ties, na_rm:

  For `mode == "rank"`: as documented on
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)

## Value

A new `tbl_gpu`, or `NULL` if this call needs the CPU fallback (caller's
responsibility to re-run the real dplyr verb)

## Details

The node sequence, in order (only the parts relevant to `mode`/
`group_mode` are actually emitted):

1.  (`mode == "rank"`, computed `order_by`) `ast_mutate`: materialize
    the computed `order_by` expression under a temp name.

2.  (`mode == "rank"`) `ast_mutate`: `..slice_na.. = is.na(order_col)` –
    the NA-last sort key `order_by` needs (see module docs on why a
    plain `desc(order_col)` arrange can't be used here).

3.  (`group_mode == "by"`) `ast_window` (UNGROUPED):
    `..slice_rowid.. = row_number()` over the whole table, then
    `ast_window` (grouped): `..slice_first.. = min(..slice_rowid..)` –
    each group's own first-appearance position, broadcast to every row
    of that group.

4.  `ast_arrange`: group order key first (`group_cols` ascending for
    `group_mode == "group_by"`; `..slice_first..` ascending for `"by"`),
    then (`mode == "rank"` only) `..slice_na..` ascending, then
    `order_col` in the requested direction.

5.  (`mode == "rank" && with_ties`) `ast_mutate`:
    `..slice_valid.. = if_else(..slice_na.., 0L, 1L)`.

6.  `ast_window` (grouped by `group_cols`): whichever of
    `..slice_gs.. = n()`, `..slice_rn.. = row_number()` (frame:
    `group_cols`, no order – physically pre-arranged above, so bare
    [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
    already reproduces `order_col`'s ranking for rank mode),
    `..slice_mr.. = min_rank(order_col)` (frame: `group_cols`,
    `order_col` in the requested direction),
    `..slice_nn.. = sum(..slice_valid..)` are needed for this
    `mode`/`with_ties` combination.

7.  (needs `..slice_gs..`) `ast_mutate`: `..slice_size.. =` the same
    `n=`/`prop=` clamp rule `compute_slice_size()` (C++) uses for the
    ungrouped path, expressed as IR (`gs` is a real per-group column
    here, not a compile-time scalar).

8.  `ast_filter`: the mode-specific predicate (see module docs' desugar
    table).

9.  `ast_select`: back down to the ORIGINAL column set (captured before
    any of the above), dropping every `..slice_*..`/computed- `order_by`
    temp column in one step.

Two-pass (plan, then emit): every failure mode that triggers a CPU
fallback (`order_by` the IR can't parse, an unsupported
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html) index
shape) is checked BEFORE any
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)
call, so a rejected call never partially executes GPU work ("never
half-lower").
