# Apply a two-input join AST node, dispatching on exec_mode

The join analogue of
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md).
Joins can't reuse
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)
as-is because it is built around a single `.data` input: it reads
exactly one `exec_mode`/`ptr`/`schema` to decide the schedule. A join
has two independent inputs (`x`/`y`) whose *either* side being lazy
forces the whole join lazy (mirroring the pre-unification behavior), and
whose eager execution needs both sides' pointers anchored – see
[`join_input_node()`](https://bbtheo.github.io/cuplyr/reference/join_input_node.md).

## Usage

``` r
push_join(
  join_type,
  x,
  y,
  join_spec,
  suffix,
  keep,
  na_matches,
  multiple = "all",
  unmatched = "drop",
  relationship = NULL,
  cross_by = FALSE
)
```

## Arguments

- join_type:

  One of `"left"`, `"inner"`, `"full"`, `"right"`, `"semi"`, `"anti"`,
  `"cross"`

- x, y:

  The two `tbl_gpu` join inputs

- join_spec:

  List with `left`/`right`/`op`/`filter` (and, when key types differ,
  `key_casts`), as returned by
  [`parse_join_by()`](https://bbtheo.github.io/cuplyr/reference/parse_join_by.md)
  /
  [`resolve_join_key_types()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_key_types.md)

- suffix, keep, na_matches, multiple, unmatched, relationship:

  As documented on the join verbs (all are unused for `"semi"`/`"anti"`,
  which have no `suffix`/`keep`/`multiple`/`unmatched`/`relationship` in
  dplyr's own signature; `"cross"` has none of `keep`/`na_matches`/
  `multiple`/`unmatched`/`relationship` either, and always passes
  `keep = TRUE` internally purely to get
  [`build_join_output_info()`](https://bbtheo.github.io/cuplyr/reference/build_join_output_info.md)'s
  all-common-names-suffixed behavior – see
  [`cross_join.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/cross_join.tbl_gpu.md),
  `R/join.R`)

- cross_by:

  `TRUE` for dplyr's deprecated `by = character()` spelling: an ordinary
  `join_type` join over a constant key (see
  [`lower_cross_by_join()`](https://bbtheo.github.io/cuplyr/reference/lower_cross_by_join.md),
  `R/lower.R`)

## Value

A new `tbl_gpu`

## Details

Groups: every dplyr join ends with `dplyr_reconstruct(out, x)`, so for
EVERY join type the result is grouped by
`intersect(x$groups, names(out))` and keeps `x`'s `.drop` setting
(`x$group_drop`); `y`'s grouping is irrelevant. A group column that gets
suffixed away (a name collision, or the key under `keep = TRUE`) simply
drops out of the intersection; a merged key that is also a group column
stays grouped (with its coerced type). Semi/anti outputs are `x`'s
schema verbatim, so all of `x`'s groups survive.
