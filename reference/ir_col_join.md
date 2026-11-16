# Create a side-tagged IR column reference for a non-equi `join_by()` condition (Phase 7 J7)

A plain
[`ir_col()`](https://bbtheo.github.io/cuplyr/reference/ir_col.md) node
is resolved against a *single* schema by name
([`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md)) –
fine for
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)/[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
which only ever see one table. A join condition (`join_by(a >= b)`)
references columns from TWO tables at once, and – crucially – a
self-join (`inner_join(x, x, join_by(a >= b))`) has IDENTICAL left/right
schemas, so a name alone can't say which side `a`/`b` belong to.
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)'s own
pre-normalized spec already tells us explicitly, though: `x`/`y` are
parallel vectors where `x[i]` is ALWAYS the left-table column and `y[i]`
is ALWAYS the right-table column for condition `i` (see
[`join_by_is_non_equi()`](https://bbtheo.github.io/cuplyr/reference/join_by_is_non_equi.md)'s
own roxygen, R/join.R).
[`join_cond_ir_from_spec()`](https://bbtheo.github.io/cuplyr/reference/join_cond_ir_from_spec.md)
(R/join.R) uses that knowledge directly when building each condition's
IR, tagging every `col` node with an explicit `side` (`0L` = left/x,
`1L` = right/y) at construction time – deliberately NOT the design doc's
original "resolve left-first-then-right by lookup order" sketch, which
silently breaks on a self-join.
[`ir_bind_join()`](https://bbtheo.github.io/cuplyr/reference/ir_bind_join.md)
then resolves `$index` against the correct side's schema using this tag,
never by cross-schema lookup.

## Usage

``` r
ir_col_join(name, side)
```

## Arguments

- name:

  Character scalar, the column name

- side:

  `0L` (left/x) or `1L` (right/y)

## Value

A `cuplyr_ir_col` node with an extra `side` field
