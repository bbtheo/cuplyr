# Build a single, `&`-folded IR condition tree from a join spec's non-equi subset (Phase 7 J7)

One `ir_call(op, list(x_col, y_col))` comparison node per condition
entry (`cond_spec$left[i] <op> cond_spec$right[i]`), each column tagged
with its side via
[`ir_col_join()`](https://bbtheo.github.io/cuplyr/reference/ir_col_join.md)
(`0L` = left/x, `1L` = right/y – straight from `cond_spec`'s own
parallel `left`/`right` vectors, which
[`parse_join_by()`](https://bbtheo.github.io/cuplyr/reference/parse_join_by.md)
populated from
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)'s own
`x`/`y` fields; see
[`ir_bind_join()`](https://bbtheo.github.io/cuplyr/reference/ir_bind_join.md)'s
docs for why this explicit tag – not a lookup-order convention – is
required for a self-join to bind correctly), folded together with `"&"`
exactly like
[`lower_filter()`](https://bbtheo.github.io/cuplyr/reference/lower_filter.md)
folds a filter node's predicate list into one expression. `cond_spec`
must be non-empty (callers only invoke this when
`length(cond_spec$op) > 0`, i.e. the join_by() has at least one
genuinely non-equi condition – see
[`lower_join()`](https://bbtheo.github.io/cuplyr/reference/lower_join.md),
R/lower.R).

## Usage

``` r
join_cond_ir_from_spec(cond_spec)
```

## Arguments

- cond_spec:

  The non-equi subset of a join spec, as returned by
  [`join_cond_spec()`](https://bbtheo.github.io/cuplyr/reference/join_cond_spec.md)

## Value

An unbound IR node (run
[`ir_bind_join()`](https://bbtheo.github.io/cuplyr/reference/ir_bind_join.md)
before lowering)
