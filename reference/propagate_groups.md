# Propagate group columns across an AST node

A `join` keeps `x`'s groups that survive in its output
([`push_join()`](https://bbtheo.github.io/cuplyr/reference/push_join.md)
computes that itself; this branch is only the defensive default).
`summarise` yields whatever `node$result_groups` says (computed by
`resolve_summarise_groups()`, R/summarise.R, from the node's `.groups=`
semantics; [`character()`](https://rdrr.io/r/base/character.html) for
`.groups = "drop"`/an ungrouped input, all pre-summarise groups for
`"keep"`, all-but-the-last for the default/ `"drop_last"`). Every other
node type keeps whichever of the existing `groups` still appear in the
node's (post-op) schema. This single rule reproduces today's per-verb
behavior (select drops de-selected group columns; filter/mutate/arrange
preserve all of them).

## Usage

``` r
propagate_groups(node, groups, schema)
```

## Arguments

- node:

  The AST node being applied

- groups:

  Character vector of the input's group columns

- schema:

  The node's output schema (post-op), as from
  [`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md)

## Value

Character vector of group columns to carry forward

## Details

`rename` needs one extra step before that final
[`intersect()`](https://generics.r-lib.org/reference/setops.html): a
group column that got renamed (e.g. `rename(gdf, grp = g)` where `g` is
grouped) must be translated to its NEW name first, or the plain
[`intersect()`](https://generics.r-lib.org/reference/setops.html) would
miss it entirely (`schema$names` has the new name, `groups` still has
the old one) and silently drop the grouping.
