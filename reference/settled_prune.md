# `prune_dead_columns()` on a settled chain

What the pass returns for a chain
[`settled_chain()`](https://bbtheo.github.io/cuplyr/reference/settled_chain.md)
accepts: the same nodes, each mutate's expressions replaced by
`exprs[keep]` with every `keep` TRUE. That subset drops the list's
attributes but its names (the parse-time flags `parse_mutate_dots()`
sets), so a mutate whose list has other attributes is rebuilt with it; a
node nothing changes, with nothing changed below it, is returned as it
is.

## Usage

``` r
settled_prune(ast)
```

## Arguments

- ast:

  A chain
  [`settled_chain()`](https://bbtheo.github.io/cuplyr/reference/settled_chain.md)
  accepted

## Value

The pruned chain
