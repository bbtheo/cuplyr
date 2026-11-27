# Optimize AST respecting barrier nodes

A plan is optimized one segment at a time: the nodes above a barrier
(down to it), then the barrier, then what is below it as a plan of its
own. Each segment is told what its consumer reads of its output, so the
columns a barrier does not need are pruned below it as well.

## Usage

``` r
optimize_with_barriers(ast, required = NULL)
```

## Arguments

- ast:

  Root AST node

- required:

  Columns the consumer of `ast` reads from its output (`NULL`: all of
  them)

## Value

Optimized AST
