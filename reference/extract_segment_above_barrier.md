# Extract segment above a barrier (returns copy)

Extract segment above a barrier (returns copy)

## Usage

``` r
extract_segment_above_barrier(ast, barrier, stub)
```

## Arguments

- ast:

  Root node

- barrier:

  Target barrier node

- stub:

  Source node that takes the barrier's place

## Value

Copy of segment with `stub` at the barrier point
