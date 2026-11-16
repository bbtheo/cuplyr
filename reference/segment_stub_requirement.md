# What a segment reads of the stub at its bottom

The columns of the pruning select that projection pushdown put directly
on the stub (`pruned = TRUE`: it exists because some stub columns are
not read), or `NULL` when the node on the stub is anything else: every
stub column is then read or passed on. A select higher up is not looked
at: one above a filter keeps fewer columns than the filter reads (the
filter case wraps itself in a select to drop its predicate columns), and
a user select on the stub keeps only what the stub delivers anyway.

## Usage

``` r
segment_stub_requirement(segment)
```

## Arguments

- segment:

  An optimized segment ending in a stub source

## Value

Character vector or `NULL`
