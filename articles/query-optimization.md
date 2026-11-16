# Query optimization in cuplyr

cuplyr tables are lazy by default. A verb on a lazy table doesn’t run.
It adds a node to a tree, the AST, that describes the query. When
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) or
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) runs
the tree, an optimizer first rewrites it into a plan that gives the same
result. An eager table, `tbl_gpu(df, lazy = FALSE)`, runs each verb at
once, so the optimizer never sees it.

This article shows what the optimizer does. It doesn’t time anything.
For timings, see
[`vignette("performance", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/performance.md).

``` r

library(cuplyr)
library(dplyr)

# Prints only the optimized plan of explain(), to keep the output short
optimized <- function(x) {
  out <- capture.output(explain(x))
  cat(out[(grep("^Optimized plan", out) + 1):length(out)], sep = "\n")
}
```

## Reading the tree

[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
prints the pending tree, one `|` per level, with the `source` node at
the bottom and the last verb at the top.
[`explain()`](https://dplyr.tidyverse.org/reference/explain.html) prints
the same tree and then the optimized plan. Neither runs anything.

``` r

pipeline <- tbl_gpu(mtcars) |>
  filter(mpg > 20) |>
  mutate(power_ratio = hp / wt) |>
  select(mpg, cyl, power_ratio) |>
  arrange(desc(mpg))

show_query(pipeline)
#> Pending operations:
#> ast_arrange [1 keys]
#> | ast_select [3 cols]
#> | | ast_mutate [1 expressions]
#> | | | ast_filter [1 predicates]
#> | | | | ast_source [11 cols]
#> 
#> AST depth: 5 
#> Node count: 5
```

Most verbs add one node of the same name. The exceptions:

| Verb | AST node |
|----|----|
| [`select()`](https://dplyr.tidyverse.org/reference/select.html) that renames | `select` with a `rename` above it |
| [`head()`](https://rdrr.io/r/utils/head.html), ungrouped [`slice()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html) | `slice` |
| [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) | `barrier` |
| [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) on columns | none, grouping is table metadata |
| `group_by(x = a + 1)` | `mutate` |
| window functions such as [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html) | `window`, plus other nodes |

Some calls add several nodes. A grouped
[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html)
records `arrange`, `window`, `mutate`, `filter` and `select`.

## The passes

The optimizer runs these passes in order:

1.  Projection pruning adds `select` nodes that drop columns nothing
    later reads.
2.  Mutate fusion merges a `mutate` with the `mutate` directly below it.
3.  Dead-column pruning drops `mutate` expressions and `select` columns
    that nothing above uses.
4.  Filter pushdown moves a `filter` below the `mutate` or `select`
    under it, or into the inputs of a join.
5.  Filter reordering merges consecutive `filter` nodes into one and
    sorts the predicates by estimated cost.
6.  Scan pushdown hands a filter on a lazy
    [`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
    table to the parquet reader.

Each pass sees what the earlier ones left, and the order matters. Fusion
runs before pushdown, so two mutates that become neighbours after a
filter moves out from between them stay separate. A pass that has
nothing to work on is skipped, which doesn’t change the plan.

## Projection pruning

Only `mpg`, `hp` and `wt` are read from the 11 columns of `mtcars` here,
so the pass adds a `select [3 cols]` above the source. The
`select [2 cols]` at the top is the one in the pipeline.

``` r

tbl_gpu(mtcars) |>
  mutate(power_ratio = hp / wt) |>
  select(mpg, power_ratio) |>
  explain()
#> Pending operations:
#> ast_select [2 cols]
#> | ast_mutate [1 expressions]
#> | | ast_source [11 cols]
#> 
#> AST depth: 3 
#> Node count: 3 
#> 
#> Optimized plan:
#> ast_select [2 cols]
#> | ast_mutate [1 expressions]
#> | | ast_select [3 cols]
#> | | | ast_source [11 cols]
```

## Mutate fusion

Three [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
calls become one node:

``` r

tbl_gpu(mtcars) |>
  mutate(a = hp + 10) |>
  mutate(b = wt * 2) |>
  mutate(c = a + b) |>
  explain()
#> Pending operations:
#> ast_mutate [1 expressions]
#> | ast_mutate [1 expressions]
#> | | ast_mutate [1 expressions]
#> | | | ast_source [11 cols]
#> 
#> AST depth: 4 
#> Node count: 4 
#> 
#> Optimized plan:
#> ast_mutate [3 expressions]
#> | ast_source [11 cols]
```

The fused node evaluates its expressions in the order you wrote them,
each one seeing the columns the earlier ones left. That’s dplyr’s order,
so `c` below uses the original `a` and `b`, as in dplyr:

``` r

df <- tibble::tibble(a = 1, b = 2, hp = 3, wt = 4)

tbl_gpu(df) |>
  mutate(c = a + b) |>
  mutate(a = hp + 10, b = wt * 2) |>
  collect()
#> # A tibble: 1 × 5
#>       a     b    hp    wt     c
#>   <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1    13     8     3     4     3

df |>
  mutate(c = a + b) |>
  mutate(a = hp + 10, b = wt * 2)
#> # A tibble: 1 × 5
#>       a     b    hp    wt     c
#>   <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1    13     8     3     4     3
```

Two mutates don’t fuse when they hold more than 8 expressions together,
or when the upper one reads more than 4 columns the lower one writes.
Overwritten columns count as written. Here 5 + 4 expressions stay as two
nodes:

``` r

tbl_gpu(mtcars) |>
  mutate(e1 = mpg + 1, e2 = mpg + 2, e3 = mpg + 3, e4 = mpg + 4, e5 = mpg + 5) |>
  mutate(e6 = mpg + 6, e7 = mpg + 7, e8 = mpg + 8, e9 = mpg + 9) |>
  optimized()
#> ast_mutate [4 expressions]
#> | ast_mutate [5 expressions]
#> | | ast_source [11 cols]
```

## Dead-column pruning

The [`select()`](https://dplyr.tidyverse.org/reference/select.html)
keeps only `mpg` and `a`, so the `mutate` keeps one of its three
expressions:

``` r

tbl_gpu(mtcars) |>
  mutate(a = hp + 10, b = wt * 2, c = mpg * 2) |>
  select(mpg, a) |>
  optimized()
#> ast_select [2 cols]
#> | ast_mutate [1 expressions]
#> | | ast_select [2 cols]
#> | | | ast_source [11 cols]
```

The pass runs after fusion, so it also prunes across
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) calls
that fused. It keeps an unused expression in three cases:

- It writes a column that a later expression overwrites, and the column
  is still needed. `mutate(a = hp + 1) |> mutate(a = wt + 1)` computes
  both.
- It’s a vector literal such as `v = c(10, 20, 30, 40)`. dplyr checks
  its length whether you use it or not.
- It does checked integer arithmetic: `+`, `-` or `*` on two integer or
  logical operands. dplyr computes it and warns “NAs produced by integer
  overflow” even if nothing reads the result. `%/%`, `%%` and an
  operation with a literal `NA` operand never warn, so they aren’t
  checked.

Here `y` is unused, but `a + b` is checked, so the node keeps both
expressions:

``` r

tbl_gpu(data.frame(a = 1:4, b = 4:1, x = 1)) |>
  mutate(y = a + b, w = x * 2) |>
  select(a, w) |>
  optimized()
#> ast_select [2 cols]
#> | ast_mutate [2 expressions]
#> | | ast_source [3 cols]
```

The pass also narrows `select` nodes. Here the first
[`select()`](https://dplyr.tidyverse.org/reference/select.html) ends up
keeping only `a`, like the second:

``` r

tbl_gpu(data.frame(a = 1:3, b = 1, c = 2)) |>
  select(a, b, c) |>
  select(a) |>
  explain()
#> Pending operations:
#> ast_select [1 cols]
#> | ast_select [3 cols]
#> | | ast_source [3 cols]
#> 
#> AST depth: 3 
#> Node count: 3 
#> 
#> Optimized plan:
#> ast_select [1 cols]
#> | ast_source [3 cols]
```

## Filter pushdown

Moving a filter down means the nodes below it see fewer rows.

### Past a mutate

A filter moves below the `mutate` under it unless the mutate:

- creates or overwrites a column the filter reads,
- holds a vector literal,
- holds checked integer arithmetic, which dplyr would compute, and maybe
  warn on, in rows the filter drops, or
- holds a rowwise integer [`sum()`](https://rdrr.io/r/base/sum.html),
  [`min()`](https://rdrr.io/r/base/Extremes.html) or
  [`max()`](https://rdrr.io/r/base/Extremes.html), or a rowwise double
  [`min()`](https://rdrr.io/r/base/Extremes.html) or
  [`max()`](https://rdrr.io/r/base/Extremes.html).

`mpg` exists before the mutate, so this filter moves:

``` r

tbl_gpu(mtcars) |>
  mutate(power_ratio = hp / wt) |>
  filter(mpg > 20) |>
  explain()
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_mutate [1 expressions]
#> | | ast_source [11 cols]
#> 
#> AST depth: 3 
#> Node count: 3 
#> 
#> Optimized plan:
#> ast_mutate [1 expressions]
#> | ast_filter [1 predicates]
#> | | ast_source [11 cols]
```

The mutate creates `power_ratio`, so this one stays:

``` r

tbl_gpu(mtcars) |>
  mutate(power_ratio = hp / wt) |>
  filter(power_ratio > 50) |>
  explain()
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_mutate [1 expressions]
#> | | ast_source [11 cols]
#> 
#> AST depth: 3 
#> Node count: 3 
#> 
#> Optimized plan:
#> ast_filter [1 predicates]
#> | ast_mutate [1 expressions]
#> | | ast_source [11 cols]
```

### Past a select

A filter moves below a `select` that keeps every column the filter
reads. The lower `select` here is the one projection pruning added to
drop `c`:

``` r

tbl_gpu(data.frame(a = 1:20, b = 1, c = 2)) |>
  select(b, a) |>
  filter(a > 10) |>
  explain()
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_select [2 cols]
#> | | ast_source [3 cols]
#> 
#> AST depth: 3 
#> Node count: 3 
#> 
#> Optimized plan:
#> ast_select [2 cols]
#> | ast_filter [1 predicates]
#> | | ast_select [2 cols]
#> | | | ast_source [3 cols]
```

A filter never moves below a `select` that projection pruning added. A
[`select()`](https://dplyr.tidyverse.org/reference/select.html) that
renames a column, such as `select(b2 = b, a)`, records a `rename` node,
which is a barrier, so a filter after it stays put.

### One step per filter

Pushdown moves each filter below at most one node. Here the filter
passes the `select` but stops above the `mutate`, even though it could
pass that too:

``` r

tbl_gpu(data.frame(k = 1:20, a = 1, b = 2)) |>
  mutate(z = a + 1) |>
  select(k, a, z) |>
  filter(k > 10) |>
  optimized()
#> ast_select [3 cols]
#> | ast_filter [1 predicates]
#> | | ast_mutate [1 expressions]
#> | | | ast_select [2 cols]
#> | | | | ast_source [3 cols]
```

This is where fusion matters. Two mutates that fused count as one node,
so the filter gets below both:

``` r

tbl_gpu(mtcars) |>
  mutate(a = hp + 10) |>
  mutate(b = wt * 2) |>
  filter(mpg > 20) |>
  optimized()
#> ast_mutate [2 expressions]
#> | ast_filter [1 predicates]
#> | | ast_source [11 cols]
```

With the 5 + 4 expressions from above, the mutates don’t fuse and the
filter gets below the top one only:

``` r

tbl_gpu(mtcars) |>
  mutate(e1 = mpg + 1, e2 = mpg + 2, e3 = mpg + 3, e4 = mpg + 4, e5 = mpg + 5) |>
  mutate(e6 = mpg + 6, e7 = mpg + 7, e8 = mpg + 8, e9 = mpg + 9) |>
  filter(mpg > 20) |>
  optimized()
#> ast_mutate [4 expressions]
#> | ast_filter [1 predicates]
#> | | ast_mutate [5 expressions]
#> | | | ast_source [11 cols]
```

### Into a join

What moves depends on the join type and its arguments. The examples use
two small tables:

``` r

x <- tbl_gpu(data.frame(k = c(1, 2, 3, 4), a = c(5, 15, 25, 35)))
y <- tbl_gpu(data.frame(k = c(1, 2, 2, 4), v = c(10, 20, 30, 40)))
```

For
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
and
[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
the whole filter moves into the left input. The `select` on the right
input is projection pruning keeping only the key:

``` r

x |> semi_join(y, by = "k") |> filter(a > 10) |> optimized()
#> ast_join [semi join; on k == k]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]
#> | ast_select [1 cols]
#> | | ast_source [2 cols]
```

For
[`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
[`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
and
[`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
the default `relationship` blocks pushdown. dplyr checks for
many-to-many matches over the full inputs, and a filter below the join
could change that check. dplyr runs the check only for joins whose
conditions are all `==`, called from your own code. So with
`relationship` and `multiple` left at their defaults, nothing moves:

``` r

x |> inner_join(y, by = "k") |> filter(a > 10) |> optimized()
#> ast_filter [1 predicates]
#> | ast_join [inner join; on k == k]
#> | | ast_source [2 cols]
#> | | ast_source [2 cols]
```

Joins dplyr doesn’t check are recorded as
`relationship = "many-to-many"`. That covers inequality, `closest()`,
[`between()`](https://dplyr.tidyverse.org/reference/between.html),
`overlaps()` and [`within()`](https://rdrr.io/r/base/with.html)
conditions, `by = character()`, and `==` joins called from inside a
package. This filter moves although no argument is set:

``` r

y_k2 <- tbl_gpu(data.frame(k2 = c(1, 2, 2, 4), v = c(10, 20, 30, 40)))

x |> inner_join(y_k2, join_by(k >= k2)) |> filter(a > 10) |> optimized()
#> ast_join [inner join; on k >= k2]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]
#> | ast_source [2 cols]
```

Nothing moves with `unmatched = "error"` or a `relationship` of
`"one-to-one"`, `"one-to-many"` or `"many-to-one"`. Otherwise, for
example with `relationship = "many-to-many"`, the filter is split by
predicate:

- A predicate on left columns only moves into the left input of an inner
  or left join.
- A predicate on right columns only moves into the right input of an
  inner or right join, but not with a `multiple` other than `"all"` or a
  `closest()` condition.
- A predicate on both sides stays above the join. So does one on an `==`
  key whose two sides have different types, such as integer and double.

In the left join, `v > 15` reads the right side, so it stays:

``` r

x |>
  inner_join(y, by = "k", relationship = "many-to-many") |>
  filter(a > 10, v > 15) |>
  optimized()
#> ast_join [inner join; on k == k]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]

x |>
  left_join(y, by = "k", relationship = "many-to-many") |>
  filter(a > 10, v > 15) |>
  optimized()
#> ast_filter [1 predicates]
#> | ast_join [left join; on k == k]
#> | | ast_filter [1 predicates]
#> | | | ast_source [2 cols]
#> | | ast_source [2 cols]
```

With `multiple = "first"`, `a > 10` still moves and `v > 15` stays:

``` r

x |>
  inner_join(y, by = "k", relationship = "many-to-many", multiple = "first") |>
  filter(a > 10, v > 15) |>
  optimized()
#> ast_filter [1 predicates]
#> | ast_join [inner join; on k == k]
#> | | ast_filter [1 predicates]
#> | | | ast_source [2 cols]
#> | | ast_source [2 cols]
```

A
[`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
follows the inner join rules on both sides. A
[`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
moves nothing.

``` r

x |>
  cross_join(y_k2) |>
  filter(a > 10, v > 15) |>
  optimized()
#> ast_join [cross join]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]
```

Checked integer arithmetic is stricter. dplyr computes every predicate
of one [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
call on the same rows, the rows the join keeps. A checked predicate
moves only into the left input of a left join or the right input of a
right join, where every row reaches the output. In a semi, anti, inner
or cross join it stays, and the rest of its
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) call
stays with it:

``` r

x_i <- tbl_gpu(data.frame(k = 1:4, a = c(5, 15, 25, 35)))
y_i <- tbl_gpu(data.frame(k = c(1L, 2L, 2L, 4L), v = c(10, 20, 30, 40)))

x_i |>
  inner_join(y_i, by = "k", relationship = "many-to-many") |>
  filter(a > 10, k + 1L > 2L) |>
  optimized()
#> ast_filter [2 predicates]
#> | ast_join [inner join; on k == k]
#> | | ast_source [2 cols]
#> | | ast_source [2 cols]
```

## Filter reordering

A chain of `filter` nodes becomes one node. A filter node runs as one
expression, its predicates joined with `&` into one row mask.

``` r

d <- data.frame(x = c(1, 5, 20, 30), y = c(0, 1, 2, 3), z = c(5, 11, 12, 1))

tbl_gpu(d) |>
  filter(x > y) |>
  filter(z > 10) |>
  explain()
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_filter [1 predicates]
#> | | ast_source [3 cols]
#> 
#> AST depth: 3 
#> Node count: 3 
#> 
#> Optimized plan:
#> ast_filter [2 predicates]
#> | ast_source [3 cols]
```

The merged predicates are sorted cheapest first. The cost is the size of
the expression: 1 for a literal, 2 for a column, and 1 plus its
arguments for a call. `z > 10` costs 4 and `x > y` costs 5, so `z > 10`
comes first. A single filter node isn’t touched, and its predicates keep
your order. Predicates that aren’t deterministic are merged but not
sorted.

Reordering runs after pushdown, so filters that pushdown brings together
are merged. Here the second filter moves below the mutate and joins the
first:

``` r

tbl_gpu(mtcars) |>
  filter(cyl > 4) |>
  mutate(r = hp / wt) |>
  filter(mpg > 15) |>
  explain()
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_mutate [1 expressions]
#> | | ast_filter [1 predicates]
#> | | | ast_source [11 cols]
#> 
#> AST depth: 4 
#> Node count: 4 
#> 
#> Optimized plan:
#> ast_mutate [1 expressions]
#> | ast_filter [2 predicates]
#> | | ast_source [11 cols]
```

Some filters don’t merge:

- A filter on a grouped table and one on an ungrouped table stay
  separate.
- A filter with checked integer arithmetic isn’t merged with the filters
  below it, because dplyr only warns about overflow in the rows those
  filters keep. It can still absorb the filters above it.
- On a rowwise table, a filter holding one of the rowwise aggregates
  listed under “Past a mutate” isn’t merged. The filters below it still
  are.

So this stays two nodes:

``` r

tbl_gpu(data.frame(a = 1:4, b = 4:1)) |>
  filter(a < 100L) |>
  filter(a + b > 0L) |>
  explain()
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]
#> 
#> AST depth: 3 
#> Node count: 3 
#> 
#> Optimized plan:
#> ast_filter [1 predicates]
#> | ast_filter [1 predicates]
#> | | ast_source [2 cols]
```

## Scan pushdown

On a lazy table from
[`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md),
a filter directly on the file scan gives the reader the comparisons it
can decide exactly, so it can skip row groups. Only comparisons on
integer, integer64, Date and POSIXct columns go to the reader. The rest
of the filter runs after the read, and the rows are the same either way.
Projection pruning also tells the reader which columns to read.
`options(cuplyr.parquet_pushdown = FALSE)` turns both off. See
[`?read_parquet_gpu`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
for the full rules.

## Barriers

The optimizer splits the plan at barrier nodes and optimizes each part
on its own, so no pass moves a node across a barrier. These nodes are
barriers:

| Node | Recorded by |
|----|----|
| `arrange` | [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) |
| `summarise` | [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) |
| `reframe` | [`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html) |
| `distinct` | [`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html) |
| `slice` | [`head()`](https://rdrr.io/r/utils/head.html), [`slice()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html) and the like |
| `rename` | [`rename()`](https://dplyr.tidyverse.org/reference/rename.html), or a [`select()`](https://dplyr.tidyverse.org/reference/select.html) that renames |
| `window` | window functions in [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) or [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) |
| `barrier` | [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) |
| `select` by position | a [`select()`](https://dplyr.tidyverse.org/reference/select.html) that picks a column twice, or any [`select()`](https://dplyr.tidyverse.org/reference/select.html) on a table with repeated names |

[`head()`](https://rdrr.io/r/utils/head.html) shows why. Filtering the
first four rows is not the same as taking the first four rows that pass
the filter:

``` r

d <- data.frame(x = 1:10, y = c(-1, 2, -3, 4, -5, 6, -7, 8, -9, 10))

q <- tbl_gpu(d) |>
  arrange(x) |>
  head(4) |>
  filter(y > 0)

explain(q)
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_slice [mode=head, n=4]
#> | | ast_arrange [1 keys]
#> | | | ast_source [2 cols]
#> 
#> AST depth: 4 
#> Node count: 4 
#> 
#> Optimized plan:
#> ast_filter [1 predicates]
#> | ast_slice [mode=rank, order_by=x, desc=FALSE, n=4, with_ties=FALSE, top_k]
#> | | ast_source [2 cols]

all.equal(
  as.data.frame(collect(q)),
  d |> arrange(x) |> head(4) |> filter(y > 0),
  check.attributes = FALSE
)
#> [1] TRUE
```

The plan has no `arrange` node. `head(n)` directly on an ungrouped
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) of one
numeric, Date or POSIXct column becomes a single top-k `slice`, which
picks the rows without sorting the whole table. It is still a barrier.

### Segments

Each part between barriers is a segment. The optimizer works from the
top down. A segment below a barrier keeps only the columns that the
barrier and everything above it read, but it keeps every `mutate`
expression, because dplyr would compute them. Here each filter moves
below its own mutate, and
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) stays
between them:

``` r

d <- data.frame(a = c(5, 12, 20, 30), c = c(1, 6, 7, 8))

q <- tbl_gpu(d) |>
  mutate(b = a + 1) |>
  filter(a > 10) |>
  arrange(b) |>
  mutate(d = c + 1) |>
  filter(c > 5)

explain(q)
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_mutate [1 expressions]
#> | | ast_arrange [1 keys]
#> | | | ast_filter [1 predicates]
#> | | | | ast_mutate [1 expressions]
#> | | | | | ast_source [2 cols]
#> 
#> AST depth: 6 
#> Node count: 6 
#> 
#> Optimized plan:
#> ast_mutate [1 expressions]
#> | ast_filter [1 predicates]
#> | | ast_arrange [1 keys]
#> | | | ast_mutate [1 expressions]
#> | | | | ast_filter [1 predicates]
#> | | | | | ast_source [2 cols]
```

A join and its two inputs form one segment, together with the nodes
above it up to the next barrier. The passes don’t split a join’s inputs
at barriers, so column pruning can reach below a barrier there.
Projection pruning drops `b` below
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) in the
left input:

``` r

d2 <- tbl_gpu(data.frame(k = c(1, 2, 3, 4), a = c(5, 15, 25, 35), b = 1:4))
y2 <- tbl_gpu(data.frame(k = c(1, 2, 2, 4), v = c(10, 20, 30, 40)))

d2 |>
  arrange(a) |>
  inner_join(y2, by = "k", relationship = "many-to-many") |>
  select(k, v) |>
  optimized()
#> ast_select [2 cols]
#> | ast_join [inner join; on k == k]
#> | | ast_arrange [1 keys]
#> | | | ast_select [2 cols]
#> | | | | ast_source [3 cols]
#> | | ast_source [2 cols]
```

To stop the optimizer at a point of your choosing, put
[`collapse()`](https://dplyr.tidyverse.org/reference/compute.html)
there. Here the two filters aren’t merged and neither crosses it:

``` r

tbl_gpu(data.frame(x = c(5, 11, 20), z = c(1, 6, 9))) |>
  filter(x > 10) |>
  mutate(y = x * 2) |>
  collapse() |>
  filter(z > 5) |>
  explain()
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_barrier
#> | | ast_mutate [1 expressions]
#> | | | ast_filter [1 predicates]
#> | | | | ast_source [2 cols]
#> 
#> AST depth: 5 
#> Node count: 5 
#> 
#> Optimized plan:
#> ast_filter [1 predicates]
#> | ast_barrier
#> | | ast_mutate [1 expressions]
#> | | | ast_filter [1 predicates]
#> | | | | ast_source [2 cols]
```

## A longer pipeline

``` r

pipeline <- tbl_gpu(mtcars) |>
  mutate(
    power_to_weight = hp / wt,
    fuel_efficiency = mpg / cyl,
    is_powerful = hp > 150
  ) |>
  filter(cyl %in% c(4, 6), mpg > 18) |>
  mutate(efficiency_score = fuel_efficiency * power_to_weight) |>
  group_by(cyl) |>
  summarise(avg_score = mean(efficiency_score), count = n()) |>
  arrange(desc(avg_score))

explain(pipeline)
#> Pending operations:
#> ast_arrange [1 keys]
#> | ast_summarise [1 groups, 2 aggs]
#> | | ast_mutate [1 expressions]
#> | | | ast_filter [2 predicates]
#> | | | | ast_mutate [3 expressions]
#> | | | | | ast_source [11 cols]
#> 
#> AST depth: 6 
#> Node count: 6 
#> 
#> Optimized plan:
#> ast_arrange [1 keys]
#> | ast_summarise [1 groups, 2 aggs]
#> | | ast_mutate [1 expressions]
#> | | | ast_select [3 cols]
#> | | | | ast_mutate [3 expressions]
#> | | | | | ast_filter [2 predicates]
#> | | | | | | ast_select [4 cols]
#> | | | | | | | ast_source [11 cols]

collect(pipeline)
#> # A tibble: 2 × 3
#>     cyl avg_score count
#>   <dbl>     <dbl> <int>
#> 1     4      255.    11
#> 2     6      136.     6
```

The filter moved below the first mutate, so the three new columns are
computed only for rows that pass it. `is_powerful` is still computed
although nothing reads it, because it sits below the
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
barrier, where mutate expressions are kept. The two mutates aren’t
fused. When fusion ran, the filter still sat between them.

## Further reading

- [`vignette("getting-started", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/getting-started.md)
  covers basic use.
- [`vignette("performance", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/performance.md)
  measures eager and lazy timings and memory.
- [`?optimize_ast`](https://bbtheo.github.io/cuplyr/reference/optimize_ast.md)
  is the optimizer’s internal entry point.
