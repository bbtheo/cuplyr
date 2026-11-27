# Query optimization in cuplyr

cuplyr tables are lazy by default. A verb on a lazy table doesn’t run.
It adds a step to a plan that describes the query. When
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) or
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) runs
the plan, an optimizer first rewrites it into one that gives the same
result. An eager table, `tbl_gpu(df, lazy = FALSE)`, runs each verb at
once, so the optimizer never sees it.

This article shows what the optimizer does. It doesn’t time anything.
For timings, see
[`vignette("performance", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/performance.md).

\
[`library`](https://rdrr.io/r/base/library.html)`(`[`cuplyr`](https://github.com/bbtheo/cuplyr)`)`\
[`library`](https://rdrr.io/r/base/library.html)`(`[`dplyr`](https://dplyr.tidyverse.org)`)`

## Reading the plan

[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html) and
[`explain()`](https://dplyr.tidyverse.org/reference/explain.html) print
the plan as R code, in the order you wrote it: the source first, then
one line per step, each step the dplyr call it stands for. Neither runs
anything. `explain(x)` prints the plan the optimizer will run.
`show_query(x, optimized = FALSE)` prints the plan as you wrote it.
`show_query(x)` prints the same text as `explain(x)`.

\
`pipeline`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``20``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``power_ratio ``=`` ``hp`` ``/`` ``wt``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``mpg``, ``cyl``, ``power_ratio``)`` ``|>`\
`  `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``mpg``)``)`\
\
[`show_query`](https://dplyr.tidyverse.org/reference/explain.html)`(``pipeline``, optimized ``=`` ``FALSE``)`\
`#> <cuplyr plan: as written>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> filter(mpg > 20)`\
`#> mutate(power_ratio = hp / wt)`\
`#> select(mpg, cyl, power_ratio)`\
`#> arrange(desc(mpg))`\
\
[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``pipeline``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 4 of 11 (mpg, cyl, hp, wt)`\
`#> filter(mpg > 20)`\
`#> mutate(power_ratio = hp / wt)`\
`#> select(mpg, cyl, power_ratio)`\
`#> arrange(desc(mpg))`

The first line says which plan it is. The source line says where the
rows come from, and `columns read:` how many of the source’s columns the
plan reads. The optimized plan reads 4 of the 11 columns of `mtcars`.

Most verbs print as the call you wrote. The exceptions:

| Verb | What the plan shows |
|----|----|
| [`select()`](https://dplyr.tidyverse.org/reference/select.html) that renames | `select(...)` with a `rename(new = old)` after it |
| [`head()`](https://rdrr.io/r/utils/head.html) | `slice_head(n = 4)` |
| [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) | [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) |
| [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) on columns | nothing, grouping is table metadata; [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) shows only before a [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) that uses it |
| `group_by(x = a + 1)` | a [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) step |
| window functions such as [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html) | a `mutate(..win1.. = row_number(), .by = g)` step, plus other steps |

Some calls show several steps. A grouped
[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html) shows
an [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html), a
window [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
a [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) and a
[`select()`](https://dplyr.tidyverse.org/reference/select.html).

A step that is longer than 80 characters prints one argument per line. A
join prints its right-hand plan under `y:`.

## The passes

The optimizer runs these passes in order:

1.  Projection pruning drops the columns nothing later reads. On a table
    the plan shows it as the source’s `columns read:` line, for example
    `columns read: 3 of 11 (mpg, hp, wt)`; elsewhere as a
    [`select()`](https://dplyr.tidyverse.org/reference/select.html)
    step.
2.  Mutate fusion merges a
    [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) with
    the [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
    directly below it.
3.  Dead-column pruning drops
    [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
    expressions and
    [`select()`](https://dplyr.tidyverse.org/reference/select.html)
    columns that nothing above uses.
4.  Filter pushdown moves a
    [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
    below the
    [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) or
    [`select()`](https://dplyr.tidyverse.org/reference/select.html)
    under it, or into the inputs of a join.
5.  Filter reordering merges consecutive
    [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
    steps into one and sorts the predicates by estimated cost.
6.  Scan pushdown hands a filter on a lazy
    [`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
    table to the parquet reader.

Each pass sees what the earlier ones left, and the order matters. Fusion
runs before pushdown, so two mutates that become neighbours after a
filter moves out from between them stay separate. A pass that has
nothing to work on is skipped, which doesn’t change the plan.

## Projection pruning

Only `mpg`, `hp` and `wt` are read from the 11 columns of `mtcars` here,
so the source line reads `columns read: 3 of 11 (mpg, hp, wt)`. The
`select(mpg, power_ratio)` at the end is the one in the pipeline.

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``power_ratio ``=`` ``hp`` ``/`` ``wt``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``mpg``, ``power_ratio``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 3 of 11 (mpg, hp, wt)`\
`#> mutate(power_ratio = hp / wt)`\
`#> select(mpg, power_ratio)`

## Mutate fusion

Three [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
calls become one step:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``a ``=`` ``hp`` ``+`` ``10``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``b ``=`` ``wt`` ``*`` ``2``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``c ``=`` ``a`` ``+`` ``b``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> mutate(a = hp + 10, b = wt * 2, c = a + b)`

The fused step evaluates its expressions in the order you wrote them,
each one seeing the columns the earlier ones left. That’s dplyr’s order,
so `c` below uses the original `a` and `b`, as in dplyr:

\
`df`` ``<-`` ``tibble``::`[`tibble`](https://tibble.tidyverse.org/reference/tibble.html)`(``a ``=`` ``1``, b ``=`` ``2``, hp ``=`` ``3``, wt ``=`` ``4``)`\
\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``df``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``c ``=`` ``a`` ``+`` ``b``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``a ``=`` ``hp`` ``+`` ``10``, b ``=`` ``wt`` ``*`` ``2``)`` ``|>`\
`  `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`#> ``# A tibble: 1 × 5`\
`#>       a     b    hp    wt     c`\
`#>   ``<dbl>`` ``<dbl>`` ``<dbl>`` ``<dbl>`` ``<dbl>`\
`#> ``1``    13     8     3     4     3`\
\
`df`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``c ``=`` ``a`` ``+`` ``b``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``a ``=`` ``hp`` ``+`` ``10``, b ``=`` ``wt`` ``*`` ``2``)`\
`#> ``# A tibble: 1 × 5`\
`#>       a     b    hp    wt     c`\
`#>   ``<dbl>`` ``<dbl>`` ``<dbl>`` ``<dbl>`` ``<dbl>`\
`#> ``1``    13     8     3     4     3`

Two mutates don’t fuse when they hold more than 8 expressions together,
or when the upper one reads more than 4 columns the lower one writes.
Overwritten columns count as written. Here 5 + 4 expressions stay as two
steps:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``e1 ``=`` ``mpg`` ``+`` ``1``, e2 ``=`` ``mpg`` ``+`` ``2``, e3 ``=`` ``mpg`` ``+`` ``3``, e4 ``=`` ``mpg`` ``+`` ``4``, e5 ``=`` ``mpg`` ``+`` ``5``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``e6 ``=`` ``mpg`` ``+`` ``6``, e7 ``=`` ``mpg`` ``+`` ``7``, e8 ``=`` ``mpg`` ``+`` ``8``, e9 ``=`` ``mpg`` ``+`` ``9``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> mutate(e1 = mpg + 1, e2 = mpg + 2, e3 = mpg + 3, e4 = mpg + 4, e5 = mpg + 5)`\
`#> mutate(e6 = mpg + 6, e7 = mpg + 7, e8 = mpg + 8, e9 = mpg + 9)`

## Dead-column pruning

The [`select()`](https://dplyr.tidyverse.org/reference/select.html)
keeps only `mpg` and `a`, so the
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) keeps
one of its three expressions:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``a ``=`` ``hp`` ``+`` ``10``, b ``=`` ``wt`` ``*`` ``2``, c ``=`` ``mpg`` ``*`` ``2``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``mpg``, ``a``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 2 of 11 (mpg, hp)`\
`#> mutate(a = hp + 10)`\
`#> select(mpg, a)`

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

Here `y` is unused, but `a + b` is checked, so the step keeps both
expressions:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``a ``=`` ``1``:``4``, b ``=`` ``4``:``1``, x ``=`` ``1``)``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``y ``=`` ``a`` ``+`` ``b``, w ``=`` ``x`` ``*`` ``2``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``a``, ``w``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 3`\
`#> mutate(y = a + b, w = x * 2)`\
`#> select(a, w)`

The pass also narrows
[`select()`](https://dplyr.tidyverse.org/reference/select.html) steps.
Here the first
[`select()`](https://dplyr.tidyverse.org/reference/select.html) ends up
keeping only `a`, like the second:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``a ``=`` ``1``:``3``, b ``=`` ``1``, c ``=`` ``2``)``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``a``, ``b``, ``c``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``a``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 1 of 3 (a)`

## Filter pushdown

Moving a filter down means the steps below it see fewer rows.

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

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``power_ratio ``=`` ``hp`` ``/`` ``wt``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``20``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> filter(mpg > 20)`\
`#> mutate(power_ratio = hp / wt)`

The mutate creates `power_ratio`, so this one stays:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``power_ratio ``=`` ``hp`` ``/`` ``wt``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``power_ratio`` ``>`` ``50``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> mutate(power_ratio = hp / wt)`\
`#> filter(power_ratio > 50)`

### Past a select

A filter moves below a
[`select()`](https://dplyr.tidyverse.org/reference/select.html) that
keeps every column the filter reads. Here projection pruning drops `c`
at the source, and the filter moves below the pipeline’s
[`select()`](https://dplyr.tidyverse.org/reference/select.html) but not
below the pruning:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``a ``=`` ``1``:``20``, b ``=`` ``1``, c ``=`` ``2``)``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``b``, ``a``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 2 of 3 (a, b)`\
`#> filter(a > 10)`\
`#> select(b, a)`

A filter never moves below the pruning of a source or a
[`select()`](https://dplyr.tidyverse.org/reference/select.html) that
projection pruning added. A
[`select()`](https://dplyr.tidyverse.org/reference/select.html) that
renames a column, such as `select(b2 = b, a)`, shows a
[`rename()`](https://dplyr.tidyverse.org/reference/rename.html) step,
which is a barrier, so a filter after it stays put.

### One step per filter

Pushdown moves each filter below at most one step. Here the filter
passes the
[`select()`](https://dplyr.tidyverse.org/reference/select.html) but
stops above the
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html), even
though it could pass that too:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k ``=`` ``1``:``20``, a ``=`` ``1``, b ``=`` ``2``)``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``z ``=`` ``a`` ``+`` ``1``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``k``, ``a``, ``z``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``k`` ``>`` ``10``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 2 of 3 (k, a)`\
`#> mutate(z = a + 1)`\
`#> filter(k > 10)`\
`#> select(k, a, z)`

This is where fusion matters. Two mutates that fused count as one step,
so the filter gets below both:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``a ``=`` ``hp`` ``+`` ``10``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``b ``=`` ``wt`` ``*`` ``2``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``20``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> filter(mpg > 20)`\
`#> mutate(a = hp + 10, b = wt * 2)`

With the 5 + 4 expressions from above, the mutates don’t fuse and the
filter gets below the top one only:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``e1 ``=`` ``mpg`` ``+`` ``1``, e2 ``=`` ``mpg`` ``+`` ``2``, e3 ``=`` ``mpg`` ``+`` ``3``, e4 ``=`` ``mpg`` ``+`` ``4``, e5 ``=`` ``mpg`` ``+`` ``5``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``e6 ``=`` ``mpg`` ``+`` ``6``, e7 ``=`` ``mpg`` ``+`` ``7``, e8 ``=`` ``mpg`` ``+`` ``8``, e9 ``=`` ``mpg`` ``+`` ``9``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``20``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> mutate(e1 = mpg + 1, e2 = mpg + 2, e3 = mpg + 3, e4 = mpg + 4, e5 = mpg + 5)`\
`#> filter(mpg > 20)`\
`#> mutate(e6 = mpg + 6, e7 = mpg + 7, e8 = mpg + 8, e9 = mpg + 9)`

### Into a join

What moves depends on the join type and its arguments. The examples use
two small tables:

\
`x`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``2``, ``3``, ``4``)``, a ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``5``, ``15``, ``25``, ``35``)``)``)`\
`y`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``2``, ``2``, ``4``)``, v ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``10``, ``20``, ``30``, ``40``)``)``)`

For
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
and
[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
the whole filter moves into the left input. The right input’s
`columns read:` line shows projection pruning keeping only the key:

\
`x`` ``|>`` `[`semi_join`](https://dplyr.tidyverse.org/reference/filter-joins.html)`(``y``, by ``=`` ``"k"``)`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``)`` ``|>`` `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a > 10)`\
`#> semi_join(y, by = join_by(k))`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: 1 of 2 (k)`

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

\
`x`` ``|>`` `[`inner_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``y``, by ``=`` ``"k"``)`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``)`` ``|>`` `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> inner_join(y, by = join_by(k))`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`\
`#> filter(a > 10)`

Joins dplyr doesn’t check are recorded as
`relationship = "many-to-many"`. That covers inequality, `closest()`,
[`between()`](https://dplyr.tidyverse.org/reference/between.html),
`overlaps()` and [`within()`](https://rdrr.io/r/base/with.html)
conditions, `by = character()`, and `==` joins called from inside a
package. This filter moves although no argument is set:

\
`y_k2`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k2 ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``2``, ``2``, ``4``)``, v ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``10``, ``20``, ``30``, ``40``)``)``)`\
\
`x`` ``|>`` `[`inner_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``y_k2``, `[`join_by`](https://dplyr.tidyverse.org/reference/join_by.html)`(``k`` ``>=`` ``k2``)``)`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``)`` ``|>`` `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a > 10)`\
`#> inner_join(y, by = join_by(k >= k2))`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`

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

\
`x`` ``|>`\
`  `[`inner_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``y``, by ``=`` ``"k"``, relationship ``=`` ``"many-to-many"``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``, ``v`` ``>`` ``15``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a > 10)`\
`#> inner_join(y, by = join_by(k), relationship = "many-to-many")`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`\
`#>     filter(v > 15)`\
\
`x`` ``|>`\
`  `[`left_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``y``, by ``=`` ``"k"``, relationship ``=`` ``"many-to-many"``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``, ``v`` ``>`` ``15``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a > 10)`\
`#> left_join(y, by = join_by(k), relationship = "many-to-many")`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`\
`#> filter(v > 15)`

With `multiple = "first"`, `a > 10` still moves and `v > 15` stays:

\
`x`` ``|>`\
`  `[`inner_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``y``, by ``=`` ``"k"``, relationship ``=`` ``"many-to-many"``, multiple ``=`` ``"first"``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``, ``v`` ``>`` ``15``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a > 10)`\
`#> inner_join(y, by = join_by(k), multiple = "first", relationship = "many-to-many")`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`\
`#> filter(v > 15)`

A
[`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
follows the inner join rules on both sides. A
[`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
moves nothing.

\
`x`` ``|>`\
`  `[`cross_join`](https://dplyr.tidyverse.org/reference/cross_join.html)`(``y_k2``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``, ``v`` ``>`` ``15``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a > 10)`\
`#> cross_join(y)`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`\
`#>     filter(v > 15)`

Checked integer arithmetic is stricter. dplyr computes every predicate
of one [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
call on the same rows, the rows the join keeps. A checked predicate
moves only into the left input of a left join or the right input of a
right join, where every row reaches the output. In a semi, anti, inner
or cross join it stays, and the rest of its
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) call
stays with it:

\
`x_i`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k ``=`` ``1``:``4``, a ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``5``, ``15``, ``25``, ``35``)``)``)`\
`y_i`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1L``, ``2L``, ``2L``, ``4L``)``, v ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``10``, ``20``, ``30``, ``40``)``)``)`\
\
`x_i`` ``|>`\
`  `[`inner_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``y_i``, by ``=`` ``"k"``, relationship ``=`` ``"many-to-many"``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``, ``k`` ``+`` ``1L`` ``>`` ``2L``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> inner_join(y, by = join_by(k), relationship = "many-to-many")`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`\
`#> filter(a > 10, k + 1L > 2L)`

## Filter reordering

A chain of
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) steps
becomes one step. A filter step runs as one expression, its predicates
joined with `&` into one row mask.

\
`d`` ``<-`` `[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``x ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``5``, ``20``, ``30``)``, y ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``0``, ``1``, ``2``, ``3``)``, z ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``5``, ``11``, ``12``, ``1``)``)`\
\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``d``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``x`` ``>`` ``y``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``z`` ``>`` ``10``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 3`\
`#> filter(z > 10, x > y)`

The merged predicates are sorted cheapest first. The cost is the size of
the expression: 1 for a literal, 2 for a column, and 1 plus its
arguments for a call. `z > 10` costs 4 and `x > y` costs 5, so `z > 10`
comes first. A single filter step isn’t touched, and its predicates keep
your order. Predicates that aren’t deterministic are merged but not
sorted.

Reordering runs after pushdown, so filters that pushdown brings together
are merged. Here the second filter moves below the mutate and joins the
first:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``cyl`` ``>`` ``4``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``r ``=`` ``hp`` ``/`` ``wt``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``15``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> filter(mpg > 15, cyl > 4)`\
`#> mutate(r = hp / wt)`

Some filters don’t merge:

- A filter on a grouped table and one on an ungrouped table stay
  separate.
- A filter with checked integer arithmetic isn’t merged with the filters
  below it, because dplyr only warns about overflow in the rows those
  filters keep. It can still absorb the filters above it.
- On a rowwise table, a filter holding one of the rowwise aggregates
  listed under “Past a mutate” isn’t merged. The filters below it still
  are.

So this stays two steps:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``a ``=`` ``1``:``4``, b ``=`` ``4``:``1``)``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``<`` ``100L``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``+`` ``b`` ``>`` ``0L``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a < 100L)`\
`#> filter(a + b > 0L)`

## Scan pushdown

On a lazy table from
[`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md),
a filter directly on the file scan gives the reader the comparisons it
can decide exactly, so it can skip row groups. Only comparisons on
integer, integer64, Date and POSIXct columns go to the reader. The rest
of the filter runs after the read, and the rows are the same either way.
Projection pruning also tells the reader which columns to read.
[`explain()`](https://dplyr.tidyverse.org/reference/explain.html) shows
both: a scan prints as

    parquet scan: flights.parquet
      columns read: 5 of 41 (carrier, distance, air_time, dep_delay, arr_delay)
      reader filter: distance > 1000L

`options(cuplyr.parquet_pushdown = FALSE)` turns both off. Then the scan
says `columns read: all 41`, and any pruning shows as a
[`select()`](https://dplyr.tidyverse.org/reference/select.html) step
above it. See
[`?read_parquet_gpu`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
for the full rules.

## Steps that run on the CPU

A call cuplyr can’t run on the GPU, such as
[`pnorm()`](https://rdrr.io/r/stats/Normal.html), runs on the CPU at the
verb: the pending plan is computed on the GPU, collected, handed to
dplyr and uploaded again. The plan keeps what happened. It shows the
verb, the expression, why it couldn’t run on the GPU and the plan the
step computed, and the steps after it follow:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``20``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``p ``=`` `[`pnorm`](https://rdrr.io/r/stats/Normal.html)`(``mpg`` ``/`` ``10``)``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``p`` ``>`` ``0.98``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> CPU fallback: mutate() for pnorm(mpg / 10)`\
`` #>   reason: `pnorm()` has no GPU implementation. ``\
`#>   columns read: all 12`\
`#>   input (computed at the mutate() call):`\
`#>     table on the GPU`\
`#>       columns read: all 11`\
`#>     filter(mpg > 20)`\
`#> filter(p > 0.98)`

The plan shows the newest three CPU steps;
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) of a
table with pending steps starts a new plan with no history. Printing the
plan doesn’t run anything.

## Barriers

The optimizer splits the plan at barrier steps and optimizes each part
on its own, so no pass moves a step across a barrier. These steps are
barriers:

| Step | Recorded by |
|----|----|
| [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) | [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) |
| [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) | [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) |
| [`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html) | [`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html) |
| [`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html) | [`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html) |
| [`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html) and the like | [`head()`](https://rdrr.io/r/utils/head.html), [`slice()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html), [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html) and the like |
| [`rename()`](https://dplyr.tidyverse.org/reference/rename.html) | [`rename()`](https://dplyr.tidyverse.org/reference/rename.html), or a [`select()`](https://dplyr.tidyverse.org/reference/select.html) that renames |
| a `mutate(..win1.. = <window function>)` step | window functions in [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) or [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) |
| [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) | [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) |
| a [`select()`](https://dplyr.tidyverse.org/reference/select.html) by position | a [`select()`](https://dplyr.tidyverse.org/reference/select.html) that picks a column twice, or any [`select()`](https://dplyr.tidyverse.org/reference/select.html) on a table with repeated names |

[`head()`](https://rdrr.io/r/utils/head.html) shows why. Filtering the
first four rows is not the same as taking the first four rows that pass
the filter:

\
`d`` ``<-`` `[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``x ``=`` ``1``:``10``, y ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``-``1``, ``2``, ``-``3``, ``4``, ``-``5``, ``6``, ``-``7``, ``8``, ``-``9``, ``10``)``)`\
\
`q`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``d``)`` ``|>`\
`  `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``x``)`` ``|>`\
`  `[`head`](https://rdrr.io/r/utils/head.html)`(``4``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``y`` ``>`` ``0``)`\
\
[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``q``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> slice_min(x, n = 4, with_ties = FALSE)  # top-k: arrange() |> head()`\
`#> filter(y > 0)`\
\
[`all.equal`](https://rdrr.io/r/base/all.equal.html)`(`\
`  `[`as.data.frame`](https://rdrr.io/r/base/as.data.frame.html)`(`[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``q``)``)``,`\
`  ``d`` ``|>`` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``x``)`` ``|>`` `[`head`](https://rdrr.io/r/utils/head.html)`(``4``)`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``y`` ``>`` ``0``)``,`\
`  check.attributes ``=`` ``FALSE`\
`)`\
`#> [1] TRUE`

The plan has no
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) step.
`head(n)` directly on an ungrouped
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) of one
numeric, Date or POSIXct column becomes a single top-k slice, which
picks the rows without sorting the whole table. It prints as
`slice_min(x, n = 4, with_ties = FALSE) # top-k: arrange() |> head()`,
and is still a barrier.

### Segments

Each part between barriers is a segment. The optimizer works from the
top down. A segment below a barrier keeps only the columns that the
barrier and everything above it read, but it keeps every `mutate`
expression, because dplyr would compute them. Here each filter moves
below its own mutate, and
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) stays
between them:

\
`d`` ``<-`` `[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``a ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``5``, ``12``, ``20``, ``30``)``, c ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``6``, ``7``, ``8``)``)`\
\
`q`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``d``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``b ``=`` ``a`` ``+`` ``1``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``a`` ``>`` ``10``)`` ``|>`\
`  `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``b``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``d ``=`` ``c`` ``+`` ``1``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``c`` ``>`` ``5``)`\
\
[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``q``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(a > 10)`\
`#> mutate(b = a + 1)`\
`#> arrange(b)`\
`#> filter(c > 5)`\
`#> mutate(d = c + 1)`

A join and its two inputs form one segment, together with the steps
above it up to the next barrier. The passes don’t split a join’s inputs
at barriers, so column pruning can reach below a barrier there.
Projection pruning drops `b` below
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) in the
left input:

\
`d2`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``2``, ``3``, ``4``)``, a ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``5``, ``15``, ``25``, ``35``)``, b ``=`` ``1``:``4``)``)`\
`y2`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``k ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``2``, ``2``, ``4``)``, v ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``10``, ``20``, ``30``, ``40``)``)``)`\
\
`d2`` ``|>`\
`  `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``a``)`` ``|>`\
`  `[`inner_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``y2``, by ``=`` ``"k"``, relationship ``=`` ``"many-to-many"``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``k``, ``v``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 2 of 3 (k, a)`\
`#> arrange(a)`\
`#> inner_join(y, by = join_by(k), relationship = "many-to-many")`\
`#>   y:`\
`#>     table on the GPU`\
`#>       columns read: all 2`\
`#> select(k, v)`

To stop the optimizer at a point of your choosing, put
[`collapse()`](https://dplyr.tidyverse.org/reference/compute.html)
there. Here the two filters aren’t merged and neither crosses it:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(`[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(``x ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``5``, ``11``, ``20``)``, z ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``1``, ``6``, ``9``)``)``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``x`` ``>`` ``10``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``y ``=`` ``x`` ``*`` ``2``)`` ``|>`\
`  `[`collapse`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``z`` ``>`` ``5``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 2`\
`#> filter(x > 10)`\
`#> mutate(y = x * 2)`\
`#> collapse()`\
`#> filter(z > 5)`

## A longer pipeline

\
`pipeline`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`    power_to_weight ``=`` ``hp`` ``/`` ``wt``,`\
`    fuel_efficiency ``=`` ``mpg`` ``/`` ``cyl``,`\
`    is_powerful ``=`` ``hp`` ``>`` ``150`\
`  ``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``cyl`` `[`%in%`](https://rdrr.io/r/base/match.html)` `[`c`](https://rdrr.io/r/base/c.html)`(``4``, ``6``)``, ``mpg`` ``>`` ``18``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``efficiency_score ``=`` ``fuel_efficiency`` ``*`` ``power_to_weight``)`` ``|>`\
`  `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``cyl``)`` ``|>`\
`  `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(``avg_score ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``efficiency_score``)``, count ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``)`` ``|>`\
`  `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``avg_score``)``)`\
\
[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``pipeline``)`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: 4 of 11 (mpg, cyl, hp, wt)`\
`#> filter(cyl %in% c(4, 6), mpg > 18)`\
`#> mutate(`\
`#>   power_to_weight = hp / wt,`\
`#>   fuel_efficiency = mpg / cyl,`\
`#>   is_powerful = hp > 150`\
`#> )`\
`#> select(cyl, power_to_weight, fuel_efficiency)`\
`#> mutate(efficiency_score = fuel_efficiency * power_to_weight)`\
`#> group_by(cyl)`\
`#> summarise(avg_score = mean(efficiency_score), count = n())`\
`#> arrange(desc(avg_score))`\
\
[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``pipeline``)`\
`#> ``# A tibble: 2 × 3`\
`#>     cyl avg_score count`\
`#>   ``<dbl>``     ``<dbl>`` ``<int>`\
`#> ``1``     4      255.    11`\
`#> ``2``     6      136.     6`

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
