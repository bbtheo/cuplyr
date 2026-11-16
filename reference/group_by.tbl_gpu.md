# Group a GPU table by one or more columns

Marks columns to group by for subsequent operations like
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html).
The grouping is stored as metadata and does not perform any computation
until an aggregation is requested.

## Usage

``` r
# S3 method for class 'tbl_gpu'
group_by(.data, ..., .add = FALSE, .drop = dplyr::group_by_drop_default(.data))
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Variables to group by: column names (`x`, `.data$x`), a top-level
  [`dplyr::pick()`](https://dplyr.tidyverse.org/reference/pick.html), or
  computed columns (`name = expression`, or an unnamed expression such
  as `a + 1`; see "Computed groups" below). Like dplyr, tidyselect
  helpers are only allowed inside
  [`pick()`](https://dplyr.tidyverse.org/reference/pick.html).

- .add:

  If `FALSE` (default), will override existing groups. If `TRUE`, will
  add to existing groups.

- .drop:

  Whether to drop unobserved factor-level combinations from
  [`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)/[`group_keys()`](https://dplyr.tidyverse.org/reference/group_data.html)/[`n_groups()`](https://dplyr.tidyverse.org/reference/group_data.html)/etc.
  Defaults to `group_by_drop_default(.data)`: `TRUE` for an ungrouped
  table, the table's current setting for a grouped one (as in dplyr).
  See "`.drop` semantics" below for the exact scope of what's honored.

## Value

A grouped `tbl_gpu` object. The object has the same data but with
grouping columns recorded for use by
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html).

## Details

Unlike operations like
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html)
or [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) does
not perform any expensive GPU aggregation itself. It simply records
which columns should be used for grouping in subsequent aggregation
operations (computed group columns, see below, DO run a real
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)).

The actual groupby computation happens when you call
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
(or one of the
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
family) on the grouped table. This lazy approach allows you to chain
multiple operations before executing the expensive groupby operation.

### Computed groups

Verified empirically against dplyr 1.2.1: a **named** argument (e.g.
`group_by(df, g = x %/% 2)`, or even a bare-symbol rename like
`group_by(df, g = x)`) is ALWAYS treated as a
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
expression – the original column (if any) is kept, a new (or
overwritten) column is added under the given name, and the table is then
grouped by that name. Later dots may reference earlier dots'
newly-computed columns, exactly like
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)'s own
chain semantics (e.g. `group_by(df, g = x %/% 2, h = g * 10)`). An
**unnamed** argument is a column reference (`x`, `.data$x`,
`.data[["x"]]`), which names an existing column, a top-level
[`pick()`](https://dplyr.tidyverse.org/reference/pick.html), which adds
the columns it selects, or – anything else – a data-masking expression
computed into a new column named
[`rlang::as_label()`](https://rlang.r-lib.org/reference/as_label.html)
of the expression (`group_by(df, a + 1)` makes a column called `a + 1`).
As soon as one dot is named or is not a column reference (a
[`pick()`](https://dplyr.tidyverse.org/reference/pick.html) or a `NULL`
counts), every dot is data-masking, so a bare name that is not a column
(a variable in the calling environment) is computed too, as in
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html). All
computed dots (named and unnamed) run as one
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) on the
ungrouped data, in argument order. Every dot's resulting group column
name(s) are combined in the order the arguments were written. The old
`add =` and `.dots =` arguments are defunct and error.

Calling
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) with
**no** arguments and `.add = FALSE` (the default) clears any existing
grouping entirely (verified empirically – this differs from simply
returning `.data` unchanged, which was this function's historical,
incorrect behavior for that case).

### `.drop` semantics

`.drop = FALSE` (verified empirically against dplyr 1.2.1) adds one
placeholder row (`.rows = integer(0)`) to
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)'s
output for every declared *factor level combination* of the FACTOR
grouping columns that has no observed row, expanded recursively like
dplyr: under each parent group a factor column takes all its levels
(plus `NA` if observed there) and a non-factor column the values
observed under that parent (a single `NA` under an empty parent); see
`expand_dropped_groups()`.

`.drop` is stored as a single boolean on the returned `tbl_gpu`
(`$group_drop`) and carried by
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)/[`push_join()`](https://bbtheo.github.io/cuplyr/reference/push_join.md)
through every verb that keeps a group column (and by
[`count()`](https://dplyr.tidyverse.org/reference/count.html),
[`add_count()`](https://dplyr.tidyverse.org/reference/count.html) and
[`slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html),
which build their result directly); it is reset to `TRUE` once no group
column is left. Like dplyr, `.drop` defaults to the table's current
setting, so regrouping a `.drop = FALSE` table keeps it.

Groups are recomputed from the data each time, and empty factor-level
groups are re-added from the declared levels. Row-subsetting verbs also
keep emptied groups of any key type, as dplyr does. On a `.drop = FALSE`
table, [`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`filter_out()`](https://dplyr.tidyverse.org/reference/filter.html),
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html),
[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html),
[`slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html),
[`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html),
[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html),
[`slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html),
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html),
[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
and [`rows_delete()`](https://dplyr.tidyverse.org/reference/rows.html)
keep every group, emptied ones included. `.preserve = TRUE` does the
same for [`slice()`](https://dplyr.tidyverse.org/reference/slice.html)
and [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) on a
`.drop = TRUE` table. The result fixes the input's group-keys table
(`$fixed_keys`), only when a group can have been lost, which
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
and friends,
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html),
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)/[`count()`](https://dplyr.tidyverse.org/reference/count.html)/
[`tally()`](https://dplyr.tidyverse.org/reference/count.html) (on the
GPU, with a row for each emptied group) and the CPU fallbacks honour.
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`select()`](https://dplyr.tidyverse.org/reference/select.html),
[`rename()`](https://dplyr.tidyverse.org/reference/rename.html) and
[`relocate()`](https://dplyr.tidyverse.org/reference/relocate.html)
carry it as long as they leave the group columns alone, and a CPU
fallback result carries whatever empty groups dplyr's own result kept
(see
[`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md):
a fallback
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) keeps
them, a fallback
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) on a
`.drop = TRUE` table drops them); every other verb, and any change of
grouping, drops it.

## See also

[`summarise.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md)
for aggregating grouped data,
[`ungroup.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/ungroup.tbl_gpu.md)
for removing grouping,
[`group_data.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/group-data.md)
for the grouping metadata query family

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Group by a single column
  by_cyl <- gpu_mtcars |>
    group_by(cyl)

  # Group by multiple columns
  by_cyl_gear <- gpu_mtcars |>
    group_by(cyl, gear)

  # Computed group column
  by_half_cyl <- gpu_mtcars |>
    group_by(half_cyl = cyl %/% 2)

  # Use with summarise for aggregation
  result <- gpu_mtcars |>
    group_by(cyl) |>
    summarise(mean_mpg = mean(mpg)) |>
    collect()
}
```
