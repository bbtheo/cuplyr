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

[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
itself does no GPU work: it records the grouping columns, and the verbs
that follow
([`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html),
the
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
family, ...) then work group by group. Computed group columns (see
below) are the exception: they run a
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) first.

### Computed groups

As in dplyr, a **named** argument (`group_by(df, g = x %/% 2)`, or even
a plain rename such as `group_by(df, g = x)`) is always computed like a
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
expression: the original column (if any) is kept, a new (or overwritten)
column is added under the given name, and the table is then grouped by
that name. A later argument may use the column of an earlier one, as in
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
(`group_by(df, g = x %/% 2, h = g * 10)`). An **unnamed** argument is a
column reference (`x`, `.data$x`, `.data[["x"]]`), which names an
existing column, a top-level
[`pick()`](https://dplyr.tidyverse.org/reference/pick.html), which adds
the columns it selects, or any other expression, which is computed into
a new column named after the expression (`group_by(df, a + 1)` makes a
column called `a + 1`). As soon as one argument is named or is not a
column reference (a
[`pick()`](https://dplyr.tidyverse.org/reference/pick.html) or a `NULL`
counts), every argument is data-masking, so a bare name that is not a
column (a variable in the calling environment) is computed too, as in
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html). All
computed arguments run as one
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) on the
ungrouped data, in argument order, and the group columns come in the
order the arguments were written. The old `add =` and `.dots =`
arguments are defunct and error.

Calling
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) with
no arguments and `.add = FALSE` (the default) removes any grouping, as
in dplyr.

### `.drop` semantics

With `.drop = FALSE`,
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
(and so
[`group_keys()`](https://dplyr.tidyverse.org/reference/group_data.html),
[`n_groups()`](https://dplyr.tidyverse.org/reference/group_data.html)
and the verbs that work per group) also has an empty group
(`.rows = integer(0)`) for every combination of factor levels of the
factor grouping columns that has no row. The combinations are expanded
as in dplyr: under each parent group a factor column takes all its
levels (plus `NA` if it occurs there) and a non-factor column the values
that occur under that parent (a single `NA` under an empty parent).

The setting stays with the table through every verb that keeps a group
column, and is reset to `TRUE` once no group column is left. As in
dplyr, `.drop` defaults to the table's current setting, so regrouping a
`.drop = FALSE` table keeps it.

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
`.drop = TRUE` table. The kept groups then show in
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
and friends,
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html), and
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
[`count()`](https://dplyr.tidyverse.org/reference/count.html) and
[`tally()`](https://dplyr.tidyverse.org/reference/count.html) (on the
GPU, with a row for each emptied group).
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`select()`](https://dplyr.tidyverse.org/reference/select.html),
[`rename()`](https://dplyr.tidyverse.org/reference/rename.html) and
[`relocate()`](https://dplyr.tidyverse.org/reference/relocate.html) keep
them as long as they leave the group columns alone. A verb that runs in
dplyr on the CPU keeps whichever empty groups dplyr's own result kept
(such a [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
keeps them, such an
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) on a
`.drop = TRUE` table drops them). Every other verb, and any change of
grouping, drops them.

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
