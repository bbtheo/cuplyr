# Create a GPU table: a data frame in GPU memory

`tbl_gpu()` copies an R data frame to GPU memory and returns a
`tbl_gpu`, cuplyr's GPU table. You work with it as with a tibble: pipe
it through dplyr verbs such as
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) or
[`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
which run on the GPU, and bring the result back to R with
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

## Usage

``` r
tbl_gpu(data, ..., lazy = NULL)

# S3 method for class 'data.frame'
tbl_gpu(data, ..., lazy = NULL)

# S3 method for class 'tbl_gpu'
tbl_gpu(data, ..., lazy = NULL)
```

## Arguments

- data:

  A data frame or tibble to transfer to GPU memory. Supported column
  types are double, integer,
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html),
  character, logical, factor, `Date` and `POSIXct` (see "Column types"
  below).

- ...:

  Additional arguments passed to methods (currently unused).

- lazy:

  Logical or character. Controls execution mode:

  - `TRUE` or `"lazy"` - Operations are deferred until
    collect()/compute()

  - `FALSE` or `"eager"` - Operations execute immediately

  - `NULL` (default) - Use the session option, then the environment
    variable, then the package default, which is lazy

  The default can be changed with `options(cuplyr.exec_mode = "eager")`
  or the environment variable `CUPLYR_EXEC_MODE=eager`; the argument
  beats the option, which beats the variable. An invalid option value
  warns once and the default is used. All options are listed on
  [cuplyr-options](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md).

## Value

A GPU table: an object of class `tbl_gpu` that holds the data in GPU
memory and works with dplyr verbs. Use
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) to get
a tibble back.

## Details

The data is copied to GPU memory as soon as `tbl_gpu()` is called (also
in lazy mode; only the verbs are deferred). The GPU memory is freed when
the R object is garbage collected.

### Working with a GPU table

A `tbl_gpu` works with the dplyr verbs. What the GPU can't do, dplyr
does on the CPU, and the result comes back to the GPU as a `tbl_gpu`
(see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)
to be told when that happens).

- [collect()](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
  runs the pending steps and brings the result back to R as a tibble (a
  plain `data.frame` if you uploaded one).

- [compute()](https://bbtheo.github.io/cuplyr/reference/compute.tbl_gpu.md)
  runs the pending steps and keeps the result on the GPU, as a `tbl_gpu`
  with nothing pending: use it for a result you use more than once.

- [explain()](https://bbtheo.github.io/cuplyr/reference/explain.tbl_gpu.md)
  and
  [show_query()](https://bbtheo.github.io/cuplyr/reference/show_query.tbl_gpu.md)
  print the pending plan, one dplyr call per step, without running it.

- [`print()`](https://rdrr.io/r/base/print.html) and
  [`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) show
  the dimensions, the column types and the first rows.

### Lazy and eager execution

A `tbl_gpu` is lazy by default: each verb adds a step to a plan, and the
GPU does the work when a result is needed:
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html),
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html),
printing,
[`nrow()`](https://rdrr.io/r/base/nrow.html)/[`dim()`](https://rdrr.io/r/base/dim.html),
[`pull()`](https://dplyr.tidyverse.org/reference/pull.html) or `$`. A
few steps need the result of the pending plan as soon as they are
called, and run it then, on the GPU (see "Plan breaks" in
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)).
Before a plan runs, cuplyr optimizes it: columns no later step reads are
dropped early, consecutive
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) calls
are combined,
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
conditions move closer to the data (below a
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) they
don't depend on, or into an input of a join when that can't change the
join's result, errors or warnings, for example below a cross join or a
join with `relationship = "many-to-many"`), and of several
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) calls in
a row the cheaper conditions run first.

Errors in a verb's arguments (a missing column, a type error) are still
raised when the verb is called. Printing a lazy table with pending
operations (or calling
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) on it)
runs the plan to show its first rows, under a `# A tbl_gpu [lazy]:`
header; the table itself is unchanged and nothing is kept, so printing
it twice runs the plan twice
([`compute()`](https://dplyr.tidyverse.org/reference/compute.html) it
first to look at it repeatedly). If the plan fails (an error, out of GPU
memory), the print shows the schema and the message instead of raising.

In eager mode (`tbl_gpu(df, lazy = FALSE)`), each dplyr verb runs on the
GPU at once. This is simple but can lead to unnecessary intermediate
allocations. `options(cuplyr.exec_mode = "eager")` (or the environment
variable `CUPLYR_EXEC_MODE=eager`) makes every new `tbl_gpu` eager, and
[`as_eager()`](https://bbtheo.github.io/cuplyr/reference/as_eager.md)
and [`as_lazy()`](https://bbtheo.github.io/cuplyr/reference/as_lazy.md)
switch one table.

For measured timings and memory use of the two modes, see
[`vignette("performance", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/performance.md).

### Column types

Each R column type is stored as a GPU type, the type the table's schema
shows:

- `numeric` (double) -\> FLOAT64

- `integer` -\> INT32

- [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  -\> INT64, exact.
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
  gives a
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  back (a double, with a warning past 2^53, when bit64 is not
  installed).

- `character` -\> STRING

- `logical` -\> BOOL8

- `factor` (ordered too) -\> DICTIONARY32: the integer codes are on the
  GPU and the levels stay in R, so
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
  gives the factor back with the same levels.

- `Date` -\> TIMESTAMP_DAYS

- `POSIXct` -\> TIMESTAMP_MICROSECONDS, the instant in UTC on the GPU
  with the time zone kept alongside, so
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
  gives the same zone back.

A
[`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/vctrs-unspecified.html)
column (all `NA`, no type yet) is stored as BOOL8. It collects back as
[`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/vctrs-unspecified.html)
only while it passes through
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html),
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html) and
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)/[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
(as a column of `x`, as in dplyr); every other verb
([`select()`](https://dplyr.tidyverse.org/reference/select.html),
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) with
computed expressions,
[`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html),
a mutating join,
[`vctrs::vec_slice()`](https://vctrs.r-lib.org/reference/vec_slice.html))
returns it as a plain logical column. An all-`NA` logical or unspecified
column joins a string, `Date`, `POSIXct` or factor key, as in dplyr: a
merged key takes that key's type; with `keep = TRUE`, in a non-equi
condition and in
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)/[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
each key column keeps its own type. A logical key with a value in it
errors. Bound to a character column in
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md),
an unspecified column errors.

A matrix column with more than one column (or more than two dimensions),
a list column, a data frame column and a complex or raw column have no
GPU form: `tbl_gpu()` raises `cuplyr_error_unsupported_class` naming the
column. A verb that ran on the CPU and returned a matrix or array raises
`cuplyr_error_unsupported_result_class` the same way; one that returned
a list or data frame column hands back a plain tibble.

### Dates and times the GPU can't hold

A GPU `Date` holds whole days in an int32 and a `POSIXct` holds int64
microseconds. A value that doesn't fit becomes `NA` when it is uploaded,
with a warning:

- `cuplyr_warning_infinite_timestamp`: `Inf` or `-Inf`.

- `cuplyr_warning_timestamp_out_of_range`: a finite value beyond about
  5.8 million years either side of 1970 (`Date`) or 292,000 years
  (`POSIXct`).

Both also inherit `cuplyr_warning_timestamp_unrepresentable`, so one
handler covers both. An upload raises at most one warning per class,
listing every affected column with its count, e.g.
`` `d` (2 values), `t` (1 value) ``. `NA` and `NaN` are ordinary missing
values and never warn. The same applies to every other way values reach
the GPU: `copy = TRUE` joins,
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md),
`$<-`, a constant in
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html), and the
result of a verb that ran on the CPU. A fractional `Date` is floored
with its own warning, `cuplyr_warning_fractional_date`.

## See also

[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
to transfer data back to R,
[`compute.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/compute.tbl_gpu.md)
to execute and keep on GPU,
[`explain.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/explain.tbl_gpu.md)
and
[`show_query.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/show_query.tbl_gpu.md)
to see the plan,
[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md),
[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md),
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for data manipulation

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  # Transfer mtcars to GPU (lazy by default)
  gpu_mtcars <- tbl_gpu(mtcars)
  print(gpu_mtcars)

  # Operations are deferred until collect()
  result <- tbl_gpu(mtcars) |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425) |>
    collect()

  # Eager mode: each verb runs at once
  eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)

  # Or make every new table eager:
  # options(cuplyr.exec_mode = "eager")
}
#> # A tbl_gpu: 32 × 11
#> $ mpg  <dbl> 21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2
#> $ cyl  <dbl> 6, 6, 4, 6, 8, 6, 8, 4, 4, 6
#> $ disp <dbl> 160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6
#> $ hp   <dbl> 110, 110, 93, 110, 175, 105, 245, 62, 95, 123
#> $ drat <dbl> 3.9, 3.9, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92
#> $ wt   <dbl> 2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44
#> $ qsec <dbl> 16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20, 22.9, 18.3
#> $ vs   <dbl> 0, 0, 1, 1, 0, 1, 0, 1, 1, 1
#> $ am   <dbl> 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
#> $ gear <dbl> 4, 4, 4, 3, 3, 3, 3, 4, 4, 4
#> $ carb <dbl> 4, 4, 1, 1, 2, 1, 4, 2, 2, 4
```
