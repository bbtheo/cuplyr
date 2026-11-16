# Create a GPU-backed data frame

cuplyr's GPU tables, called gibbles, live in GPU memory. `gibble()` is
another name for `tbl_gpu()`. It transfers an R data frame to GPU memory
using NVIDIA's libcudf library, enabling high-performance data
manipulation operations. The resulting `tbl_gpu` object can be used with
dplyr verbs like
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
[`select()`](https://dplyr.tidyverse.org/reference/select.html), and
collected back to R with
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

## Usage

``` r
tbl_gpu(data, ..., lazy = NULL)

gibble(data, ..., lazy = NULL)

# S3 method for class 'data.frame'
tbl_gpu(data, ..., lazy = NULL)

# S3 method for class 'tbl_gpu'
tbl_gpu(data, ..., lazy = NULL)
```

## Arguments

- data:

  A data frame or tibble to transfer to GPU memory. Supported column
  types include: numeric (double), integer, character, and logical.

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

A `tbl_gpu` object containing:

- `ptr` - External pointer to the GPU table

- `schema` - List with column names and types

- `lazy_ops` - AST for pending operations (lazy mode only)

- `groups` - Grouping variables

- `exec_mode` - Execution mode ("lazy" or "eager")

## Details

The data is immediately copied to GPU memory when `tbl_gpu()` is called
(also in lazy mode; only the verbs are deferred). GPU memory is
automatically freed when the R object is garbage collected.

### Execution Modes

In **lazy mode** (the default), operations build an AST (Abstract Syntax
Tree) that is optimized and executed only when
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) or
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) is
called. This enables optimizations like:

- Projection pruning (only load needed columns)

- Mutate fusion (combine multiple mutates)

- Filter reordering (cheapest filters first)

Errors in a verb's arguments (a missing column, a type error) are still
raised when the verb is called. Printing a lazy table with pending
operations (or calling
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) on it)
runs the plan to show its first rows, under a `# A gibble [lazy]:`
header; the table itself is unchanged and nothing is kept, so printing
it twice runs the plan twice
([`compute()`](https://dplyr.tidyverse.org/reference/compute.html) it
first to look at it repeatedly). If the plan fails (an error, out of GPU
memory), the print shows the schema and the message instead of raising.

In **eager mode** (`lazy = FALSE`), each dplyr verb executes immediately
on the GPU. This is simple but can lead to unnecessary intermediate
allocations.

For measured timings and memory use of the two modes, see
[`vignette("performance", package = "cuplyr")`](https://bbtheo.github.io/cuplyr/articles/performance.md).

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
returns it as a plain logical column. An unspecified column joined
against a string, `Date` or factor key, or bound to a character column
in
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md),
still errors.

Column type mappings from R to GPU:

- `numeric` -\> FLOAT64

- `integer` -\> INT32

- `character` -\> STRING

- `logical` -\> BOOL8

- `Date` -\> TIMESTAMP_DAYS

- `POSIXct` -\> TIMESTAMP_MICROSECONDS

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

### Gibbles

"Gibble" is the user-facing name of a `tbl_gpu`:
[`print()`](https://rdrr.io/r/base/print.html) starts with `# A gibble:`
and the dimensions, and `gibble()`,
[`as_gibble()`](https://bbtheo.github.io/cuplyr/reference/as_tbl_gpu.md)
and
[`is_gibble()`](https://bbtheo.github.io/cuplyr/reference/is_tbl_gpu.md)
are aliases of `tbl_gpu()`,
[`as_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/as_tbl_gpu.md)
and
[`is_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/is_tbl_gpu.md).
The class stays `tbl_gpu`, and `tbl_sum()` still returns dplyr's values.

The CRAN package gibble (geometry decomposition, Michael Sumner) also
exports a function called `gibble()`. When both packages are attached,
the one attached last masks the other; `cuplyr::gibble()` always works.

## See also

[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
to transfer data back to R,
[`compute.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/compute.tbl_gpu.md)
to execute and keep on GPU,
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
#> # A gibble: 32 × 11
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
