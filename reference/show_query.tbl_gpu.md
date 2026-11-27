# Show the plan of a lazy tbl_gpu

`tbl_gpu` methods for dplyr's
[`dplyr::show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
and
[`dplyr::explain()`](https://dplyr.tidyverse.org/reference/explain.html)
generics. Both print the same plan as R code and run nothing: no data is
read and no GPU work is done. A plan prints in pipeline order, source
first, one step per line, each step as the dplyr call it stands for:

## Usage

``` r
# S3 method for class 'tbl_gpu'
show_query(x, ..., optimized = TRUE)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Must be empty.

- optimized:

  `TRUE` (the default) prints the plan the optimizer will run, `FALSE`
  the plan as written.

## Value

`x`, invisibly.

## Details

    <cuplyr plan: optimized>
    CPU fallback: mutate() for pnorm(gain / 10)
      reason: `pnorm()` has no GPU implementation.
      columns read: 2 of 4 (carrier, z)
      input (computed at the mutate() call):
        parquet scan: flights.parquet
          columns read: 5 of 41 (carrier, distance, air_time, dep_delay, arr_delay)
          reader filter: distance > 1000L
        mutate(gain = dep_delay - arr_delay, speed = distance / air_time * 60)
        filter(speed > 400, gain > 0)
        select(carrier, speed, gain)
    filter(z > 0.5)
    select(carrier, z)

- A table on the GPU, a parquet file (`parquet scan:`) or the result of
  a step that ran on the CPU starts the plan, with the columns the plan
  reads from it: `columns read: 5 of 41 (...)` is a scan reading 5 of
  the file's 41 columns. A parquet scan also shows the comparisons the
  reader applies (`reader filter:`), written as code.

- A step that fell back to the CPU shows the verb, the expression, why
  it could not run on the GPU (the reason of the
  [cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)
  notice) and the plan it computed at that call. The newest three CPU
  steps are shown; under the oldest one the input reads
  `not shown (older CPU steps)`.
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html) of a
  table with pending steps starts a new plan with no history.

- A step that does not fit in 80 characters prints one argument per
  line.

- A join prints its right-hand plan under `y:`.

With `optimized = TRUE` the plan is what the optimizer made of the
steps: the columns nothing reads are left out, consecutive
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) calls
are fused, filters moved and, for a parquet scan, the reader's filter
and columns shown. `optimized = FALSE` shows the steps as you wrote
them. An eager table has no plan: it runs each verb when it is called.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  q <- tbl_gpu(mtcars) |>
    mutate(kpl = mpg * 0.425, hp2 = hp * 2) |>
    filter(cyl == 4) |>
    select(mpg, kpl)
  show_query(q)
  show_query(q, optimized = FALSE)
}
#> <cuplyr plan: optimized>
#> table on the GPU
#>   columns read: 2 of 11 (mpg, cyl)
#> filter(cyl == 4)
#> mutate(kpl = mpg * 0.425)
#> select(mpg, kpl)
#> select(mpg, kpl)
#> <cuplyr plan: as written>
#> table on the GPU
#>   columns read: all 11
#> mutate(kpl = mpg * 0.425, hp2 = hp * 2)
#> filter(cyl == 4)
#> select(mpg, kpl)
```
