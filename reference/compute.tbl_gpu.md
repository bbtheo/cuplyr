# Force computation of pending GPU operations

Executes any pending lazy operations and stores the result in a new GPU
table. Data remains on the GPU (unlike
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) which
transfers to R).

## Usage

``` r
# S3 method for class 'tbl_gpu'
compute(x, ..., name = NULL)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Additional arguments (unused, for compatibility).

- name:

  Ignored (included for dplyr compatibility).

## Value

A `tbl_gpu` with all operations materialized and `lazy_ops` cleared.

## Details

Use [`compute()`](https://dplyr.tidyverse.org/reference/compute.html)
when you want to:

- Force optimization and execution of a lazy pipeline

- Create a checkpoint before branching operations

- Free memory from intermediate tables

- Prepare data for non-cuplyr functions that need a materialized table

Tables are lazy by default, so
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) is how
you run a pipeline and keep the result on the GPU. In eager mode
(`tbl_gpu(df, lazy = FALSE)`),
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) is a
no-op since operations execute immediately. Printing a lazy table also
runs its plan, but keeps nothing;
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) first
if you want to look at the result repeatedly.

## See also

[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
to bring data back to R,
[`collapse.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collapse.tbl_gpu.md)
to add optimization barrier without executing

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  # Lazy pipeline
  lazy_result <- tbl_gpu(mtcars) |>  # lazy by default
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425)

  # Force execution, keep on GPU
  gpu_result <- lazy_result |> compute()

  # Now branch into two different analyses
  analysis1 <- gpu_result |> filter(cyl == 4) |> collect()
  analysis2 <- gpu_result |> filter(cyl == 6) |> collect()
}
```
