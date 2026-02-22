# Force computation of pending GPU operations

Executes any pending lazy operations and stores the result in a new GPU
table. Data remains on the GPU (unlike `collect()` which transfers to
R).

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

Use `compute()` when you want to:

- Force optimization and execution of a lazy pipeline

- Create a checkpoint before branching operations

- Free memory from intermediate tables

- Prepare data for non-cuplyr functions that need a materialized table

In eager mode, `compute()` is a no-op since operations execute
immediately.

## See also

[`collect.tbl_gpu`](collect.tbl_gpu.md) to bring data back to R,
[`collapse.tbl_gpu`](collapse.tbl_gpu.md) to add optimization barrier
without executing

## Examples

``` r
if (has_gpu()) {
  # Lazy pipeline
  lazy_result <- tbl_gpu(mtcars, lazy = TRUE) |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425)

  # Force execution, keep on GPU
  gpu_result <- lazy_result |> compute()

  # Now branch into two different analyses
  analysis1 <- gpu_result |> filter(cyl == 4) |> collect()
  analysis2 <- gpu_result |> filter(cyl == 6) |> collect()
}
#> Error in mutate(filter(tbl_gpu(mtcars, lazy = TRUE), mpg > 20), kpl = mpg *     0.425): could not find function "mutate"
```
