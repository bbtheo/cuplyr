# Transfer GPU table data back to R

Copies data from GPU memory back to R as a tibble. This is typically the
final step in a GPU data manipulation pipeline, after filtering,
mutating, and selecting the data you need.

## Usage

``` r
# S3 method for class 'tbl_gpu'
collect(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Additional arguments (ignored, included for compatibility).

## Value

A [tibble::tibble](https://tibble.tidyverse.org/reference/tibble.html)
containing the data from the GPU table. Column types are converted back
to R types:

- FLOAT64/FLOAT32 -\> numeric (double)

- INT32/INT64 -\> integer or numeric

- STRING -\> character

- BOOL8 -\> logical (TRUE/FALSE)

## Details

### Memory considerations

Collecting transfers all data from GPU to CPU memory. For large
datasets, this can be slow and memory-intensive. Best practice is to:

1.  Filter rows to reduce data volume

2.  Select only needed columns

3.  Then collect the results

### Performance

Data transfer between GPU and CPU is limited by PCIe bandwidth
(typically 16-32 GB/s). For a 1 GB dataset, expect ~50-100ms transfer
time.

## See also

[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) for
creating GPU tables,
[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md),
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for reducing data

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Process on GPU, then collect
  result <- gpu_mtcars |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425) |>
    select(mpg, kpl, hp) |>
    collect()

  # Result is a regular tibble
  class(result)
  print(result)
}
#> Error in collect(select(mutate(filter(gpu_mtcars, mpg > 20), kpl = mpg *     0.425), mpg, kpl, hp)): could not find function "collect"
```
