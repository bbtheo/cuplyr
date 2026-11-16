# Sample rows by number or fraction

[`sample_n()`](https://dplyr.tidyverse.org/reference/sample_n.html) and
[`sample_frac()`](https://dplyr.tidyverse.org/reference/sample_n.html)
(superseded by
[`slice_sample()`](https://dplyr.tidyverse.org/reference/slice.html) in
dplyr) on a GPU table. Without `weight` they draw row indices on the CPU
with the exact [`sample.int()`](https://rdrr.io/r/base/sample.html)
calls dplyr makes, so the same
[`set.seed()`](https://rdrr.io/r/base/Random.html) gives the same rows,
and gather them on the GPU like
[`slice_sample.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/slice_sample.tbl_gpu.md).
A `weight`, a `size` that reads the data (a column,
[`n()`](https://dplyr.tidyverse.org/reference/context.html)) or draws
random numbers, and any call dplyr would raise an error for (e.g. a
`size` larger than a group without `replace`) run on the CPU through
[`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md),
where dplyr raises the error and consumes the RNG itself.

## Usage

``` r
# S3 method for class 'tbl_gpu'
sample_n(tbl, size, replace = FALSE, weight = NULL, .env = NULL, ...)

# S3 method for class 'tbl_gpu'
sample_frac(tbl, size = 1, replace = FALSE, weight = NULL, .env = NULL, ...)
```

## Arguments

- tbl:

  A `tbl_gpu` object.

- size:

  Number of rows
  ([`sample_n()`](https://dplyr.tidyverse.org/reference/sample_n.html))
  or fraction of rows
  ([`sample_frac()`](https://dplyr.tidyverse.org/reference/sample_n.html))
  to sample, per group.

- replace:

  Whether to sample with replacement.

- weight:

  Sampling weights; always run on the CPU.

- .env:

  Deprecated, as in dplyr.

- ...:

  Passed on to dplyr when the call runs on the CPU.

## Value

A `tbl_gpu` object with the sampled rows, grouped like `tbl`.
