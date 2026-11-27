# Wait for the GPU to finish

Blocks until all GPU work queued by cuplyr and libcudf on the device has
completed. GPU calls return as soon as the work is queued, so a timing
taken without a wait can measure only the launch. Call `gpu_sync()`
before reading the clock, as one would use `torch.cuda.synchronize()`.

## Usage

``` r
gpu_sync()
```

## Value

`NULL`, invisibly.

## Details

Results that are collected to R
([`collect()`](https://dplyr.tidyverse.org/reference/compute.html),
[`print()`](https://rdrr.io/r/base/print.html)) are already waited for;
this is for timing code that leaves its result on the GPU.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  g <- tbl_gpu(data.frame(x = runif(1e6)), lazy = FALSE)
  t0 <- Sys.time()
  r <- dplyr::mutate(g, y = cumsum(x))
  gpu_sync()
  Sys.time() - t0
}
#> Time difference of 0.1315472 secs
```
