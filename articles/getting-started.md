# Getting started with cuplyr

cuplyr runs dplyr verbs on an NVIDIA GPU through the RAPIDS cuDF
library. You write the same dplyr code. Only the table type changes.

``` r

library(cuplyr)
library(dplyr)
```

The GPU pays off on large data, usually above 10 million rows. Below
about 100,000 rows, copying the data to the GPU costs more than it
saves, so plain dplyr is the better choice there.

## Basic workflow

Copy a data frame to the GPU with
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md), use
dplyr verbs on it, and bring the result back to R with
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html). The
original data frame is not changed.

``` r

gpu_cars <- tbl_gpu(mtcars)
gpu_cars
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

``` r

tbl_gpu(mtcars) |>
  filter(mpg > 15) |>
  mutate(power_to_weight = hp / wt) |>
  group_by(cyl) |>
  summarise(
    n_cars = n(),
    avg_mpg = mean(mpg),
    avg_power_ratio = mean(power_to_weight),
    best_mpg = max(mpg)
  ) |>
  arrange(desc(avg_mpg)) |>
  collect()
#> # A tibble: 3 × 5
#>     cyl n_cars avg_mpg avg_power_ratio best_mpg
#>   <dbl>  <int>   <dbl>           <dbl>    <dbl>
#> 1     4     11    26.7            37.9     33.9
#> 2     6      7    19.7            39.9     21.4
#> 3     8      8    16.7            50.8     19.2
```

## Common verbs

These work as they do in dplyr:

``` r

filter(x > 10, y < 5)          # conditions are combined with AND
filter(col_a > col_b)

select(mpg, cyl, hp)
select(-disp, -drat)
select(fuel_economy = mpg)     # rename while selecting
select(starts_with("d"))

mutate(a = x + y, b = a * 2)   # later columns can use earlier ones
mutate(mpg = mpg * 0.425)      # replace a column

arrange(cyl, desc(mpg))

group_by(cyl, gear) |>
  summarise(
    total = sum(mpg), average = mean(mpg),
    minimum = min(mpg), maximum = max(mpg),
    std_dev = sd(mpg), variance = var(mpg),
    count = n()
  )
```

## Column types

| R type | GPU type | Notes |
|----|----|----|
| `double` | FLOAT64 |  |
| `integer` | INT32 |  |
| `character` | STRING |  |
| `logical` | BOOL8 |  |
| `Date` | TIMESTAMP_DAYS |  |
| `POSIXct` | TIMESTAMP_MICROSECONDS |  |
| `factor` | INT32 | Integer codes on the GPU, levels kept in R. The schema labels it DICTIONARY32 |

## Lazy and eager execution

Tables are lazy by default. Each verb adds a step to a plan, and the
plan runs when you call
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) or
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html).
Before it runs, cuplyr optimizes it, for example by moving filters
earlier and fusing steps.

``` r

pipeline <- tbl_gpu(mtcars) |>
  filter(mpg > 20) |>
  mutate(x = hp * 2) |>
  filter(cyl == 4)

is_lazy(pipeline)          # TRUE
#> [1] TRUE
has_pending_ops(pipeline)  # TRUE
#> [1] TRUE
show_query(pipeline)       # the pending steps
#> Pending operations:
#> ast_filter [1 predicates]
#> | ast_mutate [1 expressions]
#> | | ast_filter [1 predicates]
#> | | | ast_source [11 cols]
#> 
#> AST depth: 4 
#> Node count: 4
```

Errors such as a missing column or a type mismatch still appear at the
verb that causes them, not at
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html):

``` r

try(tbl_gpu(mtcars) |> filter(nonexistent_column > 5))
#> Error : object 'nonexistent_column' not found
try(tbl_gpu(mtcars) |> filter(mpg > "not a number"))
#> # A gibble: 0 × 11
#> $ mpg  <dbl> 
#> $ cyl  <dbl> 
#> $ disp <dbl> 
#> $ hp   <dbl> 
#> $ drat <dbl> 
#> $ wt   <dbl> 
#> $ qsec <dbl> 
#> $ vs   <dbl> 
#> $ am   <dbl> 
#> $ gear <dbl> 
#> $ carb <dbl>
```

Printing a lazy table runs the plan and shows the first rows under a
`# A gibble [lazy]:` header. It keeps no result, so printing twice runs
the plan twice.

In eager mode each verb runs right away, which helps when you step
through a pipeline:

``` r

eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)
is_lazy(eager_tbl)  # FALSE
#> [1] FALSE
```

To make every table eager, set `options(cuplyr.exec_mode = "eager")` or
the environment variable `CUPLYR_EXEC_MODE=eager`. The `lazy` argument
of [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
overrides both.
[`cuplyr_options()`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
lists every option with its current value, and
[`?"cuplyr-options"`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
documents them.

## Reusing an intermediate result

[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) runs
the pending steps and keeps the result on the GPU. Use it when several
analyses start from the same base, so the shared steps run once:

``` r

base_data <- tbl_gpu(mtcars) |>
  filter(mpg > 15) |>
  mutate(power_ratio = hp / wt) |>
  compute()

four_cyl <- base_data |> filter(cyl == 4) |> collect()
six_cyl <- base_data |> filter(cyl == 6) |> collect()
```

## Memory

R’s garbage collector frees a table’s GPU memory once nothing refers to
it.
[`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
shows current use, and
[`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md) frees
unreferenced tables right away.

``` r

gpu_memory_state()
#> $available
#> [1] TRUE
#> 
#> $total_bytes
#> [1] 12418416640
#> 
#> $free_bytes
#> [1] 11604656128
#> 
#> $used_bytes
#> [1] 813760512
#> 
#> $total_gb
#> [1] 12.41842
#> 
#> $free_gb
#> [1] 11.60466
#> 
#> $used_gb
#> [1] 0.8137605
#> 
#> $resource
#> [1] "async"
#> 
#> $pool_bytes
#> [1] 33554432
#> 
#> $pool_used_bytes
#> [1] 1144010
```

``` r

gpu_gc()
```

## Tips

Filter and select early so later steps handle less data. Keep pipelines
lazy so the optimizer can work on them. Call
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) once,
at the end, so only the final result is copied back to R.

## Next steps

- [`vignette("query-optimization")`](https://bbtheo.github.io/cuplyr/articles/query-optimization.md)
  explains the optimizer.
- [`vignette("performance")`](https://bbtheo.github.io/cuplyr/articles/performance.md)
  has timings and memory use for lazy and eager mode, and shows where
  the GPU is slower.
- [`vignette("memory")`](https://bbtheo.github.io/cuplyr/articles/memory.md)
  covers GPU memory resources and when memory is freed.
- See also
  [`?tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
  [`?compute`](https://dplyr.tidyverse.org/reference/compute.html) and
  [`?gpu_memory_state`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md).

## System requirements

- NVIDIA GPU with compute capability 7.0 or higher
- CUDA Toolkit 12.2 or later
- RAPIDS cuDF 25.12 or later
- R 4.3.0 or later
