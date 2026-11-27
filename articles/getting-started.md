# Getting started with cuplyr

cuplyr runs dplyr verbs on an NVIDIA GPU through the RAPIDS cuDF
library. You write the same dplyr code. Only the table type changes.

\
[`library`](https://rdrr.io/r/base/library.html)`(`[`cuplyr`](https://github.com/bbtheo/cuplyr)`)`\
[`library`](https://rdrr.io/r/base/library.html)`(`[`dplyr`](https://dplyr.tidyverse.org)`)`

The GPU pays off on large data, usually above 10 million rows. Below
about 100,000 rows, copying the data to the GPU costs more than it
saves, so plain dplyr is the better choice there.

## Basic workflow

Copy a data frame to the GPU with
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md), use
dplyr verbs on it, and bring the result back to R with
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html). The
original data frame is not changed.

\
`gpu_cars`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`\
`gpu_cars`\
`#> ``# A tbl_gpu: 32 × 11`\
`#> $ mpg  <dbl> 21, 21, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2`\
`#> $ cyl  <dbl> 6, 6, 4, 6, 8, 6, 8, 4, 4, 6`\
`#> $ disp <dbl> 160, 160, 108, 258, 360, 225, 360, 146.7, 140.8, 167.6`\
`#> $ hp   <dbl> 110, 110, 93, 110, 175, 105, 245, 62, 95, 123`\
`#> $ drat <dbl> 3.9, 3.9, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92`\
`#> $ wt   <dbl> 2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44`\
`#> $ qsec <dbl> 16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20, 22.9, 18.3`\
`#> $ vs   <dbl> 0, 0, 1, 1, 0, 1, 0, 1, 1, 1`\
`#> $ am   <dbl> 1, 1, 1, 0, 0, 0, 0, 0, 0, 0`\
`#> $ gear <dbl> 4, 4, 4, 3, 3, 3, 3, 4, 4, 4`\
`#> $ carb <dbl> 4, 4, 1, 1, 2, 1, 4, 2, 2, 4`

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``15``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``power_to_weight ``=`` ``hp`` ``/`` ``wt``)`` ``|>`\
`  `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``cyl``)`` ``|>`\
`  `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`    n_cars ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``,`\
`    avg_mpg ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``mpg``)``,`\
`    avg_power_ratio ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``power_to_weight``)``,`\
`    best_mpg ``=`` `[`max`](https://rdrr.io/r/base/Extremes.html)`(``mpg``)`\
`  ``)`` ``|>`\
`  `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``avg_mpg``)``)`` ``|>`\
`  `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`#> ``# A tibble: 3 × 5`\
`#>     cyl n_cars avg_mpg avg_power_ratio best_mpg`\
`#>   ``<dbl>``  ``<int>``   ``<dbl>``           ``<dbl>``    ``<dbl>`\
`#> ``1``     4     11    26.7            37.9     33.9`\
`#> ``2``     6      7    19.7            39.9     21.4`\
`#> ``3``     8      8    16.7            50.8     19.2`

## Common verbs

These work as they do in dplyr:

\
[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``x`` ``>`` ``10``, ``y`` ``<`` ``5``)``          ``# conditions are combined with AND`\
[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``col_a`` ``>`` ``col_b``)`\
\
[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``mpg``, ``cyl``, ``hp``)`\
[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``-``disp``, ``-``drat``)`\
[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``fuel_economy ``=`` ``mpg``)``     ``# rename while selecting`\
[`select`](https://dplyr.tidyverse.org/reference/select.html)`(`[`starts_with`](https://tidyselect.r-lib.org/reference/starts_with.html)`(``"d"``)``)`\
\
[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``a ``=`` ``x`` ``+`` ``y``, b ``=`` ``a`` ``*`` ``2``)``   ``# later columns can use earlier ones`\
[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``mpg ``=`` ``mpg`` ``*`` ``0.425``)``      ``# replace a column`\
\
[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``cyl``, `[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``mpg``)``)`\
\
[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``cyl``, ``gear``)`` ``|>`\
`  `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`    total ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``mpg``)``, average ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``mpg``)``,`\
`    minimum ``=`` `[`min`](https://rdrr.io/r/base/Extremes.html)`(``mpg``)``, maximum ``=`` `[`max`](https://rdrr.io/r/base/Extremes.html)`(``mpg``)``,`\
`    std_dev ``=`` `[`sd`](https://rdrr.io/r/stats/sd.html)`(``mpg``)``, variance ``=`` `[`var`](https://rdrr.io/r/stats/cor.html)`(``mpg``)``,`\
`    count ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)`\
`  ``)`

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

\
`pipeline`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``20``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``x ``=`` ``hp`` ``*`` ``2``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``cyl`` ``==`` ``4``)`\
\
[`is_lazy`](https://bbtheo.github.io/cuplyr/reference/is_lazy.md)`(``pipeline``)``          ``# TRUE`\
`#> [1] TRUE`\
[`has_pending_ops`](https://bbtheo.github.io/cuplyr/reference/has_pending_ops.md)`(``pipeline``)``  ``# TRUE`\
`#> [1] TRUE`\
[`show_query`](https://dplyr.tidyverse.org/reference/explain.html)`(``pipeline``)``       ``# the plan, as R code`\
`#> <cuplyr plan: optimized>`\
`#> table on the GPU`\
`#>   columns read: all 11`\
`#> filter(cyl == 4, mpg > 20)`\
`#> mutate(x = hp * 2)`

Errors such as a missing column or a type mismatch still appear at the
verb that causes them, not at
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html):

\
[`try`](https://rdrr.io/r/base/try.html)`(`[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``nonexistent_column`` ``>`` ``5``)``)`\
`#> Error in filter(tbl_gpu(mtcars), nonexistent_column > 5) : `\
`#>   ``ℹ```  In argument: `nonexistent_column > ``\
`` #>   5`. ``\
`#> ``Caused by error:`\
`#> ``!`` object 'nonexistent_column' not found`\
[`try`](https://rdrr.io/r/base/try.html)`(`[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``"not a number"``)``)`\
`#> ``# A tbl_gpu: 0 × 11`\
`#> $ mpg  <dbl> `\
`#> $ cyl  <dbl> `\
`#> $ disp <dbl> `\
`#> $ hp   <dbl> `\
`#> $ drat <dbl> `\
`#> $ wt   <dbl> `\
`#> $ qsec <dbl> `\
`#> $ vs   <dbl> `\
`#> $ am   <dbl> `\
`#> $ gear <dbl> `\
`#> $ carb <dbl>`

Printing a lazy table runs the plan and shows the first rows under a
`# A tbl_gpu [lazy]:` header. It keeps no result, so printing twice runs
the plan twice.

In eager mode each verb runs right away, which helps when you step
through a pipeline:

\
`eager_tbl`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``, lazy ``=`` ``FALSE``)`\
[`is_lazy`](https://bbtheo.github.io/cuplyr/reference/is_lazy.md)`(``eager_tbl``)``  ``# FALSE`\
`#> [1] FALSE`

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

\
`base_data`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``15``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``power_ratio ``=`` ``hp`` ``/`` ``wt``)`` ``|>`\
`  `[`compute`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
\
`four_cyl`` ``<-`` ``base_data`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``cyl`` ``==`` ``4``)`` ``|>`` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`six_cyl`` ``<-`` ``base_data`` ``|>`` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``cyl`` ``==`` ``6``)`` ``|>`` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`

## Memory

R’s garbage collector frees a table’s GPU memory once nothing refers to
it.
[`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
shows current use, and
[`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md) frees
unreferenced tables right away.

\
[`gpu_memory_state`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)`(``)`\
`#> $available`\
`#> [1] TRUE`\
`#> `\
`#> $total_bytes`\
`#> [1] 12441747456`\
`#> `\
`#> $free_bytes`\
`#> [1] 11469848576`\
`#> `\
`#> $used_bytes`\
`#> [1] 971898880`\
`#> `\
`#> $total_gb`\
`#> [1] 12.44175`\
`#> `\
`#> $free_gb`\
`#> [1] 11.46985`\
`#> `\
`#> $used_gb`\
`#> [1] 0.9718989`\
`#> `\
`#> $resource`\
`#> [1] "async"`\
`#> `\
`#> $pool_bytes`\
`#> [1] 33554432`\
`#> `\
`#> $pool_used_bytes`\
`#> [1] 1144010`

\
[`gpu_gc`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)`(``)`

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

## Installation

The conda package `r-cuplyr` on the `bbtheo` channel is a binary for
RAPIDS 26.10 and CUDA 13. It needs an NVIDIA driver that supports CUDA
13, but no CUDA toolkit or compiler:

``` bash
conda install -c bbtheo -c rapidsai -c conda-forge -c nvidia r-cuplyr
```

With pixi, run `pixi add r-cuplyr` in a project with the same channels.
To build from source, use the install script; the
[README](https://github.com/bbtheo/cuplyr#installation) has both routes.
Then check the installation from R:

\
[`library`](https://rdrr.io/r/base/library.html)`(`[`cuplyr`](https://github.com/bbtheo/cuplyr)`)`\
[`verify_installation`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)`(``)``  ``# runs a small GPU round trip`\
[`check_deps`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)`(``)``           ``# lists anything missing`\
[`diagnostics`](https://bbtheo.github.io/cuplyr/reference/diagnostics.md)`(``)``          ``# system details for a bug report`

## System requirements

- NVIDIA GPU with compute capability 7.5 (Turing) or newer
- CUDA Toolkit 12.2 or later (RAPIDS 26.x is tested with CUDA 13, which
  the installers and pixi use)
- RAPIDS libcudf 26.02 or newer (developed and tested on 26.10)
- R 4.3.0 or later
