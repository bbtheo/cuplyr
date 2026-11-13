# cuplyr <img src="man/figures/logo.png" align="right" height="138" />

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![Install checks](https://github.com/bbtheo/cuplyr/actions/workflows/install-matrix.yml/badge.svg?branch=main)](https://github.com/bbtheo/cuplyr/actions/workflows/install-matrix.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bbtheo/cuplyr-material/blob/main/notebooks/try-cuplyr-colab.ipynb)
<!-- badges: end -->

cuplyr runs dplyr code on an NVIDIA GPU. It is a dplyr backend, like dbplyr, built on [RAPIDS cuDF](https://github.com/rapidsai/cudf). Upload a data frame with `tbl_gpu()`, use the dplyr verbs you already know, and `collect()` the result back into R.

```r
library(cuplyr)
library(dplyr)

set.seed(1)
n <- 1e5
sales <- data.frame(
  year    = sample(2018:2024, n, replace = TRUE),
  region  = sample(c("north", "south", "east", "west"), n, replace = TRUE),
  quarter = sample(1:4, n, replace = TRUE),
  amount  = rnorm(n, mean = 50, sd = 30),
  price   = runif(n, 1, 10)
)
regions <- data.frame(
  region  = c("north", "south", "east", "west"),
  manager = c("Ana", "Ben", "Chi", "Dev")
)

tbl_gpu(sales) |>
  filter(year >= 2020, amount > 0) |>
  mutate(revenue = amount * price) |>
  group_by(region, quarter) |>
  summarise(total = sum(revenue), .groups = "drop") |>
  inner_join(tbl_gpu(regions), by = "region") |>
  arrange(desc(total)) |>
  collect()
```

No GPU? The [Colab notebook](https://colab.research.google.com/github/bbtheo/cuplyr-material/blob/main/notebooks/try-cuplyr-colab.ipynb) installs cuplyr on a free T4 and runs a few pipelines.

cuplyr is experimental. Expect breaking changes before 1.0.

## What runs on the GPU

The goal is every dplyr verb, with dplyr's results. Any difference from dplyr is a bug. The test suite checks cuplyr against dplyr, including dplyr's own tests.

These run on the GPU:

- `filter()`, `mutate()`, `transmute()`, `select()`, `rename()`, `relocate()`, `arrange()`, `distinct()`, `pull()`
- `group_by()`, `summarise()`, `reframe()`, `count()`, `tally()`, `rowwise()`
- `slice()` and its variants, including grouped `slice_min()` and `slice_max()`
- every join: mutating, filtering and cross joins, and `join_by()` with inequality, `between()`, `within()`, `overlaps()` and `closest()` conditions
- `bind_rows()`, `bind_cols()`, the set operations and the `rows_*()` verbs
- `across()`, `if_any()` and `if_all()`
- inside expressions:
  - arithmetic, comparisons, `if_else()`, `case_when()`, `case_match()`, `coalesce()`, `na_if()` and `between()`
  - window functions: `lag()`, `lead()`, the ranking functions, `cumsum()` and the other cumulative functions
  - summaries: `first()`, `last()`, `nth()`, `n_distinct()`, `median()`, `quantile()`, `sd()` and `var()`
  - base string functions such as `paste()`, `substr()`, `nchar()`, `grepl()`, `sub()`, `gsub()` and `toupper()`
  - date parts: `year()`, `month()`, and fixed-unit `floor_date()` and `round_date()`
- rolling windows with `roll_mean()`, `roll_sum()`, `roll_median()`, `roll_min()`, `roll_max()` and `roll_sd()`
- Parquet files, read and written on the GPU with `read_parquet_gpu()` and `write_parquet_gpu()`

Anything else falls back to the CPU. cuplyr collects the data, runs dplyr and uploads the result. Your code still works, but the data makes a round trip. The fallback is silent by default. Set `options(cuplyr.fallback = "warn")` to see each fallback with its expression and reason, or `"error"` to forbid them.

Column types:

| R | GPU |
|---|---|
| logical | BOOL8 |
| integer | INT32 |
| double | FLOAT64 |
| character | STRING |
| factor | DICTIONARY32, levels kept |
| Date | TIMESTAMP_DAYS |
| POSIXct | TIMESTAMP_MICROSECONDS, time zone kept |
| integer64 | FLOAT64, with a warning: values above 2^53 lose precision |

Not supported: list columns, other column classes and more than one GPU. Uploading a `difftime`, `hms` or `ts` column raises an error. The [open issues](https://github.com/bbtheo/cuplyr/issues) track what is planned.

## Lazy by default

`tbl_gpu()` is lazy. Each verb checks its arguments and records a step, so a typo or type error still fails at the verb that caused it. Nothing runs until you `collect()` the result into R or `compute()` it into a table that stays on the GPU. Then cuplyr optimizes the whole plan, runs it and frees every intermediate column.

Printing a lazy table runs the plan to show the first rows and keeps nothing. Print it twice and it runs twice. `compute()` a table you want to look at often.

`explain()` shows the recorded steps and the plan the optimizer will run:

```r
tbl_gpu(mtcars) |>
  filter(mpg > 20) |>
  mutate(kpl = mpg * 0.425) |>
  select(cyl, kpl) |>
  explain()
```

The optimizer drops columns the result never reads, fuses consecutive `mutate()` calls, pushes filters below mutates, selects and some joins, and merges consecutive filters into one. `vignette("query-optimization")` describes each pass.

For a single pipeline, lazy is as fast as eager up to 1e6 rows and under a millisecond slower at 1e7. In exchange it holds no GPU memory between calls. Eager mode keeps intermediate results on the GPU until R's next `gc()`, and in a loop they fill the card. Over 5e7 rows, five back-to-back eager runs took 68 ms each. Lazy took 18 ms.

Eager mode runs each verb as soon as you call it, which is easier when you step through code by hand:

```r
tbl_gpu(mtcars, lazy = FALSE)          # one table
options(cuplyr.exec_mode = "eager")    # every table created afterwards
```

`cuplyr_options()` lists the other options.

## Performance

A five-verb pipeline (`filter`, `mutate`, `group_by`, `summarise`, `arrange`) on an RTX 5070, median ms per call:

| rows | cuplyr, lazy | dplyr |
|---|---|---|
| 1e3 | 1.75 | 3.50 |
| 1e6 | 2.00 | 22.0 |
| 1e7 | 5.00 | 213 |

The GPU loses when you upload a table, run one cheap verb and collect it. Uploading 1e7 rows takes 63 ms and collecting them takes 54 ms. Keep data on the GPU between steps and collect once at the end.

`vignette("performance")` has the method, eager timings, memory per call and the other cases where dplyr wins. `vignette("memory")` explains how cuplyr allocates GPU memory and when `gpu_gc()` gives it back.

## Gibbles

cuplyr's tables are called gibbles. They print as `# A gibble:`, or `# A gibble [lazy]:` with pending steps. `gibble()`, `as_gibble()` and `is_gibble()` are aliases of `tbl_gpu()`, `as_tbl_gpu()` and `is_tbl_gpu()`. The CRAN package gibble also has a `gibble()` function. `?gibble` explains which one wins when both are attached.

## Installation

You need Linux x86_64 and:

| | |
|---|---|
| NVIDIA GPU | compute capability 7.0 (Volta) or newer |
| CUDA Toolkit | 12.2 or newer |
| RAPIDS libcudf | 25.12 or newer |
| R | 4.3 or newer |

The install script clones cuplyr, finds CUDA and libcudf through pixi, conda or your system (in that order), and builds the package:

```bash
curl -fsSL https://raw.githubusercontent.com/bbtheo/cuplyr/main/install.sh | bash
```

To pick the method yourself, pass `--method=pixi`, `--method=conda` or `--method=system`. `--dry-run` shows what the script would do. From a clone, run `./install.sh` with the same options.

Then check the installation from R:

```r
library(cuplyr)
verify_installation()
```

If something is wrong, `check_deps()` lists what is missing. `diagnostics()` prints the system details to paste into a bug report.

## Learn more

- [Documentation site](https://bbtheo.github.io/cuplyr/), with the reference and the vignettes
- `vignette("getting-started")` and `vignette("complex-analysis")` for longer examples
- [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the scope, known differences from dplyr, the developer setup and how pull requests work

## Acknowledgments

cuplyr is built on [RAPIDS cuDF](https://rapids.ai/), NVIDIA's open-source GPU data frame library, by the RAPIDS team.
