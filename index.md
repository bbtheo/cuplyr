# cuplyr

cuplyr runs dplyr code on an NVIDIA GPU. It is a dplyr backend, like
dbplyr, built on [RAPIDS cuDF](https://github.com/rapidsai/cudf). Upload
a data frame with
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md), use
the dplyr verbs you already know, and
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) the
result back into R.

``` r

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

No GPU? The [Colab
notebook](https://colab.research.google.com/github/bbtheo/cuplyr-material/blob/main/notebooks/try-cuplyr-colab.ipynb)
installs cuplyr on a free T4 and runs a few pipelines.

cuplyr is experimental. Expect breaking changes before 1.0.

## What runs on the GPU

The goal is every dplyr verb, with dplyr’s results. Any difference from
dplyr is a bug. The test suite checks cuplyr against dplyr, including
dplyr’s own tests.

These run on the GPU:

- [`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
  [`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html),
  [`select()`](https://dplyr.tidyverse.org/reference/select.html),
  [`rename()`](https://dplyr.tidyverse.org/reference/rename.html),
  [`relocate()`](https://dplyr.tidyverse.org/reference/relocate.html),
  [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html),
  [`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html),
  [`pull()`](https://dplyr.tidyverse.org/reference/pull.html)
- [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html),
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
  [`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html),
  [`count()`](https://dplyr.tidyverse.org/reference/count.html),
  [`tally()`](https://dplyr.tidyverse.org/reference/count.html),
  [`rowwise()`](https://dplyr.tidyverse.org/reference/rowwise.html)
- [`slice()`](https://dplyr.tidyverse.org/reference/slice.html) and its
  variants, including grouped
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html) and
  [`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)
- every join: mutating, filtering and cross joins, and
  [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html) with
  inequality,
  [`between()`](https://dplyr.tidyverse.org/reference/between.html),
  [`within()`](https://rdrr.io/r/base/with.html), `overlaps()` and
  `closest()` conditions
- [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md),
  [`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md),
  the set operations and the `rows_*()` verbs
- [`across()`](https://dplyr.tidyverse.org/reference/across.html),
  [`if_any()`](https://dplyr.tidyverse.org/reference/across.html) and
  [`if_all()`](https://dplyr.tidyverse.org/reference/across.html)
- inside expressions:
  - arithmetic, comparisons,
    [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html),
    [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html),
    [`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html),
    [`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html),
    [`na_if()`](https://dplyr.tidyverse.org/reference/na_if.html) and
    [`between()`](https://dplyr.tidyverse.org/reference/between.html)
  - window functions:
    [`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html),
    [`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html), the
    ranking functions, [`cumsum()`](https://rdrr.io/r/base/cumsum.html)
    and the other cumulative functions
  - summaries:
    [`first()`](https://dplyr.tidyverse.org/reference/nth.html),
    [`last()`](https://dplyr.tidyverse.org/reference/nth.html),
    [`nth()`](https://dplyr.tidyverse.org/reference/nth.html),
    [`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html),
    [`median()`](https://rdrr.io/r/stats/median.html),
    [`quantile()`](https://rdrr.io/r/stats/quantile.html),
    [`sd()`](https://rdrr.io/r/stats/sd.html) and
    [`var()`](https://rdrr.io/r/stats/cor.html)
  - base string functions such as
    [`paste()`](https://rdrr.io/r/base/paste.html),
    [`substr()`](https://rdrr.io/r/base/substr.html),
    [`nchar()`](https://rdrr.io/r/base/nchar.html),
    [`grepl()`](https://rdrr.io/r/base/grep.html),
    [`sub()`](https://rdrr.io/r/base/grep.html),
    [`gsub()`](https://rdrr.io/r/base/grep.html) and
    [`toupper()`](https://rdrr.io/r/base/chartr.html)
  - date parts: `year()`, `month()`, and fixed-unit `floor_date()` and
    `round_date()`
- rolling windows with
  [`roll_mean()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_sum()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md) and
  [`roll_sd()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
- Parquet files, read and written on the GPU with
  [`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
  and
  [`write_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)

Anything else falls back to the CPU. cuplyr collects the data, runs
dplyr and uploads the result. Your code still works, but the data makes
a round trip. The fallback is silent by default. Set
`options(cuplyr.fallback = "warn")` to see each fallback with its
expression and reason, or `"error"` to forbid them.

Column types:

| R         | GPU                                                       |
|-----------|-----------------------------------------------------------|
| logical   | BOOL8                                                     |
| integer   | INT32                                                     |
| double    | FLOAT64                                                   |
| character | STRING                                                    |
| factor    | DICTIONARY32, levels kept                                 |
| Date      | TIMESTAMP_DAYS                                            |
| POSIXct   | TIMESTAMP_MICROSECONDS, time zone kept                    |
| integer64 | FLOAT64, with a warning: values above 2^53 lose precision |

Not supported: list columns, other column classes and more than one GPU.
Uploading a `difftime`, `hms` or `ts` column raises an error. The [open
issues](https://github.com/bbtheo/cuplyr/issues) track what is planned.

## Lazy by default

[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) is
lazy. Each verb checks its arguments and records a step, so a typo or
type error still fails at the verb that caused it. Nothing runs until
you [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
the result into R or
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) it
into a table that stays on the GPU. Then cuplyr optimizes the whole
plan, runs it and frees every intermediate column.

Printing a lazy table runs the plan to show the first rows and keeps
nothing. Print it twice and it runs twice.
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) a
table you want to look at often.

[`explain()`](https://dplyr.tidyverse.org/reference/explain.html) shows
the recorded steps and the plan the optimizer will run:

``` r

tbl_gpu(mtcars) |>
  filter(mpg > 20) |>
  mutate(kpl = mpg * 0.425) |>
  select(cyl, kpl) |>
  explain()
```

The optimizer drops columns the result never reads, fuses consecutive
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) calls,
pushes filters below mutates, selects and some joins, and merges
consecutive filters into one.
[`vignette("query-optimization")`](https://bbtheo.github.io/cuplyr/articles/query-optimization.md)
describes each pass.

For a single pipeline, lazy is as fast as eager up to 1e6 rows and under
a millisecond slower at 1e7. In exchange it holds no GPU memory between
calls. Eager mode keeps intermediate results on the GPU until R’s next
[`gc()`](https://rdrr.io/r/base/gc.html), and in a loop they fill the
card. Over 5e7 rows, five back-to-back eager runs took 68 ms each. Lazy
took 18 ms.

Eager mode runs each verb as soon as you call it, which is easier when
you step through code by hand:

``` r

tbl_gpu(mtcars, lazy = FALSE)          # one table
options(cuplyr.exec_mode = "eager")    # every table created afterwards
```

[`cuplyr_options()`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
lists the other options.

## Performance

A five-verb pipeline (`filter`, `mutate`, `group_by`, `summarise`,
`arrange`) on an RTX 5070, median ms per call:

| rows | cuplyr, lazy | dplyr |
|------|--------------|-------|
| 1e3  | 1.75         | 3.50  |
| 1e6  | 2.00         | 22.0  |
| 1e7  | 5.00         | 213   |

The GPU loses when you upload a table, run one cheap verb and collect
it. Uploading 1e7 rows takes 63 ms and collecting them takes 54 ms. Keep
data on the GPU between steps and collect once at the end.

[`vignette("performance")`](https://bbtheo.github.io/cuplyr/articles/performance.md)
has the method, eager timings, memory per call and the other cases where
dplyr wins.
[`vignette("memory")`](https://bbtheo.github.io/cuplyr/articles/memory.md)
explains how cuplyr allocates GPU memory and when
[`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md) gives
it back.

## Gibbles

cuplyr’s tables are called gibbles. They print as `# A gibble:`, or
`# A gibble [lazy]:` with pending steps.
[`gibble()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
[`as_gibble()`](https://bbtheo.github.io/cuplyr/reference/as_tbl_gpu.md)
and
[`is_gibble()`](https://bbtheo.github.io/cuplyr/reference/is_tbl_gpu.md)
are aliases of
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
[`as_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/as_tbl_gpu.md)
and
[`is_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/is_tbl_gpu.md).
The CRAN package gibble also has a
[`gibble()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
function.
[`?gibble`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
explains which one wins when both are attached.

## Installation

You need Linux x86_64 and:

|                |                                         |
|----------------|-----------------------------------------|
| NVIDIA GPU     | compute capability 7.0 (Volta) or newer |
| CUDA Toolkit   | 12.2 or newer                           |
| RAPIDS libcudf | 25.12 or newer                          |
| R              | 4.3 or newer                            |

The install script clones cuplyr, finds CUDA and libcudf through pixi,
conda or your system (in that order), and builds the package:

``` bash
curl -fsSL https://raw.githubusercontent.com/bbtheo/cuplyr/main/install.sh | bash
```

To pick the method yourself, pass `--method=pixi`, `--method=conda` or
`--method=system`. `--dry-run` shows what the script would do. From a
clone, run `./install.sh` with the same options.

Then check the installation from R:

``` r

library(cuplyr)
verify_installation()
```

If something is wrong,
[`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
lists what is missing.
[`diagnostics()`](https://bbtheo.github.io/cuplyr/reference/diagnostics.md)
prints the system details to paste into a bug report.

## Learn more

- [Documentation site](https://bbtheo.github.io/cuplyr/), with the
  reference and the vignettes
- [`vignette("getting-started")`](https://bbtheo.github.io/cuplyr/articles/getting-started.md)
  and
  [`vignette("complex-analysis")`](https://bbtheo.github.io/cuplyr/articles/complex-analysis.md)
  for longer examples
- [CONTRIBUTING.md](https://bbtheo.github.io/cuplyr/CONTRIBUTING.md) for
  the scope, known differences from dplyr, the developer setup and how
  pull requests work

## Acknowledgments

cuplyr is built on [RAPIDS cuDF](https://rapids.ai/), NVIDIA’s
open-source GPU data frame library, by the RAPIDS team.
