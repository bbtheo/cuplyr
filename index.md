# cuplyr

cuplyr runs dplyr code on an NVIDIA GPU. It is a dplyr backend, like
dbplyr, built on [RAPIDS cuDF](https://github.com/rapidsai/cudf). Upload
a data frame with
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md), use
the dplyr verbs you already know, and
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) the
result back into R.

\
[`library`](https://rdrr.io/r/base/library.html)`(`[`cuplyr`](https://github.com/bbtheo/cuplyr)`)`\
[`library`](https://rdrr.io/r/base/library.html)`(`[`dplyr`](https://dplyr.tidyverse.org)`)`\
\
[`set.seed`](https://rdrr.io/r/base/Random.html)`(``1``)`\
`n`` ``<-`` ``1e5`\
`sales`` ``<-`` `[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(`\
`  year    ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(``2018``:``2024``, ``n``, replace ``=`` ``TRUE``)``,`\
`  region  ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(`[`c`](https://rdrr.io/r/base/c.html)`(``"north"``, ``"south"``, ``"east"``, ``"west"``)``, ``n``, replace ``=`` ``TRUE``)``,`\
`  quarter ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(``1``:``4``, ``n``, replace ``=`` ``TRUE``)``,`\
`  amount  ``=`` `[`rnorm`](https://rdrr.io/r/stats/Normal.html)`(``n``, mean ``=`` ``50``, sd ``=`` ``30``)``,`\
`  price   ``=`` `[`runif`](https://rdrr.io/r/stats/Uniform.html)`(``n``, ``1``, ``10``)`\
`)`\
`regions`` ``<-`` `[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(`\
`  region  ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``"north"``, ``"south"``, ``"east"``, ``"west"``)``,`\
`  manager ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``"Ana"``, ``"Ben"``, ``"Chi"``, ``"Dev"``)`\
`)`\
\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``sales``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``year`` ``>=`` ``2020``, ``amount`` ``>`` ``0``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``revenue ``=`` ``amount`` ``*`` ``price``)`` ``|>`\
`  `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``region``, ``quarter``)`` ``|>`\
`  `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(``total ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``, .groups ``=`` ``"drop"``)`` ``|>`\
`  `[`inner_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(`[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``regions``)``, by ``=`` ``"region"``)`` ``|>`\
`  `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``total``)``)`` ``|>`\
`  `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`

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
  [`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md) and
  [`roll_sd()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  ([`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  runs on the CPU)
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

| R | GPU |
|----|----|
| logical | BOOL8 |
| integer | INT32 |
| double | FLOAT64 |
| character | STRING |
| factor | DICTIONARY32, levels kept |
| Date | TIMESTAMP_DAYS |
| POSIXct | TIMESTAMP_MICROSECONDS, time zone kept |
| integer64 | INT64, exact; [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) gives back [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html) |

Not supported: list columns, other column classes and more than one GPU.
Uploading a `difftime`, `hms` or `ts` column raises an error. The [open
issues](https://github.com/bbtheo/cuplyr/issues) track what is planned,
in priority order on the [roadmap
board](https://github.com/users/bbtheo/projects/1).

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

[`explain()`](https://dplyr.tidyverse.org/reference/explain.html) prints
the plan the optimizer will run as R code: the steps in order, the
columns each source reads, and any step that fell back to the CPU, with
the reason. `show_query(optimized = FALSE)` prints the steps as you
wrote them. Neither runs anything:

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``)`` ``|>`\
`  `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``mpg`` ``>`` ``20``)`` ``|>`\
`  `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``kpl ``=`` ``mpg`` ``*`` ``0.425``)`` ``|>`\
`  `[`select`](https://dplyr.tidyverse.org/reference/select.html)`(``cyl``, ``kpl``)`` ``|>`\
`  `[`explain`](https://dplyr.tidyverse.org/reference/explain.html)`(``)`

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

\
[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``mtcars``, lazy ``=`` ``FALSE``)``          ``# one table`\
[`options`](https://rdrr.io/r/base/options.html)`(``cuplyr.exec_mode ``=`` ``"eager"``)``    ``# every table created afterwards`

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

## Installation

The conda and pixi package below needs only Linux x86_64, an NVIDIA GPU
with compute capability 7.5 (Turing) or newer and a driver that supports
CUDA 13. It brings its own R and libraries, and needs no CUDA toolkit or
compiler.

To build from source (the install script), you need Linux x86_64 and:

|  |  |
|----|----|
| NVIDIA GPU | compute capability 7.5 (Turing) or newer |
| CUDA Toolkit | 12.2 or newer (RAPIDS 26.x is tested with CUDA 13, which the installers and pixi use) |
| RAPIDS libcudf | 26.02 or newer (developed and tested on 26.10) |
| R | 4.3 or newer |

Volta (7.0) GPUs are not supported: that would need a source build
against a CUDA 12 RAPIDS, which no install route provides.

The install script clones cuplyr, finds libcudf through pixi, conda or
your system (in that order), and builds the package. The pixi
environment carries libcudf and the CUDA 13 build tools (`nvcc` and the
CUDA headers), so `--method=pixi` needs only a system C++ compiler; a
system toolkit set through `CUDA_HOME` still wins. The conda method
installs `cuda-nvcc` into its environment itself:

``` bash
curl -fsSL https://raw.githubusercontent.com/bbtheo/cuplyr/main/install.sh | bash
```

To pick the method yourself, pass `--method=pixi`, `--method=conda` or
`--method=system`. `--dry-run` shows what the script would do. From a
clone, run `./install.sh` with the same options.

### conda and pixi

The conda package `r-cuplyr` is published on the `bbtheo` channel on
anaconda.org. It is a binary built against RAPIDS libcudf 26.10 and CUDA
13, so it needs no compiler or CUDA toolkit on your machine, only an
NVIDIA driver that supports CUDA 13. Install it into an active conda
environment (for a new one, run `conda create -n cuplyr` and
`conda activate cuplyr` first):

``` bash
conda install -c bbtheo -c rapidsai -c conda-forge -c nvidia r-cuplyr
```

`mamba` takes the same arguments. If your conda configuration sets
strict channel priority and the solve fails, add
`--no-channel-priority`.

With pixi, create a project with the same channels and add the package:

``` bash
pixi init -c bbtheo -c rapidsai -c conda-forge -c nvidia cuplyr-project
cd cuplyr-project
pixi add r-cuplyr
```

Then start R in the project with `pixi run R` (or `pixi shell`, then
`R`).

How the package is built and published is described in
[conda/README.md](https://github.com/bbtheo/cuplyr/blob/main/conda/README.md).

### Check the installation

Check the installation from R:

\
[`library`](https://rdrr.io/r/base/library.html)`(`[`cuplyr`](https://github.com/bbtheo/cuplyr)`)`\
[`verify_installation`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)`(``)`

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
