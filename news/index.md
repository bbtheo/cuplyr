# Changelog

## cuplyr 0.2.0

cuplyr 0.2.0 brings most of dplyr to the GPU. Nearly every dplyr verb
now works on a `tbl_gpu`, cuplyr’s GPU table. Most run on the GPU, and
the rest run in dplyr on the CPU with the same result. GPU tables are
lazy by default, so an optimizer runs the whole pipeline at
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html). Also
new: [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
joins, window and rolling functions,
[`across()`](https://dplyr.tidyverse.org/reference/across.html),
[`rowwise()`](https://dplyr.tidyverse.org/reference/rowwise.html), set
operations, Parquet files on the GPU, exact `integer64` columns and a
conda package. cuplyr now needs RAPIDS libcudf 26.02 or later and a GPU
with compute capability 7.5 or higher.

### Breaking changes

- GPU tables are lazy by default.
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
  returns a lazy table, each verb records a step, and
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html),
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html) or
  [`print()`](https://rdrr.io/r/base/print.html) run the plan. Errors in
  a verb’s arguments are still raised by that verb, but dplyr’s
  warnings, such as integer overflow or a many-to-many join, come when
  the plan runs, at every [`print()`](https://rdrr.io/r/base/print.html)
  too. For eager tables, set `options(cuplyr.exec_mode = "eager")` or
  pass `lazy = FALSE`.

- On a lazy table with pending steps,
  [`nrow()`](https://rdrr.io/r/base/nrow.html) and
  [`dim()`](https://rdrr.io/r/base/dim.html) run the plan, while
  [`gpu_object_info()`](https://bbtheo.github.io/cuplyr/reference/gpu_object_info.md)
  and
  [`gpu_memory_usage()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_usage.md)
  count the rows of the table it was made from and
  [`verify_gpu_data()`](https://bbtheo.github.io/cuplyr/reference/verify_gpu_data.md)
  can return `FALSE`. Call
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html)
  first.

- cuplyr needs RAPIDS libcudf 26.02 or later and is tested on 26.10.
  RAPIDS 25.12 is no longer supported
  ([\#94](https://github.com/bbtheo/cuplyr/issues/94)).

- The minimum GPU is compute capability 7.5 (Turing). The conda package
  and the pixi and conda install routes use a CUDA 13 build of RAPIDS,
  which has no code for Volta
  ([\#150](https://github.com/bbtheo/cuplyr/issues/150)).

- Results that differed from dplyr now match it:

  - Grouping follows dplyr.
    [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
    drops only the last grouping column by default
    (`.groups = "drop_last"`, with a message like dplyr’s),
    [`select()`](https://dplyr.tidyverse.org/reference/select.html)
    keeps the grouping columns, joins keep `x`’s groups,
    [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
    keeps the first input’s groups,
    [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
    returns a `grouped_df`, and
    [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
    keeps the groups of a `grouped_df`. In 0.1.1 each of these dropped
    the groups. `ungroup(x, col)` removes only `col`; 0.1.1 removed
    every group.
  - [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
    evaluates its arguments as dplyr does. A tidyselect helper needs
    [`pick()`](https://dplyr.tidyverse.org/reference/pick.html), as in
    `group_by(pick(all_of(cols)))`, and a string such as `group_by("x")`
    makes one constant group; 0.1.1 took both as column names.
    [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
    with no arguments removes the grouping; 0.1.1 kept it.
  - `select(new = old)` renames the column; 0.1.1 kept the old name.
  - [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
    aggregates follow R’s `na.rm = FALSE` default, so a group with an
    `NA` gives `NA`.
  - An ungrouped
    [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
    of zero rows, such as after a
    [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) that
    keeps none, returns one row, where
    [`sum()`](https://rdrr.io/r/base/sum.html) gives 0; 0.1.1 returned
    no rows.
  - `arrange(desc(x))` puts `NA` last, and `arrange(.by_group = TRUE)`
    sorts the group columns ascending, as dplyr does.
  - Grouped
    [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
    returns groups sorted by their keys, not in hash order, and keeps
    rows whose key is `NA` as a last group; 0.1.1 dropped those rows.
  - [`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
    fills the keys of rows found only in `y`; 0.1.1 left them `NA`.
  - Integer `+`, `-` and `*` return integers, not doubles. An overflow
    gives `NA` with R’s warning.
  - In
    [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
    [`sum()`](https://rdrr.io/r/base/sum.html) of integers or logicals,
    such as `sum(x > 0)`, and an ungrouped
    [`n()`](https://dplyr.tidyverse.org/reference/context.html) return
    integers, not doubles, and
    [`min()`](https://rdrr.io/r/base/Extremes.html) and
    [`max()`](https://rdrr.io/r/base/Extremes.html) of a logical column
    return integers, not logicals. A
    [`sum()`](https://rdrr.io/r/base/sum.html) too large for an integer
    is a double, as in R.
  - `NaN` stays `NaN`; it used to become `NA` on upload.
  - `integer64` columns are exact and collect to
    [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html);
    they used to be uploaded as doubles, which corrupted them.
  - `POSIXct` columns keep their time zone instead of coming back in
    UTC.
  - A GPU table made from a `data.frame` collects to a `data.frame`,
    with its attributes.
  - Factor columns stay factors after
    [`select()`](https://dplyr.tidyverse.org/reference/select.html),
    joins,
    [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md),
    [`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md),
    a lazy
    [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) and
    as
    [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
    group keys, and ordered factors stay ordered; 0.1.1 returned integer
    codes, or plain factors. Joins match factor keys by their labels;
    0.1.1 matched the codes, which paired the wrong rows when the two
    tables’ levels differed.

- Uploading a column with no GPU type (`difftime`, `hms`, `ts`,
  `POSIXlt`, a list, a matrix with several columns, …) raises an error
  that names the column, instead of losing its class. So does a verb
  that makes a `difftime` on the CPU, such as `end - start` on dates.
  Returning such columns is an open issue
  ([\#65](https://github.com/bbtheo/cuplyr/issues/65)).

- [`print()`](https://rdrr.io/r/base/print.html) uses a tibble-style
  `# A tbl_gpu: n x m` header, R type labels and factor labels.

- [`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
  is now a method for dplyr’s generic. Its argument is `x`, and it
  prints the plan as R code and returns `x` invisibly.

- `$`, `[[` and [`length()`](https://rdrr.io/r/base/length.html) of a
  GPU table give its columns, as for a data frame, and `names<-` accepts
  `NA` and empty names. In 0.1.1, `$`, `[[` and
  [`length()`](https://rdrr.io/r/base/length.html) gave the object’s
  internal fields, such as `x$schema` and `x$ptr`, and `names<-` refused
  those names ([\#4](https://github.com/bbtheo/cuplyr/issues/4)).
  [`split()`](https://rdrr.io/r/base/split.html),
  [`with()`](https://rdrr.io/r/base/with.html),
  [`subset()`](https://rdrr.io/r/base/subset.html),
  [`t()`](https://rdrr.io/r/base/t.html),
  [`as.environment()`](https://rdrr.io/r/base/as.environment.html),
  [`is.na()`](https://rdrr.io/r/base/NA.html) and
  [`anyNA()`](https://rdrr.io/r/base/NA.html) of a GPU table raise an
  error; they used to act on the internal fields.

### Installation and requirements

- cuplyr is available as a conda package, `r-cuplyr`, on the `bbtheo`
  channel on anaconda.org. It is a binary for RAPIDS 26.10 and CUDA 13
  that needs an NVIDIA driver for CUDA 13 but no CUDA toolkit or
  compiler. Install it with conda, mamba or pixi; the README has the
  commands ([\#135](https://github.com/bbtheo/cuplyr/issues/135)).

- `install.sh` and
  [`install_cuplyr()`](https://bbtheo.github.io/cuplyr/reference/install_cuplyr.md)
  build cuplyr against libcudf from pixi, conda or the system
  (`--method=` picks one). The conda route installs RAPIDS 26.10 for
  CUDA 13 with the CUDA compiler, and the pixi environments carry the
  CUDA 13 build tools, so neither needs a system toolkit
  ([\#157](https://github.com/bbtheo/cuplyr/issues/157)).

- `./configure` stops with a clear message when libcudf is older than
  26.02, and prefers the CUDA toolkit of a conda or pixi environment
  ([\#38](https://github.com/bbtheo/cuplyr/issues/38),
  [\#94](https://github.com/bbtheo/cuplyr/issues/94)).

- New
  [`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
  checks the driver, CUDA toolkit, libcudf and R packages,
  [`verify_installation()`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)
  runs a small GPU round trip, and
  [`diagnostics()`](https://bbtheo.github.io/cuplyr/reference/diagnostics.md)
  reports versions, the GPU, the memory resource and every option for a
  bug report, as text or JSON.
  [`diagnose_loader()`](https://bbtheo.github.io/cuplyr/reference/diagnose_loader.md)
  shows how cuplyr’s shared library resolves libstdc++, libcudf and
  libcuda ([\#14](https://github.com/bbtheo/cuplyr/issues/14),
  [\#49](https://github.com/bbtheo/cuplyr/issues/49)).

### New features

#### Verbs

- Anything with no GPU form runs in dplyr on the CPU instead of failing.
  `options(cuplyr.fallback = "warn")` reports each fallback with its
  reason and the data moved; `"error"` forbids them.

- Newly on the GPU:
  [`filter_out()`](https://dplyr.tidyverse.org/reference/filter.html),
  [`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html),
  [`rename()`](https://dplyr.tidyverse.org/reference/rename.html),
  [`rename_with()`](https://dplyr.tidyverse.org/reference/rename.html),
  [`relocate()`](https://dplyr.tidyverse.org/reference/relocate.html),
  [`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html),
  [`pull()`](https://dplyr.tidyverse.org/reference/pull.html),
  [`count()`](https://dplyr.tidyverse.org/reference/count.html),
  [`tally()`](https://dplyr.tidyverse.org/reference/count.html),
  [`add_count()`](https://dplyr.tidyverse.org/reference/count.html),
  [`add_tally()`](https://dplyr.tidyverse.org/reference/count.html),
  [`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html), the
  [`slice()`](https://dplyr.tidyverse.org/reference/slice.html) family,
  [`top_n()`](https://dplyr.tidyverse.org/reference/top_n.html),
  [`sample_n()`](https://dplyr.tidyverse.org/reference/sample_n.html),
  [`group_split()`](https://dplyr.tidyverse.org/reference/group_split.html),
  [`group_trim()`](https://dplyr.tidyverse.org/reference/group_trim.html),
  [`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
  and the other grouping functions,
  [`rowwise()`](https://dplyr.tidyverse.org/reference/rowwise.html), the
  `rows_*()` verbs and the `_at()`, `_all()` and `_if()` verbs.

- [`group_map()`](https://dplyr.tidyverse.org/reference/group_map.html),
  [`group_modify()`](https://dplyr.tidyverse.org/reference/group_map.html),
  [`group_nest()`](https://dplyr.tidyverse.org/reference/group_nest.html),
  [`nest_by()`](https://dplyr.tidyverse.org/reference/nest_by.html),
  [`nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)
  and [`do()`](https://dplyr.tidyverse.org/reference/do.html) run on the
  CPU.

- `.by` works in
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
  [`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
  [`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html) and
  the [`slice()`](https://dplyr.tidyverse.org/reference/slice.html)
  family, and grouped verbs run per group on the GPU.
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  takes computed keys and `.drop = FALSE`.

- [`rowwise()`](https://dplyr.tidyverse.org/reference/rowwise.html) runs
  elementwise expressions and row aggregates such as
  `sum(c_across(a:e))` on the GPU. On 1e5 rows that takes milliseconds,
  where dplyr takes about twenty seconds.

- `[`, `[<-` for whole columns,
  [`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html),
  [`as_tibble()`](https://tibble.tidyverse.org/reference/as_tibble.html),
  [`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html),
  [`head()`](https://rdrr.io/r/utils/head.html),
  [`rbind()`](https://rdrr.io/r/base/cbind.html),
  [`cbind()`](https://rdrr.io/r/base/cbind.html),
  [`vctrs::vec_slice()`](https://vctrs.r-lib.org/reference/vec_slice.html),
  [`vctrs::vec_size()`](https://vctrs.r-lib.org/reference/vec_size.html),
  [`dplyr::bind_rows()`](https://dplyr.tidyverse.org/reference/bind_rows.html)
  and
  [`purrr::map_dfr()`](https://purrr.tidyverse.org/reference/map_dfr.html)
  work on GPU tables.

#### Lazy plans

- The optimizer now also prunes unused columns below
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
  [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) and
  other barriers.

- [`explain()`](https://dplyr.tidyverse.org/reference/explain.html) and
  [`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
  print the plan as R code without running it, including the columns a
  Parquet scan reads and the reason for each CPU fallback
  ([\#10](https://github.com/bbtheo/cuplyr/issues/10)).

- Printing a lazy table runs the whole plan to show the first rows and
  keeps nothing; a plan over a Parquet file prints its schema unless it
  only selects or renames columns.

- A step whose result type or error depends on the data (an integer
  [`sum()`](https://rdrr.io/r/base/sum.html), a join with
  `unmatched = "error"`) runs at the verb, as in eager mode;
  `options(cuplyr.plan_break = "warn")` reports it.

#### Expressions

- [`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) and
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  share one expression engine that runs on the GPU:

  - arithmetic, comparisons, logic, `%in%`,
    [`is.na()`](https://rdrr.io/r/base/NA.html),
    [`is.nan()`](https://rdrr.io/r/base/is.finite.html),
    [`between()`](https://dplyr.tidyverse.org/reference/between.html),
    [`near()`](https://dplyr.tidyverse.org/reference/near.html) and math
    functions, [`round()`](https://rdrr.io/r/base/Round.html) with R’s
    rule;
  - [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html),
    [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html),
    [`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html),
    [`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html),
    [`na_if()`](https://dplyr.tidyverse.org/reference/na_if.html),
    [`recode()`](https://dplyr.tidyverse.org/reference/recode.html) and
    dplyr 1.2’s
    [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html),
    [`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html),
    [`replace_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html),
    [`when_any()`](https://dplyr.tidyverse.org/reference/when-any-all.html)
    and
    [`when_all()`](https://dplyr.tidyverse.org/reference/when-any-all.html);
  - window functions:
    [`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html),
    [`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html), the
    ranking and cumulative functions,
    [`consecutive_id()`](https://dplyr.tidyverse.org/reference/consecutive_id.html),
    [`first()`](https://dplyr.tidyverse.org/reference/nth.html),
    [`last()`](https://dplyr.tidyverse.org/reference/nth.html),
    [`nth()`](https://dplyr.tidyverse.org/reference/nth.html),
    [`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)
    and grouped aggregates such as `x - mean(x)`;
  - summaries: [`median()`](https://rdrr.io/r/stats/median.html),
    [`quantile()`](https://rdrr.io/r/stats/quantile.html),
    [`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html),
    [`any()`](https://rdrr.io/r/base/any.html),
    [`all()`](https://rdrr.io/r/base/all.html) and the rest, of any
    expression and with arithmetic around them
    ([\#22](https://github.com/bbtheo/cuplyr/issues/22));
  - base string functions:
    [`paste()`](https://rdrr.io/r/base/paste.html),
    [`substr()`](https://rdrr.io/r/base/substr.html),
    [`nchar()`](https://rdrr.io/r/base/nchar.html),
    [`toupper()`](https://rdrr.io/r/base/chartr.html),
    [`trimws()`](https://rdrr.io/r/base/trimws.html),
    [`grepl()`](https://rdrr.io/r/base/grep.html),
    [`sub()`](https://rdrr.io/r/base/grep.html),
    [`gsub()`](https://rdrr.io/r/base/grep.html),
    [`startsWith()`](https://rdrr.io/r/base/startsWith.html), …;
  - factor comparisons with labels and
    [`as.character()`](https://rdrr.io/r/base/character.html);
  - `Date` and `POSIXct` literals, casts and `date + n`, lubridate’s
    date parts (`year()`, `hour()`, `wday()`, …) and fixed-unit
    `floor_date()`, `ceiling_date()` and `round_date()`.

- [`across()`](https://dplyr.tidyverse.org/reference/across.html),
  [`if_any()`](https://dplyr.tidyverse.org/reference/across.html),
  [`if_all()`](https://dplyr.tidyverse.org/reference/across.html),
  [`cur_group_id()`](https://dplyr.tidyverse.org/reference/context.html)
  and
  [`cur_column()`](https://dplyr.tidyverse.org/reference/context.html)
  run on the GPU;
  [`pick()`](https://dplyr.tidyverse.org/reference/pick.html) as a data
  frame and
  [`cur_group()`](https://dplyr.tidyverse.org/reference/context.html)
  run on the CPU.

- A function of your own named like one cuplyr runs on the GPU
  ([`mean()`](https://rdrr.io/r/base/mean.html),
  [`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html), …) is
  the one called, as in dplyr.

#### Joins and set operations

- The mutating joins,
  [`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html),
  [`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
  and
  [`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
  run on the GPU, with
  [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
  equality, inequality,
  [`between()`](https://dplyr.tidyverse.org/reference/between.html),
  [`within()`](https://rdrr.io/r/base/with.html), `overlaps()` and
  `closest()` conditions. `multiple`, `unmatched`, `relationship`,
  `na_matches`, `keep` and `suffix` follow dplyr, and keys of different
  types combine by vctrs’ rules, an all-`NA` key included
  ([\#70](https://github.com/bbtheo/cuplyr/issues/70)).

- [`union()`](https://generics.r-lib.org/reference/setops.html),
  [`union_all()`](https://dplyr.tidyverse.org/reference/setops.html),
  [`intersect()`](https://generics.r-lib.org/reference/setops.html),
  [`setdiff()`](https://generics.r-lib.org/reference/setops.html),
  [`symdiff()`](https://dplyr.tidyverse.org/reference/setops.html) and
  [`setequal()`](https://generics.r-lib.org/reference/setops.html) run
  on the GPU.
  [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
  follows vctrs’ type rules and binds factors to the union of their
  levels. Binds and set operations return a lazy table when any input is
  lazy ([\#20](https://github.com/bbtheo/cuplyr/issues/20),
  [\#70](https://github.com/bbtheo/cuplyr/issues/70)).

#### Parquet

- New
  [`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
  and
  [`write_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
  read and write Parquet files on the GPU; only compressed bytes cross
  the PCIe bus.

- A lazy read reads only the columns the plan uses and skips row groups
  for simple filters on integer, `integer64`, `Date` and `POSIXct`
  columns. `options(cuplyr.parquet_pushdown = FALSE)` turns this off;
  use it for untrusted files, as malformed statistics can crash the
  session.

#### Types

- `integer64` comparisons, sorting, grouping, joins and binds run
  exactly on the GPU; arithmetic and aggregates follow bit64 on the CPU.

- `NaN` counts as missing in
  [`is.na()`](https://rdrr.io/r/base/NA.html), comparisons,
  [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) and
  the `na.rm = TRUE` reducers, as in R.

- An infinite or out-of-range `Date` or `POSIXct` value warns on upload
  and becomes `NA`; it used to give `NA` or a wrong date silently.

- Columns keep their element names, attributes and one-column matrix
  shape by dplyr’s rules, and tables with rows but no columns work
  ([\#19](https://github.com/bbtheo/cuplyr/issues/19),
  [\#21](https://github.com/bbtheo/cuplyr/issues/21),
  [\#56](https://github.com/bbtheo/cuplyr/issues/56)).

- Strings go to the GPU as UTF-8, and columns with more than 2 GiB of
  characters work.

#### Rolling windows

- New
  [`roll_mean()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_sum()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
  [`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md) and
  [`roll_sd()`](https://bbtheo.github.io/cuplyr/reference/roll.md) run
  on the GPU in
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) and
  [`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
  grouped or not, in time that doesn’t depend on the window width.
  [`data.table::frollmean()`](https://rdrr.io/pkg/data.table/man/froll.html)
  and its siblings work as aliases. New
  [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  matches [`stats::median()`](https://rdrr.io/r/stats/median.html)
  exactly and runs on the CPU.

#### Options, errors and memory

- New
  [`cuplyr_options()`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
  lists every option with its value and where it came from;
  [`?"cuplyr-options"`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
  documents them. An invalid value warns once, with class
  `cuplyr_warning_invalid_option`, and the default is used.

- Errors and warnings have dplyr’s class, message and call for missing
  columns, calls dplyr rejects and the
  [`min()`](https://rdrr.io/r/base/Extremes.html) and
  [`max()`](https://rdrr.io/r/base/Extremes.html) warnings for empty
  groups. Errors in
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) carry
  dplyr’s class and its `In argument:` context
  ([\#13](https://github.com/bbtheo/cuplyr/issues/13),
  [\#20](https://github.com/bbtheo/cuplyr/issues/20),
  [\#24](https://github.com/bbtheo/cuplyr/issues/24),
  [\#28](https://github.com/bbtheo/cuplyr/issues/28),
  [\#30](https://github.com/bbtheo/cuplyr/issues/30),
  [\#69](https://github.com/bbtheo/cuplyr/issues/69)).

- cuplyr installs CUDA’s stream-ordered (`"async"`) memory resource when
  it loads. It reserves a quarter of the free GPU memory, at most 1 GiB,
  and keeps the memory of freed tables for reuse, so other processes see
  it as used until
  [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)
  returns it to the device and reports it in `released_bytes`. Each call
  frees its temporary GPU tables when it returns. Before loading, set
  `options(cuplyr.memory_resource = "pool")` for RMM’s pool,
  `cuplyr.pool_initial` to reserve less, or `cuplyr.release_threshold`
  to release memory automatically
  ([`vignette("memory")`](https://bbtheo.github.io/cuplyr/articles/memory.md)).

- [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
  reports the memory resource, `pool_bytes` and `pool_used_bytes`.
  `used_bytes` counts the memory the resource holds, so it no longer
  falls when a table is freed, and the `freed_bytes` of
  [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)
  counts memory freed inside the resource.

- New
  [`gpu_sync()`](https://bbtheo.github.io/cuplyr/reference/gpu_sync.md)
  waits for queued GPU work, for timing. A step that runs out of GPU
  memory is retried once after garbage collection, and the final error
  says what helps ([\#13](https://github.com/bbtheo/cuplyr/issues/13)).

- Loading cuplyr sets the environment variable
  `LIBCUDF_USE_DEBUG_STREAM_POOL` unless it is set, so that Parquet
  reads don’t slow later verbs
  ([`?read_parquet_gpu`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)).

- New
  [`vignette("performance")`](https://bbtheo.github.io/cuplyr/articles/performance.md)
  and
  [`vignette("memory")`](https://bbtheo.github.io/cuplyr/articles/memory.md),
  and
  [`vignette("query-optimization")`](https://bbtheo.github.io/cuplyr/articles/query-optimization.md)
  matches the optimizer. Help-page examples run
  ([\#39](https://github.com/bbtheo/cuplyr/issues/39)).

### Performance

- Recording a
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) on a
  lazy table no longer slows down as the plan grows, and a column passed
  through a verb is shared instead of copied.

- Each [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
  or [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
  call runs as one fused GPU kernel where it can, and computes a
  repeated [`round()`](https://rdrr.io/r/base/Round.html), `%in%` or
  string call once ([\#17](https://github.com/bbtheo/cuplyr/issues/17)).

- A lazy `arrange(x) |> head(n)` on one numeric or date key, and
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html)
  (grouped too), use a top-k selection instead of a full sort.

### Minor improvements and bug fixes

- [`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
  works (it failed on every call) and returns dplyr’s row order and
  keys.

- [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  accepts
  [`dplyr::n()`](https://dplyr.tidyverse.org/reference/context.html) and
  [`stats::sd()`](https://rdrr.io/r/stats/sd.html), names an unnamed
  expression after its text, and works lazily with
  [`sd()`](https://rdrr.io/r/stats/sd.html) and
  [`var()`](https://rdrr.io/r/stats/cor.html).

- Lazy plans with chained joins, or a
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) that
  overwrites a column, give the right result.

- The namespace loads with only its stated dependencies (it failed when
  utils was not attached, which `R CMD check` reports),
  [`rowwise()`](https://dplyr.tidyverse.org/reference/rowwise.html)’s
  method takes `data` like the generic, and the C++ code no longer uses
  the APIs libcudf 26.10 deprecates
  ([\#166](https://github.com/bbtheo/cuplyr/issues/166)).

- [`?tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
  explains what a GPU table is and how to work with one: lazy by
  default,
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html),
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html),
  [`explain()`](https://dplyr.tidyverse.org/reference/explain.html),
  eager mode and the column types. The verbs’ help pages say what runs
  on the GPU, what runs in dplyr on the CPU and where the result differs
  from dplyr, without internal development notes
  ([\#159](https://github.com/bbtheo/cuplyr/issues/159)).

## cuplyr 0.1.1

### Bug fixes

- [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) now
  correctly restores factor columns with their original levels.
  Previously, factor columns were returned as integer codes instead of
  factors ([\#3](https://github.com/bbtheo/cuplyr/issues/3)).

- `names<-()` now validates the replacement value before assignment. It
  errors if the new names have wrong length, contain `NA` values, empty
  strings, or are not character
  ([\#4](https://github.com/bbtheo/cuplyr/issues/4)).

### Build system

- Fixed CCCL (CUDA Core Compute Libraries) header detection for RAPIDS
  25.12+ in pixi/conda environments where headers are located in a
  `rapids/` subdirectory. The configure script now automatically detects
  `<cuda/stream_ref>` and related headers.

- Reordered include paths so CUDF/RMM headers take precedence over
  system CUDA headers, ensuring consistent CCCL versions.

### Documentation

- Added pkgdown documentation site with vignettes for getting started,
  complex analysis workflows, and query optimization.

## cuplyr 0.1.0

### Lazy evaluation

- [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
  gains a `lazy` argument to enable deferred execution. Operations build
  an AST (Abstract Syntax Tree) that is optimized and executed only when
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) or
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html) is
  called. Set globally with `options(cuplyr.exec_mode = "lazy")` or
  `CUPLYR_EXEC_MODE=lazy` environment variable.

- Added an AST optimizer that applies multiple optimization passes
  before execution:

  - Projection pruning: push column selection close to data sources
  - Mutate fusion: combine consecutive mutate operations
  - Dead column pruning: remove unused intermediate columns
  - Filter pushdown: move filters earlier in the pipeline, including
    across joins
  - Filter reordering: execute cheaper filters first
  - Filter fusion: combine multiple filters into single GPU kernel

- [`compute()`](https://dplyr.tidyverse.org/reference/compute.html)
  executes pending lazy operations and keeps the result on GPU.

- [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html)
  inserts an optimization barrier without executing.

- [`as_lazy()`](https://bbtheo.github.io/cuplyr/reference/as_lazy.md)
  and
  [`as_eager()`](https://bbtheo.github.io/cuplyr/reference/as_eager.md)
  switch execution modes mid-pipeline.

- [`is_lazy()`](https://bbtheo.github.io/cuplyr/reference/is_lazy.md)
  and
  [`has_pending_ops()`](https://bbtheo.github.io/cuplyr/reference/has_pending_ops.md)
  check the current execution state.

- [`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
  displays the pending operation tree for debugging.

### Join operations

- Added
  [`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  [`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  [`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  and
  [`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
  for combining GPU tables
  ([\#2](https://github.com/bbtheo/cuplyr/issues/2)). Joins support
  automatic key detection (natural join), named vectors in `by` for
  different key names, `suffix` for column name conflicts, `keep` for
  retaining join keys, and `copy` to auto-transfer data frames to GPU.

### Bind operations

- Added
  [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
  for vertically combining GPU tables with automatic schema unification
  and type promotion.

- Added
  [`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md)
  for horizontally combining GPU tables with `.name_repair` for
  duplicate column handling.

- Both bind functions automatically materialize lazy tables before
  binding.

## cuplyr 0.0.1

Initial release of cuplyr, a GPU-accelerated dplyr backend using
NVIDIA’s libcudf library.

### Core functionality

- [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
  transfers R data frames to GPU memory, returning a `tbl_gpu` object
  that works with dplyr verbs.

- [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
  transfers GPU data back to R as a tibble.

- [`as_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/as_tbl_gpu.md)
  coerces data frames to GPU tables.

- [`is_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/is_tbl_gpu.md)
  tests if an object is a GPU table.

### dplyr verbs

- [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
  supports scalar comparisons (`x > 5`, `x == "a"`) and column-to-column
  comparisons (`x > y`). Supports boolean vectors as filter masks.

- [`select()`](https://dplyr.tidyverse.org/reference/select.html)
  supports column selection by name, position, and tidyselect helpers.

- [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
  supports arithmetic operations (`+`, `-`, `*`, `/`, `^`) with scalars
  and between columns. Supports left-associative chains (e.g.,
  `a + b + c`).

- [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html)
  sorts by one or more columns with
  [`desc()`](https://dplyr.tidyverse.org/reference/desc.html) support
  for descending order. Supports `.by_group = TRUE` for grouped tables.

- [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  sets grouping metadata for subsequent aggregation.
  [`ungroup()`](https://dplyr.tidyverse.org/reference/group_by.html)
  removes grouping.

- [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  computes grouped aggregations with support for
  [`sum()`](https://rdrr.io/r/base/sum.html),
  [`mean()`](https://rdrr.io/r/base/mean.html),
  [`min()`](https://rdrr.io/r/base/Extremes.html),
  [`max()`](https://rdrr.io/r/base/Extremes.html),
  [`n()`](https://dplyr.tidyverse.org/reference/context.html),
  [`sd()`](https://rdrr.io/r/stats/sd.html), and
  [`var()`](https://rdrr.io/r/stats/cor.html). Supports expressions
  inside aggregation functions (e.g., `sum(x > 0)`).

### Type support

- Supported R types: numeric (FLOAT64), integer (INT32), character
  (STRING), logical (BOOL8), Date (TIMESTAMP_DAYS), and POSIXct
  (TIMESTAMP_MICROSECONDS).

- factor columns are converted to INT32 codes.

- integer64 columns are converted to FLOAT64 with a warning about
  precision loss for values exceeding 2^53.

### GPU memory utilities

- [`gpu_memory_usage()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_usage.md)
  estimates GPU memory footprint of a `tbl_gpu` object.

- [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
  returns current GPU memory usage (total, free, used).

- [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)
  forces garbage collection to free GPU memory from unreferenced tables.

- [`gpu_object_info()`](https://bbtheo.github.io/cuplyr/reference/gpu_object_info.md)
  returns detailed information about a GPU table.

- [`verify_gpu_data()`](https://bbtheo.github.io/cuplyr/reference/verify_gpu_data.md)
  confirms data resides on GPU, not in R memory.

- [`gpu_size_comparison()`](https://bbtheo.github.io/cuplyr/reference/gpu_size_comparison.md)
  compares R object size vs GPU data size.

### GPU information

- [`has_gpu()`](https://bbtheo.github.io/cuplyr/reference/has_gpu.md)
  checks if a compatible GPU is available.

- [`gpu_details()`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
  returns GPU device information (name, compute capability, memory).
