# Package index

## All functions

- [`add_count(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/add_count.tbl_gpu.md)
  : Add a count column without collapsing rows

- [`arrange(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/arrange.tbl_gpu.md)
  : Order rows using column values

- [`as_eager()`](https://bbtheo.github.io/cuplyr/reference/as_eager.md)
  : Switch to eager execution mode

- [`as_lazy()`](https://bbtheo.github.io/cuplyr/reference/as_lazy.md) :
  Switch to lazy execution mode

- [`as_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/as_tbl_gpu.md)
  : Coerce to a GPU table

- [`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md)
  : Bind multiple data frames/tables by column

- [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
  : Bind multiple data frames/tables by row

- [`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
  : Check cuplyr dependencies

- [`collapse(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/collapse.tbl_gpu.md)
  : Create a subquery barrier without executing

- [`collect(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
  : Transfer GPU table data back to R

- [`compute(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/compute.tbl_gpu.md)
  : Force computation of pending GPU operations

- [`count(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md)
  : Count observations by group

- [`cross_join(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/cross_join.tbl_gpu.md)
  :

  Cross join for `tbl_gpu`

- [`cuplyr-fallback`](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)
  : CPU fallbacks and plan breaks: what cuplyr tells you, and how to
  hear it

- [`cuplyr_options()`](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
  : cuplyr options

- [`detect_environment()`](https://bbtheo.github.io/cuplyr/reference/detect_environment.md)
  : Detect the runtime environment

- [`diagnose_loader()`](https://bbtheo.github.io/cuplyr/reference/diagnose_loader.md)
  : Diagnose dynamic linker configuration for cuplyr

- [`diagnostics()`](https://bbtheo.github.io/cuplyr/reference/diagnostics.md)
  : System diagnostics for cuplyr

- [`dim(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/dim.tbl_gpu.md)
  : Get dimensions of a GPU table

- [`` `dimnames<-`( ``*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/dimnames-set-.tbl_gpu.md)
  : Set the dimnames of a GPU table

- [`dimnames(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/dimnames.tbl_gpu.md)
  : Dimnames of a GPU table

- [`distinct(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/distinct.tbl_gpu.md)
  : Keep distinct rows

- [`explain(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/explain.tbl_gpu.md)
  : Explain a tbl_gpu query

- [`filter(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md)
  : Keep rows that match a condition

- [`filter_out(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/filter_out.tbl_gpu.md)
  : Drop rows that match a condition

- [`glimpse(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/glimpse.tbl_gpu.md)
  : Get a glimpse of a GPU table's structure

- [`gpu_details()`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
  : Get detailed GPU information

- [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md) :
  Force GPU memory cleanup

- [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
  : Get GPU memory snapshot

- [`gpu_memory_usage()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_usage.md)
  : Estimate GPU memory usage of a tbl_gpu object

- [`gpu_object_info()`](https://bbtheo.github.io/cuplyr/reference/gpu_object_info.md)
  : Get detailed information about a GPU table object

- [`gpu_size_comparison()`](https://bbtheo.github.io/cuplyr/reference/gpu_size_comparison.md)
  : Compare R object size vs GPU data size

- [`gpu_sync()`](https://bbtheo.github.io/cuplyr/reference/gpu_sync.md)
  : Wait for the GPU to finish

- [`group_by(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
  : Group a GPU table by one or more columns

- [`group_split(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/group_split.tbl_gpu.md)
  : Split a GPU table into one GPU table per group (dplyr backend
  method)

- [`group_trim(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/group_trim.tbl_gpu.md)
  : Drop unused levels of the factor group columns (dplyr backend
  method)

- [`group_vars(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/group_vars.tbl_gpu.md)
  : Get grouping variables from a GPU table

- [`groups(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/groups.tbl_gpu.md)
  : Get grouping information from a GPU table

- [`has_gpu()`](https://bbtheo.github.io/cuplyr/reference/has_gpu.md) :
  Check GPU availability

- [`has_pending_ops()`](https://bbtheo.github.io/cuplyr/reference/has_pending_ops.md)
  : Check if a tbl_gpu has pending lazy operations

- [`head(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/head.tbl_gpu.md)
  : First rows of a GPU table

- [`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md)
  : Infer output schema from an AST node

- [`install_cuplyr()`](https://bbtheo.github.io/cuplyr/reference/install_cuplyr.md)
  : Install cuplyr from source

- [`is_lazy()`](https://bbtheo.github.io/cuplyr/reference/is_lazy.md) :
  Check if a tbl_gpu uses lazy execution

- [`is_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/is_tbl_gpu.md)
  : Test if an object is a GPU table

- [`mutate(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
  : Create, modify and delete columns

- [`left_join(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/mutating-joins.md)
  [`inner_join(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/mutating-joins.md)
  [`full_join(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/mutating-joins.md)
  [`right_join(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/mutating-joins.md)
  :

  Mutating joins for `tbl_gpu`
  ([`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  [`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  [`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  [`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html))

- [`na.omit(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/na.omit.tbl_gpu.md)
  : Drop rows with missing values

- [`` `names<-`( ``*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/names-set-.tbl_gpu.md)
  : Set column names of a GPU table

- [`names(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/names.tbl_gpu.md)
  : Get column names of a GPU table

- [`pull(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/pull.tbl_gpu.md)
  : Pull a single column from a GPU table

- [`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
  [`write_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md)
  : Read a parquet file straight onto the GPU

- [`reframe(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/reframe.tbl_gpu.md)
  : Summarise each group to any number of rows

- [`relocate(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/relocate.tbl_gpu.md)
  : Reorder columns of a GPU table

- [`rename(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/rename.tbl_gpu.md)
  : Rename columns of a GPU table

- [`rename_with(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/rename_with.tbl_gpu.md)
  : Rename columns of a GPU table with a function

- [`roll_mean()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  [`roll_sum()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  [`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  [`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  [`roll_sd()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
  [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md) :
  Rolling-window aggregates

- [`rowwise(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/rowwise.tbl_gpu.md)
  : Group a GPU table by rows

- [`same_src(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/same_src.tbl_gpu.md)
  [`auto_copy(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/same_src.tbl_gpu.md)
  : Source compatibility and upload for tbl_gpu

- [`sample_n(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/sample_n.tbl_gpu.md)
  [`sample_frac(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/sample_n.tbl_gpu.md)
  : Sample rows by number or fraction

- [`select(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
  : Select columns from a GPU table

- [`semi_join(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/semi-anti-joins.md)
  [`anti_join(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/semi-anti-joins.md)
  :

  Filtering joins for `tbl_gpu`
  ([`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html),
  [`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html))

- [`union(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/set-ops.md)
  [`union_all(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/set-ops.md)
  [`intersect(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/set-ops.md)
  [`setdiff(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/set-ops.md)
  [`symdiff(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/set-ops.md)
  [`setequal(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/set-ops.md)
  :

  Set operations on `tbl_gpu` tables

- [`show_gpu()`](https://bbtheo.github.io/cuplyr/reference/show_gpu.md)
  : Display GPU information

- [`show_query(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/show_query.tbl_gpu.md)
  : Show the plan of a lazy tbl_gpu

- [`slice(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/slice.tbl_gpu.md)
  : Select rows by position

- [`slice_head(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/slice_head.tbl_gpu.md)
  : Select the first rows

- [`slice_max(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/slice_max.tbl_gpu.md)
  : Select rows with the largest values of a variable

- [`slice_min(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/slice_min.tbl_gpu.md)
  : Select rows with the smallest values of a variable

- [`slice_sample(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/slice_sample.tbl_gpu.md)
  : Sample rows

- [`slice_tail(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/slice_tail.tbl_gpu.md)
  : Select the last rows

- [`` `[`( ``*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/sub-.tbl_gpu.md)
  :

  Subset a GPU table with tibble's `[` semantics

- [`summarise(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md)
  [`summarize(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md)
  : Summarise each group down to one row

- [`tail(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tail.tbl_gpu.md)
  : Last rows of a GPU table

- [`tally(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tally.tbl_gpu.md)
  : Count the rows in each group (no additional grouping)

- [`as.data.frame(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu-coerce.md)
  [`as_tibble(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu-coerce.md)
  [`as.list(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu-coerce.md)
  [`as.matrix(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu-coerce.md)
  : Coerce a GPU table to a base data frame, tibble, list or matrix

- [`rbind(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu-rbind.md)
  [`cbind(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu-rbind.md)
  :

  Base [`rbind()`](https://rdrr.io/r/base/cbind.html) /
  [`cbind()`](https://rdrr.io/r/base/cbind.html) for GPU tables

- [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) :
  Create a GPU table: a data frame in GPU memory

- [`tbl_sum(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/tbl_sum.tbl_gpu.md)
  : Summary header of a tbl_gpu (pillar)

- [`transmute(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/transmute.tbl_gpu.md)
  : Create a new GPU table with only the specified columns

- [`ungroup(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/ungroup.tbl_gpu.md)
  : Remove grouping from a GPU table

- [`unique(`*`<tbl_gpu>`*`)`](https://bbtheo.github.io/cuplyr/reference/unique.tbl_gpu.md)
  : Unique rows of a GPU table

- [`verify_gpu_data()`](https://bbtheo.github.io/cuplyr/reference/verify_gpu_data.md)
  : Verify that data resides on GPU

- [`verify_installation()`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)
  : Verify cuplyr installation
