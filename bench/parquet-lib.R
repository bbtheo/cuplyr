# Helpers of the parquet-to-summary pipeline benchmark (F121): the seeded data
# and its files, each engine's pipelines, the checks that compare their results,
# page-cache control, the CPU-load measurement, and the timing loop. The
# driver is bench/parquet-pipeline.R, the report bench/parquet-pipeline.qmd.
#
# Sourcing this file defines functions and nothing else: no library() call, no
# option or .libPaths() change. Packages are used as pkg::fn, so the caller
# decides which library they come from.

# ---- what is benchmarked -------------------------------------------------------

pp_engines <- function() c("cuplyr", "duckdb", "dplyr", "data.table")

# 3 pipelines x 2 filters. `pushed`: the filter is on a Date and an integer
# column, which cuplyr pushes into the parquet scan; the other filter (a double
# and a string column) it never pushes.
pp_variants <- function() {
  data.frame(
    variant = c("p1_push", "p1_nopush", "p2_push", "p2_nopush", "p3_push", "p3_nopush"),
    pipeline = rep(c("p1", "p2", "p3"), each = 2L),
    filter = rep(c("push", "nopush"), 3L),
    pushed = rep(c(TRUE, FALSE), 3L),
    stringsAsFactors = FALSE
  )
}

pp_constants <- function() {
  list(
    day_from = as.Date("2024-01-01"), qty_min = 2L,   # push filter: keeps about 19%
    price_min = 150, channel_drop = "phone",          # nopush filter: keeps about 19%
    top_n = 10L,
    seed = 121L, row_group_rows = 1048576L,
    tol = 1e-9,
    dt_form = "stepwise",                             # data.table's faster idiom, see forms.csv
    rowwise_cols = c("x17", "x18", "x19", "x20")      # the rowwise pipeline's columns: wide-table doubles, no NA
  )
}

pp_pipeline_of <- function(variant) {
  v <- pp_variants()
  if (length(variant) != 1L || !variant %in% v$variant) stop("unknown variant: ", paste(variant, collapse = ", "))
  v$pipeline[match(variant, v$variant)]
}

# The fact-table columns a variant needs (the dimension's `region` is not one).
pp_columns <- function(variant) {
  if (identical(variant, "rowwise")) return(c("category", pp_constants()$rowwise_cols))
  pipe <- pp_pipeline_of(variant)
  push <- grepl("_push$", variant)
  unique(c(
    if (push) "day" else "channel",
    "qty", "price", "discount",
    switch(pipe, p1 = c("store", "channel"), p2 = c("store", "channel"), p3 = "product")
  ))
}

# The columns of a variant's result, in the canonical order.
pp_result_columns <- function(variant) {
  switch(pp_pipeline_of(variant),
    p1 = c("store", "channel", "n", "revenue", "avg_price", "max_qty"),
    p2 = c("region", "channel", "n", "revenue", "avg_price", "max_qty"),
    p3 = c("product", "revenue", "n"))
}

# ---- data ----------------------------------------------------------------------

# One row group of the fact table. Seeded by (seed, group index), so the data
# depend on (n, seed, row_group_rows) and not on how many groups a write holds.
# The narrow table's 8 columns come first from the random stream; the wide
# table's filler is drawn after them, so the shared columns are identical.
pp_fact_group <- function(g, n, wide, seed, rg) {
  start <- (g - 1) * rg
  m <- min(rg, n - start)
  set.seed(seed * 1000L + g)
  i <- start + seq_len(m)
  df <- data.frame(
    day = as.Date("2020-01-01") + floor((i - 1) * 1827 / n),
    store = sample.int(1000L, m, replace = TRUE),
    product = sample.int(50000L, m, replace = TRUE),
    qty = sample.int(20L, m, replace = TRUE),
    price = round(stats::runif(m, 1, 200), 2),
    discount = ifelse(stats::runif(m) < 0.02, NA_real_, round(stats::runif(m, 0, 0.5), 2)),
    channel = sample(c("app", "phone", "store", "web"), m, replace = TRUE),
    category = sample(sprintf("cat_%02d", 1:20), m, replace = TRUE),
    stringsAsFactors = FALSE
  )
  if (wide) {
    for (j in 1:16) df[[sprintf("x%02d", j)]] <- sample.int(100L, m, replace = TRUE)
    for (j in 17:32) df[[sprintf("x%02d", j)]] <- sample(seq(0, 9.9, by = 0.1), m, replace = TRUE)
  }
  df
}

# Write the fact table with arrow (snappy, row groups of `row_group_rows` rows)
# in batches of `groups_per_write` whole row groups, so 1e8 rows never sit in R
# memory at once. Written to a temporary name and renamed.
pp_gen_fact <- function(path, n, wide = FALSE, seed = 121L, row_group_rows = 1048576L,
                        groups_per_write = 4L) {
  n <- as.numeric(n); rg <- as.numeric(row_group_rows)
  groups <- as.integer(ceiling(n / rg))
  old_seed <- get0(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (is.null(old_seed)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) rm(".Random.seed", envir = globalenv())
  } else assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp")
  on.exit(unlink(tmp), add = TRUE)
  sink <- arrow::FileOutputStream$create(tmp)
  writer <- NULL
  closed <- FALSE
  on.exit(if (!closed) { try(writer$Close(), silent = TRUE); try(sink$close(), silent = TRUE) }, add = TRUE)
  g <- 1L
  while (g <= groups) {
    last <- min(groups, g + groups_per_write - 1L)
    df <- do.call(rbind, lapply(g:last, pp_fact_group, n = n, wide = wide, seed = seed, rg = rg))
    tab <- arrow::arrow_table(df)
    if (is.null(writer)) {
      props <- arrow::ParquetWriterProperties$create(column_names = names(df), compression = "snappy")
      writer <- arrow::ParquetFileWriter$create(tab$schema, sink, properties = props)
    }
    writer$WriteTable(tab, chunk_size = as.integer(rg))
    g <- last + 1L
  }
  writer$Close(); sink$close(); closed <- TRUE
  file.rename(tmp, path)
  invisible(path)
}

# The 1000-row store dimension: `store` 1..1000 and `region`, 10 values.
pp_gen_dim <- function(path, seed = 121L) {
  old_seed <- get0(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (is.null(old_seed)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) rm(".Random.seed", envir = globalenv())
  } else assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  set.seed(seed * 1000L)
  dm <- data.frame(store = 1:1000, region = sprintf("region_%02d", sample(rep(1:10, 100L))),
                   stringsAsFactors = FALSE)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  arrow::write_parquet(dm, path)
  invisible(path)
}

# ---- engines -------------------------------------------------------------------

# Packages an engine needs loaded before its first call (counted as load time).
pp_engine_packages <- function(engine) {
  switch(engine,
    cuplyr = c("dplyr", "tibble"),
    duckdb = c("DBI", "duckdb"),
    dplyr = c("arrow", "dplyr", "tibble", "tidyselect"),
    data.table = c("arrow", "data.table", "tidyselect"),
    stop("unknown engine: ", engine))
}

# One context per engine and process: DuckDB's connection is opened once,
# untimed, the way a package is loaded. cuplyr, dplyr and data.table need none.
pp_context <- function(engine) {
  if (!engine %in% pp_engines()) stop("unknown engine: ", engine)
  ctx <- list(engine = engine)
  if (engine == "duckdb") {
    ctx$con <- DBI::dbConnect(duckdb::duckdb(shared_home = FALSE))
  }
  ctx
}

pp_close <- function(ctx) {
  if (!is.null(ctx$con)) try(DBI::dbDisconnect(ctx$con, shutdown = TRUE), silent = TRUE)
  invisible(NULL)
}

# The thread settings in effect, read back, named numeric.
pp_threads <- function(engine, ctx = NULL) {
  switch(engine,
    duckdb = c(threads = as.numeric(DBI::dbGetQuery(ctx$con, "SELECT current_setting('threads') AS t")$t)),
    data.table = c(threads = as.numeric(data.table::getDTthreads())),
    dplyr = c(arrow_cpu = as.numeric(arrow::cpu_count()), arrow_io = as.numeric(arrow::io_thread_count())),
    cuplyr = c(gpu_devices = as.numeric(tryCatch(cuplyr::gpu_details()$device_count, error = function(e) NA_real_))),
    stop("unknown engine: ", engine))
}

# Settings that bear on what a timing means, read back from the running engine
# (named character): DuckDB's local-file caching, and whether libcudf's debug
# stream pool is switched on in this process (cuplyr's .onLoad sets it).
pp_engine_settings <- function(engine, ctx = NULL) {
  get <- function(nm) as.character(DBI::dbGetQuery(ctx$con, sprintf("SELECT current_setting('%s') AS v", nm))$v)
  switch(engine,
    duckdb = c(cache_local_files = get("cache_local_files"),
               enable_external_file_cache = get("enable_external_file_cache")),
    cuplyr = {
      v <- Sys.getenv("LIBCUDF_USE_DEBUG_STREAM_POOL", NA_character_)
      c(debug_stream_pool = if (is.na(v)) "unset" else paste0("set (", v, ")"))
    },
    character())
}

# Only the needed columns are read, by arrow: dplyr and data.table both use it.
pp_read_arrow <- function(path, cols) {
  arrow::read_parquet(path, col_select = tidyselect::all_of(cols))
}

pp_read_nano <- function(path, cols) {
  nanoparquet::read_parquet(path, col_select = cols)
}

# cuplyr: a lazy read_parquet_gpu() (projection and the Date/integer filter go
# into the scan) and the dplyr verbs, as a lazy plan; pp_cuplyr() runs it with
# collect(). Not every verb stays in the plan: for p1 and p2, summarise() runs
# the plan when it is called (F07: max() of an INT32 column cannot be planned,
# because the result type depends on the data), so by the time collect() runs
# only the transfer to R is left; p3's plan is run by collect().
pp_cuplyr_plan <- function(variant, fact, dim) {
  k <- pp_constants()
  day_from <- k$day_from; qty_min <- k$qty_min
  price_min <- k$price_min; channel_drop <- k$channel_drop; topn <- k$top_n
  x <- cuplyr::read_parquet_gpu(fact)
  x <- if (grepl("_push$", variant)) {
    dplyr::filter(x, day >= day_from, qty >= qty_min)
  } else {
    dplyr::filter(x, price >= price_min, channel != channel_drop)
  }
  x <- dplyr::mutate(x, revenue = price * qty * (1 - discount))
  switch(pp_pipeline_of(variant),
    p1 = x |>
      dplyr::group_by(store, channel) |>
      dplyr::summarise(n = dplyr::n(), revenue = sum(revenue, na.rm = TRUE),
                       avg_price = mean(price), max_qty = max(qty), .groups = "drop"),
    p2 = x |>
      dplyr::inner_join(cuplyr::read_parquet_gpu(dim), by = "store") |>
      dplyr::group_by(region, channel) |>
      dplyr::summarise(n = dplyr::n(), revenue = sum(revenue, na.rm = TRUE),
                       avg_price = mean(price), max_qty = max(qty), .groups = "drop"),
    p3 = x |>
      dplyr::group_by(product) |>
      dplyr::summarise(revenue = sum(revenue, na.rm = TRUE), n = dplyr::n()) |>
      dplyr::slice_max(revenue, n = topn)
  )
}

pp_cuplyr <- function(variant, fact, dim) dplyr::collect(pp_cuplyr_plan(variant, fact, dim))

# cuplyr, probe helper (the `probe` step and the check step's scan under the
# join): the fact side of the join pipeline p2 alone, as a lazy plan: the same
# scan, filter and mutate as pp_cuplyr(), nothing after but a projection to the
# columns that p2 uses after it (p1 uses the same ones; p3 also needs `product`).
pp_cuplyr_fact_side <- function(variant, fact) {
  k <- pp_constants()
  day_from <- k$day_from; qty_min <- k$qty_min
  price_min <- k$price_min; channel_drop <- k$channel_drop
  x <- cuplyr::read_parquet_gpu(fact)
  x <- if (grepl("_push$", variant)) {
    dplyr::filter(x, day >= day_from, qty >= qty_min)
  } else {
    dplyr::filter(x, price >= price_min, channel != channel_drop)
  }
  x <- dplyr::mutate(x, revenue = price * qty * (1 - discount))
  dplyr::select(x, store, channel, price, qty, revenue)   # what the verbs after it use
}

# cuplyr, probe helper: the join pipeline (p2) with the dimension already computed
# on the device (`dim_tbl`, a tbl_gpu), so that the fact file is the plan's only
# parquet read and `parquet_last_read()` afterwards reports the fact scan as it
# runs under the join (in pp_cuplyr() the last read is the dimension file).
pp_cuplyr_under_join <- function(variant, fact, dim_tbl) {
  pp_cuplyr_fact_side(variant, fact) |>
    dplyr::inner_join(dim_tbl, by = "store") |>
    dplyr::group_by(region, channel) |>
    dplyr::summarise(n = dplyr::n(), revenue = sum(revenue, na.rm = TRUE),
                     avg_price = mean(price), max_qty = max(qty), .groups = "drop") |>
    dplyr::collect()
}

# cuplyr, probe helper: the scan alone, as a lazy plan. The columns the variant
# needs, and for a push variant the pushed filter (which the scan applies); for a
# nopush variant nothing is filtered, because that filter is not pushed.
pp_cuplyr_scan <- function(variant, fact) {
  k <- pp_constants()
  day_from <- k$day_from; qty_min <- k$qty_min
  x <- cuplyr::read_parquet_gpu(fact)
  if (grepl("_push$", variant)) x <- dplyr::filter(x, day >= day_from, qty >= qty_min)
  dplyr::select(x, dplyr::all_of(pp_columns(variant)))
}

# dplyr: arrow's reader with the needed columns, then the same verbs on a tibble.
pp_dplyr <- function(variant, fact, dim) {
  k <- pp_constants()
  day_from <- k$day_from; qty_min <- k$qty_min
  price_min <- k$price_min; channel_drop <- k$channel_drop; topn <- k$top_n
  x <- pp_read_arrow(fact, pp_columns(variant))
  x <- if (grepl("_push$", variant)) {
    dplyr::filter(x, day >= day_from, qty >= qty_min)
  } else {
    dplyr::filter(x, price >= price_min, channel != channel_drop)
  }
  x <- dplyr::mutate(x, revenue = price * qty * (1 - discount))
  switch(pp_pipeline_of(variant),
    p1 = x |>
      dplyr::group_by(store, channel) |>
      dplyr::summarise(n = dplyr::n(), revenue = sum(revenue, na.rm = TRUE),
                       avg_price = mean(price), max_qty = max(qty), .groups = "drop"),
    p2 = x |>
      dplyr::inner_join(pp_read_arrow(dim, c("store", "region")), by = "store") |>
      dplyr::group_by(region, channel) |>
      dplyr::summarise(n = dplyr::n(), revenue = sum(revenue, na.rm = TRUE),
                       avg_price = mean(price), max_qty = max(qty), .groups = "drop"),
    p3 = x |>
      dplyr::group_by(product) |>
      dplyr::summarise(revenue = sum(revenue, na.rm = TRUE), n = dplyr::n()) |>
      dplyr::slice_max(revenue, n = topn)
  )
}

# data.table: the same reader, setDT(), data.table syntax. Two idioms were
# timed (the `forms` step of the driver): form "single" puts the filter in i and
# the revenue arithmetic inside j, so the filtered rows are never copied;
# "stepwise" copies the filtered rows, adds `revenue` with `:=`, then
# aggregates. pp_constants()$dt_form is the one that was faster in most of the
# timed cells (forms.csv). In p2 the filtered facts are joined to the dimension
# first (a keyed join, `stores[facts, on =, nomatch = NULL]`, an inner join); the
# join result is a new table, so `revenue` is added to it directly (stepwise) or
# computed inside j (single).
pp_datatable <- function(variant, fact, dim, form = pp_constants()$dt_form) {
  k <- pp_constants()
  day_from <- k$day_from; qty_min <- k$qty_min
  price_min <- k$price_min; channel_drop <- k$channel_drop; topn <- k$top_n
  DT <- data.table::setDT(pp_read_arrow(fact, pp_columns(variant)))
  keep <- if (grepl("_push$", variant)) {
    DT$day >= day_from & DT$qty >= qty_min
  } else {
    DT$price >= price_min & DT$channel != channel_drop
  }
  pipe <- pp_pipeline_of(variant)
  by <- switch(pipe, p1 = c("store", "channel"), p2 = c("region", "channel"), p3 = "product")
  joined <- NULL
  if (pipe == "p2") {
    stores <- data.table::setDT(pp_read_arrow(dim, c("store", "region")))
    joined <- stores[DT[keep], on = "store", nomatch = NULL]
  }
  if (form == "single") {
    res <- if (pipe == "p2") {
      joined[, .(n = .N, revenue = sum(price * qty * (1 - discount), na.rm = TRUE),
                 avg_price = mean(price), max_qty = max(qty)), by = by]
    } else if (pipe == "p3") {
      DT[keep, .(revenue = sum(price * qty * (1 - discount), na.rm = TRUE), n = .N), by = by]
    } else {
      DT[keep, .(n = .N, revenue = sum(price * qty * (1 - discount), na.rm = TRUE),
                 avg_price = mean(price), max_qty = max(qty)), by = by]
    }
  } else {
    x <- if (pipe == "p2") joined else DT[keep]
    x[, revenue := price * qty * (1 - discount)]
    res <- if (pipe == "p3") {
      x[, .(revenue = sum(revenue, na.rm = TRUE), n = .N), by = by]
    } else {
      x[, .(n = .N, revenue = sum(revenue, na.rm = TRUE),
            avg_price = mean(price), max_qty = max(qty)), by = by]
    }
  }
  if (pipe == "p3") res <- res[data.table::frank(-revenue, ties.method = "min") <= topn]
  res
}

# The WHERE clause of a variant's filter.
pp_sql_where <- function(variant) {
  k <- pp_constants()
  if (grepl("_push$", variant)) {
    sprintf("day >= DATE '%s' AND qty >= %d", format(k$day_from), k$qty_min)
  } else {
    sprintf("price >= %s AND channel <> '%s'", format(k$price_min), k$channel_drop)
  }
}

# DuckDB: its own reader, in SQL, with explicit columns. coalesce(sum(), 0)
# is R's sum(na.rm = TRUE) for a group whose values are all missing; the top-k
# keeps ties, like slice_max().
pp_sql <- function(variant, fact, dim) {
  k <- pp_constants()
  q <- function(p) paste0("'", gsub("'", "''", p, fixed = TRUE), "'")
  where <- pp_sql_where(variant)
  rev <- "coalesce(sum(price * qty * (1 - discount)), 0)"
  aggs <- sprintf("count(*) AS n, %s AS revenue, avg(price) AS avg_price, max(qty) AS max_qty", rev)
  switch(pp_pipeline_of(variant),
    p1 = sprintf("SELECT store, channel, %s\nFROM read_parquet(%s)\nWHERE %s\nGROUP BY store, channel",
                 aggs, q(fact), where),
    p2 = sprintf(paste0("SELECT s.region AS region, f.channel AS channel, %s\n",
                        "FROM read_parquet(%s) AS f JOIN read_parquet(%s) AS s ON f.store = s.store\n",
                        "WHERE %s\nGROUP BY s.region, f.channel"),
                 aggs, q(fact), q(dim), where),
    p3 = sprintf(paste0("SELECT product, %s AS revenue, count(*) AS n\nFROM read_parquet(%s)\nWHERE %s\n",
                        "GROUP BY product\nQUALIFY rank() OVER (ORDER BY revenue DESC) <= %d"),
                 rev, q(fact), where, k$top_n))
}

# Run one engine's pipeline from the file path to an R data frame.
pp_run <- function(engine, variant, fact, dim, ctx = NULL) {
  switch(engine,
    cuplyr = {
      old <- options(cuplyr.fallback = "error")   # a CPU fallback would not be a GPU number
      on.exit(options(old), add = TRUE)
      pp_cuplyr(variant, fact, dim)
    },
    duckdb = DBI::dbGetQuery(ctx$con, pp_sql(variant, fact, dim)),
    dplyr = pp_dplyr(variant, fact, dim),
    data.table = pp_datatable(variant, fact, dim),
    stop("unknown engine: ", engine))
}

# ---- the rowwise pipeline (Theo's addition) ----------------------------------------

# A per-row maximum and sum over four numeric columns of the wide table, then a
# summary by category, so the result is small. Five forms, each engine's natural
# way to say it; the first two evaluate per row, the other three are vectorised.
pp_rowwise_forms <- function() {
  c("cuplyr_rowwise", "dplyr_rowwise", "dplyr_vectorised", "datatable_vectorised", "duckdb")
}

# The engine (pp_engines()) whose context a form needs.
pp_rowwise_engine <- function(form) {
  switch(form,
    cuplyr_rowwise = "cuplyr",
    dplyr_rowwise = , dplyr_vectorised = "dplyr",
    datatable_vectorised = "data.table",
    duckdb = "duckdb",
    stop("unknown rowwise form: ", paste(form, collapse = ", ")))
}

# DuckDB's form: greatest() and arithmetic over read_parquet(), explicit columns.
pp_rowwise_sql <- function(fact) {
  q <- function(p) paste0("'", gsub("'", "''", p, fixed = TRUE), "'")
  sprintf(paste0("SELECT category, count(*) AS n, avg(m) AS m_mean, sum(s) AS s_sum\n",
                 "FROM (SELECT category, greatest(x17, x18, x19, x20) AS m, x17 + x18 + x19 + x20 AS s\n",
                 "      FROM read_parquet(%s))\nGROUP BY category"), q(fact))
}

# A form as function(fact, ctx): the file path to an R data frame of the summary
# (category, n, m_mean, s_sum). `ctx` is the engine's pp_context() (only DuckDB
# uses it). The cuplyr form is run without cuplyr.fallback = "error": whether it
# stays on the GPU is the thing to report (pp_fallback_probe()), not to forbid.
pp_rowwise_fn <- function(form) {
  switch(form,
    cuplyr_rowwise = function(fact, ctx) {
      cols <- pp_constants()$rowwise_cols
      cuplyr::read_parquet_gpu(fact) |>
        dplyr::rowwise() |>
        dplyr::mutate(m = max(dplyr::c_across(dplyr::all_of(cols))),
                      s = sum(dplyr::c_across(dplyr::all_of(cols)))) |>
        dplyr::ungroup() |>
        dplyr::group_by(category) |>
        dplyr::summarise(n = dplyr::n(), m_mean = mean(m), s_sum = sum(s)) |>
        dplyr::collect()
    },
    dplyr_rowwise = function(fact, ctx) {
      cols <- pp_constants()$rowwise_cols
      pp_read_arrow(fact, pp_columns("rowwise")) |>
        dplyr::rowwise() |>
        dplyr::mutate(m = max(dplyr::c_across(dplyr::all_of(cols))),
                      s = sum(dplyr::c_across(dplyr::all_of(cols)))) |>
        dplyr::ungroup() |>
        dplyr::group_by(category) |>
        dplyr::summarise(n = dplyr::n(), m_mean = mean(m), s_sum = sum(s))
    },
    dplyr_vectorised = function(fact, ctx) {
      pp_read_arrow(fact, pp_columns("rowwise")) |>
        dplyr::mutate(m = pmax(x17, x18, x19, x20), s = x17 + x18 + x19 + x20) |>
        dplyr::group_by(category) |>
        dplyr::summarise(n = dplyr::n(), m_mean = mean(m), s_sum = sum(s))
    },
    datatable_vectorised = function(fact, ctx) {
      cols <- pp_constants()$rowwise_cols
      DT <- data.table::setDT(pp_read_arrow(fact, pp_columns("rowwise")))
      DT[, `:=`(m = pmax(x17, x18, x19, x20), s = rowSums(.SD)), .SDcols = cols]
      DT[, .(n = .N, m_mean = mean(m), s_sum = sum(s)), by = category]
    },
    duckdb = function(fact, ctx) DBI::dbGetQuery(ctx$con, pp_rowwise_sql(fact)),
    stop("unknown rowwise form: ", paste(form, collapse = ", ")))
}

pp_run_rowwise <- function(form, fact, ctx = NULL) pp_rowwise_fn(form)(fact, ctx)

# The canonical form of a rowwise result: a data.frame (category, n, m_mean,
# s_sum), category character, the rest double, sorted by category (radix).
pp_normalize_rowwise <- function(df) {
  want <- c("category", "n", "m_mean", "s_sum")
  df <- as.data.frame(df)
  miss <- setdiff(want, names(df))
  if (length(miss)) stop("result lacks column(s): ", paste(miss, collapse = ", "))
  out <- df[want]
  out$category <- as.character(out$category)
  for (nm in want[-1L]) out[[nm]] <- as.numeric(out[[nm]])
  out <- out[order(out$category, method = "radix"), , drop = FALSE]
  rownames(out) <- NULL
  out
}

# An independent reference in base R (no dplyr, data.table or SQL): the result
# of the rowwise pipeline for a fact file, in the canonical form.
pp_rowwise_ref <- function(fact) {
  f <- as.data.frame(pp_read_arrow(fact, pp_columns("rowwise")))
  cols <- pp_constants()$rowwise_cols
  m <- do.call(pmax, unname(as.list(f[cols])))
  s <- Reduce(`+`, f[cols])
  ix <- split(seq_len(nrow(f)), f$category)
  pp_normalize_rowwise(data.frame(
    category = names(ix), n = as.double(lengths(ix)),
    m_mean = vapply(ix, function(i) mean(m[i]), 0), s_sum = vapply(ix, function(i) sum(s[i]), 0),
    stringsAsFactors = FALSE))
}

# Run `fn()` once, untimed, and say whether cuplyr kept it on the GPU, from F88's
# `cuplyr_cpu_fallback` condition (signalled before any CPU work, whatever
# cuplyr.fallback says). Returns list(native, reason): native is TRUE when the
# condition was not signalled, FALSE when it was (reason: its reasons, joined),
# and NA when fn() failed with some other error (reason: the message). Leaves
# options(cuplyr.fallback) as it found it: under "error" the fallback's abort
# is caught here, after the condition has been seen.
pp_fallback_probe <- function(fn) {
  reasons <- character()
  err <- NULL
  withCallingHandlers(
    tryCatch(fn(), error = function(cnd) err <<- cnd),
    cuplyr_cpu_fallback = function(cnd) reasons <<- c(reasons, pp_fallback_reason(cnd)))
  if (length(reasons)) return(list(native = FALSE, reason = paste(unique(reasons), collapse = " | "), error = err))
  if (!is.null(err)) return(list(native = NA, reason = paste("error:", gsub("\\s+", " ", conditionMessage(err))), error = err))
  list(native = TRUE, reason = "", error = NULL)
}

# A cuplyr_cpu_fallback condition as text: "verb(): reason".
pp_fallback_reason <- function(cnd) {
  r <- if (!is.null(cnd$reason)) cnd$reason else conditionMessage(cnd)
  paste0(if (!is.null(cnd$verb)) paste0(cnd$verb, "(): "), r)
}

# Watch every call of the zero-argument function `fn` for F88's
# cuplyr_cpu_fallback condition, without changing what the call does (a calling
# handler only records). Returns list(fn, calls): `fn` is the watched function,
# to be timed in its place; `calls()` gives one string per call made so far, in
# call order, "" for a call that signalled nothing and the reasons otherwise. A
# call that fails or is aborted is recorded too.
pp_fallback_watch <- function(fn) {
  force(fn)   # the caller may put the watched function where `fn` was
  log <- character()
  watched <- function() {
    seen <- character()
    on.exit(log <<- c(log, paste(unique(seen), collapse = " | ")), add = TRUE)
    withCallingHandlers(fn(), cuplyr_cpu_fallback = function(cnd) seen <<- c(seen, pp_fallback_reason(cnd)))
  }
  list(fn = watched, calls = function() log)
}

# ---- comparing results ---------------------------------------------------------

pp_key_columns <- c("store", "product", "channel", "region", "category")

# The canonical form every engine's result is compared in: a plain data.frame,
# the columns in pp_result_columns() order, keys integer (store, product) or
# character (channel, region), measures double, sorted by the keys (radix, so
# C-locale for strings) or, for the top-k, by revenue descending then product.
pp_normalize <- function(df, variant) {
  if (identical(variant, "rowwise")) return(pp_normalize_rowwise(df))
  want <- pp_result_columns(variant)
  df <- as.data.frame(df)
  miss <- setdiff(want, names(df))
  if (length(miss)) stop("result lacks column(s): ", paste(miss, collapse = ", "))
  out <- df[want]
  for (nm in want) {
    out[[nm]] <- if (nm %in% c("store", "product")) as.integer(out[[nm]])
    else if (nm %in% c("channel", "region")) as.character(out[[nm]])
    else as.numeric(out[[nm]])
  }
  o <- if (pp_pipeline_of(variant) == "p3") {
    order(-out$revenue, out$product, method = "radix")
  } else {
    do.call(order, c(unname(as.list(out[want[1:2]])), list(method = "radix")))
  }
  out <- out[o, , drop = FALSE]
  rownames(out) <- NULL
  out
}

# Keys exact, measures within a relative tolerance. Both sides normalized.
pp_compare <- function(x, y, tol = 1e-9) {
  fail <- function(...) list(ok = FALSE, detail = paste0(...))
  if (!identical(names(x), names(y))) {
    return(fail("columns differ: ", paste(names(x), collapse = ","), " vs ", paste(names(y), collapse = ",")))
  }
  if (nrow(x) != nrow(y)) return(fail(nrow(x), " rows vs ", nrow(y), " rows"))
  for (nm in names(x)) {
    a <- x[[nm]]; b <- y[[nm]]
    if (!identical(is.na(a), is.na(b))) {
      i <- which(is.na(a) != is.na(b))[1L]
      return(fail(nm, ": missing values differ at row ", i))
    }
    if (nm %in% pp_key_columns) {
      bad <- which(a != b)
      if (length(bad)) return(fail(nm, ": keys differ at row ", bad[1L], " (", a[bad[1L]], " vs ", b[bad[1L]], ")"))
    } else {
      d <- abs(a - b)
      bad <- which(!is.na(d) & d > tol * pmax(abs(a), abs(b)))
      if (length(bad)) {
        i <- bad[1L]
        return(fail(nm, ": row ", i, " differs by ", signif(d[i] / max(abs(a[i]), abs(b[i])), 3), " (relative): ",
                    format(a[i], digits = 15), " vs ", format(b[i], digits = 15)))
      }
    }
  }
  list(ok = TRUE, detail = "")
}

# ---- the page cache ------------------------------------------------------------

# Drop files from the page cache: fsync (dirty pages cannot be dropped), then
# posix_fadvise(DONTNEED), as F85-a2 did. python3 is used because R has no
# posix_fadvise.
pp_evict <- function(paths) {
  paths <- paths[file.exists(paths)]
  if (!length(paths)) return(invisible(NULL))
  py <- paste0("import os, sys\nfor p in sys.argv[1:]:\n    f = os.open(p, os.O_RDONLY)\n",
               "    os.fsync(f)\n    os.posix_fadvise(f, 0, 0, os.POSIX_FADV_DONTNEED)\n    os.close(f)\n")
  st <- system2("python3", c("-I", "-c", shQuote(py), shQuote(paths)))
  if (!identical(as.integer(st), 0L)) stop("could not evict from the page cache (python3 exit status ", st, ")")
  invisible(NULL)
}

# Read the files through once so that they sit in the page cache.
pp_warm <- function(paths) {
  for (p in paths[file.exists(paths)]) system2("cat", shQuote(p), stdout = FALSE)
  invisible(NULL)
}

# The share of the files' bytes in the page cache (fincore), 0 to 1.
pp_resident <- function(paths) {
  paths <- paths[file.exists(paths)]
  if (!length(paths)) return(NA_real_)
  res <- vapply(paths, function(p) {
    out <- suppressWarnings(system2("fincore", c("-b", "-n", "-o", "RES", shQuote(p)), stdout = TRUE))
    as.numeric(trimws(out[[1L]]))
  }, 0)
  min(1, sum(res) / sum(file.size(paths)))
}

# ---- machine load --------------------------------------------------------------

# A snapshot: /proc/stat's jiffies (all CPUs: total, idle + iowait, user + nice,
# system + irq + softirq) and this process's own user and system CPU seconds
# (all its threads).
pp_cpu <- function() {
  v <- as.numeric(strsplit(trimws(sub("^cpu", "", readLines("/proc/stat", n = 1L))), " +")[[1]])
  pt <- proc.time()
  c(total = sum(v[1:8]), idle = v[4] + v[5], user = v[1] + v[2], sys = v[3] + v[6] + v[7],
    own_user = pt[["user.self"]], own_sys = pt[["sys.self"]])
}

# Between two snapshots, as shares of the whole machine (all CPUs), which add up
# to `total` (clock-tick rounding aside):
#   own      this process (user + system time);
#   foreign  other processes' user time (user + nice, minus this process's user
#            time): the load a second job puts on the CPUs, and what F110's 15%
#            rule is about;
#   kernel   system + irq + softirq time minus this process's system time. A fast
#            cold read makes kernel threads (dm-crypt, the page cache) busy on the
#            reader's behalf, so this is not another job's load: recorded, not
#            part of the rule.
# /proc/stat counts in USER_HZ = 100 per second.
pp_busy <- function(c0, c1) {
  clamp <- function(x) min(1, max(0, x))
  dt <- c1[["total"]] - c0[["total"]]
  if (!is.finite(dt) || dt <= 0) return(c(total = 0, own = 0, kernel = 0, foreign = 0))
  own_user <- (c1[["own_user"]] - c0[["own_user"]]) * 100
  own_sys <- (c1[["own_sys"]] - c0[["own_sys"]]) * 100
  c(total = clamp(1 - (c1[["idle"]] - c0[["idle"]]) / dt),
    own = clamp((own_user + own_sys) / dt),
    kernel = clamp((c1[["sys"]] - c0[["sys"]] - own_sys) / dt),
    foreign = clamp((c1[["user"]] - c0[["user"]] - own_user) / dt))
}

# Wait until other processes' load, sampled over `window` seconds with this
# process idle, is at most `busy_max`; gives up after `max_wait` seconds and
# returns the last sample either way (the call's own busy check decides).
pp_wait_quiet <- function(busy_max = 0.15, window = 0.25, max_wait = 600) {
  if (busy_max >= 1) return(invisible(0))
  t0 <- proc.time()[["elapsed"]]
  repeat {
    c0 <- pp_cpu(); Sys.sleep(window); b <- pp_busy(c0, pp_cpu())[["foreign"]]
    if (b <= busy_max || proc.time()[["elapsed"]] - t0 > max_wait) return(invisible(b))
    Sys.sleep(1)
  }
}

# ---- timing --------------------------------------------------------------------

pp_error_status <- function(cnd) {
  if (inherits(cnd, "cuplyr_error_gpu_oom")) return("oom")
  if (grepl("cannot allocate|out of memory|bad_alloc|std::bad_alloc", conditionMessage(cnd), ignore.case = TRUE)) return("oom")
  "error"
}

# Time the zero-argument functions `fns` (named by engine) in one cell.
#   rep 0     the cell's first call for every engine: recorded, not for medians;
#   rep 1..   `reps` quiet reps per engine. The engines take turns in rotating
#             order. A call whose foreign load (see pp_busy()) exceeds `busy_max`
#             is kept in the output flagged quiet = FALSE and run again (at most
#             `max_attempts` calls per engine), so the rows hold every attempt.
# One rule for every engine and rep: the wait before each call and the verdict
# after it both use `busy_max`.
# Around every call, untimed: gc(), `before(engine)` (cache control; returns the
# files' residency), `wait(busy_max)` (returns the foreign load of its last
# sample). After the call, untimed: `check(engine, result)` (TRUE = right). An
# out-of-memory or other error, or a wrong result, ends that engine's turns with
# status "oom" / "error" / "mismatch" and secs NA (error) or the time (mismatch).
# Per row: busy = foreign share during the call; own, kernel, total as in
# pp_busy(); wait_load = the last sample of the wait before the call (it can
# exceed busy_max if the wait gave up); wait_secs = how long the wait took.
# `cap` (seconds, default none): a call that runs longer is aborted (R's elapsed
# time limit, checked wherever R code runs) and recorded with status "capped",
# ok FALSE and secs = the elapsed time when the abort reached this function (so
# at least `cap`); that engine gets no more calls in the cell. A capped row is
# never quiet, so it is never part of a median. With the default Inf no time
# limit is ever set and nothing else changes.
pp_time_cell <- function(fns, reps = 5L, before = NULL, check = NULL, busy_max = 0.15,
                         max_attempts = 3L * reps, busy_fn = pp_busy, wait = pp_wait_quiet,
                         cap = Inf) {
  engines <- names(fns)
  k <- length(engines)
  rows <- list()
  alive <- rep(TRUE, k)
  quiet_n <- integer(k)
  attempts <- integer(k)
  part <- function(b, nm) if (nm %in% names(b)) unname(b[[nm]]) else NA_real_

  call_once <- function(i, rep) {
    e <- engines[[i]]
    # gc() first: the previous result may still map the file (arrow reads it
    # with mmap), and mapped pages cannot be dropped from the page cache
    gc()
    resident <- if (is.null(before)) NA_real_ else as.numeric(before(e))
    wait_load <- NA_real_; wait_secs <- 0
    if (!is.null(wait)) {
      w0 <- proc.time()[["elapsed"]]
      ws <- wait(busy_max)
      wait_secs <- proc.time()[["elapsed"]] - w0
      if (is.numeric(ws) && length(ws) == 1L) wait_load <- as.numeric(ws)
    }
    c0 <- pp_cpu()
    started <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3")
    capped <- is.finite(cap)
    if (capped) setTimeLimit(elapsed = cap, transient = FALSE)
    out <- tryCatch({
      t0 <- as.numeric(Sys.time())
      val <- fns[[i]]()
      secs <- as.numeric(Sys.time()) - t0
      c1 <- pp_cpu()
      list(val = val, secs = secs, c1 = c1)
    }, error = function(cnd) list(err = cnd), finally = if (capped) setTimeLimit(elapsed = Inf, transient = FALSE))
    status <- "ok"; detail <- ""; secs <- NA_real_
    b <- c(total = NA_real_, own = NA_real_, kernel = NA_real_, foreign = NA_real_)
    if (!is.null(out$err)) {
      hit_cap <- capped && grepl("reached elapsed time limit", conditionMessage(out$err), fixed = TRUE)
      if (hit_cap) secs <- as.numeric(Sys.time()) - t0   # the elapsed time at the abort
      status <- if (hit_cap) "capped" else pp_error_status(out$err)
      detail <- if (hit_cap) paste0("capped: the call ran over ", format(cap), " s") else gsub("\\s+", " ", conditionMessage(out$err))
      gc()
      if (isNamespaceLoaded("cuplyr")) try(cuplyr::gpu_gc(), silent = TRUE)
    } else {
      secs <- out$secs
      b <- busy_fn(c0, out$c1)
      if (!is.null(check)) {
        r <- tryCatch(check(e, out$val), error = function(cnd) paste("check failed:", conditionMessage(cnd)))
        if (!isTRUE(r)) {
          status <- "mismatch"
          detail <- if (is.character(r)) r else ""
        }
      }
    }
    quiet <- status == "ok" && isTRUE(part(b, "foreign") <= busy_max)
    data.frame(engine = e, rep = as.integer(rep), secs = secs, busy = part(b, "foreign"),
               own = part(b, "own"), kernel = part(b, "kernel"), total = part(b, "total"),
               wait_load = wait_load, wait_secs = wait_secs,
               resident = resident, quiet = quiet, ok = status == "ok",
               status = status, detail = detail, started = started, stringsAsFactors = FALSE)
  }

  for (i in seq_len(k)) {
    r <- call_once(i, 0L)
    rows[[length(rows) + 1L]] <- r
    if (r$status != "ok") alive[[i]] <- FALSE
  }
  round <- 1L
  repeat {
    todo <- alive & quiet_n < reps & attempts < max_attempts
    if (!any(todo)) break
    for (i in ((seq_len(k) - 1L + round) %% k) + 1L) {
      if (!todo[[i]]) next
      r <- call_once(i, quiet_n[[i]] + 1L)
      attempts[[i]] <- attempts[[i]] + 1L
      rows[[length(rows) + 1L]] <- r
      if (r$status != "ok") alive[[i]] <- FALSE else if (r$quiet) quiet_n[[i]] <- quiet_n[[i]] + 1L
    }
    round <- round + 1L
  }
  do.call(rbind, rows)
}

# ---- the GPU's idle cost ---------------------------------------------------------

# The GPU's performance state and clocks now, from nvidia-smi: list(pstate, e.g.
# "P8"; sm_mhz, mem_mhz). NA where nvidia-smi is missing or fails.
pp_gpu_state <- function() {
  na <- list(pstate = NA_character_, sm_mhz = NA_real_, mem_mhz = NA_real_)
  out <- tryCatch(suppressWarnings(system2("nvidia-smi", c("--query-gpu=pstate,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"),
                                           stdout = TRUE, stderr = FALSE)),
                  error = function(e) character())
  if (!length(out) || !is.null(attr(out, "status"))) return(na)
  v <- trimws(strsplit(out[[1L]], ",")[[1L]])
  if (length(v) < 3L) return(na)
  list(pstate = v[[1L]], sm_mhz = suppressWarnings(as.numeric(v[[2L]])), mem_mhz = suppressWarnings(as.numeric(v[[3L]])))
}

# Time each zero-argument function in `fns` (named) after the machine has been
# left idle for each of `idles` seconds: for every rep, every idle time and every
# function in turn, untimed: gc(), `before(name)` (cache control), one call of the
# same function (the primer: whatever ran before, the GPU is busy until it ends),
# sleep(idle), `wait(busy_max)` (the quiet-machine wait), the GPU state; then the
# timed call; then the GPU state again. So the GPU has been idle for the pause,
# not for the pause plus other functions' pauses. Rows: name, idle (asked), gap_s
# (seconds from the end of the primer to the start of the timed call: the pause,
# the wait and the state query), rep, secs, busy (foreign load) and quiet (busy
# <= busy_max), the GPU state before and after (pstate, memory clock), ok/status,
# primed (TRUE). Nothing is re-run: a probe.
pp_idle_probe <- function(fns, idles = c(0, 10, 20, 40), reps = 4L, before = NULL, busy_max = 0.15,
                          wait = pp_wait_quiet, sleep = Sys.sleep, state = pp_gpu_state, busy_fn = pp_busy) {
  rows <- list()
  for (rep in seq_len(reps)) for (idle in idles) for (nm in names(fns)) {
    gc()
    resident <- if (is.null(before)) NA_real_ else as.numeric(before(nm))
    tryCatch(fns[[nm]](), error = function(cnd) NULL)   # the primer, untimed
    last_end <- as.numeric(Sys.time())
    sleep(idle)
    if (!is.null(wait)) wait(busy_max)
    s0 <- state()
    c0 <- pp_cpu()
    started <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3")
    t0 <- as.numeric(Sys.time())
    r <- tryCatch({ fns[[nm]](); NULL }, error = function(cnd) cnd)
    t1 <- as.numeric(Sys.time())
    c1 <- pp_cpu()
    s1 <- state()
    b <- busy_fn(c0, c1)
    ok <- is.null(r)
    rows[[length(rows) + 1L]] <- data.frame(
      name = nm, idle = idle, gap_s = t0 - last_end, rep = as.integer(rep), secs = if (ok) t1 - t0 else NA_real_,
      busy = unname(b[["foreign"]]), quiet = ok && isTRUE(b[["foreign"]] <= busy_max), resident = resident,
      pstate_before = s0$pstate, mem_mhz_before = s0$mem_mhz, pstate_after = s1$pstate, mem_mhz_after = s1$mem_mhz,
      ok = ok, status = if (ok) "ok" else pp_error_status(r), detail = if (ok) "" else gsub("\\s+", " ", conditionMessage(r)),
      started = started, primed = TRUE, stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}

# ---- the first call, in a fresh process ----------------------------------------

# Run `calls` consecutive calls of one engine's pipeline in a new R process
# (callr), timing each separately, after the engine's packages are loaded
# (`load_secs`; for cuplyr that is pkgload::load_all() of `root`, which is not
# what a user with an installed package pays) and DuckDB's connection is open.
pp_first_call <- function(engine, variant, fact, dim, calls = 8L, root, lib = NULL) {
  lib_file <- file.path(root, "bench", "parquet-lib.R")
  libpath <- unique(c(lib[!is.null(lib) & nzchar(lib)], .libPaths()))
  child <- function(engine, variant, fact, dim, calls, root, lib_file) {
    sys.source(lib_file, envir = environment())
    t0 <- as.numeric(Sys.time())
    if (engine == "cuplyr") pkgload::load_all(root, quiet = TRUE, debug = FALSE, compile = FALSE)
    for (p in pp_engine_packages(engine)) loadNamespace(p)
    load_secs <- as.numeric(Sys.time()) - t0
    ctx <- pp_context(engine)
    secs <- busy <- numeric(calls)
    for (j in seq_len(calls)) {
      c0 <- pp_cpu()
      t1 <- as.numeric(Sys.time())
      res <- pp_run(engine, variant, fact, dim, ctx)
      secs[[j]] <- as.numeric(Sys.time()) - t1
      busy[[j]] <- pp_busy(c0, pp_cpu())[["foreign"]]
      rm(res)
    }
    pp_close(ctx)
    data.frame(engine = engine, variant = variant, call = seq_len(calls), secs = secs,
               load_secs = load_secs, busy = busy, pid = Sys.getpid(), stringsAsFactors = FALSE)
  }
  callr::r(child, args = list(engine, variant, fact, dim, as.integer(calls), root, lib_file),
           libpath = libpath, show = FALSE)
}
