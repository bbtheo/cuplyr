# Driver of the parquet-to-summary pipeline benchmark (F121): data, correctness,
# timings, first calls, probes, and the report. Run from the checkout through
# tools/r (every R process holds the GPU lock):
#
#   tools/r 'PP <- list(steps = "gen"); source("bench/parquet-pipeline.R")'
#
# `PP` (a list in the global environment) overrides the defaults below:
#   steps          any of "gen", "check", "rowwise_check", "readers", "time", "rowwise",
#                  "idle", "first", "probe", "render"; they run in that order whatever
#                  the order given
#   sizes, tables  fact-table rows (1e6, 1e7, 1e8) and "narrow" / "wide"
#   caches         "warm" / "cold" (the timings' page-cache state)
#   reps           quiet reps per cell: one number, or one per size; default 15/10/5
#                  for 1e6/1e7/1e8
#   row_group_rows rows per row group of the generated files
#   data_dir       the parquet files; out_dir the CSVs, reference results and the report
#   root, lib      the checkout (cuplyr is loaded from it, never installed) and the
#                  benchmark library (bench/install-deps.R)
#   busy_max       one rule for every engine, cache state and rep: a call counts as
#                  quiet when other processes' user-space CPU share (pp_busy()'s
#                  `foreign`) is at most this, and the wait before each call uses the
#                  same limit
#   warmups        untimed calls per engine and variant at the start of each size,
#                  table and cache piece of the time step, before any cell is timed
#   first_rows, first_calls   the fresh-process first-call step: table size, calls per process
#   probes         the probe step's parts: "stages", "join", "shapes"; probe_sizes the
#                  sizes of the first two (shapes uses first_rows)
#   variants, engines   optional subsets (for tests and retries)
#   rowwise_sizes  fact-table rows of the rowwise pipeline (wide table); rowwise_forms an
#                  optional subset of its forms; rowwise_reps quiet reps (one number, or
#                  one per size; default 10 for 1e5 rows or fewer, else 5); cap seconds
#                  per call, after which a call is aborted and recorded as capped (an
#                  engine capped at one size is not run at larger ones); rowwise_warm_rows
#                  the size of the file the rowwise forms are warmed up on
#   The rowwise steps: rowwise_check runs every form once per size against a base-R
#                  reference (rowwise-correctness.csv) and asks pp_fallback_probe() whether
#                  cuplyr stayed on the GPU (rowwise-native.csv); rowwise times the forms
#                  (rowwise.csv), one size and cache state per call like `time`: dplyr's
#                  per-row form in a cell of its own, the four others together
#   idle_rows, idle_secs, idle_reps   the idle step (pp_idle_probe()): cuplyr's rowwise
#                  form, a one-column cuplyr scan and DuckDB's rowwise form on the wide
#                  table at idle_rows rows (warm cache), each after an untimed call of
#                  itself and then idle_secs seconds of idle, idle_reps times (idle.csv)
# A step run on a subset (sizes, tables, caches, variants, engines) replaces exactly
# that subset's rows of its CSV files; the replaced rows of timings.csv move to
# timings-superseded.csv, so no attempt is ever deleted. The report
# (bench/parquet-pipeline.qmd) only reads the CSVs; the `time` step is meant to be run
# in pieces (one size, table and cache per call), when the machine is quiet.
# The steps that take timings (readers, time, rowwise, idle, first, probe) refuse to run from a
# bench/ tree with uncommitted changes: commit first, then time. Every step appends a
# row to provenance.csv (git HEAD, bench/ dirty flag, the commit of the package code,
# start and end).

local({
  defaults <- list(
    steps = c("gen", "check", "rowwise_check", "readers", "time", "rowwise", "idle", "first", "probe", "render"),
    sizes = c(1e6, 1e7, 1e8), tables = c("narrow", "wide"), caches = c("warm", "cold"),
    reps = NULL, row_group_rows = 1048576L,
    data_dir = "/home/theo/cuplr/scratchpad/benchmarks/parquet-pipeline/data",
    out_dir = "/home/theo/cuplr/scratchpad/benchmarks/parquet-pipeline",
    root = normalizePath(getwd()),
    lib = "/home/theo/cuplr/scratchpad/benchmarks/rlib",
    busy_max = 0.15, warmups = 2L, first_rows = 1e7, first_calls = 8L,
    probes = c("stages", "join", "shapes"), probe_sizes = c(1e7, 1e8),
    variants = NULL, engines = NULL,
    rowwise_sizes = c(1e5, 1e6, 1e7), rowwise_forms = NULL, rowwise_reps = NULL,
    cap = 60, rowwise_warm_rows = 1e5,
    idle_rows = 1e5, idle_secs = c(0, 10, 20, 40), idle_reps = 4L
  )
  user <- get0("PP", envir = globalenv(), inherits = FALSE)
  cfg <- defaults
  for (nm in names(user)) cfg[nm] <- list(user[[nm]])
  cfg$steps <- intersect(defaults$steps, cfg$steps)
  cfg$root <- normalizePath(cfg$root, mustWork = TRUE)
  dir.create(cfg$data_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)
  .libPaths(unique(c(cfg$lib[dir.exists(cfg$lib)], .libPaths())))

  env <- environment()
  source(file.path(cfg$root, "bench", "parquet-lib.R"), local = env)

  variants <- pp_variants()$variant
  if (!is.null(cfg$variants)) variants <- intersect(variants, cfg$variants)
  engines <- pp_engines()
  if (!is.null(cfg$engines)) engines <- intersect(engines, cfg$engines)
  forms_rw <- pp_rowwise_forms()
  if (!is.null(cfg$rowwise_forms)) forms_rw <- intersect(forms_rw, cfg$rowwise_forms)
  k <- pp_constants()

  fact_path <- function(table, n) file.path(cfg$data_dir, sprintf("fact-%s-%.0f.parquet", table, n))
  dim_path <- file.path(cfg$data_dir, "stores.parquet")
  out <- function(f) file.path(cfg$out_dir, f)
  reps_for <- function(n) {
    r <- cfg$reps
    if (is.null(r)) return(if (n <= 1e6) 15L else if (n <= 1e7) 10L else 5L)
    if (length(r) == 1L) return(as.integer(r))
    if (length(r) != length(cfg$sizes)) stop("PP$reps must be one number or one per size")
    as.integer(r[[match(n, cfg$sizes)]])
  }
  rw_reps_for <- function(n) {
    r <- cfg$rowwise_reps
    if (is.null(r)) return(if (n <= 1e5) 10L else 5L)
    if (length(r) == 1L) return(as.integer(r))
    if (length(r) != length(cfg$rowwise_sizes)) stop("PP$rowwise_reps must be one number or one per rowwise size")
    as.integer(r[[match(n, cfg$rowwise_sizes)]])
  }
  `%||%` <- function(a, b) if (is.null(a)) b else a
  say <- function(...) message(format(Sys.time(), "%H:%M:%S"), "  ", ...)
  stamp <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3")

  # ---- CSV files: read, replace a subset's rows, write -----------------------------
  read_csv <- function(f) {
    if (file.exists(out(f))) utils::read.csv(out(f), stringsAsFactors = FALSE) else NULL
  }
  # `a` and `b` with the union of their columns, a first
  rbind_fill <- function(a, b) {
    if (is.null(a) || !nrow(a)) return(b)
    if (is.null(b) || !nrow(b)) return(a)
    for (m in setdiff(names(a), names(b))) b[[m]] <- NA
    for (m in setdiff(names(b), names(a))) a[[m]] <- NA
    rbind(a, b[names(a)])
  }
  # Append rows to a CSV file (a log: nothing is replaced).
  csv_append <- function(f, new) {
    utils::write.csv(rbind_fill(read_csv(f), new), out(f), row.names = FALSE, na = "")
  }
  # `old` without the rows that match any row of `drop` on drop's columns, then
  # `new` appended. With `archive` (a file name), the removed rows are appended to
  # that file, stamped with the time, instead of being lost.
  csv_replace <- function(f, new, drop, archive = NULL) {
    old <- read_csv(f)
    if (!is.null(old) && nrow(old) && nrow(drop)) {
      cols <- intersect(names(drop), names(old))
      key <- function(d) do.call(paste, c(lapply(d[cols], as.character), sep = "\r"))
      gone <- key(old) %in% key(drop)
      if (!is.null(archive) && any(gone)) {
        moved <- old[gone, , drop = FALSE]
        moved$superseded_at <- stamp()
        csv_append(archive, moved)
      }
      old <- old[!gone, , drop = FALSE]
    }
    if (!is.null(new) && nrow(new)) {
      if (!is.null(old) && nrow(old)) {
        for (m in setdiff(names(old), names(new))) new[[m]] <- NA
        for (m in setdiff(names(new), names(old))) old[[m]] <- NA
        old <- old[names(new)]
        new <- rbind(old, new)
      }
    } else new <- old
    if (!is.null(new)) utils::write.csv(new, out(f), row.names = FALSE, na = "")
    invisible(new)
  }
  # NA written as "" would read back as NA only for numbers; character NAs become ""
  grid <- function(...) {
    g <- expand.grid(..., stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
    if ("rows" %in% names(g)) g$rows <- as.integer(g$rows)
    g
  }

  # ---- provenance ------------------------------------------------------------------
  git <- function(...) {
    tryCatch(suppressWarnings(system2("git", c("-C", shQuote(cfg$root), ...), stdout = TRUE, stderr = FALSE)),
             error = function(e) character())
  }
  pkg_paths <- c("R", "src", "DESCRIPTION", "NAMESPACE")
  provenance <- function() {
    list(git_head = git("rev-parse", "--short", "HEAD")[1L] %||% NA_character_,
         bench_dirty = length(git("status", "--porcelain", "--", "bench")) > 0L,
         cuplyr_commit = git("log", "-1", "--format=%h", "--", pkg_paths)[1L] %||% NA_character_,
         cuplyr_dirty = length(git("status", "--porcelain", "--", pkg_paths)) > 0L)
  }
  # Timings are taken from committed code only.
  require_clean <- function(step) {
    if (provenance()$bench_dirty) {
      stop("the ", step, " step takes timings and needs a bench/ tree without uncommitted changes: commit first (",
           paste(git("status", "--porcelain", "--", "bench"), collapse = "; "), ")", call. = FALSE)
    }
  }
  collapse <- function(x, all) if (is.null(x) || setequal(x, all)) "all" else paste(x, collapse = " ")

  ensure_cuplyr <- function() {
    if (!"cuplyr" %in% loadedNamespaces()) {
      pkgload::load_all(cfg$root, quiet = TRUE, debug = FALSE)
    }
  }
  last_read <- function() utils::getFromNamespace("parquet_last_read", "cuplyr")()

  context <- function() stats::setNames(lapply(engines, pp_context), engines)
  close_all <- function(ctx) for (c in ctx) pp_close(c)
  run_engine <- function(ctx, e, v, fact) pp_run(e, v, fact, dim_path, ctx[[e]])

  # The settings in effect in this process, read back from the engines.
  record_settings <- function(ctx, step) {
    rows <- list()
    add <- function(engine, v) {
      if (length(v)) rows[[length(rows) + 1L]] <<- data.frame(engine = engine, setting = names(v), value = as.character(v),
                                                               step = step, stringsAsFactors = FALSE)
    }
    for (e in engines) {
      add(e, pp_threads(e, ctx[[e]]))
      add(e, pp_engine_settings(e, ctx[[e]]))
    }
    add("harness", c(busy_max = cfg$busy_max, warmups = cfg$warmups))
    th <- do.call(rbind, rows)
    csv_replace("threads.csv", th, th[c("engine", "setting")])
  }

  # ---- gen -------------------------------------------------------------------------
  step_gen <- function() {
    say("gen: ", length(cfg$sizes) * length(cfg$tables), " fact files in ", cfg$data_dir)
    if (!file.exists(dim_path)) pp_gen_dim(dim_path, seed = k$seed)
    con <- DBI::dbConnect(duckdb::duckdb(shared_home = FALSE))
    on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
    rows <- list()
    for (n in cfg$sizes) for (tab in cfg$tables) {
      path <- fact_path(tab, n)
      want_groups <- as.integer(ceiling(n / cfg$row_group_rows))
      have <- if (file.exists(path)) tryCatch(arrow::ParquetFileReader$create(path)$num_row_groups, error = function(e) -1L) else -1L
      if (have != want_groups) {
        t0 <- proc.time()[["elapsed"]]
        pp_gen_fact(path, n = n, wide = tab == "wide", seed = k$seed, row_group_rows = cfg$row_group_rows)
        say("gen: wrote ", basename(path), sprintf(" in %.0f s", proc.time()[["elapsed"]] - t0))
      }
      rd <- arrow::ParquetFileReader$create(path)
      md <- DBI::dbGetQuery(con, sprintf(
        "SELECT string_agg(DISTINCT compression, ',') AS codec, max(row_group_num_rows) AS rg_max FROM parquet_metadata('%s')", path))
      rows[[length(rows) + 1L]] <- data.frame(
        table = tab, rows = as.integer(n), path = path, bytes = file.size(path),
        row_groups = rd$num_row_groups, codec = md$codec, columns = rd$GetSchema()$num_fields,
        row_group_max_rows = as.numeric(md$rg_max),
        stringsAsFactors = FALSE)
    }
    csv_replace("files.csv", do.call(rbind, rows), grid(table = cfg$tables, rows = cfg$sizes))
  }

  # ---- check -----------------------------------------------------------------------
  step_check <- function() {
    ensure_cuplyr()
    ctx <- context()
    on.exit(close_all(ctx), add = TRUE)
    record_settings(ctx, "check")
    cpu <- tryCatch(sub("^.*: ", "", grep("^model name", readLines("/proc/cpuinfo"), value = TRUE)[[1L]]), error = function(e) NA_character_)
    mem <- tryCatch(as.numeric(sub("^MemTotal:\\s+([0-9]+).*$", "\\1", readLines("/proc/meminfo", 1L))) * 1024, error = function(e) NA_real_)
    gpu <- tryCatch(cuplyr::gpu_details(), error = function(e) NULL)
    ver <- function(p) tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)
    extra <- data.frame(
      engine = c(rep("machine", 5L), "cuplyr", "cuplyr", "duckdb", "dplyr", "dplyr", "data.table"),
      setting = c("cpu_model", "logical_cpus", "ram_bytes", "gpu_name", "gpu_memory_bytes",
                  "version", "git_head", "version", "arrow_version", "version", "version"),
      value = as.character(c(cpu, parallel::detectCores(), mem, if (is.null(gpu)) NA else gpu$name,
                             if (is.null(gpu)) NA else gpu$total_memory,
                             ver("cuplyr"), provenance()$git_head, ver("duckdb"), ver("arrow"), ver("dplyr"), ver("data.table"))),
      step = "check", stringsAsFactors = FALSE)
    csv_replace("threads.csv", extra, extra[c("engine", "setting")])

    # rows in the row groups that survive the date filter's statistics, from the file's footer
    dcon <- ctx[["duckdb"]]$con %||% DBI::dbConnect(duckdb::duckdb(shared_home = FALSE))
    if (is.null(ctx[["duckdb"]]$con)) on.exit(DBI::dbDisconnect(dcon, shutdown = TRUE), add = TRUE)
    decoded_rows <- function(path, push, after, total_rows) {
      if (!push) return(as.numeric(total_rows))
      md <- DBI::dbGetQuery(dcon, sprintf(
        "SELECT row_group_id, any_value(row_group_num_rows) AS n, max(stats_max) AS mx FROM parquet_metadata('%s') WHERE path_in_schema = 'day' GROUP BY row_group_id", path))
      keep <- md$mx >= format(k$day_from)
      if (!is.na(after) && sum(keep) != after) return(NA_real_)
      as.numeric(sum(md$n[keep]))
    }

    dir.create(out("reference"), showWarnings = FALSE)
    ref_order <- intersect(c("dplyr", "data.table", "duckdb"), engines)
    for (n in cfg$sizes) for (tab in cfg$tables) {
      fact <- fact_path(tab, n)
      if (!file.exists(fact)) stop("missing ", fact, " (run the gen step)")
      cr <- list(); pd <- list()
      pd_row <- function(v, lr, plan) {
        push <- grepl("_push$", v)
        data.frame(
          variant = v, table = tab, rows = as.integer(n), source = basename(lr$paths[[1L]]), plan = plan,
          columns_read = paste(lr$columns, collapse = ","), predicates = paste(lr$predicates, collapse = " & "),
          rows_read = lr$rows_read, row_groups_total = lr$row_groups_total,
          row_groups_after_stats = lr$row_groups_after_stats,
          rows_decoded = if (startsWith(basename(lr$paths[[1L]]), "fact")) decoded_rows(fact, push, lr$row_groups_after_stats, n) else NA_real_,
          stringsAsFactors = FALSE)
      }
      for (v in variants) {
        say("check: ", tab, " ", n, " ", v)
        res <- list(); status <- list(); detail <- list()
        for (e in c(ref_order, setdiff(engines, ref_order))) {
          r <- tryCatch(run_engine(ctx, e, v, fact), error = function(cnd) cnd)
          if (inherits(r, "error")) {
            status[[e]] <- pp_error_status(r); detail[[e]] <- gsub("\\s+", " ", conditionMessage(r))
            if (e == "cuplyr") try(cuplyr::gpu_gc(), silent = TRUE)
          } else {
            res[[e]] <- tryCatch(pp_normalize(r, v), error = function(cnd) cnd)
            if (inherits(res[[e]], "error")) {
              status[[e]] <- "error"; detail[[e]] <- paste("normalize:", conditionMessage(res[[e]])); res[[e]] <- NULL
            } else {
              if (e == "cuplyr") {
                pd[[length(pd) + 1L]] <- pd_row(v, last_read(), "pipeline")
                # after a join the last read is the dimension file: compute the
                # dimension first, so that the fact file is the plan's only parquet
                # read, and take the fact scan as it runs under the join (its
                # result must equal the pipeline's)
                if (startsWith(v, "p2")) {
                  old <- options(cuplyr.fallback = "error")
                  fs <- tryCatch({
                    dm <- dplyr::compute(cuplyr::read_parquet_gpu(dim_path))
                    z <- pp_normalize(pp_cuplyr_under_join(v, fact, dm), v)
                    lr <- last_read()
                    cmp <- pp_compare(z, res[[e]], k$tol)
                    if (!cmp$ok) stop("the join with a computed dimension differs from the pipeline: ", cmp$detail)
                    pd_row(v, lr, "fact scan under the join")
                  }, error = function(cnd) { say("check: ", v, " under the join failed: ", conditionMessage(cnd)); NULL })
                  options(old)
                  if (!is.null(fs)) pd[[length(pd) + 1L]] <- fs
                  invisible(gc()); try(cuplyr::gpu_gc(), silent = TRUE)
                }
              }
              status[[e]] <- "ok"; detail[[e]] <- ""
            }
          }
          rm(r); invisible(gc())
        }
        ref_engine <- intersect(ref_order, names(res))[1L]
        if (!is.na(ref_engine)) {
          saveRDS(list(engine = ref_engine, result = res[[ref_engine]]),
                  out(file.path("reference", sprintf("%s-%s-%.0f.rds", v, tab, n))))
          for (e in names(res)) {
            if (e == ref_engine) { detail[[e]] <- "reference"; next }
            cmp <- pp_compare(res[[e]], res[[ref_engine]], k$tol)
            if (!cmp$ok) { status[[e]] <- "mismatch"; detail[[e]] <- cmp$detail }
          }
        }
        cr[[length(cr) + 1L]] <- data.frame(
          engine = engines, variant = v, table = tab, rows = as.integer(n),
          status = unlist(status[engines]), detail = unlist(detail[engines]),
          result_rows = vapply(engines, function(e) if (is.null(res[[e]])) NA_real_ else as.numeric(nrow(res[[e]])), 0),
          kept_rows = vapply(engines, function(e) if (is.null(res[[e]]) || !"n" %in% names(res[[e]]) || startsWith(v, "p3")) NA_real_ else sum(res[[e]]$n), 0),
          stringsAsFactors = FALSE, row.names = NULL)
      }
      drop <- grid(engine = engines, variant = variants, table = tab, rows = n)
      csv_replace("correctness.csv", do.call(rbind, cr), drop)
      if (length(pd)) csv_replace("pushdown.csv", do.call(rbind, pd), grid(variant = variants, table = tab, rows = n))
    }
  }

  # ---- readers and the data.table idioms ---------------------------------------------
  # The page cache is warm for both: files are read through before every call.
  warm_before <- function(paths) function(e) {
    r <- pp_resident(paths)
    if (!is.finite(r) || r < 0.99) { pp_warm(paths); r <- pp_resident(paths) }
    r
  }
  step_readers <- function() {
    ensure_cuplyr()
    require_clean("readers")
    cols <- pp_columns("p1_push")
    for (n in cfg$sizes) for (tab in cfg$tables) {
      fact <- fact_path(tab, n)
      say("readers: ", tab, " ", n)
      fns <- list(
        arrow = function() pp_read_arrow(fact, cols),
        nanoparquet = function() pp_read_nano(fact, cols))
      tm <- pp_time_cell(fns, reps = reps_for(n), before = warm_before(fact),
                         check = function(e, res) is.data.frame(res) && nrow(res) == n && ncol(res) == length(cols),
                         busy_max = cfg$busy_max)
      names(tm)[names(tm) == "engine"] <- "reader"
      csv_replace("readers.csv", cbind(tm[1L], table = tab, rows = as.integer(n), tm[-1L]),
                  grid(reader = c("arrow", "nanoparquet"), table = tab, rows = n))
      # the two data.table idioms (see pp_datatable()), on the variants that differ most
      fv <- intersect(c("p1_push", "p2_nopush", "p3_nopush"), variants)
      for (v in fv) {
        ref <- tryCatch(readRDS(out(file.path("reference", sprintf("%s-%s-%.0f.rds", v, tab, n))))$result, error = function(e) NULL)
        forms <- c("single", "stepwise")
        fns <- stats::setNames(lapply(forms, function(f) function() pp_datatable(v, fact, dim_path, form = f)), forms)
        tm <- pp_time_cell(fns, reps = reps_for(n), before = warm_before(c(fact, dim_path)),
                           check = function(e, res) is.null(ref) || {
                             z <- pp_compare(pp_normalize(res, v), ref, k$tol); if (z$ok) TRUE else z$detail },
                           busy_max = cfg$busy_max)
        names(tm)[names(tm) == "engine"] <- "form"
        csv_replace("forms.csv", cbind(tm[1L], variant = v, table = tab, rows = as.integer(n), tm[-1L]),
                    grid(form = forms, variant = v, table = tab, rows = n))
      }
    }
  }

  # ---- time ------------------------------------------------------------------------
  step_time <- function() {
    ensure_cuplyr()
    require_clean("time")
    cr <- read_csv("correctness.csv")
    if (is.null(cr)) stop("no correctness.csv: run the check step first")
    ctx <- context()
    on.exit(close_all(ctx), add = TRUE)
    record_settings(ctx, "time")
    drop <- grid(engine = engines, variant = variants, table = cfg$tables, rows = cfg$sizes, cache = cfg$caches)
    csv_replace("timings.csv", NULL, drop, archive = "timings-superseded.csv")
    csv_replace("warmup.csv", NULL, drop)
    for (n in cfg$sizes) for (tab in cfg$tables) for (cache in cfg$caches) {
      fact <- fact_path(tab, n)
      paths <- c(fact, dim_path)
      before <- if (cache == "warm") warm_before(paths) else function(e) { pp_evict(paths); pp_resident(paths) }
      ok_of <- function(v) engines[vapply(engines, function(e) {
        any(cr$engine == e & cr$variant == v & cr$table == tab & cr$rows == n & cr$status == "ok")
      }, NA)]
      # Process warm-up: before any cell of this piece is timed, every engine runs
      # every variant `warmups` times, untimed and the same for all engines, so the
      # cells measure the steady state. (The first-call cost is the first step's.)
      wu <- list()
      for (v in variants) for (e in ok_of(v)) for (j in seq_len(cfg$warmups)) {
        say("warm-up: ", cache, " ", tab, " ", n, " ", v, " ", e, " #", j)
        invisible(gc())
        before(e)
        t0 <- as.numeric(Sys.time())
        r <- tryCatch(run_engine(ctx, e, v, fact), error = function(cnd) cnd)
        secs <- as.numeric(Sys.time()) - t0
        failed <- inherits(r, "error")
        if (failed && e == "cuplyr") try(cuplyr::gpu_gc(), silent = TRUE)
        rm(r)
        wu[[length(wu) + 1L]] <- data.frame(engine = e, variant = v, table = tab, rows = as.integer(n), cache = cache,
                                            call = j, secs = secs, ok = !failed, stringsAsFactors = FALSE)
      }
      if (length(wu)) csv_replace("warmup.csv", do.call(rbind, wu), grid(engine = engines, variant = variants, table = tab, rows = n, cache = cache))
      for (v in variants) {
        ok_eng <- ok_of(v)
        ref <- tryCatch(readRDS(out(file.path("reference", sprintf("%s-%s-%.0f.rds", v, tab, n))))$result, error = function(e) NULL)
        if (!length(ok_eng) || is.null(ref)) { say("time: skipping ", cache, " ", tab, " ", n, " ", v, " (not checked)"); next }
        say("time: ", cache, " ", tab, " ", n, " ", v)
        fns <- stats::setNames(lapply(ok_eng, function(e) function() run_engine(ctx, e, v, fact)), ok_eng)
        tm <- pp_time_cell(fns, reps = reps_for(n), before = before,
                           check = function(e, res) {
                             z <- pp_compare(pp_normalize(res, v), ref, k$tol); if (z$ok) TRUE else z$detail },
                           busy_max = cfg$busy_max)
        csv_replace("timings.csv",
                    cbind(tm[1L], variant = v, table = tab, rows = as.integer(n), cache = cache, tm[-1L], busy_max = cfg$busy_max),
                    grid(engine = ok_eng, variant = v, table = tab, rows = n, cache = cache),
                    archive = "timings-superseded.csv")
      }
    }
  }

  # ---- rowwise: correctness, cuplyr's native/fallback verdict, timings -------------------
  # The rowwise pipeline (pp_rowwise_forms()) runs on the wide table only. Its sizes are
  # cfg$rowwise_sizes. A call longer than cfg$cap seconds is aborted and recorded as
  # "capped"; a form capped at one size is not run at larger ones (a "not_run" row says
  # so). `rowwise-correctness.csv` holds one row per form and size, `rowwise-native.csv`
  # cuplyr's verdict per size, `rowwise.csv` the timed calls.
  rw_context <- function() {
    e <- unique(vapply(forms_rw, pp_rowwise_engine, ""))
    stats::setNames(lapply(e, pp_context), e)
  }
  # form -> the smallest size below `n` at which it was capped, in either file
  rw_capped_before <- function(n) {
    out <- list()
    for (f in c("rowwise-correctness.csv", "rowwise.csv")) {
      d <- read_csv(f)
      if (is.null(d) || !nrow(d)) next
      d <- d[d$status == "capped" & d$rows < n, , drop = FALSE]
      for (i in seq_len(nrow(d))) out[[d$form[i]]] <- min(out[[d$form[i]]] %||% Inf, d$rows[i])
    }
    out
  }
  rw_not_run <- function(skip, n, extra) {
    if (!length(skip)) return(NULL)
    cbind(data.frame(form = names(skip), rows = as.integer(n), stringsAsFactors = FALSE), extra,
          data.frame(status = "not_run",
                     detail = paste0("capped at ", format(unlist(skip), big.mark = ",", scientific = FALSE, trim = TRUE), " rows"),
                     stringsAsFactors = FALSE))
  }

  step_rowwise_check <- function() {
    ensure_cuplyr()
    ctx <- rw_context()
    on.exit(close_all(ctx), add = TRUE)
    dir.create(out("reference"), showWarnings = FALSE)
    for (n in cfg$rowwise_sizes) {
      fact <- fact_path("wide", n)
      if (!file.exists(fact)) stop("missing ", fact, " (run the gen step for the wide table at this size)")
      say("rowwise check: ", n)
      ref <- pp_rowwise_ref(fact)
      saveRDS(list(engine = "base R", result = ref), out(file.path("reference", sprintf("rowwise-%.0f.rds", n))))
      skip <- rw_capped_before(n)
      run <- setdiff(forms_rw, names(skip))
      skip <- skip[intersect(forms_rw, names(skip))]
      rows_of <- list()
      verdict <- read <- NULL
      fns <- stats::setNames(lapply(run, function(f) function() {
        e <- ctx[[pp_rowwise_engine(f)]]
        if (!identical(f, "cuplyr_rowwise")) return(pp_run_rowwise(f, fact, e))
        r <- NULL
        verdict <<- pp_fallback_probe(function() r <<- pp_run_rowwise(f, fact, e))
        if (is.null(r)) stop(if (!is.null(verdict$error)) verdict$error else verdict$reason)   # keeps an OOM classed
        read <<- last_read()   # the scan this run did: columns read, rows returned
        r
      }), run)
      if (length(fns)) {
        tm <- pp_time_cell(fns, reps = 0L, busy_max = 1, wait = NULL, cap = cfg$cap,
                           check = function(f, res) {
                             rows_of[[f]] <<- nrow(res)
                             z <- pp_compare(pp_normalize(res, "rowwise"), ref, k$tol); if (z$ok) TRUE else z$detail })
        cr <- data.frame(form = tm$engine, rows = as.integer(n), status = tm$status,
                         detail = ifelse(tm$status == "ok", "", tm$detail),
                         result_rows = vapply(tm$engine, function(f) if (is.null(rows_of[[f]])) NA_real_ else as.numeric(rows_of[[f]]), 0),
                         stringsAsFactors = FALSE, row.names = NULL)
      } else cr <- NULL
      nr <- rw_not_run(skip, n, data.frame(result_rows = NA_real_))
      if (!is.null(nr)) cr <- rbind_fill(cr, nr[c("form", "rows", "status", "detail", "result_rows")])
      csv_replace("rowwise-correctness.csv", cr, grid(form = forms_rw, rows = n))
      if (!is.null(verdict)) {
        csv_replace("rowwise-native.csv",
                    data.frame(form = "cuplyr_rowwise", rows = as.integer(n), native = verdict$native,
                               reason = verdict$reason,
                               columns_read = if (is.null(read)) NA_character_ else paste(read$columns, collapse = ","),
                               rows_read = if (is.null(read)) NA_real_ else read$rows_read,
                               stringsAsFactors = FALSE),
                    data.frame(form = "cuplyr_rowwise", rows = as.integer(n)))
      }
      invisible(gc()); try(cuplyr::gpu_gc(), silent = TRUE)
    }
  }

  # One size and cache state per call, like `time`. dplyr's per-row form takes thousands
  # of times longer per call than the other forms, so it is timed in a cell of its own,
  # with the same cache control, check, cap and quiet-machine rule: in a shared cell each
  # of its calls would leave the GPU idle for tens of seconds between two cuplyr calls,
  # and an idle GPU drops its clocks (the idle step measures what that costs). The other
  # forms are timed together, interleaved, in one cell. Each cell's forms are warmed up
  # on the small file (cfg$rowwise_warm_rows), untimed, right before the cell. Every
  # timed cuplyr call is watched for F88's cuplyr_cpu_fallback condition
  # (pp_fallback_watch(); columns `fallback` and `fallback_reason`). `cell_forms` names
  # the forms that shared a row's cell.
  step_rowwise <- function() {
    ensure_cuplyr()
    require_clean("rowwise")
    cr <- read_csv("rowwise-correctness.csv")
    if (is.null(cr)) stop("no rowwise-correctness.csv: run the rowwise_check step first")
    ctx <- rw_context()
    on.exit(close_all(ctx), add = TRUE)
    drop <- grid(form = forms_rw, rows = cfg$rowwise_sizes, cache = cfg$caches)
    csv_replace("rowwise.csv", NULL, drop, archive = "rowwise-superseded.csv")
    csv_replace("rowwise-warmup.csv", NULL, drop)
    wfact <- fact_path("wide", cfg$rowwise_warm_rows)
    for (n in cfg$rowwise_sizes) for (cache in cfg$caches) {
      fact <- fact_path("wide", n)
      before <- if (cache == "warm") warm_before(fact) else function(e) { pp_evict(fact); pp_resident(fact) }
      skip <- rw_capped_before(n)
      skip <- skip[intersect(forms_rw, names(skip))]
      checked <- forms_rw[vapply(forms_rw, function(f) any(cr$form == f & cr$rows == n & cr$status %in% c("ok", "capped")), NA)]
      run <- setdiff(checked, names(skip))
      ref <- tryCatch(readRDS(out(file.path("reference", sprintf("rowwise-%.0f.rds", n))))$result, error = function(e) NULL)
      if (!length(run) || is.null(ref)) { say("rowwise: skipping ", cache, " ", n, " (not checked)"); next }
      cells_rw <- list(setdiff(run, "dplyr_rowwise"), intersect(run, "dplyr_rowwise"))
      cells_rw <- cells_rw[lengths(cells_rw) > 0L]
      wu <- list(); new <- NULL
      for (g in cells_rw) {
        for (f in g) for (j in seq_len(cfg$warmups)) {
          say("rowwise warm-up: ", cache, " ", n, " ", f, " #", j)
          invisible(gc())
          warm_before(wfact)(f)
          t0 <- as.numeric(Sys.time())
          r <- tryCatch(pp_run_rowwise(f, wfact, ctx[[pp_rowwise_engine(f)]]), error = function(cnd) cnd)
          secs <- as.numeric(Sys.time()) - t0
          failed <- inherits(r, "error")
          if (failed && f == "cuplyr_rowwise") try(cuplyr::gpu_gc(), silent = TRUE)
          rm(r)
          wu[[length(wu) + 1L]] <- data.frame(form = f, rows = as.integer(n), cache = cache, warm_rows = as.integer(cfg$rowwise_warm_rows),
                                              call = j, secs = secs, ok = !failed, stringsAsFactors = FALSE)
        }
        say("rowwise time: ", cache, " ", n, " ", paste(g, collapse = " "))
        fns <- stats::setNames(lapply(g, function(f) function() pp_run_rowwise(f, fact, ctx[[pp_rowwise_engine(f)]])), g)
        watch <- NULL
        if ("cuplyr_rowwise" %in% g) {
          watch <- pp_fallback_watch(fns[["cuplyr_rowwise"]])
          fns[["cuplyr_rowwise"]] <- watch$fn
        }
        tm <- pp_time_cell(fns, reps = rw_reps_for(n), before = before, busy_max = cfg$busy_max, cap = cfg$cap,
                           check = function(f, res) {
                             z <- pp_compare(pp_normalize(res, "rowwise"), ref, k$tol); if (z$ok) TRUE else z$detail })
        # one watch record per cuplyr call, in call order, which is the rows' order
        tm$fallback <- NA
        tm$fallback_reason <- NA_character_
        if (!is.null(watch)) {
          i <- which(tm$engine == "cuplyr_rowwise")
          fb <- watch$calls()
          if (length(fb) != length(i)) stop("the fallback watch saw ", length(fb), " cuplyr calls, the cell has ", length(i), " rows")
          tm$fallback[i] <- nzchar(fb)
          tm$fallback_reason[i] <- fb
        }
        names(tm)[names(tm) == "engine"] <- "form"
        new <- rbind_fill(new, cbind(tm[1L], table = "wide", rows = as.integer(n), cache = cache, cell_forms = paste(g, collapse = "+"),
                                     tm[-1L], busy_max = cfg$busy_max, cap = cfg$cap))
      }
      if (length(wu)) csv_replace("rowwise-warmup.csv", do.call(rbind, wu), grid(form = forms_rw, rows = n, cache = cache))
      nr <- rw_not_run(skip, n, data.frame(table = "wide", cache = cache))
      if (!is.null(nr)) {
        nr$rep <- 0L; nr$ok <- FALSE; nr$quiet <- FALSE; nr$busy_max <- cfg$busy_max; nr$cap <- cfg$cap
        new <- rbind_fill(new, nr)
      }
      csv_replace("rowwise.csv", new, grid(form = forms_rw, rows = n, cache = cache), archive = "rowwise-superseded.csv")
      invisible(gc()); try(cuplyr::gpu_gc(), silent = TRUE)
    }
  }

  # ---- idle: what a GPU left idle costs the next cuplyr call ----------------------------
  # cuplyr's rowwise form, the smallest cuplyr call there is (one column read and
  # collected) and DuckDB's rowwise form as a CPU control, on the wide table at
  # cfg$idle_rows rows with a warm cache, each after the machine was left idle for each
  # of cfg$idle_secs seconds after an untimed call of the same function (pp_idle_probe()),
  # with nvidia-smi's performance state and memory clock before and after every call
  # (idle.csv; a re-run moves the old rows to idle-superseded.csv). Each function first
  # runs cfg$warmups times untimed.
  step_idle <- function() {
    ensure_cuplyr()
    require_clean("idle")
    n <- cfg$idle_rows
    fact <- fact_path("wide", n)
    if (!file.exists(fact)) stop("missing ", fact, " (run the gen step for the wide table at this size)")
    ctx <- stats::setNames(list(pp_context("duckdb")), "duckdb")
    on.exit(close_all(ctx), add = TRUE)
    col <- k$rowwise_cols[[1L]]
    fns <- list(
      cuplyr_rowwise = function() pp_run_rowwise("cuplyr_rowwise", fact, NULL),
      cuplyr_scan = function() dplyr::collect(cuplyr::read_parquet_gpu(fact, columns = col)),
      duckdb = function() pp_run_rowwise("duckdb", fact, ctx$duckdb))
    pp_warm(fact)
    for (f in fns) for (j in seq_len(cfg$warmups)) invisible(f())
    say("idle: ", length(fns), " functions x ", length(cfg$idle_secs), " idle times x ", cfg$idle_reps, " reps")
    tm <- pp_idle_probe(fns, idles = cfg$idle_secs, reps = cfg$idle_reps, before = warm_before(fact), busy_max = cfg$busy_max)
    csv_replace("idle.csv", cbind(tm[1L], table = "wide", rows = as.integer(n), cache = "warm", tm[-1L], busy_max = cfg$busy_max),
                data.frame(table = "wide"), archive = "idle-superseded.csv")
    invisible(gc()); try(cuplyr::gpu_gc(), silent = TRUE)
  }

  # ---- first calls, each in a fresh R process ----------------------------------------
  step_first <- function() {
    require_clean("first")
    n <- cfg$first_rows
    fact <- fact_path("narrow", n)
    if (!file.exists(dim_path)) pp_gen_dim(dim_path, seed = k$seed)
    if (!file.exists(fact)) pp_gen_fact(fact, n = n, wide = FALSE, seed = k$seed, row_group_rows = cfg$row_group_rows)
    rows <- list()
    for (e in engines) for (v in variants) {
      say("first: ", e, " ", v)
      pp_warm(c(fact, dim_path))
      r <- tryCatch(pp_first_call(e, v, fact, dim_path, calls = cfg$first_calls, root = cfg$root, lib = cfg$lib),
                    error = function(cnd) {
                      data.frame(engine = e, variant = v, call = 1L, secs = NA_real_, load_secs = NA_real_,
                                 busy = NA_real_, pid = NA_integer_, stringsAsFactors = FALSE)
                    })
      rows[[length(rows) + 1L]] <- cbind(r[1:2], table = "narrow", rows = as.integer(n), r[-(1:2)])
    }
    csv_replace("firstcall.csv", do.call(rbind, rows), grid(engine = engines, variant = variants, table = "narrow", rows = n))
  }

  # ---- probes ------------------------------------------------------------------------
  # Three separate, narrow measurements behind claims of the report (cuplyr only,
  # warm cache, narrow table, the quiet-machine rule as everywhere):
  #   stages  where a cuplyr call's time goes: the scan alone, the whole plan run
  #           to a GPU table, collecting a table that is already computed, and the
  #           whole pipeline (the benchmark's call). Stages are separate calls;
  #           each ends with gpu_sync(), so queued GPU work is counted.
  #   join    the join of p2 alone, on resident inputs, cuplyr against DuckDB
  #           (filtered facts as a table, joined to the 1000-row dimension), and
  #           the aggregate of p1 on the same resident table for scale.
  #   shapes  consecutive calls of small cuplyr pipelines in fresh processes,
  #           each shape alone and all of them in one process in sequence.
  step_probe <- function() {
    ensure_cuplyr()
    require_clean("probe")
    old <- options(cuplyr.fallback = "error")
    on.exit(options(old), add = TRUE)
    probes <- cfg$probes
    big <- intersect(cfg$sizes, cfg$probe_sizes)
    tab <- "narrow"

    if ("stages" %in% probes) for (n in big) for (v in intersect(c("p1_push", "p1_nopush"), variants)) {
      fact <- fact_path(tab, n)
      say("probe stages: ", n, " ", v)
      done <- dplyr::compute(pp_cuplyr_plan(v, fact, dim_path))
      invisible(dplyr::collect(done))
      sync <- function(x) { force(x); cuplyr::gpu_sync(); NULL }
      fns <- list(
        read = function() sync(dplyr::compute(pp_cuplyr_scan(v, fact))),
        compute = function() sync(dplyr::compute(pp_cuplyr_plan(v, fact, dim_path))),
        collect_only = function() dplyr::collect(done),
        full = function() pp_cuplyr(v, fact, dim_path))
      # two untimed runs of every stage first, so that the stages are measured after the
      # start-up cost of their shape (the same warm-up as the time step's)
      for (f in fns) for (j in seq_len(cfg$warmups)) f()
      tm <- pp_time_cell(fns, reps = reps_for(n), before = warm_before(c(fact, dim_path)), busy_max = cfg$busy_max)
      names(tm)[names(tm) == "engine"] <- "stage"
      csv_replace("stages.csv", cbind(tm[1L], variant = v, table = tab, rows = as.integer(n), cache = "warm", tm[-1L]),
                  grid(stage = names(fns), variant = v, table = tab, rows = n))
      rm(done); invisible(gc()); try(cuplyr::gpu_gc(), silent = TRUE)
    }

    if ("join" %in% probes) {
      ctx <- stats::setNames(list(pp_context("duckdb")), "duckdb")
      on.exit(close_all(ctx), add = TRUE)
      con <- ctx$duckdb$con
      aggs <- "count(*) AS n, coalesce(sum(f.revenue), 0) AS revenue, avg(f.price) AS avg_price, max(f.qty) AS max_qty"
      for (n in big) for (v in intersect(c("p2_push", "p2_nopush"), variants)) {
        fact <- fact_path(tab, n)
        say("probe join: ", n, " ", v)
        x <- dplyr::compute(pp_cuplyr_fact_side(v, fact))
        d <- dplyr::compute(cuplyr::read_parquet_gpu(dim_path))
        DBI::dbExecute(con, sprintf(paste0("CREATE OR REPLACE TEMP TABLE f AS SELECT store, channel, price, qty, ",
                                          "price * qty * (1 - discount) AS revenue FROM read_parquet('%s') WHERE %s"),
                                    fact, pp_sql_where(v)))
        DBI::dbExecute(con, sprintf("CREATE OR REPLACE TEMP TABLE s AS SELECT store, region FROM read_parquet('%s')", dim_path))
        sync <- function(z) { force(z); cuplyr::gpu_sync(); NULL }
        agg <- function(g) dplyr::summarise(g, n = dplyr::n(), revenue = sum(revenue, na.rm = TRUE),
                                            avg_price = mean(price), max_qty = max(qty), .groups = "drop")
        fns <- list(
          `cuplyr join` = function() sync(dplyr::compute(dplyr::inner_join(x, d, by = "store"))),
          `duckdb join` = function() DBI::dbExecute(con, paste("CREATE OR REPLACE TEMP TABLE j AS SELECT f.store, f.channel,",
                                                               "f.price, f.qty, f.revenue, s.region FROM f JOIN s ON f.store = s.store")),
          `cuplyr join+summarise` = function() dplyr::collect(agg(dplyr::group_by(dplyr::inner_join(x, d, by = "store"), region, channel))),
          `duckdb join+summarise` = function() DBI::dbGetQuery(con, paste("SELECT s.region, f.channel,", aggs,
                                                                          "FROM f JOIN s ON f.store = s.store GROUP BY s.region, f.channel")),
          `cuplyr summarise` = function() dplyr::collect(agg(dplyr::group_by(x, store, channel))),
          `duckdb summarise` = function() DBI::dbGetQuery(con, paste("SELECT f.store, f.channel,", aggs,
                                                                     "FROM f GROUP BY f.store, f.channel")))
        tm <- pp_time_cell(fns, reps = reps_for(n), busy_max = cfg$busy_max)
        tm$op <- sub("^[a-z]+ ", "", tm$engine)
        tm$engine <- sub(" .*$", "", tm$engine)
        tm <- tm[c("engine", "op", setdiff(names(tm), c("engine", "op")))]
        csv_replace("joincost.csv", cbind(tm[1:2], variant = v, table = tab, rows = as.integer(n), cache = "warm", tm[-(1:2)]),
                    grid(variant = v, table = tab, rows = n))
        rm(x, d); invisible(gc()); try(cuplyr::gpu_gc(), silent = TRUE)
      }
    }

    if ("shapes" %in% probes) {
      n <- cfg$first_rows
      f <- fact_path(tab, n)
      lib_file <- file.path(cfg$root, "bench", "parquet-lib.R")
      libpath <- unique(c(cfg$lib[nzchar(cfg$lib)], .libPaths()))
      child <- function(plan, f, dim, root, lib_file) {
        sys.source(lib_file, envir = environment())
        pkgload::load_all(root, quiet = TRUE, debug = FALSE, compile = FALSE)
        options(cuplyr.fallback = "error")   # as in every other measurement: a CPU fallback is not a GPU number
        day_from <- as.Date("2024-01-01"); qty_min <- 2L
        shapes <- list(
          scan = function() dplyr::collect(cuplyr::read_parquet_gpu(f, columns = "store")),
          filter_int = function() {
            dplyr::collect(dplyr::select(dplyr::filter(cuplyr::read_parquet_gpu(f), qty >= qty_min), store, qty))
          },
          filter_date_int = function() {
            dplyr::collect(dplyr::select(dplyr::filter(cuplyr::read_parquet_gpu(f), day >= day_from, qty >= qty_min), store, qty))
          },
          summarise = function() {
            dplyr::collect(dplyr::summarise(dplyr::group_by(cuplyr::read_parquet_gpu(f), store), n = dplyr::n()))
          },
          mutate_sum = function() {
            dplyr::collect(dplyr::summarise(dplyr::mutate(cuplyr::read_parquet_gpu(f), revenue = price * qty), r = sum(revenue)))
          },
          p1_push = function() pp_cuplyr("p1_push", f, dim))
        res <- list(); idx <- 0L
        for (i in seq_len(nrow(plan))) for (j in seq_len(plan$calls[[i]])) {
          t0 <- as.numeric(Sys.time())
          z <- shapes[[plan$shape[[i]]]]()
          secs <- as.numeric(Sys.time()) - t0
          rm(z)
          idx <- idx + 1L
          res[[idx]] <- data.frame(shape = plan$shape[[i]], call = j, seq = idx, secs = secs, pid = Sys.getpid(), stringsAsFactors = FALSE)
        }
        do.call(rbind, res)
      }
      names_all <- c("scan", "filter_int", "filter_date_int", "summarise", "mutate_sum", "p1_push")
      pp_warm(c(f, dim_path))
      rows <- list()
      for (s in names_all) {
        say("probe shapes: ", s, " alone")
        r <- callr::r(child, args = list(data.frame(shape = s, calls = 5L), f, dim_path, cfg$root, lib_file), libpath = libpath, show = FALSE)
        rows[[length(rows) + 1L]] <- cbind(mode = "alone", r)
      }
      say("probe shapes: all in sequence")
      r <- callr::r(child, args = list(data.frame(shape = names_all, calls = 4L), f, dim_path, cfg$root, lib_file), libpath = libpath, show = FALSE)
      rows[[length(rows) + 1L]] <- cbind(mode = "sequence", r)
      sh <- cbind(do.call(rbind, rows), rows_in_file = as.integer(n))
      csv_replace("shapes.csv", sh, data.frame(rows_in_file = as.integer(n)))
    }
  }

  # ---- render ----------------------------------------------------------------------
  # Recent pandoc/Quarto embeds a few scripts as <script src="data:...;base64,...">.
  # That is self-contained, but a plain <script> with the text inside is the older,
  # more portable form, and it leaves no src attribute to mistake for an external
  # file; decode them back (skipping any script that could not sit inline).
  inline_module_scripts <- function(html) {
    txt <- rawToChar(readBin(html, "raw", file.size(html)))
    Encoding(txt) <- "UTF-8"
    pat <- '<script src="data:application/javascript;base64,([A-Za-z0-9+/=]+)"( type="module")?></script>'
    hits <- regmatches(txt, gregexpr(pat, txt, perl = TRUE))[[1L]]
    if (!length(hits)) return(invisible(FALSE))
    repl <- vapply(hits, function(h) {
      js <- rawToChar(jsonlite::base64_dec(sub(pat, "\\1", h, perl = TRUE)))
      Encoding(js) <- "UTF-8"
      if (grepl("</script", js, fixed = TRUE)) return(h)
      paste0("<script", if (grepl("type=\"module\"", h, fixed = TRUE)) " type=\"module\"", ">\n", js, "\n</script>")
    }, "", USE.NAMES = FALSE)
    regmatches(txt, gregexpr(pat, txt, perl = TRUE)) <- list(repl)
    writeBin(charToRaw(enc2utf8(txt)), html)
    invisible(TRUE)
  }

  # The lock log's rows (GPU jobs of every worktree, when each queued, waited and
  # held the lock), from three hours before the first recorded step, so that the
  # report can say what else used the machine while the timings ran.
  copy_lock_log <- function() {
    common <- git("rev-parse", "--git-common-dir")[1L]
    if (is.na(common)) return(invisible(NULL))
    if (!startsWith(common, "/")) common <- file.path(cfg$root, common)
    log <- file.path(common, "cuplyr-locks", "stats.tsv")
    pv <- read_csv("provenance.csv")
    if (!file.exists(log) || is.null(pv)) return(invisible(NULL))
    st <- utils::read.delim(log, header = FALSE, stringsAsFactors = FALSE,
                            col.names = c("start", "label", "worktree", "wait_s", "hold_s", "exit"))
    t0 <- as.POSIXct(min(pv$started), format = "%Y-%m-%dT%H:%M:%OS") - 3 * 3600
    when <- as.POSIXct(sub("([+-][0-9]{2}):([0-9]{2})$", "\\1\\2", st$start), format = "%Y-%m-%dT%H:%M:%S%z")
    utils::write.csv(st[!is.na(when) & when >= t0, , drop = FALSE], out("lockjobs.csv"), row.names = FALSE)
  }

  # Quarto renders a copy of the report in a scratch directory, so nothing is left
  # in bench/ (no .quarto/, no _files/, no html); the self-contained html is copied
  # to out_dir.
  step_render <- function() {
    quarto <- Sys.getenv("QUARTO_PATH", "")
    if (!nzchar(quarto)) quarto <- unname(Sys.which("quarto"))
    if (!nzchar(quarto) && file.exists(path.expand("~/.local/bin/quarto"))) quarto <- path.expand("~/.local/bin/quarto")
    if (!nzchar(quarto)) stop("the quarto CLI was not found (put it on PATH or set QUARTO_PATH)")
    copy_lock_log()
    src <- file.path(cfg$root, "bench", "parquet-pipeline.qmd")
    stage <- tempfile("parquet-pipeline-")
    dir.create(stage)
    on.exit(unlink(stage, recursive = TRUE), add = TRUE)
    file.copy(src, file.path(stage, "parquet-pipeline.qmd"))
    args <- c("render", shQuote(file.path(stage, "parquet-pipeline.qmd")),
              "-P", shQuote(paste0("root:", cfg$root)), "-P", shQuote(paste0("lib:", cfg$lib)),
              "-P", shQuote(paste0("out_dir:", cfg$out_dir)))
    say("render: quarto ", paste(args, collapse = " "))
    status <- system2(quarto, args)
    if (!identical(status, 0L)) stop("quarto render failed with exit status ", status)
    html <- file.path(stage, "parquet-pipeline.html")
    if (!file.exists(html)) stop("quarto did not produce ", html)
    inline_module_scripts(html)
    file.copy(html, file.path(cfg$out_dir, "parquet-pipeline.html"), overwrite = TRUE)
    say("render: wrote ", file.path(cfg$out_dir, "parquet-pipeline.html"))
  }

  for (s in cfg$steps) {
    t0 <- proc.time()[["elapsed"]]
    started <- stamp()
    prov <- provenance()
    outcome <- "error"
    on.exit_log <- function() {
      rw <- startsWith(s, "rowwise")
      idle <- identical(s, "idle")
      csv_append("provenance.csv", data.frame(
        step = s, sizes = if (idle) format(cfg$idle_rows) else if (rw) collapse(cfg$rowwise_sizes, defaults$rowwise_sizes) else collapse(cfg$sizes, defaults$sizes),
        tables = if (rw || idle) "wide" else collapse(cfg$tables, defaults$tables),
        caches = if (idle) "warm" else collapse(cfg$caches, defaults$caches),
        variants = if (rw || idle) "rowwise" else collapse(variants, pp_variants()$variant),
        engines = if (idle) "cuplyr_rowwise cuplyr_scan duckdb" else if (rw) collapse(forms_rw, pp_rowwise_forms()) else collapse(engines, pp_engines()),
        git_head = prov$git_head, bench_dirty = prov$bench_dirty,
        cuplyr_commit = prov$cuplyr_commit, cuplyr_dirty = prov$cuplyr_dirty,
        started = started, ended = stamp(), outcome = outcome, stringsAsFactors = FALSE))
    }
    tryCatch({
      switch(s, gen = step_gen(), check = step_check(), rowwise_check = step_rowwise_check(),
             readers = step_readers(), time = step_time(), rowwise = step_rowwise(), idle = step_idle(),
             first = step_first(), probe = step_probe(), render = step_render())
      outcome <- "ok"
    }, finally = on.exit_log())
    say("step ", s, sprintf(" done in %.0f s", proc.time()[["elapsed"]] - t0))
  }
  invisible(cfg)
})
