# F63 helpers: read bench/rolling-window.qmd and lint its prose.
#
# Shared by test-parity-F63-qmd-static.R and test-parity-F63-qmd-render.R.
# bench/ is .Rbuildignore'd, so the tests skip outside a source checkout.

f63_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = TRUE)
}

# Path of a repo file, recording a failure (then skipping) if it is missing.
f63_require <- function(rel) {
  root <- testthat::test_path("..", "..")
  if (!file.exists(file.path(root, "DESCRIPTION"))) {
    testthat::skip("not in a source checkout; bench/ is not shipped")
  }
  path <- file.path(root, rel)
  testthat::expect_true(file.exists(path), label = paste(rel, "exists"))
  if (!file.exists(path)) testthat::skip(paste(rel, "missing (failure recorded)"))
  normalizePath(path)
}

f63_qmd_lines <- function() {
  readLines(f63_require("bench/rolling-window.qmd"), warn = FALSE)
}

f63_bench_lib <- function() {
  Sys.getenv("CUPLYR_BENCH_LIB", "/home/theo/cuplr/scratchpad/benchmarks/rlib")
}

# Split a .qmd into front matter, R chunks and prose.
#   front:  the YAML lines (without the --- delimiters)
#   chunks: list of list(label, opts (named chr), code (chr), line (int))
#           for ```{r} chunks; other fenced blocks are dropped
#   prose:  chr, one entry per source line; NA for YAML and fenced lines
f63_parse_qmd <- function(lines) {
  n <- length(lines)
  prose <- lines
  front <- character()
  i <- 1L
  if (n && trimws(lines[1]) == "---") {
    close <- which(trimws(lines[-1]) %in% c("---", "..."))[1] + 1L
    if (is.na(close)) close <- n
    front <- if (close > 2L) lines[2:(close - 1L)] else character()
    prose[1:close] <- NA_character_
    i <- close + 1L
  }
  chunks <- list()
  while (i <= n) {
    m <- regmatches(lines[i], regexec("^\\s*(`{3,}|~{3,})\\s*(.*)$", lines[i]))[[1]]
    if (!length(m)) {
      i <- i + 1L
      next
    }
    fence <- m[2]
    info <- trimws(m[3])
    close_re <- paste0("^\\s*\\", substr(fence, 1, 1), "{", nchar(fence), ",}\\s*$")
    j <- i + 1L
    while (j <= n && !grepl(close_re, lines[j])) j <- j + 1L
    body <- if (j > i + 1L) lines[(i + 1L):min(j - 1L, n)] else character()
    prose[i:min(j, n)] <- NA_character_
    if (grepl("^\\{r([ ,}]|$)", info)) {
      header <- sub("^\\{r\\s*,?\\s*", "", sub("\\}\\s*$", "", info))
      label <- trimws(strsplit(header, ",")[[1]][1])
      if (is.na(label) || grepl("=", label)) label <- ""
      is_opt <- grepl("^#\\|", body)
      first_code <- which(!is_opt)[1]
      opt_lines <- if (is.na(first_code)) body else body[seq_len(first_code - 1L)]
      opts <- list()
      for (ol in opt_lines) {
        kv <- regmatches(ol, regexec("^#\\|\\s*([A-Za-z0-9_.-]+)\\s*:\\s*(.*)$", ol))[[1]]
        if (length(kv)) opts[[kv[2]]] <- trimws(kv[3])
      }
      # header options such as {r bench-main, cache=TRUE}
      hdr <- regmatches(header, gregexpr("([A-Za-z0-9_.-]+)\\s*=\\s*([^,]+)", header))[[1]]
      for (h in hdr) {
        kv <- regmatches(h, regexec("^([A-Za-z0-9_.-]+)\\s*=\\s*(.*)$", h))[[1]]
        if (!kv[2] %in% names(opts)) opts[[kv[2]]] <- trimws(kv[3])
      }
      if (!is.null(opts[["label"]])) label <- opts[["label"]]
      code <- if (is.na(first_code)) character() else body[first_code:length(body)]
      chunks[[length(chunks) + 1L]] <- list(label = label, opts = opts, code = code, line = i)
    }
    i <- j + 1L
  }
  list(front = front, chunks = chunks, prose = prose)
}

f63_chunk_cached <- function(ch) {
  v <- ch$opts[["cache"]]
  !is.null(v) && toupper(v) %in% c("TRUE", "T", "YES")
}

# A code span whose content is a bare measurement (`2.78 ms`, `216`, `8.9x`)
# is a hard-coded number in disguise.
f63_bare_number_span <- function(span) {
  grepl("^\\s*[-+~]?[0-9][0-9.,eE+-]*\\s*(ms|s|sec|min|MB|GB|KB|B|bytes|x|%)?\\s*$", span)
}

# Prose lines that still contain a hand-typed number. Returns a data frame
# (line, text); zero rows means every number is an inline R expression.
#
# Exempt: YAML front matter and fenced blocks; inline `r ...` expressions;
# other code spans (identifiers, labels, file names) unless they hold a bare
# measurement; URLs and link targets; HTML comments; {attribute} blocks;
# ISO dates (decision dates); finding/feature ids (F1, F54); "Deviation N";
# "log10"; ordered-list markers. Chunk captions (fig-cap, tbl-cap, ...) are
# prose too unless given with !expr.
f63_prose_number_lines <- function(lines) {
  p <- f63_parse_qmd(lines)
  txt <- p$prose
  keep <- !is.na(txt)
  # HTML comments, possibly multi-line: blank them out, keeping line numbers
  txt2 <- ifelse(keep, txt, "")
  in_comment <- FALSE
  for (k in seq_along(txt2)) {
    s <- txt2[k]
    out <- ""
    repeat {
      if (in_comment) {
        e <- regexpr("-->", s, fixed = TRUE)
        if (e < 0) break
        s <- substring(s, e + 3L)
        in_comment <- FALSE
      }
      st <- regexpr("<!--", s, fixed = TRUE)
      if (st < 0) {
        out <- paste0(out, s)
        break
      }
      out <- paste0(out, substr(s, 1L, st - 1L))
      s <- substring(s, st + 4L)
      in_comment <- TRUE
    }
    txt2[k] <- out
  }
  bad <- list()
  for (k in which(keep)) {
    s <- txt2[k]
    s <- gsub("`r [^`]*`", " ", s)
    spans <- regmatches(s, gregexpr("`[^`]+`", s))[[1]]
    spans <- gsub("^`|`$", "", spans)
    if (any(f63_bare_number_span(spans))) {
      bad[[length(bad) + 1L]] <- data.frame(line = k, text = lines[k])
      next
    }
    s <- gsub("`[^`]+`", " ", s)
    s <- gsub("\\]\\([^)]*\\)", "] ", s)
    s <- gsub("<https?://[^>]*>", " ", s)
    s <- gsub("https?://[^[:space:])>]+", " ", s)
    s <- gsub("\\{[^}]*\\}", " ", s)
    s <- gsub("\\b[0-9]{4}-[0-9]{2}-[0-9]{2}\\b", " ", s, perl = TRUE)
    s <- gsub("\\bF[0-9]+\\b", " ", s, perl = TRUE)
    s <- gsub("\\b[Dd]eviation [0-9]+\\b", " ", s, perl = TRUE)
    s <- gsub("\\blog10\\b", " ", s, perl = TRUE)
    s <- sub("^\\s*[0-9]+[.)]\\s", " ", s)
    if (grepl("[0-9]", s)) bad[[length(bad) + 1L]] <- data.frame(line = k, text = lines[k])
  }
  # captions inside R chunks
  cap_keys <- c("fig-cap", "fig-subcap", "fig-alt", "tbl-cap", "tbl-subcap", "caption")
  for (ch in p$chunks) {
    for (key in intersect(names(ch$opts), cap_keys)) {
      v <- ch$opts[[key]]
      if (!grepl("^!expr", v) && grepl("[0-9]", v)) {
        bad[[length(bad) + 1L]] <- data.frame(line = ch$line, text = paste0("#| ", key, ": ", v))
      }
    }
  }
  if (!length(bad)) return(data.frame(line = integer(), text = character()))
  do.call(rbind, bad)
}

f63_inline_r_count <- function(lines) {
  p <- f63_parse_qmd(lines)
  s <- p$prose[!is.na(p$prose)]
  sum(lengths(regmatches(s, gregexpr("`r [^`]+`", s))))
}

# YAML front matter as a list; !expr values are kept as strings.
f63_front <- function(lines) {
  p <- f63_parse_qmd(lines)
  yaml::yaml.load(paste(p$front, collapse = "\n"),
                  handlers = list(expr = function(x) paste("!expr", x)))
}

f63_param <- function(front, name) {
  v <- front$params[[name]]
  if (is.list(v) && !is.null(v$value)) v <- v$value
  v
}

# knitr cache files: named vector label -> hash (from <label>_<hash>.rdb).
f63_cache_hashes <- function(cache_dir) {
  f <- list.files(cache_dir, pattern = "_[0-9a-f]{32}\\.rdb$", recursive = TRUE)
  f <- basename(f)
  stats::setNames(sub("^.*_([0-9a-f]{32})\\.rdb$", "\\1", f),
                  sub("_[0-9a-f]{32}\\.rdb$", "", f))
}

f63_quarto <- function() {
  q <- Sys.getenv("QUARTO_PATH", "")
  if (!nzchar(q)) q <- Sys.which("quarto")[[1]]
  if (!nzchar(q) && file.exists(path.expand("~/.local/bin/quarto"))) q <- path.expand("~/.local/bin/quarto")
  q
}

f63_render <- function(qmd, out_dir, params_file) {
  t0 <- proc.time()[["elapsed"]]
  out <- suppressWarnings(system2(
    f63_quarto(),
    c("render", shQuote(qmd), "--output-dir", shQuote(out_dir),
      "--execute-params", shQuote(params_file)),
    stdout = TRUE, stderr = TRUE))
  list(status = attr(out, "status") %||% 0L, log = out,
       secs = proc.time()[["elapsed"]] - t0)
}
