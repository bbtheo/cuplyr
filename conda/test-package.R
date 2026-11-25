# Package test for the r-cuplyr conda package, run by both recipes with
# `Rscript --vanilla` inside the test environment. It needs no GPU: build
# machines have none. It checks that the package loads and that the compiled library finds
# libcudf and librmm inside this environment.

library(cuplyr)

so <- system.file("libs", "cuplyr.so", package = "cuplyr")
stopifnot(nzchar(so))

# The environment prefix, from R itself: <prefix>/lib/R.
prefix <- normalizePath(file.path(R.home(), "..", ".."))

ldd <- system2("ldd", shQuote(so), stdout = TRUE, stderr = TRUE)
cat(ldd, sep = "\n")

missing <- grep("not found", ldd, value = TRUE)
if (length(missing)) {
  stop("unresolved libraries:\n", paste(missing, collapse = "\n"), call. = FALSE)
}

# Resolved path of one library, or NA when ldd doesn't list it.
resolved <- function(lib) {
  hit <- grep(paste0("^\\s*", lib, "\\.so"), ldd, value = TRUE)
  if (!length(hit)) {
    return(NA_character_)
  }
  sub("^.*=>\\s*(\\S+).*$", "\\1", hit[[1]])
}

for (lib in c("libcudf", "librmm")) {
  path <- resolved(lib)
  if (is.na(path)) {
    stop(lib, " is not a dependency of cuplyr.so", call. = FALSE)
  }
  # the same directory, however it is spelled
  path <- normalizePath(path)
  if (!startsWith(path, paste0(prefix, "/"))) {
    stop(lib, " resolves to ", path, ", outside the environment ", prefix, call. = FALSE)
  }
}

cat("cuplyr loads; libcudf and librmm resolve inside", prefix, "\n")
