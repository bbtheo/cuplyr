// src/version_info.cpp
//
// The libcudf and RMM versions cuplyr was compiled against, read from the
// version headers at compile time. Needs no GPU (and no CUDA call), so it works
// on the machines diagnostics() is most useful on.
// Each header is optional (configure only warns when the cudf one is missing):
// without it that entry is NA instead of a compile error.
#if defined(__has_include)
#  if __has_include(<cudf/version_config.hpp>)
#    include <cudf/version_config.hpp>
#    define CUPLYR_HAVE_CUDF_VERSION 1
#  endif
#  if __has_include(<rmm/version_config.hpp>)
#    include <rmm/version_config.hpp>
#    define CUPLYR_HAVE_RMM_VERSION 1
#  endif
#endif

#include <cstdio>

#include <Rcpp.h>

// "MAJOR.MM.PATCH" with the minor zero-padded to two digits, as RAPIDS writes
// it (26.02.0, not 26.2.0) and as configure formats it.
static std::string format_version(int major, int minor, int patch) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%d.%02d.%d", major, minor, patch);
    return buf;
}

// [[Rcpp::export]]
Rcpp::CharacterVector cuplyr_build_versions() {
    Rcpp::CharacterVector out = Rcpp::CharacterVector::create(
        Rcpp::Named("libcudf") = NA_STRING,
        Rcpp::Named("rmm") = NA_STRING);
#ifdef CUPLYR_HAVE_CUDF_VERSION
    out["libcudf"] =
        format_version(CUDF_VERSION_MAJOR, CUDF_VERSION_MINOR, CUDF_VERSION_PATCH);
#endif
#ifdef CUPLYR_HAVE_RMM_VERSION
    out["rmm"] =
        format_version(RMM_VERSION_MAJOR, RMM_VERSION_MINOR, RMM_VERSION_PATCH);
#endif
    return out;
}
