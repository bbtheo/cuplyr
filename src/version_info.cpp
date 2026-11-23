// src/version_info.cpp
//
// The libcudf and RMM versions cuplyr was compiled against, read from the
// version headers at compile time. Needs no GPU (and no CUDA call), so it works
// on the machines diagnostics() is most useful on.
#include <cudf/version_config.hpp>
#include <rmm/version_config.hpp>

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
        Rcpp::Named("libcudf") =
            format_version(CUDF_VERSION_MAJOR, CUDF_VERSION_MINOR, CUDF_VERSION_PATCH),
        Rcpp::Named("rmm") =
            format_version(RMM_VERSION_MAJOR, RMM_VERSION_MINOR, RMM_VERSION_PATCH));
    return out;
}
