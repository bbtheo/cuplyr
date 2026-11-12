// src/r_strings.hpp
//
// R string <-> cudf string boundary helpers (F53). cudf holds UTF-8 bytes, so
// every R string that becomes cudf data goes through r_string_utf8(), and every
// string read back from cudf is marked UTF-8 (r_mk_utf8_char()). Base R and
// vctrs compare strings after translating to UTF-8, so this is what makes a
// latin1 "cafe" and a UTF-8 "cafe" the same value on the GPU too.
#ifndef CUPLYR_R_STRINGS_HPP
#define CUPLYR_R_STRINGS_HPP

#include <Rcpp.h>

#include <string>

namespace cuplyr {

// UTF-8 bytes of a (non-NA) CHARSXP. Rf_translateCharUTF8() is a flag check
// with no copy for ASCII, UTF-8-marked, and (in a UTF-8 session) unmarked
// strings; it converts latin1 and native strings of a non-UTF-8 session.
// Invalid bytes in an unmarked/UTF-8 string pass through unchanged (bytewise,
// as dplyr compares them). A "bytes"-encoded string compares unequal to every
// non-bytes string in base R, which the GPU cannot reproduce, so it is refused.
inline std::string r_string_utf8(SEXP charsxp) {
    if (Rf_getCharCE(charsxp) == CE_BYTES) {
        Rcpp::stop("a string in the \"bytes\" encoding cannot be sent to the GPU; "
                   "convert it first with iconv() or enc2utf8()");
    }
    const char* p = Rf_translateCharUTF8(charsxp);
    return std::string(p);
}

// First element of a length >= 1 character vector (not NA) as UTF-8.
inline std::string r_scalar_utf8(SEXP strsxp) {
    if (TYPEOF(strsxp) != STRSXP || Rf_xlength(strsxp) < 1) {
        Rcpp::stop("expected a character value");
    }
    return r_string_utf8(STRING_ELT(strsxp, 0));
}

// A CHARSXP for bytes read back from cudf: UTF-8-marked (ASCII stays unmarked).
inline SEXP r_mk_utf8_char(const char* p, int len) {
    return Rf_mkCharLenCE(p, len, CE_UTF8);
}

} // namespace cuplyr

#endif // CUPLYR_R_STRINGS_HPP
