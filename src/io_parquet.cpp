// src/io_parquet.cpp
//
// Parquet read/write on the GPU (F85). R/io.R owns the semantics (the type
// map, the scan record, every error and warning); this file is the libcudf
// glue:
//   gpu_parquet_probe()   footer facts plus the exact column types the reader
//                         produces (a 0-row read), so no type table of our own
//                         can disagree with the reader
//   gpu_parquet_read()    the read, with the normalisation of every column the
//                         type map changes (cast, round, decode, encode)
//   gpu_parquet_write()   the writer, footer key-values included
//   parquet_*netstring*   the footer encoding of factor levels
//   parquet_arrow_tz()    the time zones in a footer's ARROW:schema
//
// C++ never raises a condition R classifies: what R must report (a value
// that could not be converted, a rounding) is flagged in attributes on the
// returned pointer, like the collect path (F105's marker rule). libcudf's own
// exceptions pass through as C++ errors and R maps them.

#include "cuda_utils.hpp"
#include "gpu_table.hpp"
#include "r_strings.hpp"
#include "r_to_column.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join/join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/device_uvector.hpp>

#include <Rcpp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace Rcpp;

namespace {

const char* type_name(cudf::type_id id) {
    switch (id) {
        case cudf::type_id::FLOAT64: return "FLOAT64";
        case cudf::type_id::FLOAT32: return "FLOAT32";
        case cudf::type_id::INT64: return "INT64";
        case cudf::type_id::INT32: return "INT32";
        case cudf::type_id::INT16: return "INT16";
        case cudf::type_id::INT8: return "INT8";
        case cudf::type_id::BOOL8: return "BOOL8";
        case cudf::type_id::STRING: return "STRING";
        case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
        case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
        case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
        case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
        case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
        case cudf::type_id::UINT8: return "UINT8";
        case cudf::type_id::UINT16: return "UINT16";
        case cudf::type_id::UINT32: return "UINT32";
        case cudf::type_id::UINT64: return "UINT64";
        case cudf::type_id::DURATION_DAYS: return "DURATION_DAYS";
        case cudf::type_id::DURATION_SECONDS: return "DURATION_SECONDS";
        case cudf::type_id::DURATION_MILLISECONDS: return "DURATION_MILLISECONDS";
        case cudf::type_id::DURATION_MICROSECONDS: return "DURATION_MICROSECONDS";
        case cudf::type_id::DURATION_NANOSECONDS: return "DURATION_NANOSECONDS";
        case cudf::type_id::DECIMAL32: return "DECIMAL32";
        case cudf::type_id::DECIMAL64: return "DECIMAL64";
        case cudf::type_id::DECIMAL128: return "DECIMAL128";
        case cudf::type_id::DICTIONARY32: return "DICTIONARY32";
        case cudf::type_id::LIST: return "LIST";
        case cudf::type_id::STRUCT: return "STRUCT";
        default: return "UNKNOWN";
    }
}

std::string path_at(const CharacterVector& paths, R_xlen_t i) {
    return cuplyr::r_string_utf8(STRING_ELT(paths, i));
}

// kvikio, the file layer under libcudf's readers, defaults to probing for
// GPUDirect Storage: it loads libcufile, which writes a `cufile.log` into the
// working directory and starts cuFile's own threads in the process. Reads here
// are served by the page cache either way, so default to compatibility mode,
// unless the user chose a KVIKIO_COMPAT_MODE, even an empty one, which kvikio
// reads itself. Through the environment variable because kvikio's setter is out
// of reach: kvikio/defaults.hpp compiles neither with kvikio's stand-ins for
// cuFile (they leave out the batch API) nor against CUDA's cufile.h (its
// cuFileGetVersion is overloaded). A child process inherits the variable. This
// runs before the first file is opened.
void kvikio_defaults() {
    static std::once_flag once;
    std::call_once(once, [] {
        setenv("KVIKIO_COMPAT_MODE", "ON", 0);
    });
}

cudf::io::source_info make_source(const CharacterVector& paths) {
    kvikio_defaults();
    std::vector<std::string> v;
    v.reserve(paths.size());
    for (R_xlen_t i = 0; i < paths.size(); ++i) v.push_back(path_at(paths, i));
    return cudf::io::source_info(v);
}

// R strings cannot hold a NUL byte, and Rf_mkCharLenCE() raises an R error (a
// longjmp over this frame's C++ destructors) for one. Every string that comes
// out of a file is checked first.
bool has_nul(const std::string& s) { return std::memchr(s.data(), 0, s.size()) != nullptr; }

// UTF-8 marked character vector; throws a C++ error (which Rcpp turns into an
// R error after unwinding) for a string with a NUL byte.
SEXP utf8_strsxp(const std::vector<std::string>& v) {
    for (const auto& s : v) {
        if (has_nul(s)) Rcpp::stop("a string in the parquet footer contains a NUL byte");
    }
    Rcpp::CharacterVector out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        SET_STRING_ELT(out, i, cuplyr::r_mk_utf8_char(v[i].data(), static_cast<int>(v[i].size())));
    }
    return out;
}

// A join map (device_uvector<size_type>) as a column_view, no copy.
cudf::column_view map_view(const rmm::device_uvector<cudf::size_type>& map) {
    cudf::device_span<cudf::size_type const> span(map.data(), map.size());
    return cudf::column_view(span);
}

// The column's bits under another fixed-width type of the same size.
cudf::column_view reinterpret_as(const cudf::column_view& col, cudf::type_id id) {
    return cudf::column_view(cudf::data_type{id}, col.size(), col.head(), col.null_mask(),
                             col.null_count(), col.offset());
}

// A fixed-width column of `id` that takes over the buffers of `src`.
std::unique_ptr<cudf::column> retype(std::unique_ptr<cudf::column> src, cudf::type_id id) {
    const cudf::size_type n = src->size();
    const cudf::size_type nulls = src->null_count();
    auto contents = src->release();
    return std::make_unique<cudf::column>(
        cudf::data_type{id}, n, std::move(*contents.data),
        contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{}, nulls);
}

// Epoch nanoseconds to epoch microseconds, rounded half up (toward +inf); the
// remainder tells whether anything was lost. `lossy` is set when a value is not
// a whole microsecond. All in exact INT64 arithmetic: PYMOD is the floor
// remainder, so (v - m) is a multiple of 1000.
std::unique_ptr<cudf::column> ns_to_us(const cudf::column_view& col, bool& lossy) {
    const cudf::data_type i64{cudf::type_id::INT64};
    cudf::column_view v = reinterpret_as(col, cudf::type_id::INT64);
    cudf::numeric_scalar<int64_t> thousand(1000), half(500), zero(0);
    auto m = cudf::binary_operation(v, thousand, cudf::binary_operator::PYMOD, i64);
    auto ne = cudf::binary_operation(m->view(), zero, cudf::binary_operator::NOT_EQUAL,
                                     cudf::data_type{cudf::type_id::BOOL8});
    auto any = cudf::reduce(ne->view(),
                            *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::BOOL8});
    if (any->is_valid() && static_cast<cudf::numeric_scalar<bool>*>(any.get())->value()) {
        lossy = true;
    }
    auto floor_part = cudf::binary_operation(v, m->view(), cudf::binary_operator::SUB, i64);
    auto q = cudf::binary_operation(floor_part->view(), thousand, cudf::binary_operator::DIV, i64);
    auto up = cudf::binary_operation(m->view(), half, cudf::binary_operator::GREATER_EQUAL,
                                     cudf::data_type{cudf::type_id::BOOL8});
    auto up64 = cudf::cast(up->view(), i64);
    auto r = cudf::binary_operation(q->view(), up64->view(), cudf::binary_operator::ADD, i64);
    return retype(std::move(r), cudf::type_id::TIMESTAMP_MICROSECONDS);
}

// A STRING column as 1-based INT32 codes into `levels` (null where the string
// is null). `bad` is set when a non-null string is not a level.
std::unique_ptr<cudf::column> strings_to_codes(const cudf::column_view& col,
                                               const CharacterVector& levels, bool& bad) {
    const cudf::size_type n = col.size();
    const cudf::data_type i32{cudf::type_id::INT32};
    if (n == 0) return cudf::make_empty_column(i32);
    auto codes = cudf::make_numeric_column(i32, n, cudf::mask_state::ALL_NULL);
    if (levels.size() > 0 && col.null_count() < n) {
        auto lv = cuplyr::character_to_gpu(levels);
        auto [left, right] = cudf::inner_join(cudf::table_view(std::vector<cudf::column_view>{col}), cudf::table_view(std::vector<cudf::column_view>{lv->view()}),
                                              cudf::null_equality::UNEQUAL);
        cudf::numeric_scalar<int32_t> one(1);
        auto src = cudf::binary_operation(map_view(*right), one, cudf::binary_operator::ADD, i32);
        auto scattered = cudf::scatter(cudf::table_view(std::vector<cudf::column_view>{src->view()}), map_view(*left),
                                       cudf::table_view(std::vector<cudf::column_view>{codes->view()}));
        codes = std::move(scattered->release()[0]);
    }
    if (codes->null_count() > col.null_count()) bad = true;
    return codes;
}

// R's NA_integer_ and bit64's NA are INT_MIN / INT64_MIN, so R can't hold that
// value and the upload turns it into null (a GPU INT32/INT64 never holds it:
// F42 reads it as overflow). A file can hold it (arrow writes it as a plain
// value), so it becomes null here, after one min() per column: collect() would
// show NA for it anyway, and a filter or a sum must agree. Pushed predicates see
// the raw value, which R/pushdown.R accounts for.
template <typename T>
std::unique_ptr<cudf::column> null_int_min(std::unique_ptr<cudf::column> col) {
    if (col->size() == 0 || col->null_count() == col->size()) return col;
    auto mn = cudf::reduce(col->view(), *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
                           col->type());
    constexpr T lowest = std::numeric_limits<T>::min();
    if (!mn->is_valid() || static_cast<cudf::numeric_scalar<T>*>(mn.get())->value() != lowest) {
        return col;
    }
    cudf::numeric_scalar<T> sentinel(lowest);
    // null in, null out: the mask below keeps every row that is valid and not the sentinel
    auto keep = cudf::binary_operation(col->view(), sentinel, cudf::binary_operator::NOT_EQUAL,
                                       cudf::data_type{cudf::type_id::BOOL8});
    auto [mask, nulls] = cudf::bools_to_mask(keep->view());
    const cudf::data_type type = col->type();
    const cudf::size_type n = col->size();
    auto contents = col->release();
    return std::make_unique<cudf::column>(type, n, std::move(*contents.data), std::move(*mask), nulls);
}

// What the type map does to one reader column. `target` is the R-facing GPU
// type (R/io.R::parquet_type_map()); a column already of that type is passed
// through untouched.
struct Flags {
    std::vector<int> lossy;      // 1-based: values rounded (ns -> us, decimals)
    std::vector<int> u64_over;   // a UINT64 value past INT64_MAX
    std::vector<int> dec_over;   // a DECIMAL value past the double range
    std::vector<int> bad_factor; // a string that is not one of the stored levels
};

std::unique_ptr<cudf::column> normalise(std::unique_ptr<cudf::column> col, const std::string& target,
                                        SEXP levels, int idx1, Flags& fl) {
    const cudf::type_id from = col->type().id();
    const char* from_name = type_name(from);
    if (target == from_name) {
        if (from == cudf::type_id::INT32) return null_int_min<int32_t>(std::move(col));
        if (from == cudf::type_id::INT64) return null_int_min<int64_t>(std::move(col));
        return col;
    }

    auto cast_to = [&](cudf::type_id id) {
        return cudf::cast(col->view(), cudf::data_type{id});
    };
    if (target == "FLOAT64") {
        if (from == cudf::type_id::FLOAT32 || from == cudf::type_id::UINT32) {
            return cast_to(cudf::type_id::FLOAT64);
        }
        if (from == cudf::type_id::DECIMAL32 || from == cudf::type_id::DECIMAL64 ||
            from == cudf::type_id::DECIMAL128) {
            bool lossy = false, over = false;
            auto out = cuplyr::decimal_to_f64_column(col->view(), lossy, over);
            if (lossy) fl.lossy.push_back(idx1);
            if (over) fl.dec_over.push_back(idx1);
            return out;
        }
    } else if (target == "INT32") {
        if (from == cudf::type_id::INT8 || from == cudf::type_id::INT16 ||
            from == cudf::type_id::UINT8 || from == cudf::type_id::UINT16) {
            return cast_to(cudf::type_id::INT32);
        }
    } else if (target == "INT64") {
        if (from == cudf::type_id::UINT64) {
            auto mx = cudf::reduce(col->view(), *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
                                   col->type());
            if (mx->is_valid() &&
                static_cast<cudf::numeric_scalar<uint64_t>*>(mx.get())->value() >
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                fl.u64_over.push_back(idx1);
                return cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64}, col->size(),
                                                 cudf::mask_state::ALL_NULL);
            }
            return cast_to(cudf::type_id::INT64);
        }
    } else if (target == "TIMESTAMP_MICROSECONDS") {
        if (from == cudf::type_id::TIMESTAMP_SECONDS || from == cudf::type_id::TIMESTAMP_MILLISECONDS) {
            return cast_to(cudf::type_id::TIMESTAMP_MICROSECONDS);
        }
        if (from == cudf::type_id::TIMESTAMP_NANOSECONDS) {
            bool lossy = false;
            auto out = ns_to_us(col->view(), lossy);
            if (lossy) fl.lossy.push_back(idx1);
            return out;
        }
    } else if (target == "DICTIONARY32") {
        if (from == cudf::type_id::STRING && TYPEOF(levels) == STRSXP) {
            bool bad = false;
            auto out = strings_to_codes(col->view(), CharacterVector(levels), bad);
            if (bad) fl.bad_factor.push_back(idx1);
            return out;
        }
    }
    Rcpp::stop("parquet column %d: no conversion from %s to %s", idx1, from_name, target.c_str());
}

SEXP int_vec(const std::vector<int>& v) { return Rcpp::wrap(v); }

// libcudf erases ARROW:schema from the key-values it reports, so read it from
// the footer itself. The footer is thrift-compact; the KeyValue struct holds the
// key as field 1 (0x18, length, bytes) and the value as field 2 (0x18, varint
// length, bytes). The value is base64 text, and only base64 text is accepted, so
// a match inside another string cannot produce anything but "no information"
// (parquet_arrow_tz() validates what is left). Every length is checked against
// the buffer. Returns "" when there is no such entry.
std::string footer_arrow_schema(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < 12) return "";
    unsigned char tail[8];
    f.seekg(size - 8);
    f.read(reinterpret_cast<char*>(tail), 8);
    if (!f || std::memcmp(tail + 4, "PAR1", 4) != 0) return "";
    const uint64_t flen = static_cast<uint64_t>(tail[0]) | (static_cast<uint64_t>(tail[1]) << 8) |
                          (static_cast<uint64_t>(tail[2]) << 16) | (static_cast<uint64_t>(tail[3]) << 24);
    if (flen == 0 || flen > static_cast<uint64_t>(size) - 8 || flen > (256u << 20)) return "";
    std::string buf(flen, '\0');
    f.seekg(size - 8 - static_cast<std::streamoff>(flen));
    f.read(&buf[0], static_cast<std::streamsize>(flen));
    if (!f) return "";
    static const char key[] = "ARROW:schema";
    const size_t klen = sizeof(key) - 1;
    size_t at = 0;
    while ((at = buf.find(key, at, klen)) != std::string::npos) {
        size_t p = at + klen;
        at += 1;
        if (p >= buf.size() || static_cast<unsigned char>(buf[p]) != 0x18) continue;
        ++p;
        uint64_t len = 0;
        int shift = 0;
        bool ok = false;
        while (p < buf.size() && shift < 35) {
            const unsigned char c = static_cast<unsigned char>(buf[p++]);
            len |= static_cast<uint64_t>(c & 0x7F) << shift;
            shift += 7;
            if (!(c & 0x80)) { ok = true; break; }
        }
        if (!ok || len == 0 || len > buf.size() - p) continue;
        std::string v = buf.substr(p, static_cast<size_t>(len));
        bool b64 = true;
        for (unsigned char c : v) {
            if (!(std::isalnum(c) || c == '+' || c == '/' || c == '=')) { b64 = false; break; }
        }
        if (b64) return v;
    }
    return "";
}

}  // namespace

// Footer facts and the reader's column types for each file: a list with one
// entry per path (nrow, row_groups, names, types, kv). `kv` holds only the keys
// R reads: ARROW:schema and the cuplyr.* ones.
// [[Rcpp::export]]
List gpu_parquet_probe(CharacterVector paths) {
    kvikio_defaults();
    List out(paths.size());
    for (R_xlen_t i = 0; i < paths.size(); ++i) {
        cudf::io::source_info src(path_at(paths, i));
        auto md = cudf::io::read_parquet_metadata(src);

        // pandas metadata off: with it on the reader puts a pandas index column
        // first and returns it next to a selection, which is not the file's columns
        auto opts = cudf::io::parquet_reader_options::builder(src)
                        .use_pandas_metadata(false)
                        .num_rows(0)
                        .build();
        auto res = cudf::io::read_parquet(opts);
        const auto& info = res.metadata.schema_info;
        std::vector<std::string> nm, ty;
        for (size_t j = 0; j < info.size(); ++j) {
            nm.push_back(info[j].name);
            ty.push_back(type_name(res.tbl->view().column(static_cast<cudf::size_type>(j)).type().id()));
        }
        CharacterVector names = utf8_strsxp(nm), types = utf8_strsxp(ty);

        std::vector<std::string> kk, kvv;
        for (const auto& kv : md.metadata()) {
            if (kv.first == "ARROW:schema" || kv.first.rfind("cuplyr.", 0) == 0) {
                kk.push_back(kv.first);
                kvv.push_back(kv.second);
            }
        }
        if (std::string a = footer_arrow_schema(path_at(paths, i)); !a.empty()) {
            kk.push_back("ARROW:schema");
            kvv.push_back(a);
        }
        // (a value with a NUL byte is cut at it by the R string, which then
        // fails to decode: no information)
        CharacterVector kv(kvv.size());
        for (size_t j = 0; j < kvv.size(); ++j) kv[j] = kvv[j];
        kv.names() = utf8_strsxp(kk);

        out[i] = List::create(
            _["nrow"] = static_cast<double>(md.num_rows()),
            _["row_groups"] = static_cast<int>(md.num_rowgroups()),
            _["names"] = names, _["types"] = types, _["kv"] = kv);
    }
    return out;
}

namespace {

// The filter of a read: AST atoms from R, ANDed. Every atom is
// `col <op> literal`, the literal already converted to the column's file type by
// R/pushdown.R. Columns are referenced by name, so a projection can't shift them.
// The tree owns the nodes; `owned` keeps the literals' scalars alive for as long
// as the tree.
struct ReadFilter {
    cudf::ast::tree tree;
    std::vector<std::shared_ptr<void>> owned;
    const cudf::ast::expression* root = nullptr;
};

template <typename S, typename... A>
S& own_scalar(ReadFilter& f, A&&... a) {
    auto s = std::make_shared<S>(std::forward<A>(a)...);
    f.owned.push_back(s);
    return *s;
}

double whole_in(SEXP v, double lo, double hi, const char* what) {
    if (TYPEOF(v) != REALSXP || Rf_xlength(v) != 1) Rcpp::stop("a pushed %s literal must be one double", what);
    const double x = REAL(v)[0];
    if (!(x == std::floor(x)) || x < lo || x > hi) Rcpp::stop("a pushed %s literal is not exact", what);
    return x;
}

const cudf::ast::expression& filter_literal(ReadFilter& f, const std::string& type, SEXP v) {
    using cudf::ast::literal;
    if (type == "INT32") {
        return f.tree.push(literal{own_scalar<cudf::numeric_scalar<int32_t>>(
            f, static_cast<int32_t>(whole_in(v, -2147483648.0, 2147483647.0, "INT32")))});
    }
    if (type == "INT64") {
        // (R sends only |v| <= 2^53 and INT64_MIN, which a double holds exactly)
        return f.tree.push(literal{own_scalar<cudf::numeric_scalar<int64_t>>(
            f, static_cast<int64_t>(whole_in(v, -9223372036854775808.0, 9007199254740992.0, "INT64")))});
    }
    if (type == "TIMESTAMP_DAYS") {
        const int32_t d = static_cast<int32_t>(whole_in(v, -2147483648.0, 2147483647.0, "Date"));
        return f.tree.push(literal{own_scalar<cudf::timestamp_scalar<cudf::timestamp_D>>(
            f, cudf::duration_D{d}, true)});
    }
    if (type == "TIMESTAMP_MICROSECONDS") {
        const int64_t u = static_cast<int64_t>(whole_in(v, -9007199254740992.0, 9007199254740992.0, "POSIXct"));
        return f.tree.push(literal{own_scalar<cudf::timestamp_scalar<cudf::timestamp_us>>(
            f, cudf::duration_us{u}, true)});
    }
    Rcpp::stop("no pushed literal of type %s", type.c_str());
}

cudf::ast::ast_operator compare_op(const std::string& op) {
    using cudf::ast::ast_operator;
    if (op == "EQUAL") return ast_operator::EQUAL;
    if (op == "NOT_EQUAL") return ast_operator::NOT_EQUAL;
    if (op == "LESS") return ast_operator::LESS;
    if (op == "LESS_EQUAL") return ast_operator::LESS_EQUAL;
    if (op == "GREATER") return ast_operator::GREATER;
    if (op == "GREATER_EQUAL") return ast_operator::GREATER_EQUAL;
    Rcpp::stop("no pushed operator %s", op.c_str());
}

void build_filter(const List& atoms, ReadFilter& f) {
    using namespace cudf::ast;
    for (R_xlen_t k = 0; k < atoms.size(); ++k) {
        List a = atoms[k];
        const std::string col = cuplyr::r_scalar_utf8(a["col"]);
        const std::string op = as<std::string>(a["op"]);
        const expression& ref = f.tree.push(column_name_reference{col});
        const expression& lit = filter_literal(f, as<std::string>(a["type"]), a["value"]);
        const expression& atom = f.tree.push(operation{compare_op(op), ref, lit});
        f.root = f.root ? &f.tree.push(operation{ast_operator::LOGICAL_AND, *f.root, atom}) : &atom;
    }
}

}  // namespace

// Read `paths` (in order). `columns` NULL = every column; `num_rows` < 0 = all
// rows. `expect` is the name of each output column, in order: the reader's
// answer is checked against it. `targets` is the R-facing GPU type of each
// output column, `levels` the level vector of a factor column (NULL elsewhere),
// `filters` the atoms of the pushed predicate (see build_filter()). Flags for R
// are attributes on the returned pointer: rows_read, row_groups_total,
// row_groups_after_stats and row_groups_after_bloom (NA when that stage did not
// run), lossy, u64_over, dec_over,
// bad_factor.
// [[Rcpp::export]]
SEXP gpu_parquet_read(CharacterVector paths, SEXP columns, double num_rows,
                      CharacterVector targets, List levels, List filters,
                      CharacterVector expect) {
    using namespace cuplyr;

    ReadFilter filter;
    build_filter(filters, filter);

    // pandas metadata off (see gpu_parquet_probe())
    auto builder = cudf::io::parquet_reader_options::builder(make_source(paths)).use_pandas_metadata(false);
    if (!Rf_isNull(columns)) {
        CharacterVector cv(columns);
        std::vector<std::string> cols;
        for (R_xlen_t i = 0; i < cv.size(); ++i) cols.push_back(path_at(cv, i));
        builder.columns(cols);
    }
    if (num_rows >= 0) builder.num_rows(static_cast<int64_t>(num_rows));
    if (filter.root) builder.filter(*filter.root);
    auto res = cudf::io::read_parquet(builder.build());

    const auto& info = res.metadata.schema_info;
    auto cols = res.tbl->release();
    if (cols.size() != static_cast<size_t>(targets.size()) || info.size() != cols.size() ||
        cols.size() != static_cast<size_t>(expect.size())) {
        Rcpp::stop("parquet read returned %d columns, expected %d", static_cast<int>(cols.size()),
                   static_cast<int>(expect.size()));
    }
    for (size_t j = 0; j < cols.size(); ++j) {
        const std::string want = path_at(expect, static_cast<R_xlen_t>(j));
        if (info[j].name != want) {
            Rcpp::stop("parquet read returned column '%s' where '%s' was expected", info[j].name.c_str(),
                       want.c_str());
        }
    }
    Flags fl;
    std::vector<SharedColumn> out_cols;
    out_cols.reserve(cols.size());
    double rows = cols.empty() ? 0.0 : static_cast<double>(cols[0]->size());
    for (size_t j = 0; j < cols.size(); ++j) {
        out_cols.push_back(share_column(normalise(std::move(cols[j]), as<std::string>(targets[j]),
                                                  levels[j], static_cast<int>(j) + 1, fl)));
    }
    auto xp = make_gpu_table_xptr(std::move(out_cols));
    SEXP out = xp;
    Rcpp::Shield<SEXP> rr(Rcpp::wrap(rows));
    Rf_setAttrib(out, Rf_install("rows_read"), rr);
    Rcpp::Shield<SEXP> rgt(Rcpp::wrap(static_cast<int>(res.metadata.num_input_row_groups)));
    Rf_setAttrib(out, Rf_install("row_groups_total"), rgt);
    Rcpp::Shield<SEXP> rga(Rcpp::wrap(res.metadata.num_row_groups_after_stats_filter
                                          ? static_cast<int>(*res.metadata.num_row_groups_after_stats_filter)
                                          : NA_INTEGER));
    Rf_setAttrib(out, Rf_install("row_groups_after_stats"), rga);
    Rcpp::Shield<SEXP> rgb(Rcpp::wrap(res.metadata.num_row_groups_after_bloom_filter
                                          ? static_cast<int>(*res.metadata.num_row_groups_after_bloom_filter)
                                          : NA_INTEGER));
    Rf_setAttrib(out, Rf_install("row_groups_after_bloom"), rgb);
    struct { const char* nm; const std::vector<int>* v; } flags[] = {
        {"lossy", &fl.lossy}, {"u64_over", &fl.u64_over}, {"dec_over", &fl.dec_over},
        {"bad_factor", &fl.bad_factor}};
    for (const auto& f : flags) {
        if (f.v->empty()) continue;
        Rcpp::Shield<SEXP> iv(int_vec(*f.v));
        Rf_setAttrib(out, Rf_install(f.nm), iv);
    }
    return out;
}

// Write the table to `path`. `kv_keys`/`kv_vals` go in the footer.
// [[Rcpp::export]]
void gpu_parquet_write(SEXP xptr, CharacterVector names, std::string path, std::string compression,
                       int row_group_size, CharacterVector kv_keys, CharacterVector kv_vals) {
    using namespace cuplyr;
    kvikio_defaults();

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    cudf::io::compression_type ct;
    if (compression == "snappy") ct = cudf::io::compression_type::SNAPPY;
    else if (compression == "zstd") ct = cudf::io::compression_type::ZSTD;
    else if (compression == "lz4") ct = cudf::io::compression_type::LZ4;
    else if (compression == "none") ct = cudf::io::compression_type::NONE;
    else Rcpp::stop("unknown compression '%s'", compression.c_str());

    cudf::io::table_input_metadata meta(view);
    for (R_xlen_t i = 0; i < names.size() && i < static_cast<R_xlen_t>(meta.column_metadata.size()); ++i) {
        meta.column_metadata[i].set_name(r_string_utf8(STRING_ELT(names, i)));
    }
    std::map<std::string, std::string> kv;
    for (R_xlen_t i = 0; i < kv_keys.size(); ++i) {
        kv[r_string_utf8(STRING_ELT(kv_keys, i))] = r_string_utf8(STRING_ELT(kv_vals, i));
    }

    auto builder = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(path), view)
                       .metadata(std::move(meta))
                       .compression(ct)
                       .key_value_metadata({kv});
    if (row_group_size > 0) {
        // (row groups end at page-fragment boundaries, 5000 rows by default: a
        // smaller row group needs smaller fragments)
        builder.row_group_size_rows(row_group_size);
        builder.max_page_fragment_size(std::min<cudf::size_type>(row_group_size, 5000));
    }
    cudf::io::write_parquet(builder.build());
}

// Netstrings: "<bytes>:<utf8>" per string, concatenated, so a label can hold any
// character. R never sees raw footer bytes it has to trust: the decoder checks
// every length against the buffer and returns NULL for anything malformed.
// [[Rcpp::export]]
std::string parquet_encode_netstrings(CharacterVector x) {
    std::string out;
    for (R_xlen_t i = 0; i < x.size(); ++i) {
        std::string s = cuplyr::r_string_utf8(STRING_ELT(x, i));
        out += std::to_string(s.size());
        out += ':';
        out += s;
    }
    return out;
}

// [[Rcpp::export]]
SEXP parquet_decode_netstrings(std::string x) {
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < x.size()) {
        size_t colon = x.find(':', pos);
        if (colon == std::string::npos || colon == pos || colon - pos > 9) return R_NilValue;
        size_t len = 0;
        for (size_t k = pos; k < colon; ++k) {
            if (x[k] < '0' || x[k] > '9') return R_NilValue;
            len = len * 10 + static_cast<size_t>(x[k] - '0');
        }
        if (len > x.size() - colon - 1) return R_NilValue;
        parts.push_back(x.substr(colon + 1, len));
        pos = colon + 1 + len;
        // (an R string can't hold a NUL byte: no information, not an R error)
        if (has_nul(parts.back())) return R_NilValue;
    }
    return utf8_strsxp(parts);
}

namespace {

// Standard base64; whitespace and padding are skipped, anything else fails.
bool base64_decode(const std::string& in, std::vector<uint8_t>& out) {
    uint32_t val = 0;
    int bits = -8;
    for (unsigned char c : in) {
        int d;
        if (c >= 'A' && c <= 'Z') d = c - 'A';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
        else if (c >= '0' && c <= '9') d = c - '0' + 52;
        else if (c == '+') d = 62;
        else if (c == '/') d = 63;
        else if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        else return false;
        val = (val << 6) | static_cast<uint32_t>(d);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return true;
}

// A bounds-checked reader over a flatbuffer. Every accessor says false instead
// of reading outside the buffer; nothing here is trusted.
struct Fb {
    const std::vector<uint8_t>& b;
    bool u8(size_t p, uint8_t& v) const {
        if (p >= b.size()) return false;
        v = b[p];
        return true;
    }
    bool u16(size_t p, uint16_t& v) const {
        if (p > b.size() || b.size() - p < 2) return false;
        v = static_cast<uint16_t>(b[p] | (b[p + 1] << 8));
        return true;
    }
    bool u32(size_t p, uint32_t& v) const {
        if (p > b.size() || b.size() - p < 4) return false;
        v = static_cast<uint32_t>(b[p]) | (static_cast<uint32_t>(b[p + 1]) << 8) |
            (static_cast<uint32_t>(b[p + 2]) << 16) | (static_cast<uint32_t>(b[p + 3]) << 24);
        return true;
    }
    // Position of field `idx` of the table at `t`; false when the field is absent.
    bool field(size_t t, int idx, size_t& pos) const {
        uint32_t soff;
        if (!u32(t, soff)) return false;
        int64_t vt = static_cast<int64_t>(t) - static_cast<int32_t>(soff);
        if (vt < 0) return false;
        uint16_t vt_size;
        if (!u16(static_cast<size_t>(vt), vt_size)) return false;
        if (4 + 2 * (idx + 1) > vt_size) return false;
        uint16_t off;
        if (!u16(static_cast<size_t>(vt) + 4 + 2 * idx, off) || off == 0) return false;
        if (t + off >= b.size()) return false;
        pos = t + off;
        return true;
    }
    // The object an offset field at `p` points to.
    bool indirect(size_t p, size_t& target) const {
        uint32_t off;
        if (!u32(p, off)) return false;
        if (static_cast<uint64_t>(p) + off >= b.size()) return false;
        target = p + off;
        return true;
    }
    bool string(size_t p, std::string& s) const {
        size_t sp;
        uint32_t len;
        if (!indirect(p, sp) || !u32(sp, len)) return false;
        if (sp + 4 > b.size() || len > b.size() - sp - 4) return false;
        s.assign(reinterpret_cast<const char*>(&b[sp + 4]), len);
        return true;
    }
};

}  // namespace

// The time zone of every top-level Timestamp field in an ARROW:schema value
// (base64 of an Arrow IPC schema message). Returns list(names, tz) where tz is
// NA for a timestamp with no zone (isAdjustedToUTC = false). Anything that
// does not parse gives an empty list: "no information", never an error.
// [[Rcpp::export]]
List parquet_arrow_tz(std::string b64) {
    List none = List::create(_["names"] = CharacterVector(0), _["tz"] = CharacterVector(0));
    std::vector<uint8_t> buf;
    if (!base64_decode(b64, buf) || buf.size() < 16) return none;
    Fb fb{buf};

    size_t start = 0;
    uint32_t first;
    if (!fb.u32(0, first)) return none;
    if (first == 0xFFFFFFFFu) start = 8;  // continuation marker + metadata length
    else start = 4;                       // pre-0.15 framing: length only

    size_t msg;
    if (!fb.indirect(start, msg)) return none;
    // Message: version(0), header_type(1), header(2). header_type 1 = Schema.
    size_t p;
    uint8_t htype;
    if (!fb.field(msg, 1, p) || !fb.u8(p, htype) || htype != 1) return none;
    size_t schema;
    if (!fb.field(msg, 2, p) || !fb.indirect(p, schema)) return none;
    // Schema: endianness(0), fields(1)
    size_t fields;
    uint32_t nfields;
    if (!fb.field(schema, 1, p) || !fb.indirect(p, fields) || !fb.u32(fields, nfields)) return none;
    if (nfields > buf.size() / 4) return none;

    std::vector<std::string> names, tzs;
    std::vector<bool> has_tz;
    for (uint32_t i = 0; i < nfields; ++i) {
        size_t f;
        if (!fb.indirect(fields + 4 + 4 * static_cast<size_t>(i), f)) return none;
        // Field: name(0), nullable(1), type_type(2), type(3)
        uint8_t ttype;
        if (!fb.field(f, 2, p) || !fb.u8(p, ttype) || ttype != 10) continue;  // 10 = Timestamp
        std::string name;
        if (!fb.field(f, 0, p) || !fb.string(p, name) || has_nul(name)) continue;
        std::string tz;
        bool tz_set = false;
        size_t ts;
        if (fb.field(f, 3, p) && fb.indirect(p, ts)) {
            // Timestamp: unit(0), timezone(1)
            size_t tp;
            if (fb.field(ts, 1, tp) && fb.string(tp, tz) && !tz.empty()) tz_set = true;
        }
        if (has_nul(tz)) continue;  // (no information for this column)
        names.push_back(name);
        tzs.push_back(tz);
        has_tz.push_back(tz_set);
    }
    // (no string here has a NUL byte, so building the R strings cannot raise)
    CharacterVector rn(names.size()), rtz(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        rn[i] = cuplyr::r_mk_utf8_char(names[i].data(), static_cast<int>(names[i].size()));
        rtz[i] = has_tz[i] ? cuplyr::r_mk_utf8_char(tzs[i].data(), static_cast<int>(tzs[i].size()))
                           : NA_STRING;
    }
    return List::create(_["names"] = rn, _["tz"] = rtz);
}

// Row count of each file from the footer alone: no column is decoded and
// nothing is allocated on the device. Summed over the paths.
// [[Rcpp::export]]
double gpu_parquet_nrow(CharacterVector paths) {
    kvikio_defaults();
    double n = 0;
    for (R_xlen_t i = 0; i < paths.size(); ++i) {
        n += static_cast<double>(cudf::io::read_parquet_metadata(cudf::io::source_info(path_at(paths, i))).num_rows());
    }
    return n;
}

// What a read of the first `n` rows has to fetch, from the footers alone: the
// largest row count among the row groups that hold those rows (libcudf decodes
// whole row groups, so that is the size of the read, not `n`), or Inf when one of
// them holds more than `max_bytes` of column data. Files are read in order; 0 for
// an empty file set.
// [[Rcpp::export]]
double gpu_parquet_preview_rows(CharacterVector paths, double n, double max_bytes) {
    kvikio_defaults();
    double need = n, widest = 0;
    for (R_xlen_t i = 0; i < paths.size() && need > 0; ++i) {
        auto md = cudf::io::read_parquet_metadata(cudf::io::source_info(path_at(paths, i)));
        for (const auto& rg : md.rowgroup_metadata()) {
            if (need <= 0) break;
            auto it = rg.find("num_rows");
            if (it == rg.end()) return std::numeric_limits<double>::infinity();  // (unknown: not cheap)
            auto bytes = rg.find("total_byte_size");
            if (bytes != rg.end() && static_cast<double>(bytes->second) > max_bytes) {
                return std::numeric_limits<double>::infinity();
            }
            const double rows = static_cast<double>(it->second);
            widest = std::max(widest, rows);
            need -= rows;
        }
    }
    return widest;
}
