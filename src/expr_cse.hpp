// src/expr_cse.hpp
//
// Common subexpressions (#17): marks the call subtrees of the IR trees one
// evaluation (one mutate() call, or the one predicate of a filter()) will
// materialize more than once, with an integer `cse` id. The evaluator
// (expr_eval.hpp) keeps a memo from id to the slot that holds the value, so a
// tagged node is computed on first use and read from its slot after that.
//
// The pass reads the IR as it arrives from R through the C API. It lists the
// calls first and stops there unless two of them share an op name (two calls
// can only be equal when they do), which is the verb's common case: nothing is
// hashed, nothing allocated but one small vector, and the input SEXP comes
// back untouched (a verb costs ~0.4 ms; this is ~0.1-0.5 us). Otherwise every
// call gets a 64-bit structural hash bottom-up, and the input is again
// untouched unless two calls share a hash.
//
// A hash decides nothing: equal hashes are confirmed with R_compute_identical()
// on the two subtrees, with the flags identical(num.eq = FALSE, single.NA =
// FALSE) sets (read off do_identical() in R's identical.c: the other options
// are at their defaults, so IDENT_USE_CLOENV too), i.e. bitwise doubles and
// NA payloads that differ. A literal 0 and -0, or NA and NaN, are different
// subtrees; a bound `col` is its index, so a column overwritten between two
// uses (an appended index) differs from the original. The hash only has to be
// equal for identical trees, so it may be coarse (a long literal vector is
// hashed by its length and first elements).
//
// The tags are put on COPIES of the nodes (the R side keeps its own
// references to the IR), so only a pass that finds a repeat builds anything.
// Both gpu_mutate_expr()'s remap_virtual_col_indices() (it clones each node,
// so the tag rides along) and the evaluator see the tagged copies.
#ifndef CUPLYR_EXPR_CSE_HPP
#define CUPLYR_EXPR_CSE_HPP

#include <Rcpp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace cuplyr {
namespace cse {

// Measurement hook (gpu_cse_enable()): with the memo off nothing is tagged.
inline bool& enabled() {
    static bool on = true;
    return on;
}

// identical(num.eq = FALSE, single.NA = FALSE): bitwise numbers, NA payloads
// kept apart; everything else at identical()'s defaults (ignore.environment =
// FALSE sets IDENT_USE_CLOENV).
constexpr int kIdenticalFlags = IDENT_NUM_AS_BITS | IDENT_NA_AS_BITS | IDENT_USE_CLOENV;

constexpr R_xlen_t kHashElements = 64;   // elements of a vector that feed its hash
constexpr int kHashDepth = 8;

inline uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

inline uint64_t hash_chars(const char* s) {
    uint64_t h = 1469598103934665603ULL;
    for (; *s; ++s) h = (h ^ static_cast<unsigned char>(*s)) * 1099511628211ULL;
    return h;
}

// Hashes the bytes: equal for two identical() objects whose strings have the
// same encoding (bitwise numbers). Strings that are identical() in different
// encodings hash apart, which only loses a share; unequal objects may collide.
inline uint64_t hash_sexp(SEXP x, int depth = 0) {
    uint64_t h = mix(0, static_cast<uint64_t>(TYPEOF(x)));
    switch (TYPEOF(x)) {
    case LGLSXP:
    case INTSXP: {
        R_xlen_t n = XLENGTH(x);
        h = mix(h, static_cast<uint64_t>(n));
        const int* p = INTEGER(x);
        for (R_xlen_t i = 0; i < std::min(n, kHashElements); ++i) h = mix(h, static_cast<uint64_t>(p[i]));
        break;
    }
    case REALSXP: {
        R_xlen_t n = XLENGTH(x);
        h = mix(h, static_cast<uint64_t>(n));
        const double* p = REAL(x);
        for (R_xlen_t i = 0; i < std::min(n, kHashElements); ++i) {
            uint64_t bits;
            std::memcpy(&bits, p + i, sizeof bits);
            h = mix(h, bits);
        }
        break;
    }
    case STRSXP: {
        R_xlen_t n = XLENGTH(x);
        h = mix(h, static_cast<uint64_t>(n));
        for (R_xlen_t i = 0; i < std::min(n, kHashElements); ++i) {
            SEXP s = STRING_ELT(x, i);
            h = mix(h, s == NA_STRING ? 7 : hash_chars(CHAR(s)));
        }
        break;
    }
    case VECSXP: {
        R_xlen_t n = XLENGTH(x);
        h = mix(h, static_cast<uint64_t>(n));
        if (depth < kHashDepth) {
            for (R_xlen_t i = 0; i < std::min(n, kHashElements); ++i) {
                h = mix(h, hash_sexp(VECTOR_ELT(x, i), depth + 1));
            }
        }
        break;
    }
    case CPLXSXP:
    case RAWSXP:
        h = mix(h, static_cast<uint64_t>(XLENGTH(x)));
        break;
    default:
        return h;
    }
    if (depth < kHashDepth) {
        SEXP nm = Rf_getAttrib(x, R_NamesSymbol);
        if (TYPEOF(nm) == STRSXP) h = mix(h, hash_sexp(nm, depth + 1));
    }
    return h;
}

// The named elements of an IR node that matter here; looked up in one pass
// over its names.
struct node_fields {
    SEXP kind = R_NilValue, op = R_NilValue, args = R_NilValue, meta = R_NilValue;
    SEXP index = R_NilValue, value = R_NilValue, type = R_NilValue, na = R_NilValue;
    SEXP cse = R_NilValue;
    R_xlen_t args_at = -1;   // position of `args` in the node
};

inline node_fields scan(SEXP node) {
    node_fields f;
    SEXP nm = Rf_getAttrib(node, R_NamesSymbol);
    if (TYPEOF(nm) != STRSXP) return f;
    R_xlen_t n = std::min(XLENGTH(node), XLENGTH(nm));
    for (R_xlen_t i = 0; i < n; ++i) {
        const char* s = CHAR(STRING_ELT(nm, i));
        SEXP v = VECTOR_ELT(node, i);
        switch (s[0]) {
        case 'k': if (!std::strcmp(s, "kind")) f.kind = v; break;
        case 'o': if (!std::strcmp(s, "op")) f.op = v; break;
        case 'a': if (!std::strcmp(s, "args")) { f.args = v; f.args_at = i; } break;
        case 'c': if (!std::strcmp(s, "cse")) f.cse = v; break;
        case 'm': if (!std::strcmp(s, "meta")) f.meta = v; break;
        case 'i': if (!std::strcmp(s, "index")) f.index = v; break;
        case 'v': if (!std::strcmp(s, "value")) f.value = v; break;
        case 't': if (!std::strcmp(s, "type")) f.type = v; break;
        // (a bare NA literal's flag; `na` is also the name of a column, but
        // never on a call/col node)
        case 'n': if (!std::strcmp(s, "na")) f.na = v; break;
        }
    }
    return f;
}

inline bool kind_is(SEXP kind, const char* what) {
    return TYPEOF(kind) == STRSXP && XLENGTH(kind) == 1 && !std::strcmp(CHAR(STRING_ELT(kind, 0)), what);
}

// A call node: a list with kind "call" and a list of arguments. (Anything
// else is a leaf, whatever it is.)
inline bool is_call(SEXP node, node_fields& f) {
    if (TYPEOF(node) != VECSXP) return false;
    f = scan(node);
    return kind_is(f.kind, "call") && TYPEOF(f.args) == VECSXP;
}

inline uint64_t leaf_hash(SEXP node) {
    if (TYPEOF(node) != VECSXP) return hash_sexp(node);
    node_fields f = scan(node);
    uint64_t h = 0x51;
    if (kind_is(f.kind, "col")) {
        return mix(mix(h, 1), hash_sexp(f.index));
    }
    if (kind_is(f.kind, "lit")) {
        h = mix(h, 2);
        h = mix(h, hash_sexp(f.value));
        h = mix(h, hash_sexp(f.type));
        return mix(h, hash_sexp(f.na));
    }
    return mix(h, hash_sexp(f.kind));
}

struct occurrence {
    SEXP node;
    uint64_t hash;
    int parent;       // index of the enclosing call's occurrence, -1 at the top
    const char* op;   // the call's op name ("" if it has none)
};

class marker {
public:
    // `roots` is a list of IR trees, or one IR tree when `single`. Returns
    // false when nothing repeats (the input stays as it is, and nothing but
    // two small vectors was allocated); otherwise `tagged` is the list of the
    // trees (tagged copies where needed, the originals elsewhere).
    bool run(SEXP roots, bool single, Rcpp::RObject& tagged) {
        single_ = single;
        roots_ = roots;
        const R_xlen_t n_roots = single ? 1 : XLENGTH(roots);
        occs_.reserve(8);
        for (R_xlen_t i = 0; i < n_roots; ++i) collect(root(i), -1);
        const size_t n = occs_.size();
        if (n < 2) return false;

        // The verb's common case ends here, before anything is hashed: two
        // calls can only be equal when they have the same op.
        if (!ops_repeat()) return false;

        // Hash every call bottom-up; any two equal?
        int cursor = 0;
        for (R_xlen_t i = 0; i < n_roots; ++i) hash_walk(root(i), cursor);
        if (!hash_repeats()) return false;

        // Classes of identical subtrees: an equal hash, confirmed.
        std::vector<int> cls(n);
        std::vector<int> rep;       // a member of each class
        std::vector<int> count;     // occurrences of each class
        std::unordered_multimap<uint64_t, int> by_hash;
        for (size_t i = 0; i < n; ++i) {
            int found = -1;
            auto range = by_hash.equal_range(occs_[i].hash);
            for (auto it = range.first; it != range.second && found < 0; ++it) {
                if (R_compute_identical(occs_[rep[it->second]].node, occs_[i].node, kIdenticalFlags)) {
                    found = it->second;
                }
            }
            if (found < 0) {
                found = static_cast<int>(rep.size());
                rep.push_back(static_cast<int>(i));
                count.push_back(0);
                by_hash.emplace(occs_[i].hash, found);
            }
            cls[i] = found;
            ++count[found];
        }

        // Maximal repeats: a class that occurs at least twice and, in some
        // occurrence, whose parent occurs fewer times than it does (a
        // top-level occurrence has parent count 0). A repeated chain is
        // tagged at its root, not at every inner node.
        std::vector<int> id(rep.size(), 0);
        int next_id = 0;
        for (size_t i = 0; i < n; ++i) {
            const int c = cls[i];
            if (count[c] < 2 || id[c] > 0) continue;
            const int pc = occs_[i].parent < 0 ? 0 : count[cls[occs_[i].parent]];
            if (pc < count[c]) id[c] = ++next_id;
        }
        if (next_id == 0) return false;

        // Rebuild the trees (the same pre-order as walk()).
        occ_id_.resize(n);
        for (size_t i = 0; i < n; ++i) occ_id_[i] = id[cls[i]];
        Rcpp::List out(static_cast<R_xlen_t>(n_roots));
        cursor = 0;
        for (R_xlen_t i = 0; i < n_roots; ++i) {
            Rcpp::RObject t = rebuild(root(i), cursor);
            SET_VECTOR_ELT(out, i, t);
        }
        tagged = out;
        return true;
    }

private:
    bool single_ = false;
    SEXP roots_ = R_NilValue;
    std::vector<occurrence> occs_;
    std::vector<int> occ_id_;

    SEXP root(R_xlen_t i) const { return single_ ? roots_ : VECTOR_ELT(roots_, i); }

    // Records every call, in pre-order, with its parent.
    void collect(SEXP node, int parent) {
        node_fields f;
        if (!is_call(node, f)) return;
        const int me = static_cast<int>(occs_.size());
        const char* op = TYPEOF(f.op) == STRSXP && XLENGTH(f.op) == 1 ? CHAR(STRING_ELT(f.op, 0)) : "";
        occs_.push_back({node, 0, parent, op});
        for (R_xlen_t j = 0; j < XLENGTH(f.args); ++j) collect(VECTOR_ELT(f.args, j), me);
    }

    // Do two calls share an op name? (a long block says yes without looking)
    bool ops_repeat() const {
        const size_t n = occs_.size();
        if (n > 32) return true;
        for (size_t i = 1; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) {
                if (!std::strcmp(occs_[i].op, occs_[j].op)) return true;
            }
        }
        return false;
    }

    bool hash_repeats() const {
        const size_t n = occs_.size();
        if (n <= 32) {
            for (size_t i = 1; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if (occs_[i].hash == occs_[j].hash) return true;
                }
            }
            return false;
        }
        std::vector<uint64_t> hs(n);
        for (size_t i = 0; i < n; ++i) hs[i] = occs_[i].hash;
        std::sort(hs.begin(), hs.end());
        return std::adjacent_find(hs.begin(), hs.end()) != hs.end();
    }

    // The structural hash of `node`, filling in the hash of every call under
    // it (`cursor` is the pre-order index collect() gave the next call).
    uint64_t hash_walk(SEXP node, int& cursor) {
        node_fields f;
        if (!is_call(node, f)) return leaf_hash(node);
        const int me = cursor++;
        uint64_t h = mix(hash_sexp(f.op), static_cast<uint64_t>(XLENGTH(f.args)));
        for (R_xlen_t j = 0; j < XLENGTH(f.args); ++j) {
            h = mix(h, hash_walk(VECTOR_ELT(f.args, j), cursor));
        }
        SEXP arg_names = Rf_getAttrib(f.args, R_NamesSymbol);
        if (TYPEOF(arg_names) == STRSXP) h = mix(h, hash_sexp(arg_names));
        if (TYPEOF(f.meta) == VECSXP && XLENGTH(f.meta) > 0) h = mix(h, hash_sexp(f.meta));
        occs_[me].hash = h;
        return h;
    }

    // `node` with its arguments replaced (when any changed) and tagged with
    // its occurrence's id (when it has one); `node` itself when neither.
    Rcpp::RObject rebuild(SEXP node, int& cursor) {
        node_fields f;
        if (!is_call(node, f)) return Rcpp::RObject(node);
        const int me = cursor++;
        Rcpp::RObject new_args;   // a shallow copy of `args`, once a child changed
        for (R_xlen_t j = 0; j < XLENGTH(f.args); ++j) {
            SEXP a = VECTOR_ELT(f.args, j);
            Rcpp::RObject r = rebuild(a, cursor);
            if (SEXP(r) != a) {
                if (SEXP(new_args) == R_NilValue) new_args = Rf_shallow_duplicate(f.args);
                SET_VECTOR_ELT(new_args, j, r);
            }
        }
        const int tag = occ_id_[me];
        const bool args_changed = SEXP(new_args) != R_NilValue;
        if (!args_changed && tag == 0) return Rcpp::RObject(node);

        const R_xlen_t len = XLENGTH(node);
        Rcpp::Shield<SEXP> out(Rf_allocVector(VECSXP, len + (tag > 0 ? 1 : 0)));
        Rcpp::Shield<SEXP> names(Rf_allocVector(STRSXP, XLENGTH(out)));
        SEXP old_names = Rf_getAttrib(node, R_NamesSymbol);
        for (R_xlen_t i = 0; i < len; ++i) {
            SEXP v = VECTOR_ELT(node, i);
            SET_STRING_ELT(names, i, STRING_ELT(old_names, i));
            if (args_changed && i == f.args_at) v = new_args;
            SET_VECTOR_ELT(out, i, v);
        }
        if (tag > 0) {
            SET_VECTOR_ELT(out, len, Rf_ScalarInteger(tag));
            SET_STRING_ELT(names, len, Rf_mkChar("cse"));
        }
        Rf_copyMostAttrib(node, out);
        Rf_setAttrib(out, R_NamesSymbol, names);
        return Rcpp::RObject(out);
    }
};

// IR trees of one evaluation -> the same trees with the repeats tagged; `irs`
// itself when nothing repeats.
inline Rcpp::List mark_list(Rcpp::List const& irs) {
    if (!enabled() || irs.size() < 1) return irs;
    marker m;
    Rcpp::RObject tagged;
    return m.run(irs, false, tagged) ? Rcpp::List(tagged) : irs;
}

// The same for one tree.
inline Rcpp::List mark_one(Rcpp::List const& ir) {
    if (!enabled()) return ir;
    marker m;
    Rcpp::RObject tagged;
    return m.run(ir, true, tagged) ? Rcpp::List(VECTOR_ELT(tagged, 0)) : ir;
}

} // namespace cse
} // namespace cuplyr

#endif // CUPLYR_EXPR_CSE_HPP
