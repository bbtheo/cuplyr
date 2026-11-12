// src/ops_rolling.cpp
//
// F59: rolling windows whose cost does not depend on the window width.
//
// cudf::grouped_rolling_window() rescans every window: about 0.04 ms x w per
// 1e7 rows for mean/sum/min/max, 0.5 ms x w for std. Past a few hundred rows
// that loses to a constant number of passes over the column, so this file
// computes all five from segmented scans. src/ops_window.cpp keeps cudf's
// kernel for the narrow windows where it is the faster of the two
// (roll_native() holds the measured crossovers). roll_sd always comes from
// here: cudf's STD loses more than 1e-8 on data with a large offset, at every
// width.
// Everything uses cudf host APIs (groupby::scan, cudf::scan, shift, gather,
// compute_column, copy_if_else): no device code. Elementwise steps are
// compute_column ASTs (a cudf::binary_operation costs about three small
// ASTs). Per 1e7 rows an AST costs about 0.35 ms plus 0.08 ms per operator,
// and three times that over a nullable input, so every AST input here is
// null-free and long formulas are split where a subexpression repeats.
//
// Input: the frame's `work` order (sorted by group keys, original order inside
// a group), with NaN already read as null (F55). A window is the w rows
// ending at row i. It never crosses a group once r[i] >= w - 1 (r = 0-based
// row in the group); the other rows are NA.
//
// Float sums (sum, mean, sd). A prefix sum over the whole column loses
// precision (a long series, or a group at 1e9 following a group at 1e-3), so
// each group is cut into segments of w rows (segment k starts at the rows with
// row-in-group % w == 0). A window [s, i] is the head [j0, i] of row i's
// segment plus the tail [s, e] of the previous one (e = j0 - 1), or exactly
// one segment when r % w == w - 1 (the "gate" is then 0):
//     head = a forward segmented scan (it restarts at every segment);
//     tail = a segmented scan read at s = i - w + 1, a suffix sum. A forward
//            scan of the "mirrored" column (row j0 + k replaced by row
//            j0 + w - 1 - k, a within-segment reversal) is a suffix scan, so
//            head and tail share one sorted groupby, whose labels are
//            computed once per width.
// Both parts are sums of window members only, so every row is accurate
// relative to its own window however large its neighbours are. A tail is
// always a complete segment (a row with r >= w - 1 and r % w < w - 1 has
// r >= w). Ungrouped, segments are aligned multiples of w, so the per-row
// columns (segment key, gate, mirror and read offsets, first values) are
// tiles and repeats of a few values.
//
// Missing and infinite values. x0 is the value column with every NA, NaN and
// +-Inf replaced by 0. A window holding none of them only reads present
// values, so three computations cover every request:
//   C  mean / sum: plain head and tail sums of x0. A filled 0 adds nothing,
//      so one pass serves both na.rm settings; mean divides by w, or by the
//      window's count of finite values under na.rm = TRUE.
//   A  sd whose valid windows hold no NA (na.rm = FALSE, or a column without
//      NA): both parts centred on x0[j0], the first value of row i's segment
//      (the tail of segment k - 1 on the first value of segment k), a member
//      of every window ending in segment k. A window holding a filled value
//      also holds the filled row, so it is masked or overridden by the Inf
//      rule.
//   B  sd under na.rm = TRUE on a column with NA: a part's first or last
//      value may be missing, so each part is centred on its own finite
//      member, the head on cH (the first finite value of its segment), the
//      tail on cT (the last finite value of its segment), and the tail is
//      moved onto cH by dd = cT - cH (Chan's merge as a recentring) where the
//      head has a finite value; a window without one is the tail alone, about
//      cT. The centred value at a segment's last row is 0 (it is cT itself,
//      or missing), so the mirrored scan is exactly 0 at a segment's first
//      row: a row whose window is its own segment reads its tail there, and B
//      needs no gate.
// The references are members of the window, so |y| <= the window's range and
// M2 = S2 - S1 (S1 / n) loses at most log10(n) digits; |S1 / n| is at most
// the range too, so no term is larger than S2. A variance that rounding left
// slightly negative is clamped to 0.
//
// Invariant: every sum, difference and square a valid window's result is
// built from involves only members of the window. The column's range bounds
// those and nothing else: a value read from outside the window (the next
// segment's first value, the next group's first row, a filled 0, the
// reference of an empty part) may be arbitrarily far from them, and its
// square or its difference with a member may overflow. So no such value is
// ever discarded by multiplying it by a 0 gate or a zero count (0 * Inf is
// NaN): it is selected away with copy_if_else (A's tail on gate-0 rows, B's
// merge offset e). The only products with a 0 factor left multiply a single
// finite input value, exactly 0: C's gate times x0[e] (plain_sums()) and B's
// reference times a missing row's validity (general_sd()).
//
// Overflow. When the column's finite range r satisfies 8 w r^2 <= DBL_MAX no
// sum A or B forms can overflow (by the invariant they are sums over window
// members). Otherwise A's mean / sum come from C instead, and a window whose
// sums of squares overflow comes out non-finite and is recomputed by B on
// x * 2^-k (exact), its variance
// scaled back by 2^k twice before the sqrt: R's var() accumulates in long
// double and is +Inf exactly when the variance itself leaves the double range
// (sd(c(3, 1e200)) is Inf, the sd of +-1e152 over 1000 rows is 1e152).
//
// NA / Inf rules follow R/roll.R (the reference): a NaN is missing (null);
// na.rm = FALSE gives NA for a window holding a null, na.rm = TRUE drops nulls
// and gives NA for a window without a valid value (sd: fewer than two); an
// Inf is a value (sum/mean: +-Inf, or NaN for both signs; sd: NaN), kept out
// of the sums and counted separately. Window counts (present values, finite
// values, +Inf, -Inf) come from ordinary (ungrouped) inclusive INT32 prefix
// counts C: count over [s, i] = C[i] - C[s - 1], with C[s - 1] = shift(C, w)[i].
// Integer sums cannot lose precision, so they use the same prefix trick on the
// values (INT64).
//
// Min / max (van Herk / Gil-Werman): the same head and mirrored tail scans
// with MIN / MAX instead of SUM; a missing value is the type's extreme on the
// losing side, and its windows are masked by the same counts as the sums'.
//
// Memory: each N-row temporary is freed after its last use.

#include "ops_rolling.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <Rcpp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>

namespace cuplyr {

namespace {
using col_ptr = std::unique_ptr<cudf::column>;
using Op = cudf::ast::ast_operator;
using Expr = cudf::ast::expression;

const cudf::data_type kInt32{cudf::type_id::INT32};
const cudf::data_type kInt64{cudf::type_id::INT64};
const cudf::data_type kFloat64{cudf::type_id::FLOAT64};
const cudf::data_type kBool{cudf::type_id::BOOL8};
constexpr double kInf = std::numeric_limits<double>::infinity();

// The largest finite range of a column for which no window sum of squares
// (nor any term of B's merge, at most 5 w r^2) can overflow at width w.
double safe_range(int w) {
    return std::sqrt(std::numeric_limits<double>::max() / (8.0 * w));
}

// Small expression-tree helper: keeps literals alive next to the tree. A node
// used twice is evaluated twice (the AST has no common subexpressions), so a
// value needed more than once and costing more than a few operations is
// materialised by a separate eval().
struct AstBuilder {
    cudf::ast::tree tree;
    std::deque<cudf::numeric_scalar<double>> dlits;
    std::deque<cudf::numeric_scalar<int32_t>> ilits;

    const Expr& col(int i) { return tree.emplace<cudf::ast::column_reference>(i); }
    const Expr& lit(double v) {
        dlits.emplace_back(v, true);
        return tree.emplace<cudf::ast::literal>(dlits.back());
    }
    const Expr& ilit(int32_t v) {
        ilits.emplace_back(v, true);
        return tree.emplace<cudf::ast::literal>(ilits.back());
    }
    const Expr& op(Op o, const Expr& a) { return tree.emplace<cudf::ast::operation>(o, a); }
    const Expr& op(Op o, const Expr& a, const Expr& b) {
        return tree.emplace<cudf::ast::operation>(o, a, b);
    }
    const Expr& to_f64(const Expr& a) { return op(Op::CAST_TO_FLOAT64, a); }
};

col_ptr eval(const std::vector<cudf::column_view>& cols, const Expr& e) {
    return cudf::compute_column(cudf::table_view(cols), e);
}

col_ptr copy_col(cudf::column_view c) { return std::make_unique<cudf::column>(c); }

// shift(c, k) with `fill`: row i gets c[i - k] (k > 0), or c[i - k] looking
// ahead (k < 0). |k| may be >= size.
col_ptr shift_col(cudf::column_view c, cudf::size_type k, const cudf::scalar& fill) {
    if (k == 0) return copy_col(c);
    if (k >= c.size() || -k >= c.size()) return cudf::make_column_from_scalar(fill, c.size());
    return cudf::shift(c, k, fill);
}

col_ptr scan_sum(cudf::column_view c) {
    return cudf::scan(c, *cudf::make_sum_aggregation<cudf::scan_aggregation>(),
                      cudf::scan_type::INCLUSIVE, cudf::null_policy::EXCLUDE);
}

col_ptr as_type(col_ptr c, cudf::data_type t) {
    if (c->type().id() == t.id()) return c;
    return cudf::cast(c->view(), t);
}

// Inclusive prefix count (INT32) of the rows where `ind` (BOOL8, no nulls) is true.
col_ptr prefix_count(cudf::column_view ind_bool) {
    auto as_int = cudf::cast(ind_bool, kInt32);
    return as_type(scan_sum(as_int->view()), kInt32);
}

col_ptr window_count(cudf::column_view prefix, cudf::size_type w) {
    cudf::numeric_scalar<int32_t> zero(0, true);
    auto sh = shift_col(prefix, w, zero);
    AstBuilder a;
    return eval({prefix, sh->view()}, a.op(Op::SUB, a.col(0), a.col(1)));
}

col_ptr null_column_like(cudf::data_type t, cudf::size_type n) {
    auto sc = cudf::make_default_constructed_scalar(t);
    sc->set_valid_async(false);
    return cudf::make_column_from_scalar(*sc, n);
}

// valid  = r >= w - 1  [ & nv == w (na.rm = FALSE) | nv >= k (na.rm = TRUE) ]
// Both inputs are non-null INT32; `nv` may be null (no nulls in the column).
col_ptr valid_mask(cudf::column_view r, const cudf::column_view* nv, int w, bool na_rm, int k) {
    AstBuilder a;
    std::vector<cudf::column_view> cols = {r};
    const Expr& base = a.op(Op::GREATER_EQUAL, a.col(0), a.ilit(w - 1));
    if (nv == nullptr) return eval(cols, base);
    cols.push_back(*nv);
    const Expr& cond = na_rm ? a.op(Op::GREATER_EQUAL, a.col(1), a.ilit(k))
                             : a.op(Op::EQUAL, a.col(1), a.ilit(w));
    return eval(cols, a.op(Op::LOGICAL_AND, base, cond));
}

void apply_validity(col_ptr& res, const col_ptr& valid) {
    auto mask = cudf::bools_to_mask(valid->view());
    res->set_null_mask(std::move(*mask.first), mask.second);
}

// Ungrouped, no missing values: only the first w - 1 rows are NA.
void apply_prefix_validity(col_ptr& res, int w) {
    auto buf = cudf::create_null_mask(res->size(), cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(buf.data()), 0, w - 1, false);
    res->set_null_mask(std::move(buf), w - 1);
}

// Everything needed from one value column, built lazily.
class XCol {
public:
    explicit XCol(cudf::column_view x) : x_(x), n_(x.size()) {}

    cudf::column_view x() const { return x_; }
    bool has_null() {
        if (!has_null_known_) { has_null_ = x_.has_nulls(); has_null_known_ = true; }
        return has_null_;
    }

    // --- float views ---
    cudf::column_view xd() {
        scan_values();
        return xd_;
    }
    bool has_inf() { scan_values(); return has_pinf_ || has_ninf_; }

    // finite values only (null where the value is null or +-Inf)
    cudf::column_view xf() {
        if (!has_inf()) return xd();
        if (!xf_own_) {
            AstBuilder a;
            const Expr* e = nullptr;
            if (has_pinf_ && has_ninf_) {
                e = &a.op(Op::LOGICAL_OR, a.op(Op::NULL_EQUAL, a.col(0), a.lit(kInf)),
                          a.op(Op::NULL_EQUAL, a.col(0), a.lit(-kInf)));
            } else {
                e = &a.op(Op::NULL_EQUAL, a.col(0), a.lit(has_pinf_ ? kInf : -kInf));
            }
            auto is_inf = eval({xd_}, *e);
            auto nul = cudf::make_empty_scalar_like(xd_);
            xf_own_ = cudf::copy_if_else(*nul, xd_, is_inf->view());
        }
        return xf_own_->view();
    }

    // xd with every missing and infinite value replaced by 0 (see the header)
    cudf::column_view x0() {
        if (!has_null() && !has_inf()) return xd();
        if (!x0_own_) {
            cudf::numeric_scalar<double> zero(0.0, true);
            x0_own_ = cudf::replace_nulls(xf(), zero);
        }
        return x0_own_->view();
    }
    void release_x0() { x0_own_.reset(); }

    // No window sum of squares can overflow at width w (see the header).
    bool var_safe(int w) {
        auto e = finite_extremes();
        return e.second - e.first <= safe_range(w);  // an overflowing range is +Inf
    }
    // The smallest and largest finite value: (0, 0) when there is none, and
    // for an integer column, whose range is always safe.
    std::pair<double, double> finite_extremes() {
        scan_values();
        if (!finite_known_) {
            read_minmax(cudf::minmax(xf()));
            finite_known_ = true;
        }
        return {lo_, hi_};
    }

    // --- inclusive INT32 prefix counts ---
    cudf::column_view cv() {  // present values (non-null; an Inf counts)
        if (!cv_) {
            auto ind = valid_ind(x_);
            cv_ = prefix_count(ind->view());
        }
        return cv_->view();
    }
    cudf::column_view cfin() {  // finite values
        if (!has_inf()) return cv();
        if (!cf_) {
            auto ind = valid_ind(xf());
            cf_ = prefix_count(ind->view());
        }
        return cf_->view();
    }
    cudf::column_view cinf(bool positive) {
        col_ptr& slot = positive ? cp_ : cn_;
        if (!slot) {
            AstBuilder a;
            auto ind = eval({xd()}, a.op(Op::NULL_EQUAL, a.col(0), a.lit(positive ? kInf : -kInf)));
            slot = prefix_count(ind->view());
        }
        return slot->view();
    }

private:
    void scan_values() {
        if (scanned_) return;
        scanned_ = true;
        if (x_.type().id() == cudf::type_id::FLOAT64) {
            xd_ = x_;
        } else {
            xd_own_ = cudf::cast(x_, kFloat64);
            xd_ = xd_own_->view();
        }
        auto id = x_.type().id();
        if ((id == cudf::type_id::FLOAT32 || id == cudf::type_id::FLOAT64) && n_ > 0) {
            read_minmax(cudf::minmax(xd_));
            has_ninf_ = lo_ == -kInf;
            has_pinf_ = hi_ == kInf;
            finite_known_ = !has_ninf_ && !has_pinf_;
        }
    }
    void read_minmax(const std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>>& mm) {
        auto* lo = static_cast<cudf::numeric_scalar<double>*>(mm.first.get());
        auto* hi = static_cast<cudf::numeric_scalar<double>*>(mm.second.get());
        lo_ = lo->is_valid() ? lo->value() : 0.0;
        hi_ = hi->is_valid() ? hi->value() : 0.0;
    }

    static col_ptr valid_ind(cudf::column_view c) {
        if (!c.has_nulls()) {
            cudf::numeric_scalar<bool> t(true, true);  // an all-true BOOL8 column
            return cudf::make_column_from_scalar(t, c.size());
        }
        return cudf::is_valid(c);
    }

    cudf::column_view x_;
    cudf::size_type n_;
    bool has_null_ = false, has_null_known_ = false;
    bool scanned_ = false, has_pinf_ = false, has_ninf_ = false, finite_known_ = true;
    double lo_ = 0.0, hi_ = 0.0;
    col_ptr xd_own_, xf_own_, x0_own_, cv_, cf_, cp_, cn_;
    cudf::column_view xd_;
};

// ---------------------------------------------------------------------------
// Integer sum (INT32 / BOOL8): exact INT64 prefix difference.
// ---------------------------------------------------------------------------
col_ptr int_sum(RollRows& rows, XCol& X, const RollReq& rq) {
    const int w = rq.w;
    auto xi = cudf::cast(X.x(), kInt64);
    col_ptr v;
    if (xi->view().has_nulls()) {
        cudf::numeric_scalar<int64_t> zero(0, true);
        v = cudf::replace_nulls(xi->view(), zero);
    } else {
        v = std::move(xi);
    }
    auto P = scan_sum(v->view());
    v.reset();
    cudf::numeric_scalar<int64_t> zero64(0, true);
    auto Psh = shift_col(P->view(), w, zero64);
    AstBuilder a;
    auto res = eval({P->view(), Psh->view()}, a.op(Op::SUB, a.col(0), a.col(1)));
    P.reset();
    Psh.reset();
    col_ptr valid;
    if (X.has_null()) {
        auto nv = window_count(X.cv(), w);
        cudf::column_view nvv = nv->view();
        valid = valid_mask(rows.r(), &nvv, w, rq.na_rm, 1);
    } else {
        valid = valid_mask(rows.r(), nullptr, w, rq.na_rm, 1);
    }
    apply_validity(res, valid);
    return res;
}

// ---------------------------------------------------------------------------
// Float mean / sum / sd for one width.
// ---------------------------------------------------------------------------

// A segmented inclusive scan input: a running SUM (or MIN / MAX) of `v`, or
// the running count of its non-null values.
enum class Agg { Sum, Count, Min, Max };
struct ScanIn {
    cudf::column_view v;
    Agg agg;
};

// Segmented inclusive scans keyed by `keys` (INT32, equal within a segment,
// segments contiguous). The sorted groupby computes its segment labels on the
// first scan and keeps them, so a later scan() costs only the scans.
class SegScan {
public:
    explicit SegScan(cudf::column_view keys)
        : gb_(cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::YES) {}

    std::vector<col_ptr> scan(const std::vector<ScanIn>& in) {
        std::vector<cudf::groupby::scan_request> req;
        for (const auto& s : in) {
            cudf::groupby::scan_request r;
            r.values = s.v;
            switch (s.agg) {
                case Agg::Count:
                    r.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(
                        cudf::null_policy::EXCLUDE));
                    break;
                case Agg::Min:
                    r.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_scan_aggregation>());
                    break;
                case Agg::Max:
                    r.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_scan_aggregation>());
                    break;
                default:
                    r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>());
            }
            req.push_back(std::move(r));
        }
        auto sc = gb_.scan(req);
        std::vector<col_ptr> out;
        for (size_t i = 0; i < in.size(); ++i) {
            col_ptr c = std::move(sc.second[i].results[0]);
            out.push_back(in[i].agg == Agg::Count ? as_type(std::move(c), kInt32) : std::move(c));
        }
        return out;
    }

private:
    cudf::groupby::groupby gb_;
};

// Infinite values: sum/mean +-Inf or NaN (both signs); sd NaN. `np`/`nn` are
// the window counts of +Inf / -Inf.
col_ptr apply_inf(col_ptr res, const col_ptr& np, const col_ptr& nn, bool is_sd) {
    AstBuilder b2;
    std::vector<cudf::column_view> ic = {np->view(), nn->view()};
    cudf::numeric_scalar<double> nan(std::numeric_limits<double>::quiet_NaN(), true);
    if (is_sd) {
        auto any = eval(ic, b2.op(Op::GREATER, b2.op(Op::ADD, b2.col(0), b2.col(1)), b2.ilit(0)));
        return cudf::copy_if_else(nan, res->view(), any->view());
    }
    auto pos = eval(ic, b2.op(Op::LOGICAL_AND, b2.op(Op::GREATER, b2.col(0), b2.ilit(0)),
                               b2.op(Op::EQUAL, b2.col(1), b2.ilit(0))));
    auto neg = eval(ic, b2.op(Op::LOGICAL_AND, b2.op(Op::EQUAL, b2.col(0), b2.ilit(0)),
                               b2.op(Op::GREATER, b2.col(1), b2.ilit(0))));
    auto both = eval(ic, b2.op(Op::LOGICAL_AND, b2.op(Op::GREATER, b2.col(0), b2.ilit(0)),
                                b2.op(Op::GREATER, b2.col(1), b2.ilit(0))));
    cudf::numeric_scalar<double> pinf(kInf, true), ninf(-kInf, true);
    res = cudf::copy_if_else(pinf, res->view(), pos->view());
    res = cudf::copy_if_else(ninf, res->view(), neg->view());
    return cudf::copy_if_else(nan, res->view(), both->view());
}

cudf::size_type num_segments(cudf::size_type n, int w) {
    return static_cast<cudf::size_type>((static_cast<int64_t>(n) + w - 1) / w);
}

// A column that may be a view of the first n rows of a larger owned one.
struct Col {
    col_ptr own;
    cudf::column_view v;
};
Col own_col(col_ptr c) {
    Col r;
    r.v = c->view();
    r.own = std::move(c);
    return r;
}
Col first_rows(col_ptr c, cudf::size_type n) {
    Col r;
    r.own = std::move(c);
    r.v = cudf::slice(r.own->view(), {0, n})[0];
    return r;
}

// A column of n rows repeating the w host values `pattern` (w * size_of(t) bytes).
Col tile_pattern(const std::vector<char>& pattern, cudf::data_type t, cudf::size_type n) {
    const int w = static_cast<int>(pattern.size() / cudf::size_of(t));
    auto pat = cudf::make_numeric_column(t, w, cudf::mask_state::UNALLOCATED);
    auto rc = cudaMemcpy(pat->mutable_view().head<char>(), pattern.data(), pattern.size(),
                         cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) Rcpp::stop("roll_columns: cudaMemcpy failed: %s", cudaGetErrorString(rc));
    return first_rows(std::move(cudf::tile(cudf::table_view({pat->view()}), num_segments(n, w))->release()[0]), n);
}

// x - c, and a column's square
col_ptr minus(cudf::column_view x, cudf::column_view c) {
    AstBuilder a;
    return eval({x, c}, a.op(Op::SUB, a.col(0), a.col(1)));
}
col_ptr squared(cudf::column_view y) {
    AstBuilder a;
    return eval({y}, a.op(Op::MUL, a.col(0), a.col(0)));
}
// the column with its nulls (if any) as 0
Col zero_nulls(Col c) {
    if (!c.v.has_nulls()) return c;
    cudf::numeric_scalar<double> zero(0.0, true);
    return own_col(cudf::replace_nulls(c.v, zero));
}

// The segment structure of one frame for one width w: every group is cut into
// segments of w rows, the last of a group possibly shorter.
class Segs {
public:
    Segs(RollRows& rows, int w) : w_(w), n_(rows.size()), grouped_(rows.grouped()) {
        const cudf::size_type ns = num_segments(n_, w);
        cudf::numeric_scalar<int32_t> zero(0, true), one(1, true), hi(n_ - 1, true);
        if (grouped_) {
            // grouped: segment k of a group starts at j0 = i - r % w, keyed by j0
            AstBuilder a;
            j0_ = own_col(eval({rows.idx(), rows.r()},
                               a.op(Op::SUB, a.col(0), a.op(Op::MOD, a.col(1), a.ilit(w)))));
            keys_.v = j0_.v;
            AstBuilder b;
            gate_ = own_col(eval({rows.r()}, b.op(Op::LESS, b.op(Op::MOD, b.col(0), b.ilit(w)), b.ilit(w - 1))));
            AstBuilder c;
            // m = j0 + (w - 1) - (i - j0)
            auto m = eval({j0_.v, rows.idx()},
                          c.op(Op::SUB, c.op(Op::ADD, c.col(0), c.ilit(w - 1)),
                               c.op(Op::SUB, c.col(1), c.col(0))));
            // a short segment mirrors past its end: keep the index in the table
            mirror_ = own_col(cudf::clamp(m->view(), zero, hi));
        } else {
            // ungrouped: segments are aligned multiples of w, so the per-row
            // columns are repeats / tiles of a few values
            auto seg_id = cudf::sequence(ns, zero, one);
            keys_ = first_rows(std::move(cudf::repeat(cudf::table_view({seg_id->view()}), w)->release()[0]), n_);
            std::vector<char> g(static_cast<size_t>(w), 1);
            g.back() = 0;
            gate_ = tile_pattern(g, kBool, n_);
            // row i = j0 + q mirrors to j0 + w - 1 - q = i + w - 1 - 2 q
            std::vector<char> host(static_cast<size_t>(w) * sizeof(int32_t));
            auto* d = reinterpret_cast<int32_t*>(host.data());
            for (int q = 0; q < w; ++q) d[q] = w - 1 - 2 * q;
            Col off = tile_pattern(host, kInt32, n_);
            auto idx = cudf::sequence(n_, zero, one);  // (not kept: rows.idx() lives on)
            AstBuilder a;
            mirror_ = own_col(eval({idx->view(), off.v}, a.op(Op::ADD, a.col(0), a.col(1))));
            if (n_ % w != 0) {  // the short last segment mirrors onto itself
                const cudf::size_type last = n_ / w * w;
                auto mv = mirror_.own->mutable_view();
                cudf::copy_range_in_place(idx->view(), mv, last, n_, last);
            }
        }
        // The tail is read at the mirror of s = i - w + 1, or, when the window
        // is the row's own segment, at j0 (see tails()), which is the mirror
        // of i itself on those rows.
        auto rs = shift_col(mirror_.v, w - 1, zero);
        read_ = own_col(cudf::copy_if_else(rs->view(), mirror_.v, gate_.v));
    }

    cudf::column_view keys() const { return keys_.v; }
    // BOOL8: the row's window has a tail (r % w < w - 1); otherwise it is exactly one segment
    cudf::column_view gate() const { return gate_.v; }

    // the mirror index is freed by the k-th mirrored() call
    void expect_mirrors(int k) { mirrors_left_ = k; }
    // after the last scan only the gate is still read
    void drop_all_but_gate() {
        j0_ = Col{};
        keys_ = Col{};
        mirror_ = Col{};
        read_ = Col{};
    }
    // c gathered at the mirror row: a within-segment reversal, so a forward
    // scan of it is a suffix scan of c
    col_ptr mirrored(cudf::column_view c) {
        auto out = std::move(cudf::gather(cudf::table_view({c}), mirror_.v,
                                          cudf::out_of_bounds_policy::DONT_CHECK)->release()[0]);
        if (--mirrors_left_ == 0) mirror_ = Col{};
        return out;
    }
    // The scan results of mirrored columns, read at each row's window start:
    // the sum of its tail [s, e]. A row whose window is exactly its own
    // segment (gate 0) reads the first row of its own mirrored segment
    // instead, which holds the segment's last centred value. That is 0 in
    // B's columns and x0[e] (a member) in C's and D's, but x0[e] - x0[e + 1]
    // in A's, and x0[e + 1] is outside the window: A selects it away (see
    // the invariant in the header).
    std::vector<col_ptr> tails(std::vector<col_ptr>& scanned) const {
        std::vector<cudf::column_view> views;
        for (auto& c : scanned) views.push_back(c->view());
        auto t = cudf::gather(cudf::table_view(views), read_.v,
                              cudf::out_of_bounds_policy::DONT_CHECK)->release();
        scanned.clear();
        return t;
    }

    // c at the first row of each row's segment
    Col first(cudf::column_view c) {
        if (grouped_) return at(c, j0_.v);
        return repeat(per_segment(c, starts()->view())->view());
    }
    // c at the first row of the *next* segment; the table's last segment has
    // none (and is never read as a tail), it gets its own first row
    Col next_first(cudf::column_view c) {
        if (grouped_) return at(c, offset_j0(w_)->view());
        auto fs = per_segment(c, starts()->view());
        auto last = cudf::get_element(fs->view(), fs->size() - 1);
        auto next = shift_col(fs->view(), -1, *last);
        return repeat(next->view());
    }

    // B's references (see the header), per row and 0 where there is none:
    //   cH = the first finite value of the row's segment,
    //   cT = the last finite value of the row's segment,
    //   dd = cT of the previous segment - cH.
    // `xf` holds nulls at the values that are not finite. A segment without a
    // finite value takes one from a neighbour (or 0); its parts are empty, so
    // that reference is never used. Ungrouped, dd is kept per segment until
    // expand_dd() (it is read only after the scans).
    void refs(cudf::column_view xf, Col& cH, Col& cT, Col& dd) {
        cudf::numeric_scalar<double> zero_d(0.0, true);
        const bool fill = xf.has_nulls();
        if (grouped_) {
            {
                col_ptr bf = fill ? cudf::replace_nulls(xf, cudf::replace_policy::FOLLOWING) : nullptr;
                cH = zero_nulls(at(fill ? bf->view() : xf, j0_.v));
            }
            Col cTp;
            {
                col_ptr ff = fill ? cudf::replace_nulls(xf, cudf::replace_policy::PRECEDING) : nullptr;
                cudf::column_view f = fill ? ff->view() : xf;
                cT = zero_nulls(at(f, offset_j0(w_ - 1)->view()));
                cTp = zero_nulls(at(f, offset_j0(-1)->view()));
            }
            dd = own_col(minus(cTp.v, cH.v));
            return;
        }
        Col hs, ts;
        {
            col_ptr bf = fill ? cudf::replace_nulls(xf, cudf::replace_policy::FOLLOWING) : nullptr;
            hs = zero_nulls(own_col(per_segment(fill ? bf->view() : xf, starts()->view())));
        }
        {
            col_ptr ff = fill ? cudf::replace_nulls(xf, cudf::replace_policy::PRECEDING) : nullptr;
            ts = zero_nulls(own_col(per_segment(fill ? ff->view() : xf, ends()->view())));
        }
        auto prev = shift_col(ts.v, 1, zero_d);  // cT of the previous segment
        dd = own_col(minus(prev->view(), hs.v));
        cH = repeat(hs.v);
        cT = repeat(ts.v);
    }
    // dd from refs() as a per-row column
    Col expand_dd(Col dd) {
        if (grouped_) return dd;
        return repeat(dd.v);
    }

private:
    Col at(cudf::column_view c, cudf::column_view idx) {
        return own_col(std::move(cudf::gather(cudf::table_view({c}), idx,
                                              cudf::out_of_bounds_policy::DONT_CHECK)->release()[0]));
    }
    Col repeat(cudf::column_view per_segment) {
        return first_rows(std::move(cudf::repeat(cudf::table_view({per_segment}), w_)->release()[0]), n_);
    }
    // grouped: j0 + k, clamped to the table
    col_ptr offset_j0(int k) {
        AstBuilder a;
        auto e = eval({j0_.v}, a.op(Op::ADD, a.col(0), a.ilit(k)));
        cudf::numeric_scalar<int32_t> lo(0, true), hi(n_ - 1, true);
        return cudf::clamp(e->view(), lo, hi);
    }
    // ungrouped: the first / last row of every segment (the last clamped)
    col_ptr starts() {
        cudf::numeric_scalar<int32_t> zero(0, true), step(w_, true);
        return cudf::sequence(num_segments(n_, w_), zero, step);
    }
    col_ptr ends() {
        cudf::numeric_scalar<int32_t> lo(0, true), from(w_ - 1, true), step(w_, true), hi(n_ - 1, true);
        return cudf::clamp(cudf::sequence(num_segments(n_, w_), from, step)->view(), lo, hi);
    }
    col_ptr per_segment(cudf::column_view c, cudf::column_view rows) {
        return std::move(cudf::gather(cudf::table_view({c}), rows,
                                      cudf::out_of_bounds_policy::DONT_CHECK)->release()[0]);
    }

    int w_;
    cudf::size_type n_;
    bool grouped_;
    int mirrors_left_ = -1;
    Col j0_, keys_, gate_, mirror_, read_;
};

// Everything one width shares: the segments, their groupby, window counts.
struct Width {
    Width(RollRows& r, XCol& x, int w_, int stages)
        : rows(r), X(x), w(w_), sg(r, w_), gb_(std::make_unique<SegScan>(sg.keys())), stages_(stages) {}

    SegScan& gb() { return *gb_; }
    // A computation has read its last tail; after the last one the groupby
    // and the segment indexes are freed before the final formulas.
    void scans_done() {
        if (--stages_ == 0) {
            gb_.reset();
            sg.drop_all_but_gate();
        }
    }

    // present values (an Inf counts) in each row's window
    cudf::column_view nv() {
        if (!nv_) nv_ = window_count(X.cv(), w);
        return nv_->view();
    }
    // finite values in each row's window
    cudf::column_view nf() {
        if (!X.has_inf() && X.has_null()) return nv();
        if (!nf_) {
            if (X.has_inf()) {
                nf_ = window_count(X.cfin(), w);
            } else {  // nothing missing: a complete window holds w
                cudf::numeric_scalar<int32_t> ws(w, true);
                nf_ = cudf::make_column_from_scalar(ws, rows.size());
            }
        }
        return nf_->view();
    }

    RollRows& rows;
    XCol& X;
    const int w;
    Segs sg;

private:
    std::unique_ptr<SegScan> gb_;
    int stages_;
    col_ptr nv_, nf_;
};

// sqrt(max(M2, 0) / (n - 1)), n a column (`n` != nullptr) or the constant w:
// a variance that rounding left slightly negative is 0, not NaN.
col_ptr sd_from_m2(cudf::column_view m2, const cudf::column_view* n, double w) {
    AstBuilder a;
    std::vector<cudf::column_view> cols = {m2};
    const Expr& twice = a.op(Op::ADD, a.col(0), a.op(Op::ABS, a.col(0)));  // 2 max(M2, 0)
    if (n == nullptr) {
        return eval(cols, a.op(Op::SQRT, a.op(Op::MUL, twice, a.lit(0.5 / (w - 1.0)))));
    }
    cols.push_back(*n);
    const Expr& den = a.op(Op::SUB, a.to_f64(a.col(1)), a.lit(1.0));
    return eval(cols, a.op(Op::SQRT, a.op(Op::DIV, a.op(Op::MUL, twice, a.lit(0.5)), den)));
}

// ---- C: head (S) and tail (T) plain sums of x0; a window sums S + gate T --
// On a gate-0 row T is the mirrored scan's first value in the row's own
// (complete) segment, x0[e]: one value of the window, finite (x0 holds no
// NA, NaN or Inf), so gate T is exactly 0 there.
// With `release_x0`, X's copy of x0 is freed once read.
void plain_sums(Width& W, cudf::column_view x0, col_ptr& S, col_ptr& T, bool release_x0) {
    S = std::move(W.gb().scan({{x0, Agg::Sum}})[0]);
    std::vector<col_ptr> tail_in;
    {
        auto xm = W.sg.mirrored(x0);
        if (release_x0) W.X.release_x0();
        tail_in = W.gb().scan({{xm->view(), Agg::Sum}});
    }
    T = std::move(W.sg.tails(tail_in)[0]);
    W.scans_done();
}

// ---- A: sd of windows without a missing value -----------------------------
// s1 is the window sum about cH; both are kept for mean / sum (`keep_ms`).
// With `release_x0`, X's copy of x0 is freed once read.
struct CleanSd {
    col_ptr s1, sd;
    Col cH;
};
CleanSd clean_sd(Width& W, cudf::column_view x0, bool keep_ms, bool release_x0) {
    CleanSd o;
    std::vector<col_ptr> S, T;
    o.cH = W.sg.first(x0);
    {
        col_ptr y = minus(x0, o.cH.v);
        col_ptr y2 = squared(y->view());
        S = W.gb().scan({{y->view(), Agg::Sum}, {y2->view(), Agg::Sum}});
    }
    if (!keep_ms) o.cH = Col{};
    {
        // the tail of segment k - 1 about the first value of segment k
        std::vector<col_ptr> tail_in;
        {
            col_ptr yp;
            {
                auto xm = W.sg.mirrored(x0);
                Col cN = W.sg.next_first(x0);
                if (release_x0) W.X.release_x0();
                yp = minus(xm->view(), cN.v);
            }
            col_ptr yp2 = squared(yp->view());
            tail_in = W.gb().scan({{yp->view(), Agg::Sum}, {yp2->view(), Agg::Sum}});
        }
        T = W.sg.tails(tail_in);
        W.scans_done();
        // A row whose window is its own segment (gate 0) read x0[e] -
        // x0[e + 1] there, and x0[e + 1] is outside its window (the next
        // segment, or the next group): selected away, never multiplied by 0,
        // whose product with an overflowed square is NaN.
        cudf::numeric_scalar<double> zero(0.0, true);
        for (auto& t : T) t = cudf::copy_if_else(t->view(), zero, W.sg.gate());
    }
    {
        AstBuilder a;
        o.s1 = eval({S[0]->view(), T[0]->view()}, a.op(Op::ADD, a.col(0), a.col(1)));
        S[0].reset();
        T[0].reset();
    }
    col_ptr m2;
    {
        // M2 = S2 + T2 - s1 (s1 / w)
        AstBuilder a;
        std::vector<cudf::column_view> cols = {S[1]->view(), T[1]->view(), o.s1->view()};
        const Expr& s2 = a.op(Op::ADD, a.col(0), a.col(1));
        m2 = eval(cols, a.op(Op::SUB, s2, a.op(Op::MUL, a.col(2), a.op(Op::DIV, a.col(2), a.lit(W.w)))));
        S[1].reset();
        T[1].reset();
    }
    o.sd = sd_from_m2(m2->view(), nullptr, W.w);
    return o;
}

// ---- B: sd under na.rm = TRUE on a column with missing values -------------
// `xf`: the finite values (nulls elsewhere); `x0`: the same with nulls as 0.
// With `release_x0`, X's copy of x0 is freed once read.
col_ptr general_sd(Width& W, cudf::column_view xf, cudf::column_view x0, bool release_x0) {
    Col cH, cT, dd;
    W.sg.refs(xf, cH, cT, dd);
    auto v = cudf::is_valid(xf);
    // a part's values about its reference; a missing value is 0 (x0 is 0
    // there, and the reference, a finite column value or 0, times 0 is 0)
    auto centred = [&](const Col& c) {
        AstBuilder a;
        return eval({x0, c.v, v->view()},
                    a.op(Op::SUB, a.col(0), a.op(Op::MUL, a.col(1), a.to_f64(a.col(2)))));
    };
    col_ptr y = centred(cH);
    cH = Col{};
    col_ptr ytm;  // the tail values about cT, mirrored
    {
        col_ptr yt = centred(cT);
        cT = Col{};
        v.reset();
        if (release_x0) W.X.release_x0();
        ytm = W.sg.mirrored(yt->view());
    }
    col_ptr S1H, S2H, NH;
    {
        col_ptr y2 = squared(y->view());
        auto S = W.gb().scan({{y->view(), Agg::Sum}, {y2->view(), Agg::Sum}, {xf, Agg::Count}});
        S1H = std::move(S[0]);
        S2H = std::move(S[1]);
        NH = std::move(S[2]);
    }
    y.reset();
    col_ptr T1, T2;
    {
        std::vector<col_ptr> tail_in;
        {
            col_ptr yt2m = squared(ytm->view());
            tail_in = W.gb().scan({{ytm->view(), Agg::Sum}, {yt2m->view(), Agg::Sum}});
        }
        ytm.reset();
        auto T = W.sg.tails(tail_in);
        W.scans_done();
        T1 = std::move(T[0]);
        T2 = std::move(T[1]);
    }
    cudf::column_view nf = W.nf();
    // e = dd where both parts hold a finite value, else 0 (a window without
    // a finite head value is the tail alone, about cT; one without a finite
    // tail value is the head alone). Only then are cH and cT both members of
    // the window: elsewhere dd may come from outside it, so it is selected
    // away rather than multiplied by a zero count (0 * Inf is NaN).
    col_ptr e_col;
    {
        dd = W.sg.expand_dd(std::move(dd));
        AstBuilder a;
        auto both = eval({NH->view(), nf}, a.op(Op::LOGICAL_AND, a.op(Op::GREATER, a.col(0), a.ilit(0)),
                                                a.op(Op::GREATER, a.col(1), a.col(0))));
        cudf::numeric_scalar<double> zero(0.0, true);
        e_col = cudf::copy_if_else(dd.v, zero, both->view());
        dd = Col{};
    }
    // a = T1 + nT e: the tail sum about cH, nT = nf - nH
    col_ptr a_col;
    {
        AstBuilder a;
        std::vector<cudf::column_view> cols = {T1->view(), NH->view(), nf, e_col->view()};
        a_col = eval(cols, a.op(Op::ADD, a.col(0),
                                a.op(Op::MUL, a.to_f64(a.op(Op::SUB, a.col(2), a.col(1))), a.col(3))));
    }
    // S1 = S1H + a;  S2 = S2H + T2 + e (T1 + a);  M2 = S2 - S1 (S1 / nf)
    col_ptr m2;
    {
        AstBuilder a;
        std::vector<cudf::column_view> cols = {S1H->view(), S2H->view(), T1->view(), T2->view(),
                                               a_col->view(), e_col->view(), nf};
        const Expr& s2 = a.op(Op::ADD, a.op(Op::ADD, a.col(1), a.col(3)),
                              a.op(Op::MUL, a.col(5), a.op(Op::ADD, a.col(2), a.col(4))));
        const Expr& s1 = a.op(Op::ADD, a.col(0), a.col(4));
        m2 = eval(cols, a.op(Op::SUB, s2, a.op(Op::MUL, s1, a.op(Op::DIV, s1, a.to_f64(a.col(6))))));
    }
    S1H.reset(); S2H.reset(); T1.reset(); T2.reset(); a_col.reset(); e_col.reset(); NH.reset();
    return sd_from_m2(m2->view(), &nf, 0.0);
}

// ---- overflow: B on x 2^-k, scaled back (see the header) -------------------
col_ptr scaled_sd(Width& W) {
    auto ext = W.X.finite_extremes();
    const double m = std::max(std::fabs(ext.first), std::fabs(ext.second));
    // the range is below 2 m < 2^(ilogb(m) + 2); bring it under safe_range / 2
    const int k = std::max(0, std::ilogb(m) + 3 - std::ilogb(safe_range(W.w)));
    const double down = std::ldexp(1.0, -k), up = std::ldexp(1.0, k);
    col_ptr xs, x0s;
    {
        AstBuilder a;
        xs = eval({W.X.xf()}, a.op(Op::MUL, a.col(0), a.lit(down)));
        AstBuilder b;
        x0s = eval({W.X.x0()}, b.op(Op::MUL, b.col(0), b.lit(down)));
    }
    auto sds = general_sd(W, xs->view(), x0s->view(), false);
    // sqrt(var 2^k 2^k): +Inf exactly where the variance leaves the double range
    AstBuilder a;
    const Expr& var = a.op(Op::MUL, a.op(Op::MUL, a.op(Op::MUL, a.col(0), a.col(0)), a.lit(up)), a.lit(up));
    return eval({sds->view()}, a.op(Op::SQRT, var));
}

// sd where it is finite, else the scaled recomputation
col_ptr keep_finite(col_ptr sd, cudf::column_view big) {
    AstBuilder a;
    auto fin = eval({sd->view()}, a.op(Op::LESS, a.op(Op::ABS, a.col(0)), a.lit(kInf)));
    return cudf::copy_if_else(sd->view(), big, fin->view());
}

// ---- D: min / max (van Herk / Gil-Werman) --------------------------------
// The head's running min and the tail's suffix min, from the same segments
// and mirrored scans as the sums. A row whose window is its own segment reads
// its segment's last value as the tail, a member of the window: no gate. A
// missing value becomes the type's extreme on the losing side, which never
// wins over a present value; a window holding only missing values is masked
// by its count.
std::unique_ptr<cudf::scalar> losing_extreme(cudf::data_type t, bool is_min) {
    switch (t.id()) {
        case cudf::type_id::INT32:
            return std::make_unique<cudf::numeric_scalar<int32_t>>(
                is_min ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::lowest(), true);
        case cudf::type_id::INT64:
            return std::make_unique<cudf::numeric_scalar<int64_t>>(
                is_min ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int64_t>::lowest(), true);
        case cudf::type_id::FLOAT32:
            return std::make_unique<cudf::numeric_scalar<float>>(
                is_min ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity(), true);
        case cudf::type_id::FLOAT64:
            return std::make_unique<cudf::numeric_scalar<double>>(is_min ? kInf : -kInf, true);
        default:
            Rcpp::stop("roll_columns: internal: no min/max scan for this type");
    }
}

col_ptr min_max(Width& W, cudf::column_view x, bool is_min) {
    col_ptr filled;
    if (x.has_nulls()) filled = cudf::replace_nulls(x, *losing_extreme(x.type(), is_min));
    cudf::column_view xv = filled ? filled->view() : x;
    const Agg agg = is_min ? Agg::Min : Agg::Max;
    col_ptr H = std::move(W.gb().scan({{xv, agg}})[0]);
    std::vector<col_ptr> tail_in;
    {
        auto xm = W.sg.mirrored(xv);
        filled.reset();
        tail_in = W.gb().scan({{xm->view(), agg}});
    }
    col_ptr T = std::move(W.sg.tails(tail_in)[0]);
    W.scans_done();
    AstBuilder a;
    auto head_wins = eval({H->view(), T->view()},
                          a.op(is_min ? Op::LESS_EQUAL : Op::GREATER_EQUAL, a.col(0), a.col(1)));
    return cudf::copy_if_else(H->view(), T->view(), head_wins->view());
}

void width_columns(RollRows& rows, XCol& X, int w, const std::vector<size_t>& ids,
                   const std::vector<RollReq>& reqs, std::vector<col_ptr>& out, bool last_width) {
    const bool has_null = X.has_null();
    // the request's valid windows hold no missing value
    auto null_free = [&](const RollReq& rq) { return !has_null || !rq.na_rm; };
    // which computation (see the header) gives each request its values
    enum Kind { A_SD, A_SUM, A_MEAN, B_SD, C_SUM, C_MEAN_W, C_MEAN_NF, D_MIN, D_MAX, N_KIND };
    bool run_a = false, run_b = false;
    for (size_t id : ids) {
        if (reqs[id].fn == "roll_sd") (null_free(reqs[id]) ? run_a : run_b) = true;
    }
    // A's s1 is a sum of differences between window members, so mean / sum
    // take it from A only where no such difference can overflow; otherwise C
    // (sd falls back to the scaled recomputation, `fix`)
    const bool safe = !(run_a || run_b) || X.var_safe(w);
    const bool a_ms = run_a && safe;
    std::vector<Kind> kind;
    int uses[N_KIND] = {0};
    for (size_t id : ids) {
        const RollReq& rq = reqs[id];
        const bool nfree = null_free(rq);
        Kind k;
        if (rq.fn == "roll_min") k = D_MIN;
        else if (rq.fn == "roll_max") k = D_MAX;
        else if (rq.fn == "roll_sd") k = nfree ? A_SD : B_SD;
        else if (rq.fn == "roll_sum") k = nfree && a_ms ? A_SUM : C_SUM;
        else k = nfree && a_ms ? A_MEAN : (nfree ? C_MEAN_W : C_MEAN_NF);
        kind.push_back(k);
        ++uses[k];
    }
    const bool run_c = uses[C_SUM] + uses[C_MEAN_W] + uses[C_MEAN_NF] > 0;
    const bool uses_inf_rule = run_a || run_b || run_c;
    const bool fix = !safe;

    const int stages = int(run_a) + int(run_b) + int(run_c) + int(fix) +
                       int(uses[D_MIN] > 0) + int(uses[D_MAX] > 0);
    Width W(rows, X, w, stages);
    W.sg.expect_mirrors(stages);
    col_ptr raw[N_KIND];
    if (uses[D_MIN] > 0) raw[D_MIN] = min_max(W, X.x(), true);
    if (uses[D_MAX] > 0) raw[D_MAX] = min_max(W, X.x(), false);
    // the stages run C, A, B, fix; the last one reading x0 frees it
    const bool x0_free_c = last_width && !run_a && !run_b && !fix;
    const bool x0_free_a = last_width && !run_b && !fix;
    const bool x0_free_b = last_width && !fix;

    if (run_c) {
        col_ptr S, T;
        plain_sums(W, X.x0(), S, T, x0_free_c);
        for (Kind k : {C_SUM, C_MEAN_W, C_MEAN_NF}) {
            if (uses[k] == 0) continue;
            AstBuilder a;
            std::vector<cudf::column_view> cols = {S->view(), T->view(), W.sg.gate()};
            const Expr& sum = a.op(Op::ADD, a.col(0), a.op(Op::MUL, a.to_f64(a.col(2)), a.col(1)));
            if (k == C_SUM) {
                raw[k] = eval(cols, sum);
            } else if (k == C_MEAN_W) {
                raw[k] = eval(cols, a.op(Op::DIV, sum, a.lit(w)));
            } else {
                cols.push_back(W.nf());
                raw[k] = eval(cols, a.op(Op::DIV, sum, a.to_f64(a.col(3))));
            }
        }
    }
    if (run_a) {
        const bool ms = uses[A_SUM] + uses[A_MEAN] > 0;
        CleanSd A = clean_sd(W, X.x0(), ms, x0_free_a);
        if (uses[A_SUM] > 0) {  // s1 + w cH
            AstBuilder a;
            raw[A_SUM] = eval({A.s1->view(), A.cH.v}, a.op(Op::ADD, a.col(0), a.op(Op::MUL, a.col(1), a.lit(w))));
        }
        if (uses[A_MEAN] > 0) {  // cH + s1 / w
            AstBuilder a;
            raw[A_MEAN] = eval({A.s1->view(), A.cH.v}, a.op(Op::ADD, a.col(1), a.op(Op::DIV, a.col(0), a.lit(w))));
        }
        raw[A_SD] = std::move(A.sd);
    }
    if (run_b) raw[B_SD] = general_sd(W, X.xf(), X.x0(), x0_free_b);
    if (fix) {
        col_ptr big = scaled_sd(W);
        for (Kind k : {A_SD, B_SD}) {
            if (raw[k]) raw[k] = keep_finite(std::move(raw[k]), big->view());
        }
    }

    // NA and Inf rules
    col_ptr NP, NN;
    if (uses_inf_rule && X.has_inf()) {
        NP = window_count(X.cinf(true), w);
        NN = window_count(X.cinf(false), w);
    }
    std::map<std::pair<bool, int>, col_ptr> valid_cache;
    for (size_t q = 0; q < ids.size(); ++q) {
        const RollReq& rq = reqs[ids[q]];
        const Kind k = kind[q];
        // the last request of a kind takes its column, the others copy it
        col_ptr res = --uses[k] == 0 ? std::move(raw[k]) : copy_col(raw[k]->view());
        const bool is_sd = rq.fn == "roll_sd";
        if (NP && k != D_MIN && k != D_MAX) res = apply_inf(std::move(res), NP, NN, is_sd);
        if (!has_null && !rows.grouped()) {
            apply_prefix_validity(res, w);
        } else {
            const bool nfree = null_free(rq);
            const int kmin = is_sd ? 2 : 1;
            auto key = std::make_pair(nfree, nfree ? 0 : kmin);
            auto it = valid_cache.find(key);
            if (it == valid_cache.end()) {
                col_ptr valid;
                if (!has_null) {
                    valid = valid_mask(rows.r(), nullptr, w, true, 1);
                } else {
                    cudf::column_view nvv = W.nv();
                    valid = valid_mask(rows.r(), &nvv, w, !nfree, kmin);
                }
                it = valid_cache.emplace(key, std::move(valid)).first;
            }
            apply_validity(res, it->second);
        }
        out[ids[q]] = std::move(res);
    }
}

bool is_numeric_id(cudf::type_id t) {
    switch (t) {
        case cudf::type_id::INT8: case cudf::type_id::INT16: case cudf::type_id::INT32:
        case cudf::type_id::INT64: case cudf::type_id::UINT8: case cudf::type_id::UINT16:
        case cudf::type_id::UINT32: case cudf::type_id::UINT64: case cudf::type_id::FLOAT32:
        case cudf::type_id::FLOAT64: case cudf::type_id::BOOL8:
            return true;
        default:
            return false;
    }
}

}  // namespace

cudf::column_view RollRows::r() {
    if (!grouped_) return idx();  // r == the row index
    if (!r_) {
        std::vector<cudf::groupby::scan_request> req(1);
        req[0].values = key0_;
        req[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(
            cudf::null_policy::INCLUDE));
        auto sc = gb_->scan(req);
        AstBuilder a;  // (a compute_column is faster than a column-scalar binary_operation)
        r_ = eval({sc.second[0].results[0]->view()}, a.op(Op::SUB, a.col(0), a.ilit(1)));
    }
    return r_->view();
}

cudf::column_view RollRows::idx() {
    if (!idx_) {
        cudf::numeric_scalar<int32_t> zero(0, true);
        idx_ = cudf::sequence(n_, zero);
    }
    return idx_->view();
}

cudf::size_type RollRows::weighted_group_size() {
    if (!grouped_) return n_;
    if (wgs_ < 0) {
        // the sum over groups of size^2 is the sum over rows of 2 r + 1
        auto s = cudf::reduce(r(), *cudf::make_sum_aggregation<cudf::reduce_aggregation>(), kInt64);
        const double sum_r = static_cast<double>(static_cast<cudf::numeric_scalar<int64_t>*>(s.get())->value());
        const double size = n_ > 0 ? (2.0 * sum_r + n_) / n_ : 0.0;
        wgs_ = static_cast<cudf::size_type>(std::min<double>(size, n_));
    }
    return wgs_;
}

// Crossover widths: the window width above which the scans beat cudf's
// kernel, measured at 1e6 and 1e7 rows, ungrouped and in 1e3 groups, clean
// and with 1% NA, for na.rm = FALSE / TRUE (cudf needs a second pass for
// TRUE), the lower of the 1e6 and 1e7 values (scratchpad, F59 round 3,
// cross.txt). cudf costs about 0.04 ms x min(w, group size) per 1e7 rows on
// top of the common window overhead; the scans cost a constant (1e7 rows,
// mean / sum / min / max alike: 6.1 ms clean and 8.0 ms with NA ungrouped,
// 18.4 and 19.9 ms in 1e3 groups, whose row numbers and segment indexes take
// longer to build).
struct Crossover {
    cudf::size_type clean, clean_na_rm, nulls, nulls_na_rm;
};
constexpr Crossover kUngrouped{96, 56, 136, 96};
constexpr Crossover kGrouped{128, 96, 160, 128};

bool roll_native(const std::string& fn, cudf::type_id t, cudf::size_type w, bool na_rm,
                 const RollShape& shape) {
    // the internal switch cuplyr.roll_path (R/roll.R) forces the choice
    // (tests and benchmarks; roll_sd and the integer sum have no cudf
    // alternative worth forcing)
    if (shape.path == RollPath::Cudf && fn != "roll_sd") return false;
    if (shape.path == RollPath::Scans && is_numeric_id(t)) {
        return !(fn == "roll_min" || fn == "roll_max") ||
               t == cudf::type_id::INT32 || t == cudf::type_id::INT64 ||
               t == cudf::type_id::FLOAT32 || t == cudf::type_id::FLOAT64;
    }
    if (fn == "roll_min" || fn == "roll_max") {
        // the scans keep the value type: integers and floats only
        if (!(t == cudf::type_id::INT32 || t == cudf::type_id::INT64 ||
              t == cudf::type_id::FLOAT32 || t == cudf::type_id::FLOAT64)) {
            return false;
        }
    } else {
        if (!is_numeric_id(t)) return false;
        if (fn == "roll_sd") return true;
        // an integer sum is one exact INT64 scan at any width
        if (fn == "roll_sum" && (t == cudf::type_id::INT32 || t == cudf::type_id::BOOL8)) return true;
    }
    const Crossover& c = shape.grouped ? kGrouped : kUngrouped;
    const cudf::size_type limit = shape.has_nulls ? (na_rm ? c.nulls_na_rm : c.nulls)
                                                  : (na_rm ? c.clean_na_rm : c.clean);
    // a window is never wider than its group; the group size (one scan) is
    // only needed when the width alone would pick the scans
    if (w <= limit) return false;
    return !shape.grouped || shape.rows->weighted_group_size() > limit;
}

std::vector<std::unique_ptr<cudf::column>> roll_columns(
        RollRows& rows, cudf::column_view x, const std::vector<RollReq>& reqs) {
    const cudf::size_type n = x.size();
    std::vector<col_ptr> out(reqs.size());
    const cudf::type_id tid = x.type().id();
    const bool int_input = tid == cudf::type_id::INT32 || tid == cudf::type_id::BOOL8;

    auto out_type = [&](const RollReq& rq) {
        if (rq.fn == "roll_min" || rq.fn == "roll_max") return x.type();
        return kFloat64;
    };

    XCol X(x);
    std::map<int, std::vector<size_t>> by_w;  // everything else, by width
    for (size_t i = 0; i < reqs.size(); ++i) {
        const RollReq& rq = reqs[i];
        if (n == 0 || rq.w > n) {
            // no complete window anywhere
            out[i] = null_column_like(out_type(rq), n);
        } else if (rq.w == 1) {
            if (rq.fn == "roll_sd") {
                out[i] = null_column_like(kFloat64, n);
            } else {
                out[i] = cudf::cast(x, out_type(rq));  // min/max/sum/mean of one value
            }
        } else if (rq.fn == "roll_sum" && int_input) {
            // exact in INT64, handed on as the double R's roll_sum() gives
            out[i] = cudf::cast(*int_sum(rows, X, rq), kFloat64);
        } else if (rq.fn == "roll_sum" || rq.fn == "roll_mean" || rq.fn == "roll_sd" ||
                   rq.fn == "roll_min" || rq.fn == "roll_max") {
            by_w[rq.w].push_back(i);
        } else {
            Rcpp::stop("roll_columns: internal: unknown rolling fn '%s'", rq.fn.c_str());
        }
    }
    size_t left = by_w.size();
    for (auto& kv : by_w) width_columns(rows, X, kv.first, kv.second, reqs, out, --left == 0);
    return out;
}

}  // namespace cuplyr
