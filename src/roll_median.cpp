// src/roll_median.cpp
//
// F59: CPU sliding-window median behind R/roll.R's roll_median_sorted() (the
// default, exact engine of roll_median(); F66). Same semantics as
// roll_apply_generic(x, n, stats::median, na.rm): right-aligned windows of n
// rows, a NaN is missing (F55), na.rm = FALSE gives NA for a window holding
// any missing value, na.rm = TRUE drops them (an empty window is NA).
//
// Algorithm: every non-missing value gets a unique rank (stable sort by
// value, then position); a Fenwick tree over the ranks counts the window's
// members, so each row costs one insert, one delete and one or two k-th
// element descents: O(n log n) overall, independent of the window width.
//
// The median of an even count is R's mean() of the two middle values, in
// long double, with R's own refinement pass (src/main/summary.c real_mean()),
// so the result is identical() to the reference, including the Inf + -Inf =
// NaN case and midpoints that would overflow a double sum.

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

double mean2(double a, double b) {
    long double s = static_cast<long double>(a) + static_cast<long double>(b);
    s /= 2;
    if (std::isfinite(static_cast<double>(s))) {
        long double t = (static_cast<long double>(a) - s) + (static_cast<long double>(b) - s);
        s += t / 2;
    }
    return static_cast<double>(s);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::NumericVector roll_median_cpp(Rcpp::NumericVector x, int n, bool na_rm) {
    const R_xlen_t nx = x.size();
    Rcpp::NumericVector out(nx, NA_REAL);
    if (nx == 0 || n > nx) return out;

    // rank[i] = 1-based rank of x[i] among the non-missing values, 0 if missing.
    std::vector<int> order;
    order.reserve(static_cast<size_t>(nx));
    for (R_xlen_t i = 0; i < nx; ++i) {
        if (!std::isnan(x[i])) order.push_back(static_cast<int>(i));
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x[a] < x[b]; });
    const int m = static_cast<int>(order.size());
    std::vector<int> rank(static_cast<size_t>(nx), 0);
    std::vector<double> sorted_vals(static_cast<size_t>(m) + 1);
    for (int k = 0; k < m; ++k) {
        rank[static_cast<size_t>(order[static_cast<size_t>(k)])] = k + 1;
        sorted_vals[static_cast<size_t>(k) + 1] = x[order[static_cast<size_t>(k)]];
    }
    std::vector<int>().swap(order);

    std::vector<int> bit(static_cast<size_t>(m) + 1, 0);
    int log2m = 1;
    while ((1 << log2m) <= m) ++log2m;
    auto add = [&](int pos, int delta) {
        for (; pos <= m; pos += pos & -pos) bit[static_cast<size_t>(pos)] += delta;
    };
    // smallest rank whose prefix count reaches k (1 <= k <= window count)
    auto kth = [&](int k) {
        int pos = 0;
        for (int b = log2m; b >= 0; --b) {
            int nxt = pos + (1 << b);
            if (nxt <= m && bit[static_cast<size_t>(nxt)] < k) {
                pos = nxt;
                k -= bit[static_cast<size_t>(nxt)];
            }
        }
        return sorted_vals[static_cast<size_t>(pos) + 1];
    };

    int count = 0;     // non-missing values in the window
    int n_missing = 0;
    for (R_xlen_t i = 0; i < nx; ++i) {
        int r = rank[static_cast<size_t>(i)];
        if (r) { add(r, 1); ++count; } else { ++n_missing; }
        if (i >= n) {
            int ro = rank[static_cast<size_t>(i - n)];
            if (ro) { add(ro, -1); --count; } else { --n_missing; }
        }
        if (i < n - 1) continue;
        if (!na_rm && n_missing > 0) continue;
        if (count == 0) continue;
        if (count % 2 == 1) {
            out[i] = kth((count + 1) / 2);
        } else {
            out[i] = mean2(kth(count / 2), kth(count / 2 + 1));
        }
    }
    return out;
}
