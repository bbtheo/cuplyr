// src/ops_rolling.hpp
//
// F59: width-independent rolling windows for gpu_window()'s "rolling" family
// (roll_sum / roll_mean / roll_sd / roll_min / roll_max). See ops_rolling.cpp
// for the algorithm.
#ifndef CUPLYR_OPS_ROLLING_HPP
#define CUPLYR_OPS_ROLLING_HPP

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cuplyr {

struct RollReq {
    std::string fn;       // "roll_mean" | "roll_sum" | "roll_sd" | "roll_min" | "roll_max"
    cudf::size_type w;    // window width, >= 1
    bool na_rm;
};

// The internal switch cuplyr.roll_path (R/roll.R): the measured choice, or
// one path forced (tests and benchmarks).
enum class RollPath { Auto, Scans, Cudf };

class RollRows;

// What roll_native() needs to know about the frame besides one spec.
struct RollShape {
    bool grouped;       // real group columns
    bool has_nulls;     // the value column holds nulls (or NaN)
    RollRows* rows;     // its row-weighted group size, asked only when the width needs it
    RollPath path;
};

// True when roll_columns() computes `fn` over a value column of type `t` with
// window `w`; otherwise the caller keeps cudf::grouped_rolling_window()
// (min/max of timestamps and booleans, and the narrow windows where cudf's
// direct kernel is the faster of the two: it costs about 0.04 ms x w per 1e7
// rows, a window cannot be wider than its group, and the scans cost a
// constant -- crossover widths in roll_native()'s definition). roll_sd is
// always native: cudf's STD loses accuracy on offset data at every width.
bool roll_native(const std::string& fn, cudf::type_id t, cudf::size_type w, bool na_rm,
                 const RollShape& shape);

// Per-frame row bookkeeping shared by every rolling spec of a frame: the
// 0-based row number within its group and the global row index. `gb` is the
// frame's sorted-keys groupby over `work` order; `grouped` is false when the
// keys are the substituted constant column (then the row number is the index).
class RollRows {
public:
    RollRows(cudf::groupby::groupby& gb, bool grouped, cudf::column_view key0, cudf::size_type n)
        : gb_(&gb), grouped_(grouped), key0_(key0), n_(n) {}
    cudf::size_type size() const { return n_; }
    bool grouped() const { return grouped_; }
    cudf::column_view r();    // INT32, 0-based row within the group (== idx() when ungrouped)
    cudf::column_view idx();  // INT32, 0..n-1
    // sum(size^2) / n over the groups: the size of the group an average row
    // is in, which is what a window's cost depends on (one 9e6-row group next
    // to 1e5 groups of 10 rows weighs 8.1e6, not 90)
    cudf::size_type weighted_group_size();
private:
    cudf::groupby::groupby* gb_;
    bool grouped_;
    cudf::column_view key0_;
    cudf::size_type n_;
    cudf::size_type wgs_ = -1;
    std::unique_ptr<cudf::column> r_, idx_;
};

// One result column per request, in `work` order. `x` is the value column
// (NaN already read as null, F55). Every request must satisfy roll_native().
std::vector<std::unique_ptr<cudf::column>> roll_columns(
    RollRows& rows, cudf::column_view x, const std::vector<RollReq>& reqs);

}  // namespace cuplyr

#endif  // CUPLYR_OPS_ROLLING_HPP
