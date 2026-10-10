// src/ops_gather.cpp
//
// Primitives behind the vctrs contract on tbl_gpu (R/vctrs.R, feature F02).
//
// - gpu_handle_src(): the `.src` column of the row-handle frame
//   vec_proxy.tbl_gpu() returns (the `.row` column is R's own compact 1:n).
//   It is the constant integer `id` repeated `n` times, as an ALTREP vector, so
//   vec_size() on a table costs O(1) memory and nothing is materialized until
//   something actually reads the elements (e.g. vec_slice() of the proxy). Once
//   materialized (DATAPTR), element reads see the buffer, so a write through
//   DATAPTR is never hidden by a stale Elt()/Get_region(). The ALTREP object
//   also holds `keep`, an R object (the computed tbl_gpu a lazy table's proxy
//   points at) that outlives the handles as long as this vector does.
// - gpu_gather_rows(): gather rows addressed by (source, row) handles out of
//   one or several tables of identical layout. Several sources are
//   concatenated on the device and gathered once; NA indices yield all-null
//   rows (cudf NULLIFY policy). gpu_slice_indices() can't be used for this: it
//   has slice() semantics and drops NA.

#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <R_ext/Altrep.h>
#include <R_ext/Rdynload.h>
#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace Rcpp;

namespace {

R_altrep_class_t src_class;

// data1 = list(c(id, n), keep); data2 = materialized INTSXP or NULL
inline int src_id(SEXP x) {
    return static_cast<int>(REAL(VECTOR_ELT(R_altrep_data1(x), 0))[0]);
}

R_xlen_t src_length(SEXP x) {
    return static_cast<R_xlen_t>(REAL(VECTOR_ELT(R_altrep_data1(x), 0))[1]);
}

int src_elt(SEXP x, R_xlen_t i) {
    SEXP m = R_altrep_data2(x);
    return m == R_NilValue ? src_id(x) : INTEGER(m)[i];
}

R_xlen_t src_get_region(SEXP x, R_xlen_t i, R_xlen_t n, int* buf) {
    R_xlen_t len = src_length(x);
    R_xlen_t m = (len - i < n) ? len - i : n;
    SEXP d = R_altrep_data2(x);
    if (d == R_NilValue) {
        std::fill(buf, buf + m, src_id(x));
    } else {
        const int* p = INTEGER(d);
        std::copy(p + i, p + i + m, buf);
    }
    return m;
}

void* src_dataptr(SEXP x, Rboolean) {
    SEXP m = R_altrep_data2(x);
    if (m == R_NilValue) {
        R_xlen_t len = src_length(x);
        m = PROTECT(Rf_allocVector(INTSXP, len));
        std::fill(INTEGER(m), INTEGER(m) + len, src_id(x));
        R_set_altrep_data2(x, m);
        UNPROTECT(1);
    }
    return INTEGER(m);
}

const void* src_dataptr_or_null(SEXP x) {
    SEXP m = R_altrep_data2(x);
    return m == R_NilValue ? nullptr : static_cast<const void*>(INTEGER(m));
}

} // namespace

// Registered from R_init_cuplyr by Rcpp (needs the package's DllInfo).
// [[Rcpp::init]]
void cuplyr_init_altrep(DllInfo* dll) {
    src_class = R_make_altinteger_class("cuplyr_row_src", "cuplyr", dll);
    R_set_altrep_Length_method(src_class, src_length);
    R_set_altvec_Dataptr_method(src_class, src_dataptr);
    R_set_altvec_Dataptr_or_null_method(src_class, src_dataptr_or_null);
    R_set_altinteger_Elt_method(src_class, src_elt);
    R_set_altinteger_Get_region_method(src_class, src_get_region);
}

// [[Rcpp::export]]
SEXP gpu_handle_src(double id, double n, SEXP keep) {
    SEXP d = PROTECT(Rf_allocVector(VECSXP, 2));
    SEXP bn = PROTECT(Rf_allocVector(REALSXP, 2));
    REAL(bn)[0] = id;
    REAL(bn)[1] = n;
    SET_VECTOR_ELT(d, 0, bn);
    SET_VECTOR_ELT(d, 1, keep);
    SEXP out = R_new_altrep(src_class, d, R_NilValue);
    UNPROTECT(2);
    return out;
}

// idx: 0-based row indices into the row-wise concatenation of `table_ptrs`
// (NA = all-null output row). Every table must have the same layout.
// [[Rcpp::export]]
SEXP gpu_gather_rows(List table_ptrs, NumericVector idx) {
    using namespace cuplyr;

    if (table_ptrs.size() == 0) {
        Rcpp::stop("gpu_gather_rows() needs at least one source table");
    }

    std::vector<cudf::table_view> views;
    views.reserve(table_ptrs.size());
    for (int i = 0; i < table_ptrs.size(); ++i) {
        Rcpp::XPtr<GpuTablePtr> p(table_ptrs[i]);
        views.push_back(get_table_view(p));
    }

    std::unique_ptr<cudf::table> joined;
    cudf::table_view view = views[0];
    if (views.size() > 1) {
        joined = cudf::concatenate(views);
        view = joined->view();
    }

    if (idx.size() == 0) {
        return make_gpu_table_xptr(cudf::empty_like(view));
    }

    const cudf::size_type total = view.num_rows();
    std::vector<cudf::size_type> host(static_cast<size_t>(idx.size()));
    for (R_xlen_t i = 0; i < idx.size(); ++i) {
        double v = idx[i];
        if (NumericVector::is_na(v) || v < 0 || v >= static_cast<double>(total)) {
            host[static_cast<size_t>(i)] = total;  // out of range -> NULLIFY
        } else {
            host[static_cast<size_t>(i)] = static_cast<cudf::size_type>(v);
        }
    }

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    rmm::device_uvector<cudf::size_type> d_idx(host.size(), stream);
    check_cuda(
        cudaMemcpy(d_idx.data(), host.data(), host.size() * sizeof(cudf::size_type),
                   cudaMemcpyHostToDevice),
        "row gather-map upload");

    cudf::column_view gather_map(
        cudf::device_span<cudf::size_type const>(d_idx.data(), d_idx.size()));

    auto result = cudf::gather(view, gather_map, cudf::out_of_bounds_policy::NULLIFY,
                               stream, mr);
    return make_gpu_table_xptr(std::move(result));
}
