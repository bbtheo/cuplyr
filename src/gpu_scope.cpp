// src/gpu_scope.cpp
//
// Release scopes (F58): an eager call's intermediate tables are freed when the
// call returns instead of at R's next GC. Two kinds of scope share one stack:
//
// - A step scope, opened by wrap_gpu_call() (R/utils.R) around every GPU step.
//   While it is the innermost scope, make_gpu_table_xptr() records each table
//   it creates in it. Closing it releases what it recorded that is not
//   reachable from the step's value; the tables that are reachable move down
//   to the scope below.
// - A call scope, one per user-level call (a cuplyr function called by code
//   that is not cuplyr's: the console, a user function, a data mask, another
//   package). wrap_gpu_call() opens it, below the call's first step scope, and
//   registers its close on that call's frame (gpu_call_scope_ensure(),
//   R/utils.R). It records nothing itself; it collects what the call's steps
//   kept, and on exit releases those that are not reachable from the call's
//   return value. What the call returns belongs to the user: it is not handed
//   down to any scope below, so an enclosing call can never free it (a table
//   user code makes while a verb evaluates its arguments or predicates).
//
// Only tables a step created are ever released, so a table the user holds (an
// input, a lazy source, the table filter(TRUE) hands back) is never touched.
// Released = the table drops its column references (a column shared with a
// surviving table lives on, F57 ownership rule) and is flagged, so a stale
// XPtr raises an error from get_table_view() rather than look empty.
//
// Scopes record weak_ptr<GpuTable>: if R finalizes an intermediate's XPtr
// mid-scope its memory is freed then, as before, and nothing dangles.

// The package supports R >= 4.3.0; R_mapAttrib() is new in R 4.6.0 and
// R_getVarEx() in R 4.5.0. Older R gets the fallbacks below (R_mapAttrib's is
// the one "Writing R Extensions" prints under "Some backports"). They use
// ATTRIB() and Rf_findVarInFrame(), which R 4.6 declares only under
// ENABLE_LEGACY_NONAPI_FUNS, so that is defined before any R header when a
// fallback is compiled. CUPLYR_FORCE_R_BACKPORTS compiles the fallbacks on any
// R, to test them.
#include <Rversion.h>
#if R_VERSION < R_Version(4, 6, 0) || defined(CUPLYR_FORCE_R_BACKPORTS)
#define CUPLYR_BACKPORT_MAP_ATTRIB 1
#endif
#if R_VERSION < R_Version(4, 5, 0) || defined(CUPLYR_FORCE_R_BACKPORTS)
#define CUPLYR_BACKPORT_GET_VAR 1
#endif
#if (defined(CUPLYR_BACKPORT_MAP_ATTRIB) || defined(CUPLYR_BACKPORT_GET_VAR)) && \
    !defined(ENABLE_LEGACY_NONAPI_FUNS)
#define ENABLE_LEGACY_NONAPI_FUNS 1
#endif

#include "gpu_table.hpp"

#include <Rcpp.h>
#include <R_ext/Altrep.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace {

// Call FUN(tag, value, data) on each attribute of `x` until it returns non-NULL.
#ifdef CUPLYR_BACKPORT_MAP_ATTRIB
SEXP map_attrib(SEXP x, SEXP (*FUN)(SEXP, SEXP, void*), void* data) {
  PROTECT_INDEX api;
  SEXP a = ATTRIB(x);
  SEXP val = NULL;
  PROTECT_WITH_INDEX(a, &api);
  while (a != R_NilValue) {
    SEXP tag = PROTECT(TAG(a));
    SEXP attr = PROTECT(CAR(a));
    val = FUN(tag, attr, data);
    UNPROTECT(2);
    if (val != NULL) break;
    REPROTECT(a = CDR(a), api);
  }
  UNPROTECT(1);
  return val;
}
#else
SEXP map_attrib(SEXP x, SEXP (*FUN)(SEXP, SEXP, void*), void* data) {
  return R_mapAttrib(x, FUN, data);
}
#endif

// The value `sym` is bound to in `frame` itself (no enclosures), or
// R_NilValue if it is unbound there. Never errors. The fallback returns a
// promise unforced (R_getVarEx forces one); the only variable read here is
// one this file defines, which is never a promise.
#ifdef CUPLYR_BACKPORT_GET_VAR
SEXP frame_get(SEXP sym, SEXP frame) {
  SEXP v = Rf_findVarInFrame(frame, sym);
  return v == R_UnboundValue ? R_NilValue : v;
}
#else
SEXP frame_get(SEXP sym, SEXP frame) {
  return R_getVarEx(sym, frame, FALSE, R_NilValue);
}
#endif

using cuplyr::GpuTable;
using cuplyr::GpuTablePtr;

struct Scope {
  double id;   // exact in a double up to 2^53, never reused
  bool call;   // call scope (barrier) or step scope
  std::vector<std::weak_ptr<GpuTable>> tables;
};

std::vector<Scope> g_scopes;
std::int64_t g_next_scope_id = 0;

// The variable a call scope's id is kept in, in the frame of the call it
// belongs to (a cuplyr function's frame, so the name can't meet user code).
SEXP call_scope_sym() {
  static SEXP sym = Rf_install(".__cuplyr_call_scope__");
  return sym;
}

void push_child(std::vector<SEXP>& stack, SEXP x) {
  if (x != R_NilValue) stack.push_back(x);
}

SEXP push_attribute(SEXP /* tag */, SEXP value, void* data) {
  push_child(*static_cast<std::vector<SEXP>*>(data), value);
  return NULL;  // NULL: go on to the next attribute
}

// Collect the table of every GpuTable XPtr reachable from `root` through
// lists, pairlists, attributes (a list_of's ptype) and ALTREP payloads (the
// row-handle vector of vec_proxy() holds its source table). Iterative, so a
// long lazy chain (an AST nested once per verb) has no depth limit.
// Environments, closures and calls are not entered: a table held only there
// is not found, and a scope would release it.
// A list, pairlist or ALTREP reached twice (a subtree shared by a lazy plan of
// repeated self-joins) is walked once.
void collect_kept(SEXP root, std::unordered_set<const GpuTable*>& kept) {
  std::vector<SEXP> stack;
  std::unordered_set<SEXP> visited;
  push_child(stack, root);
  while (!stack.empty()) {
    SEXP x = stack.back();
    stack.pop_back();
    int type = TYPEOF(x);
    if (type == EXTPTRSXP) {
      if (R_ExternalPtrTag(x) == cuplyr::gpu_table_xptr_tag()) {
        auto* p = static_cast<GpuTablePtr*>(R_ExternalPtrAddr(x));
        if (p != nullptr && *p) kept.insert(p->get());
      }
      continue;
    }
    if (type == LISTSXP) {
      if (!visited.insert(x).second) continue;
      // A pairlist can be dotted (a deferred string's ALTREP payload is
      // CONS(arg, info)): its last CDR is then a value, not a cell.
      SEXP c = x;
      for (; TYPEOF(c) == LISTSXP; c = CDR(c)) push_child(stack, CAR(c));
      push_child(stack, c);
      continue;
    }
    bool vector = type == VECSXP || type == LGLSXP || type == INTSXP ||
                  type == REALSXP || type == CPLXSXP || type == STRSXP || type == RAWSXP;
    if (!vector) continue;
    if ((type == VECSXP || ALTREP(x)) && !visited.insert(x).second) continue;
    map_attrib(x, push_attribute, &stack);
    if (ALTREP(x)) {
      // not its elements: they may be computed on access
      push_child(stack, R_altrep_data1(x));
      push_child(stack, R_altrep_data2(x));
    } else if (type == VECSXP) {
      R_xlen_t n = Rf_xlength(x);
      for (R_xlen_t i = 0; i < n; ++i) push_child(stack, VECTOR_ELT(x, i));
    }
  }
}

bool any_live(const Scope& s) {
  for (const auto& w : s.tables) {
    if (!w.expired()) return true;
  }
  return false;
}

double open_scope(bool call) {
  double id = static_cast<double>(++g_next_scope_id);
  g_scopes.push_back({id, call, {}});
  return id;
}

bool is_open(double id) {
  for (const auto& s : g_scopes) {
    if (s.id == id) return true;
  }
  return false;
}

// Close scope `id` and any scope left open above it (an interrupt between an
// open and the handler that closes it). Each releases what it holds except the
// tables reachable from `keep`; a step scope hands those to the scope below, a
// call scope stops tracking them. A call scope left open above `id` (its close
// never ran) is dropped without releasing anything: what it holds can be what
// its call already returned to the user. `keep` is walked at most once, and
// only if a closing scope still holds a live table. An `id` that is not open
// is a no-op.
void close_scope(double id, SEXP keep) {
  if (!is_open(id)) return;
  std::unordered_set<const GpuTable*> kept;
  bool walked = false;
  while (!g_scopes.empty()) {
    Scope scope = std::move(g_scopes.back());
    g_scopes.pop_back();
    bool orphaned_call = scope.call && scope.id != id;
    if (!orphaned_call && any_live(scope)) {
      if (!walked) {
        collect_kept(keep, kept);
        walked = true;
      }
      for (auto& w : scope.tables) {
        auto t = w.lock();
        if (!t) continue;
        if (kept.count(t.get())) {
          if (!scope.call && !g_scopes.empty()) g_scopes.back().tables.push_back(w);
        } else {
          t->columns.clear();
          t->columns.shrink_to_fit();
          t->released = true;
        }
      }
    }
    if (scope.id == id) break;
  }
}

}  // namespace

namespace cuplyr {

SEXP gpu_table_xptr_tag() {
  static SEXP tag = Rf_install("cuplyr_gpu_table");
  return tag;
}

// Only a step scope records: a table made outside any GPU step (directly in a
// call scope, or with no scope open) is left to R's GC.
void gpu_scope_record(const GpuTablePtr& table) {
  if (!g_scopes.empty() && !g_scopes.back().call) g_scopes.back().tables.push_back(table);
}

}  // namespace cuplyr

// Open a step scope; returns its id.
// [[Rcpp::export(rng = false)]]
double gpu_scope_open() {
  return open_scope(false);
}

// Close scope `id`, keeping the tables in `keep` (any R value; see
// collect_kept()).
// [[Rcpp::export(rng = false)]]
void gpu_scope_close(double id, SEXP keep) {
  close_scope(id, keep);
}

// Does the call running in `frame` have an open call scope?
// [[Rcpp::export(rng = false)]]
bool gpu_call_scope_active(SEXP frame) {
  if (TYPEOF(frame) != ENVSXP) return false;
  SEXP v = frame_get(call_scope_sym(), frame);
  return TYPEOF(v) == REALSXP && Rf_xlength(v) == 1 && is_open(REAL(v)[0]);
}

// Open a call scope for the call running in `frame` and store its id there,
// in one step (no interrupt can fall between the two). The caller registers
// gpu_call_scope_exit() on that frame first.
// [[Rcpp::export(rng = false)]]
double gpu_call_scope_open(SEXP frame) {
  if (TYPEOF(frame) != ENVSXP) Rcpp::stop("gpu_call_scope_open(): `frame` must be an environment");
  double id = open_scope(true);
  Rf_defineVar(call_scope_sym(), Rf_ScalarReal(id), frame);
  return id;
}

// Close the call scope of the call running in `frame` (if it has an open one),
// keeping `keep`: the call's return value, NULL when it failed.
// [[Rcpp::export(rng = false)]]
void gpu_call_scope_exit(SEXP frame, SEXP keep) {
  if (TYPEOF(frame) != ENVSXP) return;
  SEXP v = frame_get(call_scope_sym(), frame);
  if (TYPEOF(v) == REALSXP && Rf_xlength(v) == 1) close_scope(REAL(v)[0], keep);
}

// [[Rcpp::export(rng = false)]]
int gpu_scope_depth() {
  return static_cast<int>(g_scopes.size());
}
