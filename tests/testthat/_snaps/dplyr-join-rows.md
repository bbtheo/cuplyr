# many-to-many auto-warning matches dplyr exactly

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Detected an unexpected many-to-many relationship between `x` and `y`.
      i Row 3 of `x` matches multiple rows in `y`.
      i Row 1 of `y` matches multiple rows in `x`.
      i If a many-to-many relationship is expected, set `relationship = "many-to-many"` to silence this warning. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Detected an unexpected many-to-many relationship between `x` and `y`.
      i Row 3 of `x` matches multiple rows in `y`.
      i Row 1 of `y` matches multiple rows in `x`.
      i If a many-to-many relationship is expected, set `relationship = "many-to-many"` to silence this warning. 

# relationship = 'one-to-one' violation (y-side duplicate) matches dplyr

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 

# relationship = 'one-to-one' violation (x-side duplicate) matches dplyr

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 1 of `y` matches multiple rows in `x`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 1 of `y` matches multiple rows in `x`. 

# relationship = 'one-to-one' tie-break (both sides violate at the same row) matches dplyr

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 

# relationship = 'one-to-many' violation (x-side duplicate) matches dplyr

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 1 of `y` matches multiple rows in `x`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 1 of `y` matches multiple rows in `x`. 

# relationship = 'many-to-one' violation (y-side duplicate) matches dplyr

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 

# unmatched='error' x-side (matches_nothing) matches dplyr for inner_join()/right_join()

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 2 of `x` does not have a match. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 2 of `x` does not have a match. 

---

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 2 of `x` does not have a match. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 2 of `x` does not have a match. 

# unmatched='error' y-side (matches_remaining) matches dplyr for left_join()/inner_join()

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row of `y` must be matched by `x`.
      i Row 3 of `y` was not matched. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row of `y` must be matched by `x`.
      i Row 3 of `y` was not matched. 

---

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row of `y` must be matched by `x`.
      i Row 3 of `y` was not matched. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row of `y` must be matched by `x`.
      i Row 3 of `y` was not matched. 

# precedence: relationship violation fires before unmatched= violation

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 

---

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 1 of `x` matches multiple rows in `y`. 

# precedence: x-side unmatched= fires before y-side unmatched= (inner_join)

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 2 of `x` does not have a match. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 2 of `x` does not have a match. 

# first-offender row numbers stay correct under a shuffled row order

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 4 of `y` matches multiple rows in `x`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 4 of `y` matches multiple rows in `x`. 

---

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 7 of `y` matches multiple rows in `x`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `y` must match at most 1 row in `x`.
      i Row 7 of `y` matches multiple rows in `x`. 

# first-offender row numbers stay correct for unmatched= under a shuffled row order

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 5 of `x` does not have a match. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row of `x` must have a match in `y`.
      i Row 5 of `x` does not have a match. 

# na_matches = 'na' (default): NA keys count toward multiplicity/relationship checks

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Detected an unexpected many-to-many relationship between `x` and `y`.
      i Row 2 of `x` matches multiple rows in `y`.
      i Row 2 of `y` matches multiple rows in `x`.
      i If a many-to-many relationship is expected, set `relationship = "many-to-many"` to silence this warning. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Detected an unexpected many-to-many relationship between `x` and `y`.
      i Row 2 of `x` matches multiple rows in `y`.
      i Row 2 of `y` matches multiple rows in `x`.
      i If a many-to-many relationship is expected, set `relationship = "many-to-many"` to silence this warning. 

---

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 2 of `x` matches multiple rows in `y`. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row in `x` must match at most 1 row in `y`.
      i Row 2 of `x` matches multiple rows in `y`. 

# na_matches = 'never': NA keys never match, so they surface as unmatched= violations instead

    Code
      cat("dplyr:\n")
    Output
      dplyr:
    Code
      cat(rlang::cnd_message(dp_cnd), "\n")
    Output
      Each row of `y` must be matched by `x`.
      i Row 2 of `y` was not matched. 
    Code
      cat("cuplyr:\n")
    Output
      cuplyr:
    Code
      cat(rlang::cnd_message(cu_cnd), "\n")
    Output
      Each row of `y` must be matched by `x`.
      i Row 2 of `y` was not matched. 

