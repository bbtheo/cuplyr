# F67 acceptance-test helpers: round(x, digits) and signed-zero literals.
#
# expect_same_as_dplyr() compares with a tolerance and treats 0 and -0 as
# equal, so the F67 checks also require identical() results and the same sign
# of every zero (read from 1 / v, since identical(0, -0) is TRUE).

# Positions of negative zeros in each double column.
f67_neg_zeros <- function(tbl) {
  lapply(tbl, function(v) if (is.double(v)) which(!is.na(v) & v == 0 & 1 / v < 0) else integer())
}

# A short description of where two results differ, or "" when they agree
# exactly (values, types, NA vs NaN, sign of zero).
f67_diff <- function(got, want) {
  if (!identical(names(got), names(want))) {
    return(paste("names differ:", toString(names(got)), "vs", toString(names(want))))
  }
  if (nrow(got) != nrow(want)) {
    return(sprintf("row counts differ: %d vs %d", nrow(got), nrow(want)))
  }
  out <- character()
  for (nm in names(want)) {
    a <- got[[nm]]
    b <- want[[nm]]
    if (!identical(typeof(a), typeof(b)) || !identical(class(a), class(b))) {
      out <- c(out, sprintf("%s: type %s/%s vs %s/%s", nm, typeof(a), toString(class(a)),
                            typeof(b), toString(class(b))))
      next
    }
    if (is.double(a)) {
      same <- (is.na(a) & is.na(b) & is.nan(a) == is.nan(b)) |
        (!is.na(a) & !is.na(b) & a == b & (a != 0 | (1 / a > 0) == (1 / b > 0)))
      same[is.na(same)] <- FALSE
    } else {
      same <- (is.na(a) & is.na(b)) | (!is.na(a) & !is.na(b) & a == b)
      same[is.na(same)] <- FALSE
    }
    bad <- which(!same)
    if (length(bad) > 0) {
      k <- utils::head(bad, 5)
      out <- c(out, sprintf(
        "%s: %d of %d rows differ, e.g. row %s: GPU %s, R %s", nm, length(bad), length(a),
        toString(k), toString(format(a[k], digits = 17)), toString(format(b[k], digits = 17))
      ))
    }
  }
  if (length(out) == 0 && !identical(got, want)) {
    out <- "results not identical() (attributes)"
  }
  paste(out, collapse = "; ")
}

# Run `pipeline` with dplyr on a tibble and on a tbl_gpu, eager and lazy,
# through the parity oracle, then require identical() results and the same
# zero signs. `native = TRUE` runs the GPU side with cuplyr.fallback = "error",
# so a CPU fallback can't hide a wrong GPU result.
f67_check <- function(df, pipeline, native = TRUE) {
  if (native) {
    withr::local_options(cuplyr.fallback = "error")
  }
  res <- list(
    eager = expect_same_as_dplyr(df, pipeline),
    lazy = expect_same_as_dplyr_lazy(df, pipeline)
  )
  for (mode in names(res)) {
    d <- f67_diff(res[[mode]]$gpu, res[[mode]]$oracle)
    expect(identical(d, ""), sprintf("GPU (%s) differs from dplyr: %s", mode, d))
    expect_identical(f67_neg_zeros(res[[mode]]$gpu), f67_neg_zeros(res[[mode]]$oracle),
                     info = sprintf("signs of zero (%s)", mode))
  }
  invisible(res)
}
