# F53 acceptance tests added after the round-3 review (B7).
# dplyr behaviour verified by the reviewer's probe
# (scratchpad/parity/features/F53/runs/review-3/probe1.log) and by the PM with tools/r.
#
# Only the is.nan() result is compared: a NaN that comes from a NaN literal
# reaches the device as a null (pre-existing on dev, review-3 N1), so the
# intermediate column `y` may come back NA where dplyr has NaN. That gap is
# outside F53. What F53 must not do is turn the is.nan() answer, which dev got
# right through the CPU route, into a silent FALSE.

# ---- B7: is.nan() of a column built from a NaN literal earlier in the same call --

test_that("is.nan() of an earlier dot built from a NaN literal matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  th <- NaN
  v <- c(NaN, 1, NaN)
  pipes <- list(
    p1 = function(d) dplyr::mutate(d, y = x * th, z = is.nan(y)),
    p2 = function(d) dplyr::mutate(d, y = dplyr::if_else(x > 1, NaN, x), z = is.nan(y)),
    p3 = function(d) dplyr::mutate(d, y = NaN, z = is.nan(y)),
    p4 = function(d) dplyr::mutate(d, y = 0/0, z = is.nan(y)),
    p5 = function(d) dplyr::transmute(d, y = x * th, z = is.nan(y)),
    p6 = function(d) dplyr::mutate(d, y = th, z = is.nan(y)),
    p7 = function(d) dplyr::mutate(d, y = v, z = is.nan(y)),
    # the null carries forward through a later dot
    p8 = function(d) dplyr::mutate(d, y = x * th, y2 = y + 1, z = is.nan(y2)),
    p9 = function(d) dplyr::mutate(d, y = x * th, z = !is.nan(y) & x > 1),
    p10 = function(d) dplyr::mutate(dplyr::group_by(d, g = x > 1), y = x * th, z = is.nan(y))
  )
  for (nm in names(pipes)) {
    p <- pipes[[nm]]
    want <- p(df)
    for (lazy in c(FALSE, TRUE)) {
      got <- collect(p(tbl_gpu(df, lazy = lazy)))
      expect_identical(got$z, want$z, info = paste(nm, "lazy =", lazy))
    }
  }
})

# F53-a2: this test used to pin a native is.nan() (cuplyr.fallback = "error")
# and the NaN in `y`. is.nan() is back on dev's CPU route, which re-uploads `y`
# and so maps its NaN to null (pre-existing, F55). Only dplyr's `z` is pinned.
test_that("is.nan() of an earlier dot holding a computed NaN matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  pipe <- function(d) dplyr::mutate(d, y = (x - x) / (x - x), z = is.nan(y))
  want <- pipe(df)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(pipe(tbl_gpu(df, lazy = lazy)))
    expect_identical(got$z, want$z, info = paste("lazy =", lazy))
  }
})
