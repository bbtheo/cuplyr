# The reason a verb call's dot goes to the CPU

Reads what the call's own decision recorded (see the collector notes in
`R/fallback.R`): the reason claimed for the failing dot, with that dot's
text as the `"expr"` attribute (which
[`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md)
takes as the notice's `expr`). Call it only right after the judge of the
dots said one fails.

## Usage

``` r
fallback_failed_reason(dots, default)
```

## Arguments

- dots:

  The dots the claim's index refers to (the user's)

- default:

  The verb's own reason, used when the call claimed no dot or the claim
  names no cause (then without, or with, an `expr`)

## Value

A reason string, with an `"expr"` attribute when a dot is known
