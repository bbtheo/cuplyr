# dplyr's lifecycle warning for `by = character()`

`lifecycle` derives the package named in "was deprecated in dplyr 1.1.0"
from `env`, so `env` is dplyr's namespace (not cuplyr's), the same call
`dplyr:::warn_join_cross_by()` makes. `user_env` is the user's calling
frame (the frame that called the join verb).

## Usage

``` r
warn_join_cross_by(user_env)
```
