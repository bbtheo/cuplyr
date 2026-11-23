# filter() missing-column errors (snapshot)

    Code
      filter(d, xx > 1)
    Condition
      Error in `filter()`:
      i In argument: `xx > 1`.
      Caused by error:
      ! object 'xx' not found

---

    Code
      filter(l, xx > 1)
    Condition
      Error in `filter()`:
      i In argument: `xx > 1`.
      Caused by error:
      ! object 'xx' not found

---

    Code
      filter(group_by(d, g), xx > 1)
    Condition
      Error in `filter()`:
      i In argument: `xx > 1`.
      i In group 1: `g = "a"`.
      Caused by error:
      ! object 'xx' not found

---

    Code
      filter(l, x > 1, .data$xx > 1)
    Condition
      Error in `filter()`:
      i In argument: `.data$xx > 1`.
      Caused by error in `.data$xx`:
      ! Column `xx` not found in `.data`.

# arrange() missing-column errors (snapshot)

    Code
      arrange(d, xx)
    Condition
      Error in `arrange()`:
      i In argument: `..1 = xx`.
      Caused by error:
      ! object 'xx' not found

---

    Code
      arrange(l, x, desc(xx))
    Condition
      Error in `arrange()`:
      i In argument: `..2 = xx`.
      Caused by error:
      ! object 'xx' not found

# summarise() missing-column errors (snapshot)

    Code
      summarise(d, m = mean(xx))
    Condition
      Error in `summarise()`:
      i In argument: `m = mean(xx)`.
      Caused by error:
      ! object 'xx' not found

---

    Code
      summarise(group_by(l, g), m = mean(xx))
    Condition
      Error in `summarise()`:
      i In argument: `m = mean(xx)`.
      i In group 1: `g = "a"`.
      Caused by error:
      ! object 'xx' not found

---

    Code
      summarise(d, across(xx, mean))
    Condition
      Error in `summarise()`:
      i In argument: `across(xx, mean)`.
      Caused by error in `across()`:
      ! Can't select columns that don't exist.
      x Column `xx` doesn't exist.

---

    Code
      count(l, g, wt = xx)
    Condition
      Error in `count()`:
      i In argument: `n = base::sum(xx, na.rm = TRUE)`.
      i In group 1: `g = "a"`.
      Caused by error:
      ! object 'xx' not found

# mutate() and group_by() missing-column errors (snapshot)

    Code
      mutate(d, z = xx + 1)
    Condition
      Error in `mutate()`:
      i In argument: `z = xx + 1`.
      Caused by error:
      ! object 'xx' not found

---

    Code
      mutate(l, z = .data$xx)
    Condition
      Error in `mutate()`:
      i In argument: `z = .data$xx`.
      Caused by error in `.data$xx`:
      ! Column `xx` not found in `.data`.

---

    Code
      group_by(d, xx)
    Condition
      Error in `group_by()`:
      ! Must group by variables found in `.data`.
      x Column `xx` is not found.

---

    Code
      group_by(l, xx, yy)
    Condition
      Error in `group_by()`:
      ! Must group by variables found in `.data`.
      Column `xx` is not found.
      Column `yy` is not found.

# tidyselect missing-column errors (snapshot)

    Code
      select(d, xx)
    Condition
      Error in `select()`:
      ! Can't select columns that don't exist.
      x Column `xx` doesn't exist.

---

    Code
      relocate(l, xx)
    Condition
      Error in `relocate()`:
      ! Can't select columns that don't exist.
      x Column `xx` doesn't exist.

---

    Code
      rename(d, a = xx)
    Condition
      Error in `rename()`:
      ! Can't rename columns that don't exist.
      x Column `xx` doesn't exist.

# join missing-column errors (snapshot)

    Code
      left_join(d, d2, by = "xx")
    Condition
      Error in `left_join()`:
      ! Join columns in `x` must be present in the data.
      x Problem with `xx`.

---

    Code
      inner_join(l, l2, by = c(x = "zz"))
    Condition
      Error in `inner_join()`:
      ! Join columns in `y` must be present in the data.
      x Problem with `zz`.

---

    Code
      left_join(d, d2, by = c("xx", "yy"))
    Condition
      Error in `left_join()`:
      ! Join columns in `x` must be present in the data.
      x Problem with `xx` and `yy`.

