# explain() shows the optimized plan with the scan's columns and the CPU step

    Code
      explain(q)
    Output
      <cuplyr plan: optimized>
      CPU fallback: mutate() for pnorm(gain / 10)
        reason: `pnorm()` has no GPU implementation.
        columns read: 2 of 4 (carrier, z)
        input (computed at the mutate() call):
          parquet scan: flights.parquet
            columns read: 5 of 41 (carrier, distance, air_time, dep_delay, arr_delay)
            reader filter: distance > 1000L
          mutate(gain = dep_delay - arr_delay, speed = distance / air_time * 60)
          filter(speed > 400, gain > 0)
          select(carrier, speed, gain)
      filter(z > 0.5)
      select(carrier, z)

# explain(optimized = FALSE) shows the plan as written

    Code
      explain(q, optimized = FALSE)
    Output
      <cuplyr plan: as written>
      CPU fallback: mutate() for pnorm(gain / 10)
        reason: `pnorm()` has no GPU implementation.
        columns read: all 4
        input (computed at the mutate() call):
          parquet scan: flights.parquet
            columns read: all 41
          filter(distance > 1000L)
          mutate(gain = dep_delay - arr_delay)
          mutate(speed = distance / air_time * 60)
          filter(speed > 400, gain > 0)
          select(carrier, speed, gain)
      filter(z > 0.5)
      select(carrier, z)

# show_query() prints the same plan as explain()

    Code
      show_query(q)
    Output
      <cuplyr plan: optimized>
      CPU fallback: mutate() for pnorm(gain / 10)
        reason: `pnorm()` has no GPU implementation.
        columns read: 2 of 4 (carrier, z)
        input (computed at the mutate() call):
          parquet scan: flights.parquet
            columns read: 5 of 41 (carrier, distance, air_time, dep_delay, arr_delay)
            reader filter: distance > 1000L
          mutate(gain = dep_delay - arr_delay, speed = distance / air_time * 60)
          filter(speed > 400, gain > 0)
          select(carrier, speed, gain)
      filter(z > 0.5)
      select(carrier, z)

---

    Code
      show_query(q, optimized = FALSE)
    Output
      <cuplyr plan: as written>
      CPU fallback: mutate() for pnorm(gain / 10)
        reason: `pnorm()` has no GPU implementation.
        columns read: all 4
        input (computed at the mutate() call):
          parquet scan: flights.parquet
            columns read: all 41
          filter(distance > 1000L)
          mutate(gain = dep_delay - arr_delay)
          mutate(speed = distance / air_time * 60)
          filter(speed > 400, gain > 0)
          select(carrier, speed, gain)
      filter(z > 0.5)
      select(carrier, z)

# the plan before the CPU step is shown, fused and pruned, while it is pending

    Code
      explain(q)
    Output
      <cuplyr plan: optimized>
      parquet scan: flights.parquet
        columns read: 5 of 41 (carrier, distance, air_time, dep_delay, arr_delay)
        reader filter: distance > 1000L
      mutate(gain = dep_delay - arr_delay, speed = distance / air_time * 60)
      filter(speed > 400, gain > 0)
      select(carrier, speed, gain)
    Code
      explain(q, optimized = FALSE)
    Output
      <cuplyr plan: as written>
      parquet scan: flights.parquet
        columns read: all 41
      filter(distance > 1000L)
      mutate(gain = dep_delay - arr_delay)
      mutate(speed = distance / air_time * 60)
      filter(speed > 400, gain > 0)
      select(carrier, speed, gain)

# each step prints as the dplyr call, with its expressions as R code

    Code
      show_query(q, optimized = FALSE)
    Output
      <cuplyr plan: as written>
      table on the GPU
        columns read: all 4
      filter(!is.na(x), x %in% c(1, 3) | y == "q")
      mutate(r = if_else(x > 1, 10, 20), l = log(x) + 1, k = -i)
      arrange(desc(x), y)
      group_by(g)
      summarise(n = n(), m = mean(x, na.rm = TRUE))

# a step longer than 80 characters is split one argument per line

    Code
      show_query(q, optimized = FALSE)
    Output
      <cuplyr plan: as written>
      table on the GPU
        columns read: all 4
      mutate(
        first_long_name = x * 2 + 1,
        second_long_name = log(x) - 3,
        third_long_name = -i
      )

# a join shows its right side as `y` under the join

    Code
      show_query(q, optimized = FALSE)
    Output
      <cuplyr plan: as written>
      table on the GPU
        columns read: all 2
      left_join(y, by = join_by(k))
        y:
          table on the GPU
            columns read: all 2
          filter(b != "q")
      filter(a > 1)

# a CPU step on a table with nothing pending is still shown

    Code
      explain(q)
    Output
      <cuplyr plan: optimized>
      CPU fallback: mutate() for pnorm(x)
        reason: `pnorm()` has no GPU implementation.
        columns read: all 5
        input (computed at the mutate() call):
          table on the GPU
            columns read: all 4

# CPU steps nest: the input of a CPU step can hold an earlier one

    Code
      explain(q)
    Output
      <cuplyr plan: optimized>
      CPU fallback: mutate() for qnorm(z)
        reason: `qnorm()` has no GPU implementation.
        columns read: all 6
        input (computed at the mutate() call):
          CPU fallback: mutate() for pnorm(x)
            reason: `pnorm()` has no GPU implementation.
            columns read: all 5
            input (computed at the mutate() call):
              table on the GPU
                columns read: all 4
          filter(z > 0)

# an eager table and a lazy table with nothing pending

    Code
      explain(tbl_gpu(df, lazy = FALSE))
    Output
      <cuplyr plan>
      No pending operations: an eager table runs each verb when it is called.
    Code
      explain(tbl_gpu(df, lazy = TRUE))
    Output
      <cuplyr plan: optimized>
      table on the GPU
        columns read: all 4
    Code
      show_query(tbl_gpu(df, lazy = TRUE), optimized = FALSE)
    Output
      <cuplyr plan: as written>
      table on the GPU
        columns read: all 4

