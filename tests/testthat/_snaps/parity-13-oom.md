# OOM error messages (snapshot)

    Code
      filter(d, x > 0.5)
    Condition
      Error in `filter()`:
      ! GPU operation `filter` ran out of device memory.
      i cuplyr freed unused memory (`gc()` and `gpu_gc()`) and retried once; the retry ran out too.
      i Device memory: 12 GB total, 1.5 GB free.
      i This session: 9 GB in use, 9.5 GB held by its "async" memory resource.
      i To use less memory: filter rows and select columns early; keep tables lazy (`options(cuplyr.exec_mode = "lazy")`, the default) so a plan drops the columns it never reads; remove tables you no longer need and call `gpu_gc()`.
      i See `vignette("memory", package = "cuplyr")`.
      Caused by error:
      ! std::bad_alloc: out_of_memory: injected out_of_memory (gpu_oom_inject)

---

    Code
      collect(l)
    Condition
      Error in `collect()`:
      ! GPU operation `compute` ran out of device memory.
      i cuplyr freed unused memory (`gc()` and `gpu_gc()`) and retried once; the retry ran out too.
      i Device memory: 12 GB total, 1.5 GB free.
      i This session: 9 GB in use, 9.5 GB held by its "async" memory resource.
      i To use less memory: filter rows and select columns early; keep tables lazy (`options(cuplyr.exec_mode = "lazy")`, the default) so a plan drops the columns it never reads; remove tables you no longer need and call `gpu_gc()`.
      i See `vignette("memory", package = "cuplyr")`.
      Caused by error:
      ! std::bad_alloc: out_of_memory: injected out_of_memory (gpu_oom_inject)

---

    Code
      filter(d, x > 0.5)
    Condition
      Error in `filter()`:
      ! GPU operation `filter` ran out of device memory.
      i cuplyr freed unused memory (`gc()` and `gpu_gc()`) and retried once; the retry ran out too.
      i Device memory: 12 GB total, 1.5 GB free.
      i This session: 9 GB in use, 9.5 GB held by its "pool" memory resource.
      i To use less memory: filter rows and select columns early; keep tables lazy (`options(cuplyr.exec_mode = "lazy")`, the default) so a plan drops the columns it never reads; remove tables you no longer need and call `gpu_gc()`.
      i The "pool" resource keeps all the memory it grew to and can fragment; the default "async" resource (`options(cuplyr.memory_resource = "async")`, read when cuplyr loads) may fit more.
      i See `vignette("memory", package = "cuplyr")`.
      Caused by error:
      ! std::bad_alloc: out_of_memory: injected out_of_memory (gpu_oom_inject)

---

    Code
      filter(d, x > 0.5)
    Condition
      Error in `filter()`:
      ! GPU operation `filter` ran out of device memory.
      i cuplyr freed unused memory (`gc()` and `gpu_gc()`) and retried once; the retry ran out too.
      i GPU memory state unavailable.
      i To use less memory: filter rows and select columns early; keep tables lazy (`options(cuplyr.exec_mode = "lazy")`, the default) so a plan drops the columns it never reads; remove tables you no longer need and call `gpu_gc()`.
      i See `vignette("memory", package = "cuplyr")`.
      Caused by error:
      ! std::bad_alloc: out_of_memory: injected out_of_memory (gpu_oom_inject)

