# Complex data analysis with cuplyr

This article works through longer pipelines on one synthetic sales
table: multi-step aggregations, several analyses branching off a shared
base, dates, joins, and reusable functions.

\
[`library`](https://rdrr.io/r/base/library.html)`(`[`cuplyr`](https://github.com/bbtheo/cuplyr)`)`\
[`library`](https://rdrr.io/r/base/library.html)`(`[`dplyr`](https://dplyr.tidyverse.org)`)`

## The data

Ten thousand transactions over one year.

\
[`set.seed`](https://rdrr.io/r/base/Random.html)`(``42``)`\
`n`` ``<-`` ``10000`\
\
`sales_data`` ``<-`` `[`data.frame`](https://rdrr.io/r/base/data.frame.html)`(`\
` transaction_id ``=`` `[`seq_len`](https://rdrr.io/r/base/seq.html)`(``n``)``,`\
` date ``=`` `[`as.Date`](https://rdrr.io/r/base/as.Date.html)`(``"2023-01-01"``)`` ``+`` `[`sample`](https://rdrr.io/r/base/sample.html)`(``0``:``364``, ``n``, replace ``=`` ``TRUE``)``,`\
` customer_id ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(``1``:``500``, ``n``, replace ``=`` ``TRUE``)``,`\
` product_category ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(`[`c`](https://rdrr.io/r/base/c.html)`(``"Electronics"``, ``"Clothing"``, ``"Food"``, ``"Home"``, ``"Sports"``)``,`\
`                           ``n``, replace ``=`` ``TRUE``)``,`\
` region ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(`[`c`](https://rdrr.io/r/base/c.html)`(``"North"``, ``"South"``, ``"East"``, ``"West"``)``, ``n``, replace ``=`` ``TRUE``)``,`\
` quantity ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(``1``:``10``, ``n``, replace ``=`` ``TRUE``)``,`\
` unit_price ``=`` `[`round`](https://rdrr.io/r/base/Round.html)`(`[`runif`](https://rdrr.io/r/stats/Uniform.html)`(``n``, ``5``, ``500``)``, ``2``)``,`\
` discount_pct ``=`` `[`sample`](https://rdrr.io/r/base/sample.html)`(`[`c`](https://rdrr.io/r/base/c.html)`(``0``, ``5``, ``10``, ``15``, ``20``)``, ``n``, replace ``=`` ``TRUE``,`\
`                       prob ``=`` `[`c`](https://rdrr.io/r/base/c.html)`(``0.5``, ``0.2``, ``0.15``, ``0.1``, ``0.05``)``)`\
`)`\
\
`gpu_sales`` ``<-`` `[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)`(``sales_data``)`

## A multi-step pipeline

Derive revenue, drop small transactions, then aggregate by category and
region. Nothing runs until
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

\
`revenue_analysis`` ``<-`` ``gpu_sales`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`   gross_amount ``=`` ``quantity`` ``*`` ``unit_price``,`\
`   discount_amount ``=`` ``gross_amount`` ``*`` ``discount_pct`` ``/`` ``100``,`\
`   net_revenue ``=`` ``gross_amount`` ``-`` ``discount_amount`\
` ``)`` ``|>`\
` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``net_revenue`` ``>`` ``50``)`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``product_category``, ``region``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`   total_revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``net_revenue``)``,`\
`   total_transactions ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``,`\
`   avg_transaction ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``net_revenue``)``,`\
`   total_quantity ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``quantity``)`\
` ``)`` ``|>`\
` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``total_revenue``)``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`` #> `summarise()` has regrouped the output by dropping the last grouping variable. ``\
`#> Summaries were computed grouped by product_category, region.`\
`#> Output is grouped by product_category.`\
`` #> Use `summarise(.groups = "drop_last")` to silence this message, or `.groups = "keep"`/`"drop"` to retain/drop every grouping level. ``\
\
[`head`](https://rdrr.io/r/utils/head.html)`(``revenue_analysis``, ``10``)`\
`#> ``# A tibble: 10 × 6`\
`#> ``# Groups:   product_category [5]`\
`#>    product_category region total_revenue total_transactions avg_transaction`\
`#>    ``<chr>``            ``<chr>``          ``<dbl>``              ``<int>``           ``<dbl>`\
`#> `` 1`` Sports           West         ``720``716.                531           ``1``357.`\
`#> `` 2`` Food             West         ``711``706.                517           ``1``377.`\
`#> `` 3`` Clothing         South        ``706``586.                509           ``1``388.`\
`#> `` 4`` Food             East         ``701``067.                514           ``1``364.`\
`#> `` 5`` Sports           East         ``698``596.                501           ``1``394.`\
`#> `` 6`` Sports           South        ``691``472.                507           ``1``364.`\
`#> `` 7`` Electronics      South        ``683``388.                528           ``1``294.`\
`#> `` 8`` Clothing         North        ``664``092.                487           ``1``364.`\
`#> `` 9`` Electronics      North        ``663``955.                504           ``1``317.`\
`#> ``10`` Home             East         ``663``267.                504           ``1``316.`\
`#> ``# ℹ 1 more variable: total_quantity <int>`

## Branching from a shared base

When several analyses start from the same prepared data, call
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) on the
base. It runs the pending pipeline once and keeps the result on the GPU,
so each branch starts from that result instead of repeating the work.

\
`base_analysis`` ``<-`` ``gpu_sales`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`   revenue ``=`` ``quantity`` ``*`` ``unit_price`` ``*`` ``(``1`` ``-`` ``discount_pct`` ``/`` ``100``)``,`\
`   is_high_value ``=`` ``unit_price`` ``>`` ``200``,`\
`   is_bulk ``=`` ``quantity`` ``>=`` ``5`\
` ``)`` ``|>`\
` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``revenue`` ``>`` ``0``)`` ``|>`\
` `[`compute`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`

Regional performance:

\
`base_analysis`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``region``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`   revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``,`\
`   orders ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``,`\
`   high_value_orders ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``is_high_value``)``,`\
`   bulk_orders ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``is_bulk``)`\
` ``)`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`   high_value_pct ``=`` ``high_value_orders`` ``*`` ``100`` ``/`` ``orders``,`\
`   bulk_pct ``=`` ``bulk_orders`` ``*`` ``100`` ``/`` ``orders`\
` ``)`` ``|>`\
` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``revenue``)``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`#> ``# A tibble: 4 × 7`\
`#>   region  revenue orders high_value_orders bulk_orders high_value_pct bulk_pct`\
`#>   ``<chr>``     ``<dbl>``  ``<int>``             ``<int>``       ``<int>``          ``<dbl>``    ``<dbl>`\
`#> ``1`` South  3``394``025.   ``2``538              ``1``547        ``1``551           61.0     61.1`\
`#> ``2`` East   3``296``018.   ``2``515              ``1``533        ``1``475           61.0     58.6`\
`#> ``3`` North  3``268``140.   ``2``505              ``1``536        ``1``448           61.3     57.8`\
`#> ``4`` West   3``234``758.   ``2``442              ``1``499        ``1``484           61.4     60.8`

Revenue per order by category:

\
`base_analysis`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``product_category``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`   revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``,`\
`   avg_price ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``unit_price``)``,`\
`   avg_quantity ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``quantity``)``,`\
`   orders ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)`\
` ``)`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``revenue_per_order ``=`` ``revenue`` ``/`` ``orders``)`` ``|>`\
` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``revenue_per_order``)``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`#> ``# A tibble: 5 × 6`\
`#>   product_category  revenue avg_price avg_quantity orders revenue_per_order`\
`#>   ``<chr>``               ``<dbl>``     ``<dbl>``        ``<dbl>``  ``<int>``             ``<dbl>`\
`#> ``1`` Food             2``700``522.      252.         5.51   ``2``021             ``1``336.`\
`#> ``2`` Home             2``588``937.      254.         5.51   ``1``954             ``1``325.`\
`#> ``3`` Sports           2``772``454.      253.         5.53   ``2``098             ``1``321.`\
`#> ``4`` Electronics      2``591``229.      253.         5.42   ``1``982             ``1``307.`\
`#> ``5`` Clothing         2``539``799.      254.         5.43   ``1``945             ``1``306.`

Share of revenue at each discount level:

\
`base_analysis`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``discount_pct``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`   orders ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``,`\
`   total_revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``,`\
`   avg_quantity ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``quantity``)`\
` ``)`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``revenue_share ``=`` ``total_revenue`` ``*`` ``100`` ``/`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``total_revenue``)``)`` ``|>`\
` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``discount_pct``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`#> ``# A tibble: 5 × 5`\
`#>   discount_pct orders total_revenue avg_quantity revenue_share`\
`#>          ``<dbl>``  ``<int>``         ``<dbl>``        ``<dbl>``         ``<dbl>`\
`#> ``1``            0   ``5``022      6``962``233.         5.48         52.8 `\
`#> ``2``            5   ``1``973      2``557``858.         5.40         19.4 `\
`#> ``3``           10   ``1``505      1``903``060.         5.54         14.4 `\
`#> ``4``           15   ``1``020      1``213``285.         5.55          9.20`\
`#> ``5``           20    480       ``556``506.         5.47          4.22`

## Dates

Date columns stay dates on the GPU, so you can filter on them directly.

\
`gpu_sales`` ``|>`\
` `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``date`` ``>=`` `[`as.Date`](https://rdrr.io/r/base/as.Date.html)`(``"2023-01-01"``)``, ``date`` ``<`` `[`as.Date`](https://rdrr.io/r/base/as.Date.html)`(``"2023-04-01"``)``)`` ``|>`\
` `[`count`](https://dplyr.tidyverse.org/reference/count.html)`(``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`#>      n`\
`#> 1 2458`

lubridate’s `month()` and `quarter()` also run on the GPU. Call them
with the `lubridate::` prefix or with lubridate attached.

\
`gpu_sales`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`   quarter ``=`` ``lubridate``::`[`quarter`](https://lubridate.tidyverse.org/reference/quarter.html)`(``date``)``,`\
`   revenue ``=`` ``quantity`` ``*`` ``unit_price`` ``*`` ``(``1`` ``-`` ``discount_pct`` ``/`` ``100``)`\
` ``)`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``quarter``, ``region``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(``revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``, orders ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``)`` ``|>`\
` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``quarter``, `[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``revenue``)``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`` #> `summarise()` has regrouped the output by dropping the last grouping variable. ``\
`#> Summaries were computed grouped by quarter, region.`\
`#> Output is grouped by quarter.`\
`` #> Use `summarise(.groups = "drop_last")` to silence this message, or `.groups = "keep"`/`"drop"` to retain/drop every grouping level. ``\
`#> ``# A tibble: 16 × 4`\
`#> ``# Groups:   quarter [4]`\
`#>    quarter region revenue orders`\
`#>      ``<int>`` ``<chr>``    ``<dbl>``  ``<int>`\
`#> `` 1``       1 South  ``830``856.    620`\
`#> `` 2``       1 West   ``823``113.    622`\
`#> `` 3``       1 North  ``803``029.    623`\
`#> `` 4``       1 East   ``742``242.    593`\
`#> `` 5``       2 South  ``891``756.    667`\
`#> `` 6``       2 North  ``805``111.    609`\
`#> `` 7``       2 East   ``803``270.    619`\
`#> `` 8``       2 West   ``776``899.    600`\
`#> `` 9``       3 East   ``867``393.    645`\
`#> ``10``       3 South  ``856``960.    639`\
`#> ``11``       3 North  ``812``817.    625`\
`#> ``12``       3 West   ``812``114.    631`\
`#> ``13``       4 East   ``883``113.    658`\
`#> ``14``       4 North  ``847``184.    648`\
`#> ``15``       4 West   ``822``632.    589`\
`#> ``16``       4 South  ``814``453.    612`

A half-year comparison uses
[`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html) to
label each row:

\
`gpu_sales`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`   period ``=`` `[`if_else`](https://dplyr.tidyverse.org/reference/if_else.html)`(``date`` ``<`` `[`as.Date`](https://rdrr.io/r/base/as.Date.html)`(``"2023-07-01"``)``, ``"H1"``, ``"H2"``)``,`\
`   revenue ``=`` ``quantity`` ``*`` ``unit_price`` ``*`` ``(``1`` ``-`` ``discount_pct`` ``/`` ``100``)`\
` ``)`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``period``, ``product_category``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`   avg_order ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``revenue``)``,`\
`   revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``,`\
`   orders ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)`\
` ``)`` ``|>`\
` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``product_category``, ``period``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`` #> `summarise()` has regrouped the output by dropping the last grouping variable. ``\
`#> Summaries were computed grouped by period, product_category.`\
`#> Output is grouped by period.`\
`` #> Use `summarise(.groups = "drop_last")` to silence this message, or `.groups = "keep"`/`"drop"` to retain/drop every grouping level. ``\
`#> ``# A tibble: 10 × 5`\
`#> ``# Groups:   period [2]`\
`#>    period product_category avg_order  revenue orders`\
`#>    ``<chr>``  ``<chr>``                ``<dbl>``    ``<dbl>``  ``<int>`\
`#> `` 1`` H1     Clothing             ``1``296. 1``188``439.    917`\
`#> `` 2`` H2     Clothing             ``1``315. 1``351``360.   ``1``028`\
`#> `` 3`` H1     Electronics          ``1``269. 1``251``712.    986`\
`#> `` 4`` H2     Electronics          ``1``345. 1``339``517.    996`\
`#> `` 5`` H1     Food                 ``1``370. 1``380``907.   ``1``008`\
`#> `` 6`` H2     Food                 ``1``303. 1``319``615.   ``1``013`\
`#> `` 7`` H1     Home                 ``1``301. 1``289``023.    991`\
`#> `` 8`` H2     Home                 ``1``350. 1``299``914.    963`\
`#> `` 9`` H1     Sports               ``1``300. 1``366``195.   ``1``051`\
`#> ``10`` H2     Sports               ``1``343. 1``406``259.   ``1``047`

`avg_order` comes before `revenue` because, as in dplyr, a later
expression in
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
sees the summarised value of an earlier one.

## Customer segments

Aggregate per customer, then label each one with
[`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html).
Both steps run on the GPU.

\
`customer_segments`` ``<-`` ``gpu_sales`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``revenue ``=`` ``quantity`` ``*`` ``unit_price`` ``*`` ``(``1`` ``-`` ``discount_pct`` ``/`` ``100``)``)`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``customer_id``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`   total_spent ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``,`\
`   order_count ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``,`\
`   avg_order_value ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``revenue``)`\
` ``)`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`   segment ``=`` `[`case_when`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)`(`\
`     ``total_spent`` ``>`` ``5000`` ``&`` ``order_count`` ``>`` ``20`` ``~`` ``"VIP"``,`\
`     ``total_spent`` ``>`` ``2000`` ``|`` ``order_count`` ``>`` ``10`` ``~`` ``"Regular"``,`\
`     ``order_count`` ``>`` ``5`` ``~`` ``"Occasional"``,`\
`     .default ``=`` ``"New"`\
`   ``)`\
` ``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
\
[`table`](https://rdrr.io/r/base/table.html)`(``customer_segments``$``segment``)`\
`#> `\
`#> Regular     VIP `\
`#>     283     217`

## Two passes with a join

To put each region’s revenue against its category total, aggregate once,
then join the totals back. Both tables are `tbl_gpu`, so the join runs
on the GPU too.

\
`with_revenue`` ``<-`` ``gpu_sales`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``revenue ``=`` ``quantity`` ``*`` ``unit_price`` ``*`` ``(``1`` ``-`` ``discount_pct`` ``/`` ``100``)``)`\
\
`category_totals`` ``<-`` ``with_revenue`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``product_category``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(``category_total ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``)`\
\
`with_revenue`` ``|>`\
` `[`left_join`](https://dplyr.tidyverse.org/reference/mutate-joins.html)`(``category_totals``, by ``=`` ``"product_category"``)`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``product_category``, ``region``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`   region_revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``,`\
`   category_total ``=`` `[`max`](https://rdrr.io/r/base/Extremes.html)`(``category_total``)`\
` ``)`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``region_share ``=`` ``region_revenue`` ``*`` ``100`` ``/`` ``category_total``)`` ``|>`\
` `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(``product_category``, `[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``region_share``)``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`` #> `summarise()` has regrouped the output by dropping the last grouping variable. ``\
`#> Summaries were computed grouped by product_category, region.`\
`#> Output is grouped by product_category.`\
`` #> Use `summarise(.groups = "drop_last")` to silence this message, or `.groups = "keep"`/`"drop"` to retain/drop every grouping level. ``\
`#> ``# A tibble: 20 × 5`\
`#> ``# Groups:   product_category [5]`\
`#>    product_category region region_revenue category_total region_share`\
`#>    ``<chr>``            ``<chr>``           ``<dbl>``          ``<dbl>``        ``<dbl>`\
`#> `` 1`` Clothing         South         ``706``925.       2``539``799.         27.8`\
`#> `` 2`` Clothing         North         ``664``283.       2``539``799.         26.2`\
`#> `` 3`` Clothing         East          ``649``561.       2``539``799.         25.6`\
`#> `` 4`` Clothing         West          ``519``030.       2``539``799.         20.4`\
`#> `` 5`` Electronics      South         ``683``738.       2``591``229.         26.4`\
`#> `` 6`` Electronics      North         ``664``510.       2``591``229.         25.6`\
`#> `` 7`` Electronics      West          ``660``805.       2``591``229.         25.5`\
`#> `` 8`` Electronics      East          ``582``175.       2``591``229.         22.5`\
`#> `` 9`` Food             West          ``712``050.       2``700``522.         26.4`\
`#> ``10`` Food             East          ``701``645.       2``700``522.         26.0`\
`#> ``11`` Food             South         ``659``368.       2``700``522.         24.4`\
`#> ``12`` Food             North         ``627``458.       2``700``522.         23.2`\
`#> ``13`` Home             East          ``663``610.       2``588``937.         25.6`\
`#> ``14`` Home             South         ``652``333.       2``588``937.         25.2`\
`#> ``15`` Home             North         ``651``189.       2``588``937.         25.2`\
`#> ``16`` Home             West          ``621``806.       2``588``937.         24.0`\
`#> ``17`` Sports           West          ``721``067.       2``772``454.         26.0`\
`#> ``18`` Sports           East          ``699``027.       2``772``454.         25.2`\
`#> ``19`` Sports           South         ``691``661.       2``772``454.         24.9`\
`#> ``20`` Sports           North         ``660``699.       2``772``454.         23.8`

## Reusable functions

A `tbl_gpu` goes through ordinary functions like any data frame. Pass
grouping columns through `...`.

\
`revenue_metrics`` ``<-`` ``function``(``data``, ``...``, ``min_value`` ``=`` ``0``)`` ``{`\
` ``data`` ``|>`\
`   `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`     gross ``=`` ``quantity`` ``*`` ``unit_price``,`\
`     net ``=`` ``gross`` ``*`` ``(``1`` ``-`` ``discount_pct`` ``/`` ``100``)`\
`   ``)`` ``|>`\
`   `[`filter`](https://dplyr.tidyverse.org/reference/filter.html)`(``net`` ``>=`` ``min_value``)`` ``|>`\
`   `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``...``)`` ``|>`\
`   `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(`\
`     gross_revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``gross``)``,`\
`     net_revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``net``)``,`\
`     order_count ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``,`\
`     avg_order ``=`` `[`mean`](https://rdrr.io/r/base/mean.html)`(``net``)`\
`   ``)`` ``|>`\
`   `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``discount_rate ``=`` ``(``gross_revenue`` ``-`` ``net_revenue``)`` ``*`` ``100`` ``/`` ``gross_revenue``)`` ``|>`\
`   `[`arrange`](https://dplyr.tidyverse.org/reference/arrange.html)`(`[`desc`](https://dplyr.tidyverse.org/reference/desc.html)`(``net_revenue``)``)`` ``|>`\
`   `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`}`\
\
`revenue_metrics``(``gpu_sales``, ``region``)`\
`#> ``# A tibble: 4 × 6`\
`#>   region gross_revenue net_revenue order_count avg_order discount_rate`\
`#>   ``<chr>``          ``<dbl>``       ``<dbl>``       ``<int>``     ``<dbl>``         ``<dbl>`\
`#> ``1`` South       3``570``728.    3``394``025.        ``2``538     ``1``337.          4.95`\
`#> ``2`` East        3``468``555.    3``296``018.        ``2``515     ``1``311.          4.97`\
`#> ``3`` North       3``438``415.    3``268``140.        ``2``505     ``1``305.          4.95`\
`#> ``4`` West        3``414``554.    3``234``758.        ``2``442     ``1``325.          5.27`\
`revenue_metrics``(``gpu_sales``, ``product_category``, min_value ``=`` ``500``)`\
`#> ``# A tibble: 5 × 6`\
`#>   product_category gross_revenue net_revenue order_count avg_order discount_rate`\
`#>   ``<chr>``                    ``<dbl>``       ``<dbl>``       ``<int>``     ``<dbl>``         ``<dbl>`\
`#> ``1`` Sports                2``752``146.    2``613``453.        ``1``459     ``1``791.          5.04`\
`#> ``2`` Food                  2``668``530.    2``542``690.        ``1``398     ``1``819.          4.72`\
`#> ``3`` Electronics           2``567``281.    2``444``307.        ``1``379     ``1``773.          4.79`\
`#> ``4`` Home                  2``576``501.    2``442``535.        ``1``387     ``1``761.          5.20`\
`#> ``5`` Clothing              2``529``950.    2``395``459.        ``1``362     ``1``759.          5.32`\
[`head`](https://rdrr.io/r/utils/head.html)`(``revenue_metrics``(``gpu_sales``, ``product_category``, ``region``)``, ``10``)`\
`` #> `summarise()` has regrouped the output by dropping the last grouping variable. ``\
`#> Summaries were computed grouped by product_category, region.`\
`#> Output is grouped by product_category.`\
`` #> Use `summarise(.groups = "drop_last")` to silence this message, or `.groups = "keep"`/`"drop"` to retain/drop every grouping level. ``\
`#> ``# A tibble: 10 × 7`\
`#> ``# Groups:   product_category [5]`\
`#>    product_category region gross_revenue net_revenue order_count avg_order`\
`#>    ``<chr>``            ``<chr>``          ``<dbl>``       ``<dbl>``       ``<int>``     ``<dbl>`\
`#> `` 1`` Sports           West         ``761``348.     ``721``067.         543     ``1``328.`\
`#> `` 2`` Food             West         ``749``524.     ``712``050.         530     ``1``343.`\
`#> `` 3`` Clothing         South        ``746``336.     ``706``925.         518     ``1``365.`\
`#> `` 4`` Food             East         ``734``231.     ``701``645.         533     ``1``316.`\
`#> `` 5`` Sports           East         ``734``639.     ``699``027.         517     ``1``352.`\
`#> `` 6`` Sports           South        ``728``101.     ``691``661.         516     ``1``340.`\
`#> `` 7`` Electronics      South        ``718``194.     ``683``738.         540     ``1``266.`\
`#> `` 8`` Electronics      North        ``695``830.     ``664``510.         521     ``1``275.`\
`#> `` 9`` Clothing         North        ``701``002.     ``664``283.         493     ``1``347.`\
`#> ``10`` Home             East         ``700``863.     ``663``610.         515     ``1``289.`\
`#> ``# ℹ 1 more variable: discount_rate <dbl>`

## Finishing in R

Once the GPU has reduced the data to a few dozen rows,
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) it and
carry on with any R code. Here the tiers use quantiles of the aggregated
revenue.

\
`aggregated`` ``<-`` ``gpu_sales`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(``revenue ``=`` ``quantity`` ``*`` ``unit_price`` ``*`` ``(``1`` ``-`` ``discount_pct`` ``/`` ``100``)``)`` ``|>`\
` `[`group_by`](https://dplyr.tidyverse.org/reference/group_by.html)`(``product_category``, ``region``, ``discount_pct``)`` ``|>`\
` `[`summarise`](https://dplyr.tidyverse.org/reference/summarise.html)`(``revenue ``=`` `[`sum`](https://rdrr.io/r/base/sum.html)`(``revenue``)``, orders ``=`` `[`n`](https://dplyr.tidyverse.org/reference/context.html)`(``)``)`` ``|>`\
` `[`collect`](https://dplyr.tidyverse.org/reference/compute.html)`(``)`\
`` #> `summarise()` has regrouped the output by dropping the last grouping variable. ``\
`#> Summaries were computed grouped by product_category, region, discount_pct.`\
`#> Output is grouped by product_category, region.`\
`` #> Use `summarise(.groups = "drop_last")` to silence this message, or `.groups = "keep"`/`"drop"` to retain/drop every grouping level. ``\
\
[`nrow`](https://rdrr.io/r/base/nrow.html)`(``aggregated``)`\
`#> [1] 100`\
\
`aggregated`` ``|>`\
` `[`ungroup`](https://dplyr.tidyverse.org/reference/group_by.html)`(``)`` ``|>`\
` `[`mutate`](https://dplyr.tidyverse.org/reference/mutate.html)`(`\
`   performance ``=`` `[`case_when`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)`(`\
`     ``revenue`` ``>`` `[`quantile`](https://rdrr.io/r/stats/quantile.html)`(``revenue``, ``0.9``)`` ``~`` ``"Top 10%"``,`\
`     ``revenue`` ``>`` `[`quantile`](https://rdrr.io/r/stats/quantile.html)`(``revenue``, ``0.5``)`` ``~`` ``"Above median"``,`\
`     ``revenue`` ``>`` `[`quantile`](https://rdrr.io/r/stats/quantile.html)`(``revenue``, ``0.25``)`` ``~`` ``"Below median"``,`\
`     .default ``=`` ``"Bottom 25%"`\
`   ``)`\
` ``)`` ``|>`\
` `[`count`](https://dplyr.tidyverse.org/reference/count.html)`(``performance``)`\
`#> ``# A tibble: 4 × 2`\
`#>   performance      n`\
`#>   ``<chr>``        ``<int>`\
`#> ``1`` Above median    40`\
`#> ``2`` Below median    25`\
`#> ``3`` Bottom 25%      25`\
`#> ``4`` Top 10%         10`

## Large data

On big tables, filter and
[`select()`](https://dplyr.tidyverse.org/reference/select.html) early so
fewer rows and columns reach the later steps. Check GPU memory with
[`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
and release unused memory with
[`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md):

\
[`gpu_memory_state`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)`(``)`\
`#> $available`\
`#> [1] TRUE`\
`#> `\
`#> $total_bytes`\
`#> [1] 12441747456`\
`#> `\
`#> $free_bytes`\
`#> [1] 11436294144`\
`#> `\
`#> $used_bytes`\
`#> [1] 1005453312`\
`#> `\
`#> $total_gb`\
`#> [1] 12.44175`\
`#> `\
`#> $free_gb`\
`#> [1] 11.43629`\
`#> `\
`#> $used_gb`\
`#> [1] 1.005453`\
`#> `\
`#> $resource`\
`#> [1] "async"`\
`#> `\
`#> $pool_bytes`\
`#> [1] 67108864`\
`#> `\
`#> $pool_used_bytes`\
`#> [1] 1124618`\
[`gpu_gc`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)`(``)`

On a table this small, plain dplyr is often faster because the data has
to reach the GPU first. The GPU pays off at millions of rows.

## See also

- [`vignette("getting-started")`](https://bbtheo.github.io/cuplyr/articles/getting-started.md)
  for the basics.
- [`vignette("query-optimization")`](https://bbtheo.github.io/cuplyr/articles/query-optimization.md)
  for what
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html) and
  the optimizer do.
- [`vignette("memory")`](https://bbtheo.github.io/cuplyr/articles/memory.md)
  and
  [`vignette("performance")`](https://bbtheo.github.io/cuplyr/articles/performance.md)
  for memory use and benchmarks.
