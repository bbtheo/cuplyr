# Getting help with cuplyr

Thanks for using cuplyr. Before you file an issue, work through the steps
below. They make the problem faster to fix.

## Make a reprex

Start with a minimal reproducible example, made with the
[reprex](https://reprex.tidyverse.org/) package. If you haven't used reprex
before, the [Get help!](https://www.tidyverse.org/help/) page on the tidyverse
site explains it in a few minutes. Build the data inside the example, and say
whether the table is eager or lazy.

## Check your installation

Most first problems are installation problems. Run these and read the output:

```r
cuplyr::verify_installation()
cuplyr::diagnostics()
```

Paste the output of `diagnostics()` into any bug report.

## Where to ask

With your reprex ready, decide [where to ask](https://www.tidyverse.org/help/#where-to-ask).

* Questions go to [forum.posit.co](https://forum.posit.co/) or Stack Overflow.
  More people can answer there.
* Bugs go here. [File an issue](https://github.com/bbtheo/cuplyr/issues/new).
* Not sure? File an issue anyway and say so.

Before you open a new issue, [search issues and pull requests](https://github.com/bbtheo/cuplyr/issues)
to check that the bug isn't already reported or fixed in the development
version. The search starts with `is:issue is:open`. Remove `is:open` to include
closed issues.

## What happens next

cuplyr has one maintainer, so a reply can take a few days. The maintainer
starts by running your reprex, and a bug that can't be reproduced can't be
fixed.
