# Getting help with cuplyr

Thanks for using cuplyr!
Before filing an issue, there are a few places to explore and pieces to put together to make the process as smooth as possible.

## Make a reprex

Start by making a minimal **repr**oducible **ex**ample using the [reprex](https://reprex.tidyverse.org/) package.
If you haven't used reprex before, the [Get help!](https://www.tidyverse.org/help/) section of the tidyverse site explains it in a few minutes.
Build the data inside the example, and say whether the table is eager or lazy.

## Check your installation

Most first problems are installation problems. Run these and read the output:

```r
cuplyr::verify_installation()
cuplyr::diagnostics()
```

Paste the output of `diagnostics()` into any bug report.

## Where to ask?

Armed with your reprex, the next step is to figure out [where to ask](https://www.tidyverse.org/help/#where-to-ask).

*   If it's a question: start with [forum.posit.co](https://forum.posit.co/) or StackOverflow. There are more people there to answer questions.

*   If it's a bug: you're in the right place, [file an issue](https://github.com/bbtheo/cuplyr/issues/new).

*   If you're not sure: file an issue anyway and say so.

Before opening a new issue, [search issues and pull requests](https://github.com/bbtheo/cuplyr/issues) to make sure the bug hasn't been reported or already fixed in the development version.
The search starts with `is:issue is:open`. Remove `is:open` to search closed issues too.

## What happens next?

cuplyr has one maintainer, so a reply can take a few days.
A good reprex matters because the first thing the maintainer does is run it.
If we can't reproduce the bug, we can't fix it.
