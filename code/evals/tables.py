"""tables.py -- writing a DataFrame as markdown without depending on `tabulate`.

`DataFrame.to_markdown` is a one-liner that raises `ImportError: Missing optional
dependency 'tabulate'` inside the container, which does not have it. That is a bad failure
in a scoring script: it happens at the *end*, after every expensive thing has been
computed, so the job dies having thrown away whatever was still unwritten. It killed all
three `attribution.dag` jobs on 4 Sep 2026 after the GPU work had finished and the CSVs
were already on disk -- the markdown summary and the figure were lost and the jobs had to
be resubmitted for nothing.

Two scripts had each grown their own answer to this (`nongauss.py`'s `_md`, with the
fallback, and `delta_blocks.py`'s hand-rolled `to_markdown`, without needing one). This is
the shared version, so a third script does not have to rediscover it.

The markdown path is preferred when `tabulate` is present, because a markdown table is what
gets pasted into a document. The fallback is a fenced fixed-width block: worse to paste,
and much better than losing the run.
"""

from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = ".4f", empty: str = "_(nothing to report)_"):
    """`df` as a markdown table, degrading to a fenced fixed-width block.

    Never raises on a missing optional dependency, which is the whole point: callers write
    this last, after the CSVs, and a formatter is not worth a job.
    """
    if df is None or df.empty:
        return empty + "\n"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except ImportError:
        # A one-parameter callable, not the format string: `to_string(float_format="%.4f")`
        # is accepted by some pandas versions and not others, and this is the fallback
        # path -- it does not get to have a version problem of its own.
        def fmt(v):
            return ("%" + floatfmt) % v

        return "```\n" + df.to_string(index=False, float_format=fmt) + "\n```\n"
