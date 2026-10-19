"""summarise_scores(). Port of ``R/summarise_scores.R``."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ._score import Scores, _get_metrics_scores, new_scores


def summarise_scores(
    scores: pd.DataFrame,
    by: str | Sequence[str] = "model",
    fun: Callable = np.mean,
    **kwargs,
) -> Scores:
    """Summarise scores by group (default: mean per model).

    ``fun`` receives a 1-D float numpy array per group and metric, plus any
    extra keyword arguments. Like R's ``mean``, ``np.mean`` returns NaN if a
    group contains NaN. Boolean score columns (interval coverage) are
    converted to 0 / 1 first. Note that ``np.std`` defaults to ``ddof=0``
    while R's ``sd`` uses n - 1.
    """
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("Assertion on 'scores' failed: Must be of type 'data.frame'.")
    if by is None:
        raise ValueError(
            "Assertion on 'by' failed: Must be a subset of the column names, "
            "not 'NULL'."
        )
    by = [by] if isinstance(by, str) else list(by)
    missing = [b for b in by if b not in scores.columns]
    if not by or missing:
        raise ValueError(
            "Assertion on 'by' failed: Must be a subset of "
            f"{{{','.join(repr(c) for c in scores.columns)}}}, but has additional "
            f"elements {{{','.join(repr(m) for m in missing)}}}."
        )
    if not callable(fun):
        raise TypeError("Assertion on 'fun' failed: Must be a function.")
    metrics = _get_metrics_scores(scores, error=True)
    by_metrics = [b for b in by if b in metrics]
    if by_metrics:
        quoted = " and ".join(f'"{b}"' for b in by_metrics)
        raise ValueError(
            f"Cannot summarise scores by a metric column: {quoted}. `by` must "
            "only contain columns that identify groups of forecasts, not the "
            f"score columns themselves. Remove {quoted} from `by`."
        )
    metric_cols = [c for c in scores.columns if c in metrics]
    if not metric_cols:
        raise ValueError(
            "No score columns to summarise. The scores object has no columns "
            "matching its `metrics` attribute. This usually means every metric "
            "passed to `score()` failed (e.g. warned and returned no values)."
        )

    df = pd.DataFrame(scores)[by + metric_cols].copy()
    for col in metric_cols:
        df[col] = df[col].to_numpy(dtype=float, na_value=np.nan)
    grouped = df.groupby(by, sort=False, dropna=False)
    out = grouped[metric_cols].agg(lambda s: fun(s.to_numpy(), **kwargs))
    out = out.reset_index()
    return new_scores(out, metrics)


summarize_scores = summarise_scores
