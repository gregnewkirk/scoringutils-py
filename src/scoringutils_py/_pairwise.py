"""Pairwise comparisons and relative skill.

Port of ``R/pairwise-comparisons.R`` (scoringutils 2.3.0).

How relative skill works, in short: for every pair of models, take the
forecasts both made, and divide the sum of model A's scores by the sum of
model B's. A model's relative skill is the geometric mean of its ratios
against all models (including itself, ratio 1). Values below 1 mean better
than average. With a ``baseline``, relative skills are divided by the
baseline's, so the baseline gets 1.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._checks import inform
from ._forecast import get_forecast_unit
from ._rcompat import p_adjust_holm, wilcox_paired_pvalue
from ._score import Scores, _get_metrics_scores, new_scores

_DEFAULT_METRICS = ("wis", "crps", "brier_score")


def _as_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def get_pairwise_comparisons(
    scores: pd.DataFrame,
    compare: str = "model",
    by: str | Sequence[str] | None = None,
    metric: str | None = None,
    baseline: str | None = None,
    test_type: str | None = "non_parametric",
    one_sided: bool = False,
    n_permutations: int = 999,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Compare every pair of models (or other ``compare`` groups).

    Returns one row per ordered pair with ``mean_scores_ratio``, a p-value
    (``pval``, Wilcoxon signed rank by default, ``test_type="permutation"``
    or ``None`` for no test), Holm-adjusted ``adj_pval``, and
    ``<metric>_relative_skill`` (plus ``<metric>_scaled_relative_skill`` if a
    ``baseline`` is given). ``by`` runs the comparison separately per group.

    ``rng`` seeds the permutation test. Its random numbers differ from R's,
    so permutation p-values cannot match R exactly.
    """
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("Assertion on 'scores' failed: Must be of type 'data.frame'.")
    if compare not in scores.columns:
        raise ValueError(
            "Assertion on 'compare' failed: Must be a subset of "
            f"{{{','.join(repr(c) for c in scores.columns)}}}, but has additional "
            f"elements {{'{compare}'}}."
        )
    by = _as_list(by)
    if compare in by:
        raise ValueError(
            f"Assertion on 'by' failed: Must be disjunct from {{'{compare}'}}."
        )
    metrics = _get_metrics_scores(scores, error=True)
    if metric is None:
        metric_list = [m for m in _DEFAULT_METRICS if m in scores.columns]
    else:
        metric_list = [metric] if isinstance(metric, str) else list(metric)
    if not metric_list or any(m not in metrics for m in metric_list):
        raise ValueError(
            "Assertion on 'metric' failed: Must be a subset of "
            f"{{{','.join(repr(m) for m in metrics)}}}."
        )
    if len(metric_list) != 1:
        raise ValueError("Assertion on 'metric' failed: Must have length 1.")
    metric = metric_list[0]
    missing_by = [b for b in by if b not in scores.columns]
    if missing_by:
        raise ValueError(
            f"Not all columns specified in `by` are present: {', '.join(missing_by)}"
        )
    comparators = list(pd.unique(scores[compare]))
    if baseline is not None and baseline not in comparators:
        raise ValueError(
            "Assertion on 'baseline' failed: Must be a subset of "
            f"{{{','.join(repr(str(c)) for c in comparators)}}}, but has "
            f"additional elements {{'{baseline}'}}."
        )

    if scores[metric].isna().any():
        scores = scores[scores[metric].notna()]
        if len(scores) == 0:
            raise ValueError(
                f'After removing "NA" values for `{metric}`, no values were left.'
            )
        warnings.warn(
            f"Some values for the metric `{metric}` are NA. These have been "
            "removed. Maybe choose a different metric?",
            UserWarning,
            stacklevel=2,
        )
    values = scores[metric].to_numpy(dtype=float)
    if np.any(np.sign(values) < 0) and np.any(np.sign(values) > 0):
        raise ValueError(
            f"To compute pairwise comparisons, all values of `{metric}` must "
            "have the same sign."
        )

    forecast_unit = get_forecast_unit(scores)
    if set([compare]) == set(forecast_unit):
        warnings.warn(
            "`compare` is set to the unit of a single forecast. This doesn't "
            "look right. All relative skill scores will be equal to 1.",
            UserWarning,
            stacklevel=2,
        )
    elif set([compare] + by) == set(forecast_unit):
        inform(
            "relative skill can only be computed if the combination of "
            "`compare` and `by` is different from the unit of a single "
            "forecast. `by` was set to an empty character vector"
        )
        by = []

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if by:
        groups = [g for _, g in scores.groupby(by, sort=False, dropna=False)]
    else:
        groups = [scores]
    results = [
        _pairwise_one_group(
            g,
            metric=metric,
            baseline=baseline,
            compare=compare,
            by=by,
            test_type=test_type,
            one_sided=one_sided,
            n_permutations=n_permutations,
            rng=rng,
        )
        for g in groups
    ]
    return pd.concat(results, ignore_index=True)


def _pivot_scores(scores: pd.DataFrame, compare: str, metric: str) -> pd.DataFrame:
    forecast_unit = get_forecast_unit(scores)
    merge_by = [c for c in forecast_unit if c != compare]
    df = pd.DataFrame(scores).drop_duplicates()
    if df.duplicated(subset=forecast_unit).any():
        raise ValueError(
            "Found more than one score for the same forecast unit and element of "
            f"`{compare}`. Pairwise comparisons require exactly one score per "
            "forecast unit and comparator. Consider summarising the scores first "
            "using `summarise_scores()`."
        )
    df = df[merge_by + [compare, metric]].copy()
    df[compare] = df[compare].astype(str)
    if not merge_by:
        df["_all"] = 0
        merge_by = ["_all"]
    return df.pivot(index=merge_by, columns=compare, values=metric)


def compare_scores(
    values_x,
    values_y,
    one_sided: bool = False,
    test_type: str | None = "non_parametric",
    n_permutations: int = 999,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Ratio of summed scores plus a p-value for one pair of comparators."""
    values_x = np.asarray(values_x, dtype=float)
    values_y = np.asarray(values_y, dtype=float)
    ratio = np.sum(values_x) / np.sum(values_y)
    if test_type is None:
        pval = np.nan
    elif test_type == "permutation":
        pval = permutation_test(
            values_x,
            values_y,
            n_permutation=n_permutations,
            one_sided=one_sided,
            comparison_mode="difference",
            rng=rng,
        )
    elif test_type == "non_parametric":
        pval = wilcox_paired_pvalue(values_x, values_y)
    else:
        raise ValueError('\'arg\' should be one of "non_parametric", "permutation"')
    return {"mean_scores_ratio": float(ratio), "pval": float(pval)}


def permutation_test(
    scores1,
    scores2,
    n_permutation: int = 999,
    one_sided: bool = False,
    comparison_mode: str = "difference",
    rng: np.random.Generator | int | None = None,
) -> float:
    """Permutation test for paired scores (port of the internal R function)."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    s1 = np.asarray(scores1, dtype=float)
    s2 = np.asarray(scores2, dtype=float)
    n = s1.size
    m1, m2 = s1.mean(), s2.mean()
    if comparison_mode == "ratio":
        observed = m1 / m2 if one_sided else max(m1 / m2, m2 / m1)
    elif comparison_mode == "difference":
        observed = m1 - m2 if one_sided else abs(m1 - m2)
    else:
        raise ValueError("'comparison_mode' must be 'difference' or 'ratio'")
    permuted = np.empty(n_permutation)
    for k in range(n_permutation):
        sel = rng.integers(0, 2, size=n)
        g1 = (s1[sel == 0].sum() + s2[sel == 1].sum()) / n
        g2 = (s1[sel == 1].sum() + s2[sel == 0].sum()) / n
        if comparison_mode == "ratio":
            permuted[k] = g1 / g2 if one_sided else max(g1 / g2, g2 / g1)
        else:
            permuted[k] = g1 - g2 if one_sided else abs(g1 - g2)
    return float((1 + np.sum(permuted >= observed)) / (n_permutation + 1))


def _geometric_mean(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    with np.errstate(divide="ignore"):
        return float(np.exp(np.mean(np.log(x)))) if x.size else np.nan


def _pairwise_one_group(
    scores: pd.DataFrame,
    metric: str,
    baseline,
    compare: str,
    by: list[str],
    **test_args,
) -> pd.DataFrame:
    if compare not in scores.columns:
        raise ValueError("pairwise comparisons require a column as given by `compare`")
    comparators = [str(c) for c in pd.unique(scores[compare])]
    if len(comparators) < 2:
        raise ValueError("There are not enough comparators to do any comparison")

    pairs = [
        (comparators[i], comparators[j])
        for i in range(len(comparators))
        for j in range(i + 1, len(comparators))
    ]
    matrix = _pivot_scores(scores, compare=compare, metric=metric)
    ratios = np.full(len(pairs), np.nan)
    pvals = np.full(len(pairs), np.nan)
    for k, (a, b) in enumerate(pairs):
        x = matrix[a].to_numpy(dtype=float)
        y = matrix[b].to_numpy(dtype=float)
        overlap = ~np.isnan(x) & ~np.isnan(y)
        if not overlap.any():
            continue
        res = compare_scores(x[overlap], y[overlap], **test_args)
        ratios[k] = res["mean_scores_ratio"]
        pvals[k] = res["pval"]

    combos = pd.DataFrame(
        {
            "..compare": [p[0] for p in pairs],
            "compare_against": [p[1] for p in pairs],
            "ratio": ratios,
            "pval": pvals,
        }
    )
    combos = combos.sort_values("ratio", kind="stable", na_position="last")
    combos["adj_pval"] = p_adjust_holm(combos["pval"].to_numpy())
    mirrored = combos.rename(
        columns={"..compare": "compare_against", "compare_against": "..compare"}
    )
    with np.errstate(divide="ignore"):
        mirrored["ratio"] = 1 / mirrored["ratio"]
    equal = pd.DataFrame(
        {
            "..compare": comparators,
            "compare_against": comparators,
            "ratio": 1.0,
            "pval": 1.0,
            "adj_pval": 1.0,
        }
    )
    cols = ["..compare", "compare_against", "ratio", "pval", "adj_pval"]
    result = pd.concat([combos[cols], mirrored[cols], equal[cols]], ignore_index=True)
    result = result.rename(columns={"..compare": compare})
    theta = result.groupby(compare, sort=False)["ratio"].agg(
        lambda s: _geometric_mean(s.to_numpy())
    )
    result["theta"] = result[compare].map(theta)
    if baseline is not None:
        baseline_theta = theta.get(str(baseline))
        if baseline_theta is None:
            raise ValueError(f"Baseline comparator `{baseline}` missing.")
        result["rel_to_baseline"] = result["theta"] / baseline_theta

    keep = list(dict.fromkeys(by + [compare]))
    left = pd.DataFrame(scores)[keep].drop_duplicates().copy()
    left[compare] = left[compare].astype(str)
    out = left.merge(result, on=compare, how="outer", sort=False)
    out = out.sort_values(compare, kind="stable").reset_index(drop=True)
    out = out[[compare] + [c for c in out.columns if c != compare]]
    renames = {"ratio": "mean_scores_ratio", "theta": f"{metric}_relative_skill"}
    if baseline is not None:
        renames["rel_to_baseline"] = f"{metric}_scaled_relative_skill"
    return out.rename(columns=renames)


def add_relative_skill(
    scores: pd.DataFrame,
    compare: str = "model",
    by: str | Sequence[str] | None = None,
    metric: str | None = None,
    baseline: str | None = None,
    test_type: str | None = None,
    **kwargs,
) -> Scores:
    """Add ``<metric>_relative_skill`` (and scaled) columns to a scores table.

    Runs :func:`get_pairwise_comparisons` and merges the relative skill onto
    each row. No significance test is run by default.
    """
    pairwise = get_pairwise_comparisons(
        scores,
        compare=compare,
        by=by,
        metric=metric,
        baseline=baseline,
        test_type=test_type,
        **kwargs,
    )
    if metric is None:
        metric = next(m for m in _DEFAULT_METRICS if m in scores.columns)
    metrics = _get_metrics_scores(scores) or []
    pairwise = pairwise.drop(
        columns=["compare_against", "mean_scores_ratio", "pval", "adj_pval"]
    ).drop_duplicates()
    keys = get_forecast_unit(pairwise)
    left = pd.DataFrame(scores).copy()
    left[compare] = left[compare].astype(str)
    merged = left.merge(pairwise, how="left", on=keys)
    merged = merged.sort_values(keys, kind="stable").reset_index(drop=True)
    merged = merged[keys + [c for c in merged.columns if c not in keys]]
    new_metrics = [f"{metric}_{s}" for s in ("relative_skill", "scaled_relative_skill")]
    new_metrics = [m for m in new_metrics if m in merged.columns]
    return new_scores(merged, metrics + new_metrics)
