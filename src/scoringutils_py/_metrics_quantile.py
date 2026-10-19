"""Metrics for quantile-based forecasts.

Port of ``R/metrics-quantile.R`` (scoringutils 2.3.0). Every function takes

- ``observed``: 1-D array of length n,
- ``predicted``: (n x N) matrix, one row per forecast (a 1-D vector is
  accepted when n == 1),
- ``quantile_level``: 1-D array of length N, shared by all rows,

and returns one value per forecast.
"""

from __future__ import annotations

import numpy as np

from ._checks import check_input_quantile, inform
from ._interval import (
    get_range_from_quantile,
    interval_score,
    quantile_to_interval_matrix,
)


def wis(
    observed,
    predicted,
    quantile_level,
    separate_results: bool = False,
    weigh: bool = True,
    count_median_twice: bool = False,
    na_rm: bool = False,
):
    """Weighted interval score (WIS).

    The WIS is a weighted average of the interval scores of the central
    prediction intervals that the quantiles form, plus the absolute error of
    the median (weight 0.5, or 1 with ``count_median_twice=True``). With the
    default ``weigh=True`` it approximates the CRPS.

    With ``separate_results=True`` returns a dict with ``wis``,
    ``dispersion``, ``underprediction`` and ``overprediction``; the three
    components add up to ``wis``.

    ``na_rm=True`` drops incomplete or missing intervals instead of failing
    (R argument ``na.rm``).
    """
    obs, pred, ql = check_input_quantile(observed, predicted, quantile_level)
    ranges, lower, upper = quantile_to_interval_matrix(pred, ql)

    non_median = ql != 0.5
    interval_ranges = get_range_from_quantile(ql[non_median])
    vals, counts = np.unique(interval_ranges, return_counts=True)
    complete = np.isin(interval_ranges, vals[counts > 1])
    if not np.all(complete) and not na_rm:
        incomplete = ql[non_median][~complete]
        raise ValueError(
            "Not all quantile levels specified form symmetric prediction "
            "intervals. The following quantile levels miss a corresponding "
            f"lower/upper bound: {', '.join(_fmt(q) for q in incomplete)}. "
            "You can drop incomplete prediction intervals using `na_rm = True`."
        )

    res = interval_score(
        obs[:, None],
        lower,
        upper,
        ranges[None, :],
        weigh=weigh,
        separate_results=True,
    )
    if count_median_twice:
        w = np.ones(ranges.size)
    else:
        w = np.where(ranges == 0, 0.5, 1.0)

    def wmean(x: np.ndarray) -> np.ndarray:
        return _weighted_row_mean(x, w, na_rm)

    if separate_results:
        return {
            "wis": wmean(res["interval_score"]),
            "dispersion": wmean(res["dispersion"]),
            "underprediction": wmean(res["underprediction"]),
            "overprediction": wmean(res["overprediction"]),
        }
    return wmean(res["interval_score"])


def _weighted_row_mean(x: np.ndarray, w: np.ndarray, na_rm: bool) -> np.ndarray:
    """Row-wise stats::weighted.mean(x, w, na.rm)."""
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        xi = x[i]
        wi = w
        if na_rm:
            keep = ~np.isnan(xi)
            xi = xi[keep]
            wi = wi[keep]
        nz = wi != 0
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.sum((xi * wi)[nz]) / np.sum(wi)
    return out


def _fmt(q: float) -> str:
    return f"{q:g}"


def dispersion_quantile(observed, predicted, quantile_level, **kwargs):
    """Dispersion component of the WIS (the width of the intervals)."""
    kwargs["separate_results"] = True
    return wis(observed, predicted, quantile_level, **kwargs)["dispersion"]


def overprediction_quantile(observed, predicted, quantile_level, **kwargs):
    """Overprediction component of the WIS (observation below the intervals)."""
    kwargs["separate_results"] = True
    return wis(observed, predicted, quantile_level, **kwargs)["overprediction"]


def underprediction_quantile(observed, predicted, quantile_level, **kwargs):
    """Underprediction component of the WIS (observation above the intervals)."""
    kwargs["separate_results"] = True
    return wis(observed, predicted, quantile_level, **kwargs)["underprediction"]


def interval_coverage(observed, predicted, quantile_level, interval_range: float = 50):
    """Whether the observation falls inside the central prediction interval.

    Returns a boolean array (float with NaN if any bound is missing).
    """
    obs, pred, ql = check_input_quantile(observed, predicted, quantile_level)
    r = np.asarray(interval_range, dtype=float)
    if r.size != 1:
        raise ValueError("Assertion on 'interval_range' failed: Must have length 1.")
    r = float(r.ravel()[0])
    necessary = np.array([(100 - r) / 2, 100 - (100 - r) / 2]) / 100
    if not np.all(np.isin(np.round(necessary, 10), np.round(ql, 10))):
        raise ValueError(
            "To compute the interval coverage for an interval range of "
            f'"{r:g}%", the {_fmt(necessary[0])} and {_fmt(necessary[1])} '
            "quantiles are required"
        )
    ranges, lower, upper = quantile_to_interval_matrix(pred, ql)
    j = np.flatnonzero(ranges == r)
    if j.size == 0:
        return np.zeros(0, dtype=bool)
    lo = lower[:, j[0]]
    up = upper[:, j[0]]
    covered = (obs >= lo) & (obs <= up)
    missing = np.isnan(lo) | np.isnan(up) | np.isnan(obs)
    if np.any(missing):
        out = covered.astype(float)
        out[missing] = np.nan
        return out
    return covered


def bias_quantile(observed, predicted, quantile_level, na_rm: bool = True):
    """Bias of a quantile forecast, between -1 (under) and 1 (over-prediction).

    If the median is missing it is interpolated from the two innermost
    quantiles (a message is emitted).
    """
    obs, pred, ql = check_input_quantile(observed, predicted, quantile_level)
    if not np.any(ql[~np.isnan(ql)] <= 0.5):
        raise ValueError(
            "Assertion on 'quantile_level[quantile_level <= 0.5]' failed: "
            "Must have length >= 1, but has length 0."
        )
    if not np.any(ql[~np.isnan(ql)] >= 0.5):
        raise ValueError(
            "Assertion on 'quantile_level[quantile_level >= 0.5]' failed: "
            "Must have length >= 1, but has length 0."
        )
    if not np.any(ql == 0.5):
        inform(
            "Median not available, interpolating median from the two innermost "
            "quantiles in order to compute bias."
        )
    return np.array(
        [_bias_quantile_single(obs[i], pred[i], ql, na_rm) for i in range(obs.size)],
        dtype=float,
    )


def _bias_quantile_single(observed, predicted, quantile_level, na_rm):
    if np.isnan(observed):
        raise ValueError("Assertion on 'observed' failed: May not be NA.")
    predicted = np.asarray(predicted, dtype=float)
    quantile_level = np.asarray(quantile_level, dtype=float)
    if np.any(np.isnan(predicted)) or np.any(np.isnan(quantile_level)):
        if not na_rm:
            return np.nan
        keep = ~np.isnan(predicted)
        quantile_level = quantile_level[keep]
        predicted = predicted[keep]
        keep = ~np.isnan(quantile_level)
        predicted = predicted[keep]
        quantile_level = quantile_level[keep]
        if not np.any(quantile_level <= 0.5) or not np.any(quantile_level >= 0.5):
            return np.nan
    order = np.argsort(quantile_level, kind="stable")
    predicted = predicted[order]
    quantile_level = quantile_level[order]
    if not np.all(np.diff(predicted) >= 0):
        raise ValueError(
            "Predictions must not be decreasing with increasing quantile level."
        )
    median_prediction = interpolate_median(predicted, quantile_level)
    if observed == median_prediction:
        return 0.0
    if observed < median_prediction:
        if observed < np.min(predicted):
            return 1.0
        q = np.max(quantile_level[predicted <= observed])
        return 1 - 2 * q
    if observed > np.max(predicted):
        return -1.0
    q = np.min(quantile_level[predicted >= observed])
    return 1 - 2 * q


def interpolate_median(predicted, quantile_level) -> float:
    """Median prediction, linearly interpolated if 0.5 is not a quantile level.

    Port of the internal ``interpolate_median()``, including its naming of the
    two innermost quantiles.
    """
    predicted = np.asarray(predicted, dtype=float)
    quantile_level = np.asarray(quantile_level, dtype=float)
    if np.any(quantile_level == 0.5):
        return float(predicted[quantile_level == 0.5][0])
    upper_q_level = np.max(quantile_level[quantile_level < 0.5])
    lower_q_level = np.min(quantile_level[quantile_level > 0.5])
    upper_value = predicted[quantile_level == upper_q_level][0]
    lower_value = predicted[quantile_level == lower_q_level][0]
    w = (0.5 - lower_q_level) / (upper_q_level - lower_q_level)
    return float(lower_value + w * (upper_value - lower_value))


def ae_median_quantile(observed, predicted, quantile_level):
    """Absolute error of the median (the 0.5 quantile must be present)."""
    obs, pred, ql = check_input_quantile(observed, predicted, quantile_level)
    if not np.any(ql == 0.5):
        raise ValueError(
            "In order to compute the absolute error of the median, 0.5 must be "
            "among the quantiles given"
        )
    med = pred[:, ql == 0.5]
    if med.shape[1] != 1:
        raise ValueError("More than one 0.5 quantile level given.")
    return np.abs(obs - med[:, 0])


def quantile_score(observed, predicted, quantile_level, weigh: bool = True):
    """Mean quantile score across quantile levels.

    Note: this is **twice** the textbook pinball loss (and twice what
    ``scoringrules.quantile_score`` returns), so that averaging quantile
    scores over symmetric quantile levels gives the WIS. This follows
    scoringutils. With ``weigh=False`` each score is rescaled to the
    unweighted interval score of its central interval.
    """
    obs, pred, ql = check_input_quantile(observed, predicted, quantile_level)
    obs_smaller_pred = obs[:, None] <= pred
    tau_diff = obs_smaller_pred - ql[None, :]
    error = pred - obs[:, None]
    score = 2 * tau_diff * error
    if weigh:
        return np.mean(score, axis=1)
    central_interval = np.abs(0.5 - ql) * 2
    alpha = 1 - central_interval
    with np.errstate(divide="ignore", invalid="ignore"):
        unweighted = 2 * score / alpha[None, :]
    unweighted[~np.isnan(score) & (score == 0)] = 0.0
    return np.mean(unweighted, axis=1)
