"""Quantile <-> central prediction interval helpers (internal).

Port of ``R/helper-quantile-interval-range.R`` and ``R/metrics-interval-range.R``.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._checks import as_numeric


def get_range_from_quantile(quantile_level) -> np.ndarray:
    """Interval range (in percent) that a quantile level belongs to.

    0.05 and 0.95 both belong to the 90% central interval; 0.5 is range 0.
    """
    q = np.asarray(quantile_level, dtype=float)
    lower = q <= 0.5
    out = np.where(
        lower,
        np.round((1 - 2 * q) * 100, 10),
        np.round((2 * q - 1) * 100, 10),
    )
    return out


def quantile_to_interval_matrix(
    predicted: np.ndarray, quantile_level: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape an (n x N) quantile matrix into lower / upper interval bounds.

    Port of ``quantile_to_interval_numeric()`` (wide format). Returns
    ``(interval_range, lower, upper)`` where ``interval_range`` holds the K
    unique ranges in increasing order and ``lower`` / ``upper`` are (n x K)
    with NaN where a bound is missing. The median, if present, is range 0 with
    lower == upper.
    """
    ranges = get_range_from_quantile(quantile_level)
    is_lower = quantile_level <= 0.5
    has_lower = np.any(is_lower)
    # the median is added a second time as an "upper" boundary
    has_upper = np.any(~is_lower) or np.any(quantile_level == 0.5)
    if not (has_lower and has_upper):
        raise ValueError(
            "No valid forecast intervals found. A forecast interval comprises "
            "two quantiles with quantile levels symmetric around the median "
            "(e.g. 0.25 and 0.75)"
        )
    unique_ranges = np.unique(ranges[~np.isnan(ranges)])
    n = predicted.shape[0]
    k = unique_ranges.size
    lower = np.full((n, k), np.nan)
    upper = np.full((n, k), np.nan)
    pos = np.searchsorted(unique_ranges, ranges)
    for j in range(quantile_level.size):
        if np.isnan(ranges[j]):
            continue
        if is_lower[j]:
            lower[:, pos[j]] = predicted[:, j]
        if not is_lower[j] or quantile_level[j] == 0.5:
            upper[:, pos[j]] = predicted[:, j]
    return unique_ranges, lower, upper


_small_range_warned = False


def _check_input_interval(observed, lower, upper, interval_range) -> None:
    global _small_range_warned
    diff = upper - lower
    diff = diff[~np.isnan(diff)]
    if np.any(diff < 0):
        raise ValueError(
            "All values in `upper` need to be greater than or equal to the "
            "corresponding values in `lower`"
        )
    r = np.asarray(interval_range, dtype=float)
    if np.any((r > 0) & (r < 1)) and not _small_range_warned:
        _small_range_warned = True
        warnings.warn(
            "Found interval ranges between 0 and 1. Are you sure that's right? "
            "An interval range of 0.5 e.g. implies a (49.75%, 50.25%) prediction "
            "interval. If you want to score a (25%, 75%) prediction interval, set "
            "`interval_range = 50`.",
            UserWarning,
            stacklevel=3,
        )


def interval_score(
    observed,
    lower,
    upper,
    interval_range,
    weigh: bool = True,
    separate_results: bool = False,
):
    """Interval score of a central prediction interval.

    Port of the internal ``scoringutils:::interval_score()``. Works
    element-wise; ``interval_range`` (percent) broadcasts against the bounds.
    With ``weigh=True`` the score is multiplied by alpha / 2 so that the
    average over intervals is the WIS.
    """
    observed = as_numeric(observed, "observed").astype(float)
    lower = as_numeric(lower, "lower").astype(float)
    upper = as_numeric(upper, "upper").astype(float)
    interval_range = np.asarray(interval_range, dtype=float)
    _check_input_interval(observed, lower, upper, interval_range)
    alpha = (100 - interval_range) / 100
    with np.errstate(invalid="ignore", divide="ignore"):
        dispersion = upper - lower
        overprediction = (lower - observed) * (observed < lower)
        underprediction = (observed - upper) * (observed > upper)
        # NaN bounds must stay NaN (R: NA * FALSE is NA)
        overprediction = np.where(np.isnan(lower), np.nan, overprediction)
        underprediction = np.where(np.isnan(upper), np.nan, underprediction)
        if weigh:
            dispersion = dispersion * alpha / 2
        else:
            overprediction = np.where(
                overprediction == 0, 0.0, 2 / alpha * overprediction
            )
            underprediction = np.where(
                underprediction == 0, 0.0, 2 / alpha * underprediction
            )
        score = dispersion + underprediction + overprediction
    if separate_results:
        return {
            "interval_score": score,
            "dispersion": dispersion,
            "underprediction": underprediction,
            "overprediction": overprediction,
        }
    return score
