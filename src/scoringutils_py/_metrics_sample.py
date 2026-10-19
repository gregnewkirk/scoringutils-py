"""Metrics for sample-based forecasts.

Port of ``R/metrics-sample.R`` (scoringutils 2.3.0). Every function takes

- ``observed``: 1-D array of length n,
- ``predicted``: (n x m) matrix of m samples per forecast (a 1-D vector is
  accepted when n == 1),

and returns one value per forecast.

Where R scoringutils delegates to the R package scoringRules, this module
uses the Python package scoringrules with the settings that reproduce R
exactly (checked by the parity tests):

- CRPS: ``crps_ensemble(estimator="qd")`` equals R ``crps_sample(method = "edf")``
- log score: ``logs_ensemble(bw=bw.nrd)`` equals R ``logs_sample()``
- DSS: ``dssuv_ensemble(bias=True)`` equals R ``dss_sample()``
"""

from __future__ import annotations

import warnings

import numpy as np
import scoringrules as sr

from ._checks import check_input_sample
from ._rcompat import bw_nrd, get_type, r_mad


def bias_sample(observed, predicted):
    """Bias of a sample forecast, between -1 and 1.

    Uses the empirical CDF; for integer-valued predictions it uses the
    discrete version ``1 - (P(X <= y) + P(X <= y - 1))``.
    """
    obs, pred = check_input_sample(observed, predicted)
    n_pred = pred.shape[1]
    o = obs[:, None]
    if get_type(pred) == "continuous":
        p_lt = np.sum(pred < o, axis=1) / n_pred
        p_eq = np.sum(pred == o, axis=1) / n_pred
        return 1 - 2 * (p_lt + 0.5 * p_eq)
    p_x = np.sum(pred <= o, axis=1) / n_pred
    p_xm1 = np.sum(pred <= (o - 1), axis=1) / n_pred
    return 1 - (p_x + p_xm1)


def ae_median_sample(observed, predicted):
    """Absolute error of the sample median."""
    obs, pred = check_input_sample(observed, predicted)
    return np.abs(obs - np.median(pred, axis=1))


def se_mean_sample(observed, predicted):
    """Squared error of the sample mean."""
    obs, pred = check_input_sample(observed, predicted)
    return (obs - np.mean(pred, axis=1)) ** 2


def logs_sample(observed, predicted, bw=None):
    """Log score via Gaussian kernel density estimation.

    The bandwidth defaults to R's ``bw.nrd`` per forecast. Warns for
    integer-valued predictions, as R does.
    """
    obs, pred = check_input_sample(observed, predicted)
    if get_type(pred) == "integer":
        warnings.warn(
            "Predictions appear to be integer-valued. The log score uses kernel "
            "density estimation, which may not be appropriate for integer-valued "
            "forecasts. See the scoringrules package for alternatives for "
            "discrete probability distributions.",
            UserWarning,
            stacklevel=2,
        )
    pred = pred.astype(float)
    if bw is None:
        bw = np.array([bw_nrd(row) for row in pred])
    bw = np.broadcast_to(np.asarray(bw, dtype=float), obs.shape).copy()
    zero = bw == 0
    out = np.empty(obs.size)
    if np.any(~zero):
        # R returns Inf silently when the density underflows to 0
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            out[~zero] = sr.logs_ensemble(obs[~zero], pred[~zero], bw=bw[~zero])
    # A zero bandwidth (all samples equal) makes each kernel a point mass.
    # R then returns Inf (observation elsewhere) or -Inf (observation on it);
    # scoringrules returns NaN, so this case is handled here.
    for i in np.flatnonzero(zero):
        out[i] = -np.inf if np.any(pred[i] == obs[i]) else np.inf
    return out


def dss_sample(observed, predicted):
    """Dawid-Sebastiani score from the sample mean and (n - 1) variance."""
    obs, pred = check_input_sample(observed, predicted)
    pred = pred.astype(float)
    if pred.shape[1] < 2:
        return np.full(obs.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.asarray(sr.dssuv_ensemble(obs, pred, bias=True))


def crps_sample(observed, predicted, separate_results: bool = False):
    """CRPS of the empirical distribution of the samples.

    With ``separate_results=True`` returns a dict with ``crps``,
    ``dispersion`` (CRPS of the median against the samples),
    ``underprediction`` and ``overprediction``; components add up to ``crps``.
    """
    obs, pred = check_input_sample(observed, predicted)
    pred = pred.astype(float)
    crps = np.asarray(sr.crps_ensemble(obs, pred, estimator="qd"))
    if not separate_results:
        return crps
    medians = np.median(pred, axis=1)
    dispersion = np.asarray(sr.crps_ensemble(medians, pred, estimator="qd"))
    difference = crps - dispersion
    overprediction = np.where(obs < medians, difference, 0.0)
    underprediction = np.where(obs > medians, difference, 0.0)
    return {
        "crps": crps,
        "dispersion": dispersion,
        "underprediction": underprediction,
        "overprediction": overprediction,
    }


def dispersion_sample(observed, predicted):
    """Dispersion component of the CRPS."""
    return crps_sample(observed, predicted, separate_results=True)["dispersion"]


def overprediction_sample(observed, predicted):
    """Overprediction component of the CRPS."""
    return crps_sample(observed, predicted, separate_results=True)["overprediction"]


def underprediction_sample(observed, predicted):
    """Underprediction component of the CRPS."""
    return crps_sample(observed, predicted, separate_results=True)["underprediction"]


def mad_sample(observed=None, predicted=None):
    """Median absolute deviation of the samples (R ``stats::mad``, x 1.4826).

    ``observed`` is ignored; it is accepted so that the function has the same
    signature as the other sample metrics.
    """
    if predicted is None:
        raise TypeError('argument "predicted" is missing, with no default')
    pred = np.asarray(predicted)
    n = pred.shape[0] if pred.ndim == 2 else 1
    _, pred = check_input_sample(np.full(n, np.nan), pred)
    return np.array([r_mad(row) for row in pred.astype(float)])
