"""Metrics for binary forecasts.

Port of ``R/metrics-binary.R`` (scoringutils 2.3.0).

``observed`` is a pandas Categorical (the equivalent of an R factor) with
exactly two categories. ``predicted`` is the probability that the outcome is
the second (highest) category. With ``pd.Categorical(x, categories=[0, 1])``
that is the probability of a 1.
"""

from __future__ import annotations

import numpy as np

from ._checks import check_input_binary


def brier_score(observed, predicted):
    """Brier score, ``(predicted - outcome) ** 2`` with outcome 0 or 1.

    Lower is better: 0 is a perfect forecast, 1 the worst.
    """
    obs, pred = check_input_binary(observed, predicted)
    return (obs - pred) ** 2


def logs_binary(observed, predicted):
    """Log score for binary outcomes, ``-log(p)`` with p the probability
    assigned to what happened.

    Lower is better: 0 is a perfect forecast; a probability of 0 for the
    observed outcome gives ``inf``, as in R.
    """
    obs, pred = check_input_binary(observed, predicted)
    with np.errstate(divide="ignore"):
        return -np.log(1 - np.abs(obs - pred))
